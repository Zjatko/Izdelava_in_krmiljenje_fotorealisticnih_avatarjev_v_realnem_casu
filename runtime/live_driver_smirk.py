# live_driver_smirk.py
# Live webcam -> SMIRK encoder -> FLAME params -> GaussianAvatars render.
#
# WHY THIS EXISTS: the original live_driver.py drives the avatar from MediaPipe ARKit
# blendshapes through a tiny linear MLP. That mapping is generic and noisy, so expressions
# look approximate and the mouth smears. SMIRK (CVPR 2024) instead REGRESSES FLAME parameters
# directly from the face image (analysis-by-neural-synthesis), giving far cleaner expression,
# jaw, head pose, AND real eyelid offsets. This driver feeds those into your avatar.
#
# PIPELINE
#   webcam frame
#     -> MediaPipe Face Landmarker (478 pts) -> similarity crop to 224x224
#     -> SMIRK encoder -> { expression_params(50), jaw_params(3), pose_params(3), eyelid_params(2) }
#     -> expression_50 (FLAME2020) projected to expression_100 (FLAME2023) via smirk_to_flame2023.npz
#     -> jaw/pose copied directly (shared FLAME joints)
#     -> eyelid offsets injected into FLAME static_offset  => REAL blink
#     -> FlameGaussianModel.update_mesh_by_param_dict(...) -> render
#
# SETUP (one time): see SMIRK_SETUP.md. In short:
#   1) clone SMIRK next to this repo, download its checkpoint + assets, place FLAME2020 generic_model.pkl
#   2) pip install timm           (into the gaussian-avatars env)
#   3) python smirk_precompute.py --flame2020 <generic_model.pkl>   -> smirk_to_flame2023.npz
#
# RUN (gaussian-avatars env + build env active, from GaussianAvatars repo root):
#   python live_driver_smirk.py ^
#       --point_path "output\<avatar>\point_cloud\iteration_30000\point_cloud.ply" ^
#       --smirk_root "..\smirk" --drive_head
#
# Keys: q quit | r reset cam | +/- zoom | h head-pose driving | m mirror
#       1/2 expr intensity | 3/4 jaw intensity | 5/6 blink gain | 7/8 gaze strength

import sys
import time
import argparse
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import cv2

# --- GaussianAvatars (run from repo root so these resolve) ---
from gaussian_renderer import FlameGaussianModel, render
from utils.viewer_utils import OrbitCamera

# --- MediaPipe Tasks API (for the face crop SMIRK expects) ---
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


@dataclass
class PipelineConfig:
    debug: bool = False
    compute_cov3D_python: bool = False
    convert_SHs_python: bool = False


def build_cam(orbit: OrbitCamera):
    """Lightweight camera object the rasterizer expects (mirrors local_viewer.prepare_camera)."""
    class Cam:
        FoVx = float(np.radians(orbit.fovx))
        FoVy = float(np.radians(orbit.fovy))
        image_height = orbit.image_height
        image_width = orbit.image_width
        world_view_transform = torch.tensor(orbit.world_view_transform).float().cuda().T
        full_proj_transform = torch.tensor(orbit.full_proj_transform).float().cuda().T
        camera_center = torch.tensor(orbit.pose[:3, 3]).float().cuda()
    return Cam


def similarity_crop(frame_rgb, landmarks_xy, scale=1.4, image_size=224):
    """Reproduce SMIRK's crop_face: tight box around landmarks, similarity-warped to 224.
    landmarks_xy: [N,2] pixel coords. Returns the 224x224 RGB crop."""
    left, right = landmarks_xy[:, 0].min(), landmarks_xy[:, 0].max()
    top, bottom = landmarks_xy[:, 1].min(), landmarks_xy[:, 1].max()
    old_size = (right - left + bottom - top) / 2.0
    cx = right - (right - left) / 2.0
    cy = bottom - (bottom - top) / 2.0
    size = old_size * scale
    src = np.array([[cx - size / 2, cy - size / 2],
                    [cx - size / 2, cy + size / 2],
                    [cx + size / 2, cy - size / 2]], dtype=np.float32)
    dst = np.array([[0, 0], [0, image_size - 1], [image_size - 1, 0]], dtype=np.float32)
    M = cv2.getAffineTransform(src, dst)
    crop = cv2.warpAffine(frame_rgb, M, (image_size, image_size), flags=cv2.INTER_LINEAR)
    return crop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--point_path", required=True, type=str, help="trained GaussianAvatars point_cloud.ply")
    ap.add_argument("--smirk_root", required=True, type=str, help="path to the cloned SMIRK repo")
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="SMIRK checkpoint (default: <smirk_root>/trained_models/SMIRK_em1.pt)")
    ap.add_argument("--projection", type=str, default="smirk_to_flame2023.npz",
                    help="expression projection from smirk_precompute.py")
    ap.add_argument("--task_path", type=str,
                    default=r"face_landmarker.task",
                    help="MediaPipe face_landmarker.task model file")
    ap.add_argument("--cam_id", type=int, default=0)
    ap.add_argument("--res", type=int, default=512, help="render resolution (square)")
    ap.add_argument("--cam_radius", type=float, default=1.0)
    ap.add_argument("--cam_fovy", type=float, default=20.0)
    ap.add_argument("--smooth", type=float, default=0.6, help="EMA smoothing 0..1 (higher=smoother/laggier)")
    ap.add_argument("--drive_head", action="store_true", help="drive head rotation from SMIRK pose")
    ap.add_argument("--head_signs", type=str, default="1,-1,-1",
                    help="per-axis sign for head pose (pitch,yaw,roll). Flip a sign if it turns wrong.")
    ap.add_argument("--head_scale", type=float, default=1.0,
                    help="pomnozi rotacijo glave (<1 omeji obrat, da ostane v opazovanem obsegu in se ne pokazejo luknje)")
    ap.add_argument("--expr_scale", type=float, default=1.0, help="scale projected expression")
    ap.add_argument("--jaw_scale", type=float, default=1.0, help="scale jaw")
    ap.add_argument("--blink_gain", type=float, default=1.0, help="scale eyelid offset (1.0 = SMIRK's full close)")
    ap.add_argument("--gaze_scale", type=float, default=0.5,
                    help="eye-gaze strength. SMIRK has no gaze, so we drive FLAME eyes from MediaPipe eyeLook blendshapes. 0=off")
    ap.add_argument("--gaze_signs", type=str, default="1,1",
                    help="sign for (pitch,yaw) gaze; flip one if the eyes look the wrong way")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--half", action="store_true", help="run the SMIRK encoder in fp16 (faster on GPU)")
    ap.add_argument("--profile", action="store_true", help="print per-frame timing split (encoder vs render)")
    ap.add_argument("--mp_max_side", type=int, default=0,
                    help="downscale the frame fed to MediaPipe landmarking to this max side (0=off). Speeds up landmark detection; crop still uses full-res pixels.")
    args = ap.parse_args()

    device = args.device

    # ---- SMIRK encoder ----
    smirk_root = Path(args.smirk_root)
    sys.path.insert(0, str(smirk_root))
    from src.smirk_encoder import SmirkEncoder  # only needs timm + torch (no FLAME/renderer)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else smirk_root / "trained_models" / "SMIRK_em1.pt"
    print(f"Loading SMIRK encoder from {ckpt_path} ...")
    encoder = SmirkEncoder().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    enc_sd = {k.replace("smirk_encoder.", ""): v for k, v in ckpt.items() if "smirk_encoder" in k}
    encoder.load_state_dict(enc_sd)
    encoder.eval()
    # We never use shape/identity (the avatar has its own identity), so don't run that backbone.
    del encoder.shape_encoder
    enc_dtype = torch.float16 if args.half else torch.float32
    if args.half:
        encoder.half()
    print(f"Encoder ready (dtype={enc_dtype}, shape-encoder skipped).")

    # ---- expression projection (FLAME2020 50 -> FLAME2023 100) ----
    proj = np.load(args.projection)
    M_proj = torch.from_numpy(proj["M"]).float().to(device)          # [100, 50]
    print(f"Loaded expression projection M {tuple(M_proj.shape)}")

    # ---- eyelid offset vectors (SMIRK assets, same FLAME topology) ----
    l_eyelid = np.load(smirk_root / "assets" / "l_eyelid.npy")        # [1,5023,3]
    r_eyelid = np.load(smirk_root / "assets" / "r_eyelid.npy")
    l_eyelid = torch.from_numpy(l_eyelid).float().squeeze(0).to(device)   # [5023,3]
    r_eyelid = torch.from_numpy(r_eyelid).float().squeeze(0).to(device)

    # ---- GaussianAvatars model ----
    print("Loading avatar ...")
    gaussians = FlameGaussianModel(sh_degree=3)
    gaussians.load_ply(Path(args.point_path), has_target=False, motion_path=None, disable_fid=[])
    assert gaussians.binding is not None, "This .ply is not FLAME-bound (train with --bind_to_mesh)."
    n_expr = gaussians.n_expr
    base_static = gaussians.flame_param["static_offset"].to(device).clone()   # [1,V,3] or [V,3]
    if base_static.dim() == 3:
        base_static = base_static[0]                                          # -> [V,3]
    num_verts = base_static.shape[0]
    n_eye = min(l_eyelid.shape[0], num_verts)
    print(f"Avatar: {gaussians._xyz.shape[0]} gaussians, n_expr={n_expr}, verts={num_verts}")
    print(f"CUDA available: {torch.cuda.is_available()} | encoder device: {next(encoder.parameters()).device}")

    pipe = PipelineConfig()
    bg = torch.tensor([1.0, 1.0, 1.0]).cuda()
    cam = OrbitCamera(args.res, args.res, r=args.cam_radius, fovy=args.cam_fovy,
                      convention="opencv", save_path="live_camera.json")

    flame_param = {
        'expr': torch.zeros(1, n_expr),
        'rotation': torch.zeros(1, 3),
        'neck': torch.zeros(1, 3),
        'jaw': torch.zeros(1, 3),
        'eyes': torch.zeros(1, 6),
        'translation': torch.zeros(1, 3),
        'static_offset': base_static,   # replaced per-frame with base + eyelid
    }

    # ---- MediaPipe landmarker (for the crop) ----
    base_options = mp_python.BaseOptions(model_asset_path=args.task_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,   # needed for eyeLook -> gaze (SMIRK has no gaze)
        output_facial_transformation_matrixes=False,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(args.cam_id)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam {args.cam_id}")

    # smoothing + live-tunable state
    ema = {'expr': np.zeros(50, np.float32), 'jaw': np.zeros(3, np.float32),
           'pose': np.zeros(3, np.float32), 'eyelid': np.zeros(2, np.float32),
           'eyes': np.zeros(6, np.float32)}
    a = float(args.smooth)
    drive_head = args.drive_head
    mirror = True
    expr_scale = float(args.expr_scale)
    jaw_scale = float(args.jaw_scale)
    blink_gain = float(args.blink_gain)
    gaze_scale = float(args.gaze_scale)
    gsign = np.array([float(x) for x in args.gaze_signs.split(",")], dtype=np.float32)  # (pitch,yaw)
    signs = np.array([float(x) for x in args.head_signs.split(",")], dtype=np.float32)

    win = "GaussianAvatars - Live (SMIRK)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, args.res * 2, args.res)
    print("Running. Focus the window and press keys. q to quit.")

    t0 = time.time()
    last = time.time()
    with torch.no_grad():
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            h, w = frame_bgr.shape[:2]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            # MediaPipe landmarks are normalized, so we can detect on a downscaled copy for speed;
            # the crop still uses full-res pixels (we multiply normalized coords by full w,h).
            if args.mp_max_side and max(h, w) > args.mp_max_side:
                s = args.mp_max_side / max(h, w)
                mp_rgb = cv2.resize(frame_rgb, (int(w * s), int(h * s)))
            else:
                mp_rgb = frame_rgb
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(mp_rgb))
            ts_ms = int((time.time() - t0) * 1000)
            result = landmarker.detect_for_video(mp_image, ts_ms)

            have_face = bool(result.face_landmarks)
            if have_face:
                lms = result.face_landmarks[0]
                pts = np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float32)  # [478,2]
                crop = similarity_crop(frame_rgb, pts, scale=1.4, image_size=224)

                inp = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).to(device).to(enc_dtype) / 255.0
                if args.profile:
                    torch.cuda.synchronize(); t_enc0 = time.perf_counter()
                # only the pose + expression backbones (shape/identity unused)
                out = {}
                out.update(encoder.pose_encoder(inp))
                out.update(encoder.expression_encoder(inp))
                if args.profile:
                    torch.cuda.synchronize(); enc_ms = (time.perf_counter() - t_enc0) * 1000
                exp50 = out['expression_params'][0].float().detach().cpu().numpy()    # (50,)
                jaw3 = out['jaw_params'][0].detach().cpu().numpy()            # (3,)
                pose3 = out['pose_params'][0].detach().cpu().numpy()          # (3,)
                eyel2 = out['eyelid_params'][0].detach().cpu().numpy()        # (2,) in 0..1

                # EMA smoothing
                ema['expr'] = a * ema['expr'] + (1 - a) * exp50
                ema['jaw'] = a * ema['jaw'] + (1 - a) * jaw3
                ema['pose'] = a * ema['pose'] + (1 - a) * pose3
                ema['eyelid'] = a * ema['eyelid'] + (1 - a) * eyel2

                # expression: project FLAME2020(50) -> FLAME2023(100)
                e50 = torch.from_numpy(ema['expr']).float().to(device)
                e100 = (M_proj @ e50) * expr_scale                            # [100]
                flame_param['expr'][0, :n_expr] = e100[:n_expr].cpu()

                # jaw: direct (shared joint); jaw[0] = opening
                flame_param['jaw'][0] = torch.from_numpy(ema['jaw'] * jaw_scale)

                # head pose
                if drive_head:
                    flame_param['rotation'][0] = torch.from_numpy(ema['pose'] * signs * args.head_scale)
                else:
                    flame_param['rotation'][0] = torch.zeros(3)

                # blink: inject eyelid vertex offsets into static_offset -> moves eyelid Gaussians
                eyelid_off = torch.zeros(num_verts, 3, device=device)
                eyelid_off[:n_eye] = (l_eyelid[:n_eye] * float(ema['eyelid'][0]) +
                                      r_eyelid[:n_eye] * float(ema['eyelid'][1])) * blink_gain
                flame_param['static_offset'] = base_static + eyelid_off

                # gaze: SMIRK has none, so derive it from MediaPipe eyeLook blendshapes and
                # drive the FLAME eye joints. Indices (52-blendshape order):
                #   11/12 lookDown L/R, 13/14 lookIn L/R, 15/16 lookOut L/R, 17/18 lookUp L/R
                if result.face_blendshapes:
                    bs = np.array([c.score for c in result.face_blendshapes[0]], dtype=np.float32)
                    pitchL = bs[17] - bs[11]; yawL = bs[15] - bs[13]   # up-down ; out-in (left eye)
                    pitchR = bs[18] - bs[12]; yawR = bs[14] - bs[16]   # up-down ; in-out (right eye)
                    gaze6 = np.array([gsign[0]*pitchL, gsign[1]*yawL, 0.0,
                                      gsign[0]*pitchR, gsign[1]*yawR, 0.0], dtype=np.float32)
                    ema['eyes'] = a * ema['eyes'] + (1 - a) * gaze6
                    flame_param['eyes'][0] = torch.from_numpy(ema['eyes'] * gaze_scale)

            if args.profile:
                torch.cuda.synchronize(); t_rnd0 = time.perf_counter()
            gaussians.update_mesh_by_param_dict(flame_param)

            # render
            Cam = build_cam(cam)
            img = render(Cam, gaussians, pipe, bg)["render"]
            img = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            if args.profile:
                torch.cuda.synchronize(); rnd_ms = (time.perf_counter() - t_rnd0) * 1000
                if have_face:
                    print(f"enc {enc_ms:5.1f} ms | mesh+render {rnd_ms:5.1f} ms")
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            cam_disp = cv2.resize(frame_bgr, (args.res, args.res))
            if mirror:
                cam_disp = cv2.flip(cam_disp, 1)
            combo = np.hstack([cam_disp, img_bgr])

            now = time.time()
            fps = 1.0 / max(now - last, 1e-6)
            last = now
            hud = (f"{fps:4.1f}FPS head:{'on' if drive_head else 'off'} "
                   f"expr(1/2):{expr_scale:.2f} jaw(3/4):{jaw_scale:.2f} "
                   f"blink(5/6):{blink_gain:.2f} gaze(7/8):{gaze_scale:.2f} dist(+/-):{cam.radius:.2f}"
                   + ("" if have_face else "  [no face]"))
            cv2.putText(combo, hud, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow(win, combo)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('r'):
                cam.reset()
            elif k in (ord('+'), ord('=')):
                cam.scale(1)
            elif k in (ord('-'), ord('_')):
                cam.scale(-1)
            elif k == ord('h'):
                drive_head = not drive_head
            elif k == ord('m'):
                mirror = not mirror
            elif k == ord('1'):
                expr_scale = max(0.0, round(expr_scale - 0.05, 2))
            elif k == ord('2'):
                expr_scale = min(2.0, round(expr_scale + 0.05, 2))
            elif k == ord('3'):
                jaw_scale = max(0.0, round(jaw_scale - 0.05, 2))
            elif k == ord('4'):
                jaw_scale = min(2.0, round(jaw_scale + 0.05, 2))
            elif k == ord('5'):
                blink_gain = max(0.0, round(blink_gain - 0.1, 2))
            elif k == ord('6'):
                blink_gain = min(3.0, round(blink_gain + 0.1, 2))
            elif k == ord('7'):
                gaze_scale = max(0.0, round(gaze_scale - 0.05, 2))
            elif k == ord('8'):
                gaze_scale = min(2.0, round(gaze_scale + 0.05, 2))

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
