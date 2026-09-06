# live_driver_smirk_dual.py
# Two avatars, ONE driving signal: webcam -> SMIRK -> identical FLAME params -> both avatars.
#
# WHY THIS EXISTS: for the domain-matching comparison in the thesis we need to rule out the
# objection "the expression was simply different in the second take". Running the two avatars
# one after another cannot rule that out. Here both avatars receive the *same* parameter vector
# from the *same* frame at the *same* instant, so any difference in the result comes from the
# avatars themselves -- i.e. from the recording each one was built from.
#
# The encoder runs ONCE per frame; only the mesh update and rasterization happen twice.
#
# RUN (gaussian-avatars env, from the GaussianAvatars repo root):
#   python live_driver_smirk_dual.py ^
#       --point_path_a "output\<avatar>\point_cloud\iteration_30000\point_cloud.ply" ^
#       --point_path_b "output\<avatar>\point_cloud\iteration_30000\point_cloud.ply" ^
#       --label_a "spletna kamera" --label_b "telefon" ^
#       --smirk_root "..\smirk" --half --drive_head
#
# Keys: q quit | s SHOT (saves the figure panels) | r reset cam | +/- zoom
#       h head-pose driving | m mirror preview | 1/2 expr | 3/4 jaw | 5/6 blink | 7/8 gaze
#
# NOTE ON MIRRORING: the preview is mirrored so it feels like a mirror while you pose, but the
# SAVED camera panel is un-mirrored. SMIRK sees the un-mirrored frame, so the avatar's left/right
# matches the un-mirrored image -- saving the mirrored one would put the panels out of step.

import sys
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import cv2

from gaussian_renderer import FlameGaussianModel, render
from utils.viewer_utils import OrbitCamera

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

# reuse the single-avatar driver's helpers so the crop and camera stay identical
from live_driver_smirk import PipelineConfig, build_cam, similarity_crop


def load_avatar(point_path, device):
    """Load one trained avatar and everything that is specific to it."""
    g = FlameGaussianModel(sh_degree=3)
    g.load_ply(Path(point_path), has_target=False, motion_path=None, disable_fid=[])
    assert g.binding is not None, f"{point_path} is not FLAME-bound (train with --bind_to_mesh)."
    base_static = g.flame_param["static_offset"].to(device).clone()
    if base_static.dim() == 3:
        base_static = base_static[0]
    print(f"  {Path(point_path).parts[-4]}: {g._xyz.shape[0]} gaussians, "
          f"n_expr={g.n_expr}, verts={base_static.shape[0]}")
    return {"g": g, "base_static": base_static, "n_expr": g.n_expr,
            "num_verts": base_static.shape[0]}


def blank_param(av):
    """Per-avatar parameter dict. Identity (shape) stays inside the model itself."""
    return {
        'expr': torch.zeros(1, av["n_expr"]),
        'rotation': torch.zeros(1, 3),
        'neck': torch.zeros(1, 3),
        'jaw': torch.zeros(1, 3),
        'eyes': torch.zeros(1, 6),
        'translation': torch.zeros(1, 3),
        'static_offset': av["base_static"],
    }


def label_panel(img, text):
    """Caption strip under a panel, so the saved strip is self-explanatory."""
    strip = np.full((26, img.shape[1], 3), 255, np.uint8)
    cv2.putText(strip, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([img, strip])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--point_path_a", required=True, type=str, help="avatar A (left)")
    ap.add_argument("--point_path_b", required=True, type=str, help="avatar B (right)")
    ap.add_argument("--label_a", type=str, default="A")
    ap.add_argument("--label_b", type=str, default="B")
    ap.add_argument("--smirk_root", required=True, type=str)
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--projection", type=str, default="smirk_to_flame2023.npz")
    ap.add_argument("--task_path", type=str,
                    default=r"face_landmarker.task")
    ap.add_argument("--outdir", type=str, default="domena_shots",
                    help="where 's' saves the panels")
    ap.add_argument("--cam_id", type=int, default=0)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--cam_radius", type=float, default=1.0)
    ap.add_argument("--cam_fovy", type=float, default=20.0)
    ap.add_argument("--radius_b", type=float, default=None,
                    help="separate camera distance for B; use it when the two avatars differ in "
                         "scale, so both heads fill the frame equally")
    ap.add_argument("--smooth", type=float, default=0.6)
    ap.add_argument("--drive_head", action="store_true")
    ap.add_argument("--head_signs", type=str, default="1,-1,-1")
    ap.add_argument("--head_scale", type=float, default=1.0)
    ap.add_argument("--expr_scale", type=float, default=1.0)
    ap.add_argument("--jaw_scale", type=float, default=1.0)
    ap.add_argument("--blink_gain", type=float, default=1.0)
    ap.add_argument("--gaze_scale", type=float, default=0.5)
    ap.add_argument("--gaze_signs", type=str, default="1,1")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--mp_max_side", type=int, default=0)
    args = ap.parse_args()

    device = args.device
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---- SMIRK encoder (one instance drives both avatars) ----
    smirk_root = Path(args.smirk_root)
    sys.path.insert(0, str(smirk_root))
    from src.smirk_encoder import SmirkEncoder

    # NOTE: the checkpoint lives in pretrained_models/ -- live_driver_smirk.py defaults to
    # trained_models/, which is why that one always needs --checkpoint passed by hand.
    ckpt_path = Path(args.checkpoint) if args.checkpoint else smirk_root / "pretrained_models" / "SMIRK_em1.pt"
    print(f"Loading SMIRK encoder from {ckpt_path} ...")
    encoder = SmirkEncoder().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict({k.replace("smirk_encoder.", ""): v
                             for k, v in ckpt.items() if "smirk_encoder" in k})
    encoder.eval()
    del encoder.shape_encoder
    enc_dtype = torch.float16 if args.half else torch.float32
    if args.half:
        encoder.half()

    proj = np.load(args.projection)
    M_proj = torch.from_numpy(proj["M"]).float().to(device)          # [100, 50]

    l_eyelid = torch.from_numpy(np.load(smirk_root / "assets" / "l_eyelid.npy")).float().squeeze(0).to(device)
    r_eyelid = torch.from_numpy(np.load(smirk_root / "assets" / "r_eyelid.npy")).float().squeeze(0).to(device)

    # ---- both avatars ----
    print("Loading avatars ...")
    A = load_avatar(args.point_path_a, device)
    B = load_avatar(args.point_path_b, device)
    param = {"A": blank_param(A), "B": blank_param(B)}

    pipe = PipelineConfig()
    bg = torch.tensor([1.0, 1.0, 1.0]).cuda()
    camA = OrbitCamera(args.res, args.res, r=args.cam_radius, fovy=args.cam_fovy,
                       convention="opencv", save_path="live_camera_a.json")
    camB = OrbitCamera(args.res, args.res, r=args.radius_b if args.radius_b else args.cam_radius,
                       fovy=args.cam_fovy, convention="opencv", save_path="live_camera_b.json")

    base_options = mp_python.BaseOptions(model_asset_path=args.task_path)
    landmarker = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1))

    cap = cv2.VideoCapture(args.cam_id)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam {args.cam_id}")

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
    gsign = np.array([float(x) for x in args.gaze_signs.split(",")], dtype=np.float32)
    signs = np.array([float(x) for x in args.head_signs.split(",")], dtype=np.float32)
    shot = 0

    win = "Ujemanje domene - isti signal, dva avatarja"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, args.res * 3, args.res)
    print(f"Running. 's' saves a shot into {outdir.resolve()}, 'q' quits.")

    t0 = time.time()
    last = time.time()
    with torch.no_grad():
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            h, w = frame_bgr.shape[:2]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if args.mp_max_side and max(h, w) > args.mp_max_side:
                s = args.mp_max_side / max(h, w)
                mp_rgb = cv2.resize(frame_rgb, (int(w * s), int(h * s)))
            else:
                mp_rgb = frame_rgb
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(mp_rgb))
            result = landmarker.detect_for_video(mp_image, int((time.time() - t0) * 1000))

            have_face = bool(result.face_landmarks)
            if have_face:
                pts = np.array([[lm.x * w, lm.y * h] for lm in result.face_landmarks[0]],
                               dtype=np.float32)
                crop = similarity_crop(frame_rgb, pts, scale=1.4, image_size=224)
                inp = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).to(device).to(enc_dtype) / 255.0

                # ---- ONE encoder pass for both avatars ----
                out = {}
                out.update(encoder.pose_encoder(inp))
                out.update(encoder.expression_encoder(inp))
                exp50 = out['expression_params'][0].float().detach().cpu().numpy()
                jaw3 = out['jaw_params'][0].detach().cpu().numpy()
                pose3 = out['pose_params'][0].detach().cpu().numpy()
                eyel2 = out['eyelid_params'][0].detach().cpu().numpy()

                ema['expr'] = a * ema['expr'] + (1 - a) * exp50
                ema['jaw'] = a * ema['jaw'] + (1 - a) * jaw3
                ema['pose'] = a * ema['pose'] + (1 - a) * pose3
                ema['eyelid'] = a * ema['eyelid'] + (1 - a) * eyel2

                e100 = (M_proj @ torch.from_numpy(ema['expr']).float().to(device)) * expr_scale
                jaw_t = torch.from_numpy(ema['jaw'] * jaw_scale)
                rot_t = (torch.from_numpy(ema['pose'] * signs * args.head_scale)
                         if drive_head else torch.zeros(3))

                gaze_t = None
                if result.face_blendshapes:
                    bs = np.array([c.score for c in result.face_blendshapes[0]], dtype=np.float32)
                    pitchL = bs[17] - bs[11]; yawL = bs[15] - bs[13]
                    pitchR = bs[18] - bs[12]; yawR = bs[14] - bs[16]
                    gaze6 = np.array([gsign[0]*pitchL, gsign[1]*yawL, 0.0,
                                      gsign[0]*pitchR, gsign[1]*yawR, 0.0], dtype=np.float32)
                    ema['eyes'] = a * ema['eyes'] + (1 - a) * gaze6
                    gaze_t = torch.from_numpy(ema['eyes'] * gaze_scale)

                # ---- the SAME numbers into both avatars ----
                for key, av in (("A", A), ("B", B)):
                    p = param[key]
                    n = av["n_expr"]
                    p['expr'][0, :n] = e100[:n].cpu()
                    p['jaw'][0] = jaw_t
                    p['rotation'][0] = rot_t
                    if gaze_t is not None:
                        p['eyes'][0] = gaze_t
                    off = torch.zeros(av["num_verts"], 3, device=device)
                    ne = min(l_eyelid.shape[0], av["num_verts"])
                    off[:ne] = (l_eyelid[:ne] * float(ema['eyelid'][0]) +
                                r_eyelid[:ne] * float(ema['eyelid'][1])) * blink_gain
                    p['static_offset'] = av["base_static"] + off

            panels = []
            for key, av, orbit in (("A", A, camA), ("B", B, camB)):
                av["g"].update_mesh_by_param_dict(param[key])
                img = render(build_cam(orbit), av["g"], pipe, bg)["render"]
                img = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                panels.append(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

            cam_raw = cv2.resize(frame_bgr, (args.res, args.res))       # un-mirrored: what SMIRK saw
            cam_disp = cv2.flip(cam_raw, 1) if mirror else cam_raw
            combo = np.hstack([cam_disp] + panels)

            now = time.time()
            fps = 1.0 / max(now - last, 1e-6)
            last = now
            hud = (f"{fps:4.1f}FPS  A={args.label_a}  B={args.label_b}  "
                   f"head:{'on' if drive_head else 'off'} expr(1/2):{expr_scale:.2f} "
                   f"jaw(3/4):{jaw_scale:.2f} blink(5/6):{blink_gain:.2f} gaze(7/8):{gaze_scale:.2f}"
                   + ("" if have_face else "  [no face]"))
            cv2.putText(combo, hud, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(combo, "s = shrani", (8, combo.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imshow(win, combo)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('s'):
                shot += 1
                tag = f"domena_{shot:02d}"
                cv2.imwrite(str(outdir / f"{tag}_kamera.png"), cam_raw)
                cv2.imwrite(str(outdir / f"{tag}_a.png"), panels[0])
                cv2.imwrite(str(outdir / f"{tag}_b.png"), panels[1])
                strip = np.hstack([label_panel(cam_raw, "kamera"),
                                   label_panel(panels[0], args.label_a),
                                   label_panel(panels[1], args.label_b)])
                cv2.imwrite(str(outdir / f"{tag}_vse.png"), strip)
                print(f"  shranjeno: {tag}_kamera/_a/_b/_vse.png")
            elif k == ord('r'):
                camA.reset(); camB.reset()
            elif k in (ord('+'), ord('=')):
                camA.scale(1); camB.scale(1)
            elif k in (ord('-'), ord('_')):
                camA.scale(-1); camB.scale(-1)
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
    print(f"Konec. Posnetki so v {outdir.resolve()}")


if __name__ == "__main__":
    main()
