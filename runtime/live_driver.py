# live_driver.py
# Live webcam -> MediaPipe blendshapes -> FLAME params -> GaussianAvatars render.
#
# Pipeline:
#   webcam frame (OpenCV)
#     -> MediaPipe Face Landmarker (52 blendshapes + head transform)
#     -> mediapipe-blendshapes-to-flame MLP (bs2exp/bs2jaw/bs2eye)
#     -> FLAME params {expr, jaw, eyes, (optional) head rotation}
#     -> FlameGaussianModel.update_mesh_by_param_dict(...)  (rig deforms)
#     -> render(...)  -> show in an OpenCV window in real time
#
# RUN FROM THE GaussianAvatars REPO ROOT, in the `gaussian-avatars` conda env WITH the
# build env active (nvdiffrast/rasterizer compile at runtime):
#
#   conda activate gaussian-avatars
#   . ".\setup\enter_build_env.ps1"
#   cd ".\GaussianAvatars"
#   pip install mediapipe opencv-python            # one-time, into this env
#   python live_driver.py --point_path "output\<avatar>\point_cloud\iteration_30000\point_cloud.ply"
#
# Keys (focus the avatar window):
#   q=quit  r=reset camera  +/-=zoom  h=toggle head-pose driving  m=toggle mirror
#   1/2 = expression intensity down/up   3/4 = jaw intensity down/up   (tune out the mouth smear live)
#   b=toggle BLINK-CALIBRATION  [ ]=change expr index  , .=change value  0=zero  s=lock as blink driver
#
# NOTE ON QUALITY: the MLP maps MediaPipe blendshapes -> a FLAME expression basis. jaw/eyes
# (pose) transfer cleanly; the 100-dim expression coefficients are an approximation (the MLP
# was fit to a FLAME expression space that may differ slightly from FLAME-2023), so expressions
# are indicative rather than exact. Good enough for a live demo of the rig.
#
# EYELID BLINK: neither MediaPipe's eyeBlink blendshapes nor FLAME's expression basis carry
# eyelid closing through the MLP, so blink never transfers on its own (verified: even the
# pretrained subject-306 avatar won't blink). We fix this by reading the eyeBlinkLeft/Right
# scores DIRECTLY and pushing a FLAME expression component that closes the lids. Which
# component does that differs per FLAME fit, so use calibration mode ('b') once to find it:
# scan with '[' / ']' until the eyes close, note the index (and whether + or - closes them),
# then either press 's' to lock it in live, or relaunch with --blink_idx <i> --blink_sign <+/-1>.

import sys
import time
import argparse
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import cv2
from scipy.spatial.transform import Rotation as R

# --- GaussianAvatars (run from repo root so these resolve) ---
from gaussian_renderer import FlameGaussianModel, render
from utils.viewer_utils import OrbitCamera

# --- MediaPipe Tasks API ---
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


@dataclass
class PipelineConfig:
    debug: bool = False
    compute_cov3D_python: bool = False
    convert_SHs_python: bool = False


def build_cam(orbit: OrbitCamera):
    """Build the lightweight camera object the gaussian rasterizer expects (mirrors
    local_viewer.prepare_camera)."""
    class Cam:
        FoVx = float(np.radians(orbit.fovx))
        FoVy = float(np.radians(orbit.fovy))
        image_height = orbit.image_height
        image_width = orbit.image_width
        world_view_transform = torch.tensor(orbit.world_view_transform).float().cuda().T
        full_proj_transform = torch.tensor(orbit.full_proj_transform).float().cuda().T
        camera_center = torch.tensor(orbit.pose[:3, 3]).float().cuda()
    return Cam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--point_path", required=True, type=str, help="trained GaussianAvatars point_cloud.ply")
    ap.add_argument("--task_path", type=str,
                    default=r"face_landmarker.task",
                    help="MediaPipe face_landmarker.task model file")
    ap.add_argument("--mappings_path", type=str,
                    default=r"../mediapipe-blendshapes-to-flame",
                    help="folder containing mp_2_flame.py and mappings/")
    ap.add_argument("--cam_id", type=int, default=0, help="webcam index")
    ap.add_argument("--res", type=int, default=512, help="render resolution (square)")
    ap.add_argument("--cam_radius", type=float, default=1.0)
    ap.add_argument("--cam_fovy", type=float, default=20.0)
    ap.add_argument("--smooth", type=float, default=0.5, help="EMA smoothing 0..1 (higher=smoother/laggier)")
    ap.add_argument("--drive_head", action="store_true", help="also drive head rotation from webcam (experimental)")
    ap.add_argument("--head_scale", type=float, default=1.0, help="scale on head rotation when --drive_head")
    ap.add_argument("--blink_idx", type=int, default=-1,
                    help="FLAME expression index that closes the eyelids (-1=off). Find it with calibration mode 'b'.")
    ap.add_argument("--blink_sign", type=float, default=1.0, help="sign for blink injection (+1 or -1)")
    ap.add_argument("--blink_gain", type=float, default=2.5, help="scale eyeBlink score (0..1) -> expr units")
    ap.add_argument("--expr_scale", type=float, default=0.8,
                    help="scale the MLP expression output (<1 = calmer, closer to local_viewer; tune live with 1/2)")
    ap.add_argument("--jaw_scale", type=float, default=0.8,
                    help="scale jaw opening (<1 stops the mouth over-opening into a smear; tune live with 3/4)")
    ap.add_argument("--eye_scale", type=float, default=1.0, help="scale eye-gaze pose")
    ap.add_argument("--expr_clamp", type=float, default=2.5,
                    help="clamp |expression coeff| to this (0=off); keeps the mesh inside the well-modeled range")
    args = ap.parse_args()

    # MediaPipe 52-blendshape order: index 9 = eyeBlinkLeft, 10 = eyeBlinkRight
    BLINK_L, BLINK_R = 9, 10

    # --- the blendshape -> FLAME MLP ---
    sys.path.append(args.mappings_path)
    from mp_2_flame import MP_2_FLAME
    mp2flame = MP_2_FLAME(mappings_path=str(Path(args.mappings_path) / "mappings"))

    # --- GaussianAvatars model ---
    print("Loading avatar...")
    gaussians = FlameGaussianModel(sh_degree=3)
    gaussians.load_ply(Path(args.point_path), has_target=False, motion_path=None, disable_fid=[])
    assert gaussians.binding is not None, "This .ply is not FLAME-bound (train with --bind_to_mesh)."
    n_expr = gaussians.n_expr
    print(f"Avatar loaded: {gaussians._xyz.shape[0]} gaussians, n_expr={n_expr}")

    pipe = PipelineConfig()
    bg = torch.tensor([1.0, 1.0, 1.0]).cuda()  # white background

    cam = OrbitCamera(args.res, args.res, r=args.cam_radius, fovy=args.cam_fovy, convention="opencv",
                      save_path="live_camera.json")

    # neutral FLAME param dict (CPU tensors, like local_viewer)
    flame_param = {
        'expr': torch.zeros(1, n_expr),
        'rotation': torch.zeros(1, 3),
        'neck': torch.zeros(1, 3),
        'jaw': torch.zeros(1, 3),
        'eyes': torch.zeros(1, 6),
        'translation': torch.zeros(1, 3),
    }

    # --- MediaPipe Face Landmarker (video mode) ---
    base_options = mp_python.BaseOptions(model_asset_path=args.task_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(args.cam_id)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam {args.cam_id}")

    # smoothing state
    ema = {'expr': np.zeros(n_expr, np.float32), 'jaw': np.zeros(3, np.float32), 'eyes': np.zeros(6, np.float32)}
    ema_blink = 0.0
    a = float(args.smooth)
    drive_head = args.drive_head
    mirror = True

    # live-tunable intensity (keys 1/2 = expr, 3/4 = jaw)
    expr_scale = float(args.expr_scale)
    jaw_scale = float(args.jaw_scale)
    eye_scale = float(args.eye_scale)

    # blink-calibration state (press 'b' to toggle)
    calib = False
    calib_idx = 0
    calib_val = 0.0

    win = "GaussianAvatars - Live (q quit, r reset, +/- zoom, h head, m mirror)"
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
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int((time.time() - t0) * 1000)
            result = landmarker.detect_for_video(mp_image, ts_ms)

            if result.face_blendshapes:
                scores = np.array([c.score for c in result.face_blendshapes[0]], dtype=np.float32)  # (52,)
                exp, jaw, eye = mp2flame.convert(scores[None])  # (1,100),(1,3),(1,6)
                exp, jaw, eye = exp[0], jaw[0], eye[0]

                # Tame the MLP output toward the avatar's clean, in-distribution range.
                # The webcam->MLP mapping overshoots vs local_viewer's tracked params, which
                # pushes the mesh outside what it was trained on (the mouth/chin smear).
                exp = exp * expr_scale
                jaw = jaw * jaw_scale
                eye = eye * eye_scale
                if args.expr_clamp > 0:
                    exp = np.clip(exp, -args.expr_clamp, args.expr_clamp)

                # EMA smoothing to reduce jitter
                ema['expr'] = a * ema['expr'] + (1 - a) * exp[:n_expr]
                ema['jaw'] = a * ema['jaw'] + (1 - a) * jaw
                ema['eyes'] = a * ema['eyes'] + (1 - a) * eye

                flame_param['expr'][0, :len(ema['expr'])] = torch.from_numpy(ema['expr'])
                flame_param['jaw'][0] = torch.from_numpy(ema['jaw'])
                flame_param['eyes'][0] = torch.from_numpy(ema['eyes'])

                # --- explicit eyelid blink injection ---
                # The MLP / FLAME expression basis do NOT carry eyelid closing, so blink never
                # transfers. Read MediaPipe's eyeBlink scores directly and push the FLAME
                # expression component that closes the lids (index found via calibration 'b').
                blink_raw = float(max(scores[BLINK_L], scores[BLINK_R]))
                ema_blink = a * ema_blink + (1 - a) * blink_raw
                if args.blink_idx >= 0 and not calib:
                    flame_param['expr'][0, args.blink_idx] += args.blink_sign * args.blink_gain * ema_blink

                if drive_head and result.facial_transformation_matrixes:
                    M = np.array(result.facial_transformation_matrixes[0])  # 4x4 head pose
                    rotvec = R.from_matrix(M[:3, :3]).as_rotvec().astype(np.float32) * args.head_scale
                    # Drive the FLAME ROOT rotation (full head turn). MediaPipe cam frame -> FLAME
                    # needs per-axis sign flips; these are THE tweak points if it turns the wrong way:
                    # sx=pitch(look up/down), sy=yaw(turn left/right), sz=roll(tilt).
                    # Flip a sign to invert that axis if it goes the wrong way:
                    sx, sy, sz = 1.0, -1.0, 1.0
                    flame_param['rotation'][0] = torch.tensor(
                        [sx * rotvec[0], sy * rotvec[1], sz * rotvec[2]], dtype=torch.float32)
                else:
                    flame_param['rotation'][0] = torch.zeros(3)

            # --- blink calibration override ---
            # Isolate ONE expression component on a neutral face so you can SEE which index
            # closes the eyes on YOUR avatar. '[' / ']' change the index, ',' / '.' the value.
            # When the eyes shut, press 's' to lock it in as the live blink driver.
            if calib:
                flame_param['expr'].zero_()
                flame_param['jaw'].zero_()
                flame_param['eyes'].zero_()
                flame_param['rotation'].zero_()
                flame_param['expr'][0, calib_idx] = calib_val

            gaussians.update_mesh_by_param_dict(flame_param)

            # render the avatar
            Cam = build_cam(cam)
            img = render(Cam, gaussians, pipe, bg)["render"]  # (3,H,W)
            img = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # side-by-side: webcam | avatar
            cam_disp = cv2.resize(frame_bgr, (args.res, args.res))
            if mirror:
                cam_disp = cv2.flip(cam_disp, 1)
            combo = np.hstack([cam_disp, img_bgr])

            now = time.time()
            fps = 1.0 / max(now - last, 1e-6)
            last = now
            if calib:
                hud = f"CALIB  expr[{calib_idx}] = {calib_val:+.1f}   ([ ] idx   , . val   0 zero   s lock   b exit)"
            else:
                hud = (f"{fps:4.1f}FPS head:{'on' if drive_head else 'off'} blink:{args.blink_idx} "
                       f"expr(1/2):{expr_scale:.2f} jaw(3/4):{jaw_scale:.2f}")
            cv2.putText(combo, hud, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
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
            elif k == ord('b'):
                calib = not calib
                if calib:
                    print("CALIB ON: '[' ']' change expr index, ',' '.' change value, '0' zero, 's' lock, 'b' exit")
            elif k == ord('['):
                calib_idx = max(0, calib_idx - 1)
            elif k == ord(']'):
                calib_idx = min(n_expr - 1, calib_idx + 1)
            elif k in (ord('.'), ord('>')):
                calib_val += 0.5
            elif k in (ord(','), ord('<')):
                calib_val -= 0.5
            elif k == ord('0'):
                calib_val = 0.0
            elif k == ord('1'):
                expr_scale = max(0.0, round(expr_scale - 0.05, 2))
            elif k == ord('2'):
                expr_scale = min(2.0, round(expr_scale + 0.05, 2))
            elif k == ord('3'):
                jaw_scale = max(0.0, round(jaw_scale - 0.05, 2))
            elif k == ord('4'):
                jaw_scale = min(2.0, round(jaw_scale + 0.05, 2))
            elif k == ord('s'):
                # lock the current calibration component as the live blink driver
                args.blink_idx = calib_idx
                args.blink_sign = 1.0 if calib_val >= 0 else -1.0
                print(f"blink driver locked: --blink_idx {calib_idx} --blink_sign {args.blink_sign:+.0f}")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()


if __name__ == "__main__":
    main()
