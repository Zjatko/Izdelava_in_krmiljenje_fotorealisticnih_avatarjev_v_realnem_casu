# compare_drivers.py
# Primerjava obeh gonilnikov na ISTI vhodni slicici, drug ob drugem:
#   [ webcam | MediaPipe-MLP (live_driver.py) | SMIRK (live_driver_smirk.py) ]
# Namenjeno za sliko v diplomi -- razlika pri ocesih (mezik).
#
# Zajem: mezikni, nato pritisni 's' -> shrani trojico kot PNG.
# Tipke: s = shrani sliko | q = izhod
#
# Zazeni iz mape GaussianAvatars (okolje gaussian-avatars + build okolje):
#   python compare_drivers.py --point_path "output\<avatar>\point_cloud\iteration_30000\point_cloud.ply" ^
#       --smirk_root "..\smirk" --checkpoint "..\smirk\pretrained_models\SMIRK_em1.pt" --half

import os
import sys
import time
import argparse
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import torch
import cv2

from gaussian_renderer import FlameGaussianModel, render
from utils.viewer_utils import OrbitCamera

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


@dataclass
class PipelineConfig:
    debug: bool = False
    compute_cov3D_python: bool = False
    convert_SHs_python: bool = False


def build_cam(orbit):
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
    left, right = landmarks_xy[:, 0].min(), landmarks_xy[:, 0].max()
    top, bottom = landmarks_xy[:, 1].min(), landmarks_xy[:, 1].max()
    old = (right - left + bottom - top) / 2.0
    cx = right - (right - left) / 2.0
    cy = bottom - (bottom - top) / 2.0
    size = old * scale
    src = np.array([[cx-size/2, cy-size/2], [cx-size/2, cy+size/2], [cx+size/2, cy-size/2]], np.float32)
    dst = np.array([[0, 0], [0, image_size-1], [image_size-1, 0]], np.float32)
    M = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(frame_rgb, M, (image_size, image_size), flags=cv2.INTER_LINEAR)


def neutral_param(n_expr, base_static):
    return {
        'expr': torch.zeros(1, n_expr), 'rotation': torch.zeros(1, 3),
        'neck': torch.zeros(1, 3), 'jaw': torch.zeros(1, 3),
        'eyes': torch.zeros(1, 6), 'translation': torch.zeros(1, 3),
        'static_offset': base_static,
    }


def render_avatar(gaussians, flame_param, cam, pipe, bg):
    gaussians.update_mesh_by_param_dict(flame_param)
    img = render(build_cam(cam), gaussians, pipe, bg)["render"]
    img = (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--point_path", required=True)
    ap.add_argument("--smirk_root", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--projection", default="smirk_to_flame2023.npz")
    ap.add_argument("--mappings_path", default=r"../mediapipe-blendshapes-to-flame")
    ap.add_argument("--task_path", default=r"face_landmarker.task")
    ap.add_argument("--cam_id", type=int, default=0)
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--cam_radius", type=float, default=1.0)
    ap.add_argument("--cam_fovy", type=float, default=20.0)
    ap.add_argument("--out_dir", default="primerjava_slike")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smooth", type=float, default=0.6,
                    help="EMA glajenje SMIRK 0..1 (visje=bolj gladko/z zakasnitvijo)")
    ap.add_argument("--gaze_scale", type=float, default=0.5,
                    help="jakost pogleda (hibridno: pogled iz MediaPipe). 0=izklop")
    ap.add_argument("--gaze_signs", type=str, default="1,1", help="predznak (pitch,yaw) pogleda")
    args = ap.parse_args()
    device = args.device
    BLINK_L, BLINK_R = 9, 10

    # --- MLP (MP_2_FLAME) ---
    sys.path.append(args.mappings_path)
    from mp_2_flame import MP_2_FLAME
    mp2flame = MP_2_FLAME(mappings_path=str(Path(args.mappings_path) / "mappings"))

    # --- SMIRK encoder ---
    smirk_root = Path(args.smirk_root)
    sys.path.insert(0, str(smirk_root))
    from src.smirk_encoder import SmirkEncoder
    ckpt = Path(args.checkpoint) if args.checkpoint else smirk_root / "trained_models" / "SMIRK_em1.pt"
    encoder = SmirkEncoder().to(device)
    sd = torch.load(ckpt, map_location=device)
    encoder.load_state_dict({k.replace("smirk_encoder.", ""): v for k, v in sd.items() if "smirk_encoder" in k})
    encoder.eval()
    del encoder.shape_encoder
    enc_dtype = torch.float16 if args.half else torch.float32
    if args.half:
        encoder.half()

    M_proj = torch.from_numpy(np.load(args.projection)["M"]).float().to(device)
    l_eyelid = torch.from_numpy(np.load(smirk_root/"assets"/"l_eyelid.npy")).float().squeeze(0).to(device)
    r_eyelid = torch.from_numpy(np.load(smirk_root/"assets"/"r_eyelid.npy")).float().squeeze(0).to(device)

    # --- avatar ---
    print("Nalagam avatar ...")
    gaussians = FlameGaussianModel(sh_degree=3)
    gaussians.load_ply(Path(args.point_path), has_target=False, motion_path=None, disable_fid=[])
    n_expr = gaussians.n_expr
    base_static = gaussians.flame_param["static_offset"].to(device).clone()
    if base_static.dim() == 3:
        base_static = base_static[0]
    num_verts = base_static.shape[0]
    n_eye = min(l_eyelid.shape[0], num_verts)

    pipe = PipelineConfig()
    bg = torch.tensor([1.0, 1.0, 1.0]).cuda()
    cam = OrbitCamera(args.res, args.res, r=args.cam_radius, fovy=args.cam_fovy, convention="opencv", save_path="cmp.json")

    base = mp_python.BaseOptions(model_asset_path=args.task_path)
    opts = vision.FaceLandmarkerOptions(base_options=base, output_face_blendshapes=True,
                                        output_facial_transformation_matrixes=True,
                                        running_mode=vision.RunningMode.VIDEO, num_faces=1)
    landmarker = vision.FaceLandmarker.create_from_options(opts)

    cap = cv2.VideoCapture(args.cam_id)
    if not cap.isOpened():
        raise RuntimeError(f"Ne morem odpreti kamere {args.cam_id}")
    os.makedirs(args.out_dir, exist_ok=True)

    win = "Primerjava gonilnikov (s=shrani, q=izhod)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, args.res * 3, args.res)
    print("Tece. Mezikni in pritisni 's' za shranjevanje.")

    t0 = time.time()
    saved = 0
    a_s = float(args.smooth)
    gsign = np.array([float(x) for x in args.gaze_signs.split(",")], dtype=np.float32)
    ema = {'expr': None, 'jaw': None, 'eyel': None, 'eyes': np.zeros(6, np.float32)}
    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                                              int((time.time()-t0)*1000))

            img_mlp = np.zeros((args.res, args.res, 3), np.uint8)
            img_smk = np.zeros((args.res, args.res, 3), np.uint8)

            if res.face_landmarks:
                lms = res.face_landmarks[0]
                pts = np.array([[lm.x*w, lm.y*h] for lm in lms], np.float32)
                scores = np.array([c.score for c in res.face_blendshapes[0]], np.float32) if res.face_blendshapes else np.zeros(52, np.float32)

                # ----- pot 1: MediaPipe -> MP_2_FLAME (brez mezika) -----
                exp, jaw, eye = mp2flame.convert(scores[None])
                fp = neutral_param(n_expr, base_static)
                fp['expr'][0, :n_expr] = torch.from_numpy(exp[0][:n_expr])
                fp['jaw'][0] = torch.from_numpy(jaw[0])
                fp['eyes'][0] = torch.from_numpy(eye[0])
                img_mlp = render_avatar(gaussians, fp, cam, pipe, bg)

                # ----- pot 2: SMIRK (z mezikom) -----
                crop = similarity_crop(rgb, pts)
                inp = torch.from_numpy(crop).permute(2, 0, 1).unsqueeze(0).to(device).to(enc_dtype) / 255.0
                out = {}
                out.update(encoder.pose_encoder(inp))
                out.update(encoder.expression_encoder(inp))
                e50 = out['expression_params'][0].float()
                jaw3 = out['jaw_params'][0].float().cpu().numpy()
                eyel = out['eyelid_params'][0].float().cpu().numpy()
                # EMA glajenje (kot v live_driver_smirk.py) -- zmanjsa utripanje
                if ema['expr'] is None:
                    ema['expr'] = e50.detach().clone(); ema['jaw'] = jaw3.copy(); ema['eyel'] = eyel.copy()
                else:
                    ema['expr'] = a_s * ema['expr'] + (1 - a_s) * e50
                    ema['jaw']  = a_s * ema['jaw']  + (1 - a_s) * jaw3
                    ema['eyel'] = a_s * ema['eyel'] + (1 - a_s) * eyel
                e50, jaw3, eyel = ema['expr'], ema['jaw'], ema['eyel']
                e100 = (M_proj @ e50).cpu()
                fp2 = neutral_param(n_expr, base_static)
                fp2['expr'][0, :n_expr] = e100[:n_expr]
                fp2['jaw'][0] = torch.from_numpy(jaw3)
                # mezik prek static_offset
                eyoff = torch.zeros(num_verts, 3, device=device)
                eyoff[:n_eye] = l_eyelid[:n_eye]*float(eyel[0]) + r_eyelid[:n_eye]*float(eyel[1])
                fp2['static_offset'] = base_static + eyoff
                # pogled iz MediaPipe eyeLook (hibridni del -- SMIRK ga sam nima)
                bs = scores
                pitchL = bs[17] - bs[11]; yawL = bs[15] - bs[13]
                pitchR = bs[18] - bs[12]; yawR = bs[14] - bs[16]
                gaze6 = np.array([gsign[0]*pitchL, gsign[1]*yawL, 0.0,
                                  gsign[0]*pitchR, gsign[1]*yawR, 0.0], dtype=np.float32)
                ema['eyes'] = a_s * ema['eyes'] + (1 - a_s) * gaze6
                fp2['eyes'][0] = torch.from_numpy(ema['eyes'] * args.gaze_scale)
                img_smk = render_avatar(gaussians, fp2, cam, pipe, bg)

            cam_disp = cv2.resize(cv2.flip(frame, 1), (args.res, args.res))
            combo = np.hstack([label(cam_disp, "webcam"),
                               label(img_mlp, "MediaPipe MLP"),
                               label(img_smk, "SMIRK")])
            cv2.imshow(win, combo)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('s'):
                p = os.path.join(args.out_dir, f"primerjava_{saved:02d}.png")
                cv2.imwrite(p, combo)
                # shranim tudi posamezne panele (za diplomo)
                cv2.imwrite(os.path.join(args.out_dir, f"mlp_{saved:02d}.png"), img_mlp)
                cv2.imwrite(os.path.join(args.out_dir, f"smirk_{saved:02d}.png"), img_smk)
                print(f"shranjeno: {p}")
                saved += 1

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print(f"\nShranjenih slik: {saved}  (mapa: {os.path.abspath(args.out_dir)})")


if __name__ == "__main__":
    main()
