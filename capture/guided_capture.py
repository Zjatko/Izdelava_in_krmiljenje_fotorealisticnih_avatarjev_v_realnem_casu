# guided_capture.py
# Vodeni zajem obraza za VHAP/GaussianAvatars.
#
# Program s spletno kamero te v zivo vodi, da enakomerno pokrijes vse poze glave
# (yaw/pitch), zajames kljucne izraze (mezik, nasmeh z zobmi, odprta usta, obrvi)
# in se izognes zamegljenim slikam. Ob koncu shrani .mp4, ki gre naravnost v
# obstojeci cevovod:
#   python vhap/preprocess_video.py --input data/monocular/<ime>.mp4 --matting_method robust_video_matting
#
# Zazeni v okolju z mediapipe + opencv (npr. gaussian-avatars):
#   python guided_capture.py --out "videos/my_capture.mp4"
#
# Tipke: q = koncaj in shrani | r = ponastavi pokritost | SPACE = pavza snemanja

import os
import time
import argparse
from collections import OrderedDict

import numpy as np
import cv2

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


# ---- imena blendshapeov za detekcijo izrazov ----
def bs_value(bs_dict, *names):
    return max((bs_dict.get(n, 0.0) for n in names), default=0.0)


def head_yaw_pitch(M, mirror):
    """Iz 4x4 transformacijske matrike glave izracuna yaw in pitch v stopinjah."""
    R = np.array(M)[:3, :3]
    fwd = R @ np.array([0.0, 0.0, -1.0])            # smer gledanja
    yaw = np.degrees(np.arctan2(fwd[0], -fwd[2]))
    pitch = np.degrees(np.arctan2(fwd[1], np.hypot(fwd[0], fwd[2])))
    if mirror:
        yaw = -yaw
    return yaw, pitch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="videos/guided_capture.mp4",
                    help="kam shraniti posneti mp4")
    ap.add_argument("--task_path", type=str,
                    default=r"face_landmarker.task",
                    help="MediaPipe face_landmarker.task model")
    ap.add_argument("--cam_id", type=int, default=0)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--yaw_max", type=float, default=30.0, help="najvecji obrat levo/desno (stopinje)")
    ap.add_argument("--yaw_step", type=float, default=15.0)
    ap.add_argument("--pitch_max", type=float, default=15.0, help="najvecji nagib gor/dol (stopinje)")
    ap.add_argument("--pitch_step", type=float, default=15.0)
    ap.add_argument("--dwell", type=float, default=0.5, help="koliko sekund drzati pozo, da se steje")
    ap.add_argument("--sharp_thresh", type=float, default=15.0,
                    help="prag ostrine (varianca Laplaciana); nizje = manj zavraca (za slabe webcame)")
    ap.add_argument("--mirror", action="store_true", default=True)
    ap.add_argument("--no-mirror", dest="mirror", action="store_false")
    ap.add_argument("--flip_yaw", action="store_true", help="obrni predznak yaw, ce vodenje kaze narobe")
    ap.add_argument("--flip_pitch", action="store_true")
    args = ap.parse_args()

    # ---- ciljne poze (mreza yaw x pitch) ----
    yaws = list(np.arange(-args.yaw_max, args.yaw_max + 1e-3, args.yaw_step))
    pitches = list(np.arange(-args.pitch_max, args.pitch_max + 1e-3, args.pitch_step))
    targets = [(round(y, 1), round(p, 1)) for p in pitches for y in yaws]
    dwell_time = {t: 0.0 for t in targets}
    covered = set()

    # ---- ciljni izrazi ----
    expr = OrderedDict([
        ("mezik",           False),   # eyeBlink
        ("nasmeh (zobje)",  False),   # mouthSmile
        ("odpri usta",      False),   # jawOpen
        ("obrvi gor",       False),   # browOuterUp / browInnerUp
    ])

    # ---- MediaPipe ----
    base = mp_python.BaseOptions(model_asset_path=args.task_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(args.cam_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    if not cap.isOpened():
        raise RuntimeError(f"Ne morem odpreti kamere {args.cam_id}")

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Kamera ne vraca slik.")
    H, W = frame.shape[:2]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))

    win = "Vodeni zajem  (q=koncaj, r=ponastavi, SPACE=pavza)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(W, 1280), int(min(W, 1280) * H / W))

    t0 = time.time()
    t_prev = time.time()
    recording = True
    print("Zajem tece. Sledi navodilom na zaslonu. 'q' koncaj in shrani.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.time()
        dt = now - t_prev
        t_prev = now

        # snemamo SUROVO (nezrcaljeno) sliko v polni locljivosti za VHAP
        if recording:
            writer.write(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(mp_img, int((now - t0) * 1000))

        # prikaz je zrcaljen (selfie)
        disp = cv2.flip(frame, 1) if args.mirror else frame.copy()

        have = bool(result.face_landmarks)
        sharp = False
        yaw = pitch = 0.0
        cur_target = None

        if have:
            lms = result.face_landmarks[0]
            xs = np.array([lm.x for lm in lms]) * W
            ys = np.array([lm.y for lm in lms]) * H
            x0, x1 = int(max(xs.min(), 0)), int(min(xs.max(), W))
            y0, y1 = int(max(ys.min(), 0)), int(min(ys.max(), H))
            # ostrina na predelu obraza
            if x1 > x0 + 10 and y1 > y0 + 10:
                gray = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
                sharp_val = cv2.Laplacian(gray, cv2.CV_64F).var()
                sharp = sharp_val >= args.sharp_thresh
            # poza glave
            if result.facial_transformation_matrixes:
                yaw, pitch = head_yaw_pitch(result.facial_transformation_matrixes[0], args.mirror)
                if args.flip_yaw:
                    yaw = -yaw
                if args.flip_pitch:
                    pitch = -pitch
                # najblizji cilj
                cur_target = min(targets, key=lambda t: (t[0]-yaw)**2 + (t[1]-pitch)**2)
                dy = abs(cur_target[0]-yaw); dp = abs(cur_target[1]-pitch)
                in_cell = dy <= args.yaw_step*0.6 and dp <= args.pitch_step*0.6
                if in_cell and sharp and cur_target not in covered:
                    dwell_time[cur_target] += dt
                    if dwell_time[cur_target] >= args.dwell:
                        covered.add(cur_target)
                elif not in_cell:
                    if cur_target in dwell_time:
                        dwell_time[cur_target] = max(0.0, dwell_time[cur_target]-dt)

            # izrazi iz blendshapeov
            if result.face_blendshapes and sharp:
                bs = {c.category_name: c.score for c in result.face_blendshapes[0]}
                if bs_value(bs, "eyeBlinkLeft", "eyeBlinkRight") > 0.5:
                    expr["mezik"] = True
                if bs_value(bs, "mouthSmileLeft", "mouthSmileRight") > 0.5:
                    expr["nasmeh (zobje)"] = True
                if bs_value(bs, "jawOpen") > 0.45:
                    expr["odpri usta"] = True
                if bs_value(bs, "browOuterUpLeft", "browOuterUpRight", "browInnerUp") > 0.45:
                    expr["obrvi gor"] = True

        # ---------------- risanje HUD ----------------
        _draw_hud(disp, targets, covered, dwell_time, cur_target, yaw, pitch,
                  have, sharp, expr, recording, args)

        cv2.imshow(win, disp)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('r'):
            covered.clear()
            for t in dwell_time: dwell_time[t] = 0.0
            for e in expr: expr[e] = False
        elif k == ord(' '):
            recording = not recording

    writer.release()
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
    print(f"\nShranjeno: {os.path.abspath(args.out)}")
    print(f"Pokritih poz: {len(covered)}/{len(targets)} | izrazi: "
          + ", ".join(f"{k}:{'OK' if v else 'X'}" for k, v in expr.items()))
    print("Naslednji korak (okolje VHAP):")
    print(f"  python vhap/preprocess_video.py --input {args.out} --matting_method robust_video_matting")


def _draw_hud(img, targets, covered, dwell_time, cur, yaw, pitch, have, sharp, expr, recording, args):
    H, W = img.shape[:2]
    green = (60, 200, 60); gray = (110, 110, 110); red = (60, 60, 220)
    white = (245, 245, 245); yellow = (40, 210, 210)

    # 1) mreza pokritosti (desno zgoraj)
    gx, gy, cw, ch = W - 260, 30, 44, 44
    yaws = sorted(set(t[0] for t in targets)); pitches = sorted(set(t[1] for t in targets))
    for pi, p in enumerate(pitches):
        for yi, y in enumerate(yaws):
            x = gx + yi*(cw+4); yy = gy + pi*(ch+4)
            col = green if (y, p) in covered else gray
            cv2.rectangle(img, (x, yy), (x+cw, yy+ch), col, -1)
            if cur == (y, p):
                cv2.rectangle(img, (x, yy), (x+cw, yy+ch), yellow, 3)
    cv2.putText(img, "pokritost poz", (gx, gy-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, white, 1)
    pct = int(100*len(covered)/max(len(targets), 1))
    cv2.putText(img, f"{pct}%", (gx+200, gy-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, green if pct==100 else yellow, 2)

    # 2) navodilo (sredina zgoraj)
    msg = ""
    if not have:
        msg = "obraz ni zaznan"
    elif not sharp:
        msg = "premikaj se pocasneje (zamegljeno)"
    elif cur is not None and cur not in covered:
        ty, tp = cur
        parts = []
        if ty - yaw > 5:  parts.append("obrni DESNO" if args.mirror else "obrni LEVO")
        elif ty - yaw < -5: parts.append("obrni LEVO" if args.mirror else "obrni DESNO")
        if tp - pitch > 5:  parts.append("poglej GOR")
        elif tp - pitch < -5: parts.append("poglej DOL")
        if not parts: parts.append("drzi ...")
        msg = " + ".join(parts)
    elif len(covered) < len(targets):
        msg = "obrni glavo v sivo polje"
    else:
        msg = "poze pokrite! izvedi se izraze spodaj"
    cv2.putText(img, msg, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, white, 2)

    # 3) ostrina + snemanje (levo zgoraj, pod navodilom)
    cv2.putText(img, "OSTRO" if sharp else "ZAMEGLJENO", (30, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, green if sharp else red, 2)
    if recording:
        cv2.circle(img, (W-30, H-30), 12, red, -1)
        cv2.putText(img, "REC", (W-70, H-24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, red, 2)
    else:
        cv2.putText(img, "PAVZA (SPACE)", (W-190, H-24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, yellow, 2)

    # 4) kontrolni seznam izrazov (levo spodaj)
    y = H - 30 - 28*len(expr)
    cv2.putText(img, "izrazi:", (30, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, white, 1)
    for i, (name, done) in enumerate(expr.items()):
        col = green if done else gray
        cv2.putText(img, f"[{'x' if done else ' '}] {name}", (30, y+22*i+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)


if __name__ == "__main__":
    main()
