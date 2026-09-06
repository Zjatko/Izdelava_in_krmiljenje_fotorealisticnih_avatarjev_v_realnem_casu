# orbit_render.py -- avtomatski obhod kamere okoli avatarja (brez GUI-ja).
# Naloži naučen GaussianAvatars avatar in ga izriše iz N kotov po krožnici (360 stopinj),
# vsako sličico shrani kot PNG. Nato jih sestaviš v video s ffmpeg.
#
# Primer:
#   python orbit_render.py ^
#     --point_path output/<avatar>/point_cloud/iteration_30000/point_cloud.ply ^
#     --out viewer_output/orbit_guided_1 --frames 180 --white
#
# Nato:
#   ffmpeg -framerate 30 -i "viewer_output/orbit_guided_1/%05d.png" -c:v libx264 -pix_fmt yuv420p orbit.mp4

import math
from pathlib import Path
from dataclasses import dataclass
from argparse import ArgumentParser

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from gaussian_renderer import render, GaussianModel, FlameGaussianModel
from utils.general_utils import safe_state
from utils.viewer_utils import OrbitCamera


@dataclass
class PipelineConfig:
    debug: bool = False
    compute_cov3D_python: bool = False
    convert_SHs_python: bool = False


def build_cam(cam: OrbitCamera):
    """Iz trenutnega stanja OrbitCamera zgradi objekt, ki ga pričakuje render()."""
    @dataclass
    class Cam:
        FoVx = float(np.radians(cam.fovx))
        FoVy = float(np.radians(cam.fovy))
        image_height = cam.image_height
        image_width = cam.image_width
        world_view_transform = torch.tensor(cam.world_view_transform).float().cuda().T
        full_proj_transform = torch.tensor(cam.full_proj_transform).float().cuda().T
        camera_center = torch.tensor(cam.pose[:3, 3]).cuda()
    return Cam


def main():
    ap = ArgumentParser(description="Avtomatski orbit render GaussianAvatars avatarja.")
    ap.add_argument("--point_path", type=Path, required=True,
                    help="pot do point_cloud.ply (v mapi mora biti tudi flame_param.npz)")
    ap.add_argument("--out", type=Path, default=Path("viewer_output/orbit"),
                    help="mapa za shranjevanje sličic")
    ap.add_argument("--frames", type=int, default=180, help="stevilo sličic v celotnem obhodu")
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=960)
    ap.add_argument("--radius", type=float, default=1.0, help="oddaljenost kamere (vec = dlje)")
    ap.add_argument("--fovy", type=float, default=20.0, help="vidni kot (manj = blize/zoom)")
    ap.add_argument("--elev", type=float, default=0.0, help="nagib gor/dol v stopinjah")
    ap.add_argument("--timestep", type=int, default=0, help="katera poza FLAME (0 = prva); ce ni --animate")
    ap.add_argument("--animate", action="store_true",
                    help="med obhodom predvajaj mimiko iz posnetka (menja poze FLAME)")
    ap.add_argument("--anim_cycles", type=float, default=1.0,
                    help="kolikokrat gre mimika skozi cel posnetek med enim obhodom")
    ap.add_argument("--sh_degree", type=int, default=3)
    ap.add_argument("--white", action="store_true", help="belo ozadje (sicer crno)")
    args = ap.parse_args()

    safe_state(True)

    with torch.no_grad():
        # naloži avatar (Flame, ce je zraven flame_param.npz)
        if (args.point_path.parent / "flame_param.npz").exists():
            gaussians = FlameGaussianModel(args.sh_degree)
        else:
            gaussians = GaussianModel(args.sh_degree)

        if not args.point_path.exists():
            raise FileNotFoundError(f"Ni datoteke: {args.point_path}")
        gaussians.load_ply(args.point_path, has_target=False)

        n_ts = int(getattr(gaussians, "num_timesteps", 1) or 1)
        print(f"Na voljo je {n_ts} poz (timesteps) iz posnetka.")
        if gaussians.binding is not None and not args.animate:
            gaussians.select_mesh_by_timestep(min(args.timestep, n_ts - 1))

        bg = torch.tensor([1, 1, 1] if args.white else [0, 0, 0],
                          dtype=torch.float32, device="cuda")
        pipe = PipelineConfig()

        cam = OrbitCamera(args.width, args.height, r=args.radius,
                          fovy=args.fovy, convention="opencv")
        if args.elev != 0.0:
            cam.orbit_x(math.radians(args.elev))

        args.out.mkdir(parents=True, exist_ok=True)
        step = 2 * math.pi / args.frames  # kot med zaporednima sličicama (radiani)

        print(f"Izrisujem {args.frames} sličic v {args.out} ...")
        for i in tqdm(range(args.frames)):
            if gaussians.binding is not None and args.animate:
                ts = int((i / args.frames) * args.anim_cycles * n_ts) % n_ts
                gaussians.select_mesh_by_timestep(ts)
            Cam = build_cam(cam)
            img = render(Cam, gaussians, pipe, bg)["render"]
            arr = (img.permute(1, 2, 0).clip(0, 1).cpu().numpy() * 255).astype(np.uint8)
            Image.fromarray(arr).save(args.out / f"{i:05d}.png")
            cam.orbit_y(step)  # zavrti kamero za naslednjo sličico

        print(f"Koncano. {args.frames} sličic v: {args.out}")
        print("Sestavi video npr. z:")
        print(f'  ffmpeg -framerate 30 -i "{args.out}/%05d.png" -c:v libx264 -pix_fmt yuv420p orbit.mp4')


if __name__ == "__main__":
    main()
