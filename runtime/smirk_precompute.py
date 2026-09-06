# smirk_precompute.py
# Build the fixed linear map that converts SMIRK's FLAME-2020 expression coefficients (50)
# into your avatar's FLAME-2023 expression coefficients (100).
#
# WHY: SMIRK regresses expression in FLAME-2020's PCA basis (50 dims). Your GaussianAvatars
# avatar was tracked in FLAME-2023's basis (100 dims). The two bases are DIFFERENT, so you
# cannot copy coefficients. But both models share the same mesh topology (5023 verts), so we
# can match them in *vertex-displacement* space: find the FLAME-2023 coefficients whose vertex
# offset best reproduces SMIRK's FLAME-2020 vertex offset. That is a least-squares projection:
#
#     v_offset_2020 = B2020_50 @ e50          (B2020_50: [V*3, 50])
#     e100 = argmin || B2023_100 @ e100 - v_offset_2020 ||   ->   e100 = pinv(B2023_100) @ B2020_50 @ e50
#     M    = pinv(B2023_100) @ B2020_50        (shape [100, 50])
#
# At runtime: expr_2023 = M @ expr_smirk_50.
#
# RUN (in the gaussian-avatars conda env, from the GaussianAvatars repo root):
#   python smirk_precompute.py ^
#       --flame2020 "..\smirk\assets\FLAME2020\generic_model.pkl" ^
#       --flame2023 "flame_model\assets\flame\flame2023.pkl" ^
#       --out "smirk_to_flame2023.npz"
#
# generic_model.pkl is the standard FLAME 2020 model (same file SMIRK uses). If you already
# placed it under smirk/assets/FLAME2020/ during SMIRK setup, point --flame2020 there.

import argparse
import pickle
import numpy as np


def load_shapedirs(path):
    """Load FLAME 'shapedirs' as a float64 numpy array of shape [V, 3, n_shape+n_expr].
    Handles both chumpy-backed (FLAME 2020) and plain-ndarray (FLAME 2023) pickles."""
    # FLAME 2020 pickles reference numpy aliases removed in numpy>=1.24; restore them so
    # the chumpy arrays inside unpickle cleanly.
    for alias, real in [('bool', np.bool_), ('int', np.int_), ('float', np.float64),
                        ('complex', np.complex128), ('object', np.object_),
                        ('unicode', np.str_), ('str', np.str_)]:
        if not hasattr(np, alias):
            setattr(np, alias, real)

    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    sd = data['shapedirs']
    sd = np.asarray(sd, dtype=np.float64)  # chumpy.Ch -> ndarray via __array__, or passthrough
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--flame2020', required=True, help='generic_model.pkl (FLAME 2020, what SMIRK uses)')
    ap.add_argument('--flame2023', default=r'flame_model\assets\flame\flame2023.pkl',
                    help='flame2023.pkl (your avatar model)')
    ap.add_argument('--n_exp_smirk', type=int, default=50, help='SMIRK expression dims (FLAME2020)')
    ap.add_argument('--n_exp_avatar', type=int, default=100, help='avatar expression dims (FLAME2023)')
    ap.add_argument('--n_shape', type=int, default=300, help='shape dims before expression block in shapedirs')
    ap.add_argument('--out', default='smirk_to_flame2023.npz')
    args = ap.parse_args()

    print('Loading FLAME 2020 (SMIRK) ...')
    sd2020 = load_shapedirs(args.flame2020)
    print('Loading FLAME 2023 (avatar) ...')
    sd2023 = load_shapedirs(args.flame2023)

    V = sd2020.shape[0]
    assert sd2023.shape[0] == V, f'vertex count mismatch: {sd2020.shape} vs {sd2023.shape}'
    print(f'  vertices: {V}  | sd2020 {sd2020.shape}  sd2023 {sd2023.shape}')

    s, e2020 = args.n_shape, args.n_shape + args.n_exp_smirk
    e2023 = args.n_shape + args.n_exp_avatar

    B2020 = sd2020[:, :, s:e2020].reshape(V * 3, args.n_exp_smirk)      # [V*3, 50]
    B2023 = sd2023[:, :, s:e2023].reshape(V * 3, args.n_exp_avatar)     # [V*3, 100]

    # M = pinv(B2023) @ B2020   ->  [100, 50]
    print('Solving least-squares projection (this takes a few seconds) ...')
    M = np.linalg.pinv(B2023) @ B2020                                  # [100, 50]
    M = M.astype(np.float32)

    # quick sanity: how well does the projection reproduce a unit FLAME2020 expression in
    # vertex space? (1.0 = perfect, lower = basis can't fully represent it)
    recon = B2023 @ M                                                  # [V*3, 50]
    num = np.linalg.norm(recon, axis=0)
    den = np.linalg.norm(B2020, axis=0) + 1e-8
    frac = (num / den)
    print(f'  per-component reconstruction fraction: mean {frac.mean():.3f}  min {frac.min():.3f}  max {frac.max():.3f}')

    np.savez(args.out, M=M, n_exp_smirk=args.n_exp_smirk, n_exp_avatar=args.n_exp_avatar)
    print(f'Saved projection to {args.out}  (M shape {M.shape})')


if __name__ == '__main__':
    main()
