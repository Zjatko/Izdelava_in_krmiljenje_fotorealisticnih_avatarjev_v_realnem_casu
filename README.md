# Fotorealistični krmiljivi avatar iz ene kamere

Koda k diplomskemu delu **»Izdelava in krmiljenje fotorealističnih avatarjev v realnem
času«** (UL FRI). Repozitorij vsebuje **moje skripte za oba cevovoda**; tuji projekti
(GaussianAvatars, VHAP, SMIRK, MP\_2\_FLAME, FLAME, MediaPipe) se **ne** distribuirajo tukaj —
navodila za njihovo namestitev so spodaj.

## Cevovoda

**A) Pripravljalni cevovod (enkrat na osebo):**
```
vodeni zajem  →  VHAP sledenje  →  izvoz zbirke  →  učenje GaussianAvatars
(guided_capture.py)  (05_...)       (patchan izvoz)   (06_...)
```

**B) Sprotni cevovod (krmiljenje v živo):**
```
spletna kamera → MediaPipe → { SMIRK (hibridno) | MP_2_FLAME } → parametri FLAME → izris
(live_driver_smirk.py / live_driver.py)
```

## Vsebina repozitorija

```
capture/         guided_capture.py            # vodeni zajem posnetka
runtime/         live_driver_smirk.py         # hibridni krmilnik (SMIRK + pogled MediaPipe)
                 live_driver.py               # pristop MP_2_FLAME
                 compare_drivers.py           # primerjava obeh pristopov (za sliko)
                 live_driver_smirk_dual.py    # dva avatarja, isti signal
                 smirk_precompute.py          # projekcijska matrika FLAME 2020 -> 2023
                 orbit_render.py              # orbit render avatarja
vhap_modified/   export_as_nerf_dataset.py    # popravek izvoza (številčenje sličic) + nastavitve
                 config_base.py               # -> nadomestita datoteki v klonu VHAP
setup/           00..07 *.ps1                 # namestitev okolij in zagon (Windows/conda)
```

> Skripte v `runtime/` in `smirk_precompute.py` so mišljene, da jih **prekopiraš v koren
> kloniranega repozitorija GaussianAvatars** (uvažajo njegove module).

## Zahteve

- Windows + Anaconda, NVIDIA GPU (razvito na RTX 3070 Ti).
- CUDA 12.1 + PyTorch 2.2, Visual Studio 2022 z MSVC **14.38** (za prevajanje rasterizatorja
  in nvdiffrast). Podrobnosti so v `setup/01_env_gaussianavatars.ps1` in `02_env_vhap.ps1`.

## Namestitev

> Podroben vodič korak-za-korakom je v **[`SETUP.md`](SETUP.md)**. Spodaj je hiter povzetek.

1. **Kloniraj tuja projekta** (skripta `setup/00_clone_repos.ps1`):
   ```powershell
   git clone https://github.com/ShenhanQian/GaussianAvatars.git --recursive
   git clone https://github.com/ShenhanQian/VHAP.git
   ```
   ter **SMIRK** in **MP\_2\_FLAME**:
   ```powershell
   git clone https://github.com/georgeretsi/smirk.git
   git clone https://github.com/PeizhiYan/mediapipe-blendshapes-to-flame.git
   ```

2. **Ustvari conda okolji:** `setup/01_env_gaussianavatars.ps1` (env `gaussian-avatars`) in
   `setup/02_env_vhap.ps1` (env `VHAP`).

3. **Pridobi modele (vsak pod svojo licenco — glej `THIRD_PARTY.md`):**
   - **FLAME** (registracija na <https://flame.is.tue.mpg.de>): FLAME 2020 `generic_model.pkl`,
     FLAME 2023 `flame2023.pkl` in `FLAME_masks.pkl`. Postavi v:
     `VHAP/asset/flame/`, `GaussianAvatars/flame_model/assets/flame/`, `smirk/assets/FLAME2020/`.
   - **SMIRK uteži** `SMIRK_em1.pt` → `smirk/pretrained_models/`.
   - **MP\_2\_FLAME** (`mlp.pth` + `mappings/`) so že v kloniranem repozitoriju.
   - **MediaPipe** model:
     <https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task>

4. **Vnesi moje datoteke:**
   - vsebino `runtime/` prekopiraj v koren `GaussianAvatars/`;
   - datoteki iz `vhap_modified/` nadomestita `VHAP/vhap/export_as_nerf_dataset.py` in
     `VHAP/vhap/config/base.py` (sta predelava VHAP — glej licenco).

## Zagon

**A – priprava avatarja**
```powershell
# 1) vodeni zajem
python capture/guided_capture.py --out "videos/moj_zajem.mp4"
# 2) sledenje + izvoz (VHAP)      -> setup/05_track_my_video.ps1
# 3) učenje avatarja (GaussianAvatars, ~30k iteracij) -> setup/06_train_avatar.ps1
```

**B – krmiljenje v živo** (iz korena GaussianAvatars, env `gaussian-avatars`)
```powershell
# enkraten predizračun projekcije FLAME 2020 -> 2023
python smirk_precompute.py --flame2020 "..\smirk\assets\FLAME2020\generic_model.pkl" ^
    --flame2023 "flame_model\assets\flame\flame2023.pkl" --out "smirk_to_flame2023.npz"

# hibridni krmilnik
python live_driver_smirk.py --point_path "output\<avatar>\point_cloud\iteration_30000\point_cloud.ply" ^
    --smirk_root "..\smirk" --half
```

> **Opomba:** nekatere skripte imajo privzete poti do modelov (npr. `--mappings_path`,
> `--task_path`). Prilagodi jih svoji postavitvi prek argumentov ali uredi privzete vrednosti.

## Licenca in viri

Moja koda: **CC BY-NC-SA 4.0** (nekomercialno, priznanje avtorstva, deljenje pod enakimi
pogoji) — enako kot GaussianAvatars/VHAP, na katerih temelji. Seznam vseh tujih komponent,
njihovih licenc in kaj je treba pridobiti, je v **`THIRD_PARTY.md`**.
