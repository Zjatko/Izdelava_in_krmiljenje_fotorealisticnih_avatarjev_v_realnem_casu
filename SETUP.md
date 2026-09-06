# Navodila za namestitev (korak za korakom)

Podroben vodič za postavitev obeh cevovodov na Windows. Za hiter pregled glej `README.md`,
za tuje licence/prenose pa `THIRD_PARTY.md`.

---

## 0. Predpogoji

- **Windows 10/11**, **NVIDIA GPU** (razvito na RTX 3070 Ti) z gonilniki.
- **Anaconda / Miniconda** (uporabljaj *Anaconda PowerShell Prompt*).
- **Git**.
- **Visual Studio 2022** z delovno obremenitvijo *Desktop development with C++* in
  **MSVC toolset v14.38** (v VS Installer: »MSVC v143 … build tools (v14.38-17.8)«).
  Nujno za prevajanje rasterizatorja 3DGS in nvdiffrast.
- **CUDA 12.1** (kombinacija CUDA 12.1 + PyTorch 2.2 + MSVC 14.38 je preverjena).

> Opomba: CUDA 12.1 ne dela z novejšimi MSVC (14.4x) — zato je pomemben toolset 14.38.

---

## 1. Kloniraj repozitorije

V koren projekta (npr. `C:\projekt\avatar`) kloniraj:

```powershell
git clone https://github.com/ShenhanQian/GaussianAvatars.git --recursive
git clone https://github.com/ShenhanQian/VHAP.git
git clone https://github.com/georgeretsi/smirk.git
git clone https://github.com/PeizhiYan/mediapipe-blendshapes-to-flame.git
```

(Prva dva lahko tudi prek `setup/00_clone_repos.ps1`.) Pomembno: GaussianAvatars kloniraj
**`--recursive`** (potrebuje podmodule, med njimi rasterizator 3DGS).

Struktura naj bo:
```
<koren projekta>/
  GaussianAvatars/
  VHAP/
  smirk/
  mediapipe-blendshapes-to-flame/
```

## 2. Nastavi koren v skriptah

V vseh `setup/*.ps1` uredi vrstico:
```powershell
$ROOT = "C:\pot\do\projekta"   # <-- svoj koren projekta
```

## 3. Ustvari conda okolji

```powershell
.\setup\01_env_gaussianavatars.ps1   # okolje: gaussian-avatars
.\setup\02_env_vhap.ps1              # okolje: VHAP
```
Skripti sami vstopita v razvojno okolje VS2022 (vcvars) in prevedeta razširitve.

## 4. Pridobi modele (glej THIRD_PARTY.md za licence)

- **FLAME** — registracija na <https://flame.is.tue.mpg.de>, nato:
  - `generic_model.pkl` (FLAME 2020) → `smirk/assets/FLAME2020/`
  - `flame2023.pkl` in `FLAME_masks.pkl` → `VHAP/asset/flame/` **in** `GaussianAvatars/flame_model/assets/flame/`
- **SMIRK uteži** `SMIRK_em1.pt` → `smirk/pretrained_models/`
- **MediaPipe** model `face_landmarker.task` (prenesi):
  <https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task>
  → postavi tja, od koder poganjaš (npr. koren GaussianAvatars) ali podaj prek `--task_path`.
- **MP\_2\_FLAME** (`mlp.pth`, `mappings/`) je že v kloniranem repozitoriju.

## 5. Vnesi moje datoteke

- Vsebino **`runtime/`** prekopiraj v koren **`GaussianAvatars/`** (skripte uvažajo njegove module).
- Datoteki iz **`vhap_modified/`** nadomestita:
  - `vhap_modified/export_as_nerf_dataset.py` → `VHAP/vhap/export_as_nerf_dataset.py`
  - `vhap_modified/config_base.py` → `VHAP/vhap/config/base.py`
- **`capture/guided_capture.py`** lahko poganjaš iz korena projekta.

---

## 6. Cevovod A — priprava avatarja

```powershell
# 1) vodeni zajem (okolje z mediapipe/opencv, npr. gaussian-avatars)
python capture\guided_capture.py --out "videos\my_capture.mp4" --task_path face_landmarker.task

# 2) sledenje + izvoz (VHAP)     -> uredi $SRC/$SEQ v skripti
.\setup\05_track_my_video.ps1

# 3) učenje avatarja (~30k iteracij, GaussianAvatars)
.\setup\06_train_avatar.ps1

# (ogled) .\setup\07_view_avatar.ps1
```

## 7. Cevovod B — krmiljenje v živo

Iz korena **GaussianAvatars/**, okolje `gaussian-avatars`:

```powershell
# enkraten predizračun projekcije FLAME 2020 -> 2023
python smirk_precompute.py --flame2020 "..\smirk\assets\FLAME2020\generic_model.pkl" ^
    --flame2023 "flame_model\assets\flame\flame2023.pkl" --out "smirk_to_flame2023.npz"

# hibridni krmilnik (SMIRK + pogled MediaPipe)
python live_driver_smirk.py ^
    --point_path "output\<avatar>\point_cloud\iteration_30000\point_cloud.ply" ^
    --smirk_root "..\smirk" --half

# primerjava obeh pristopov (za sliko z mežikom)
python compare_drivers.py ^
    --point_path "output\<avatar>\point_cloud\iteration_30000\point_cloud.ply" ^
    --smirk_root "..\smirk" --checkpoint "..\smirk\pretrained_models\SMIRK_em1.pt" --half
```
