# Tuje komponente, licence in prenos

Ta repozitorij vsebuje **samo mojo kodo**. Spodnje komponente **niso** vključene — pridobiš
jih sam, vsako pod njeno licenco. Modelov in uteži ne redistribuiram.

| Komponenta | Licenca | Kje dobiš | Opomba |
|---|---|---|---|
| **GaussianAvatars** | CC BY-NC-SA 4.0 | https://github.com/ShenhanQian/GaussianAvatars | nekomercialno; moje `runtime/` skripte se vstavijo vanj |
| **VHAP** | CC BY-NC-SA 4.0 | https://github.com/ShenhanQian/VHAP | datoteki iz `vhap_modified/` sta predelava VHAP (ista licenca) |
| **3D Gaussian Splatting** | licenca Inria (nekomercialna, raziskovalna) | podmodul GaussianAvatars (`--recursive`) | rasterizator |
| **SMIRK** | MIT (koda) | https://github.com/georgeretsi/smirk | uteži `SMIRK_em1.pt` pridobi sam |
| **MP\_2\_FLAME** | za raziskave/izobraževanje (»as-is«) | https://github.com/PeizhiYan/mediapipe-blendshapes-to-flame | uteži učene na javnih zbirkah; ne za komercialno rabo |
| **FLAME** (model) | MPI, **nekomercialno, brez redistribucije**, registracija | https://flame.is.tue.mpg.de | FLAME 2020 + 2023 + maske |
| **MediaPipe** (Face Landmarker) | Apache-2.0 | https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task | model prenese uporabnik |

## Kaj je NAMENOMA izpuščeno

- Vsi modeli FLAME (`*.pkl`), ker jih licenca MPI **prepoveduje razširjati**.
- Vse tuje uteži (`SMIRK_em1.pt`, `mlp.pth`, `face_landmarker.task`).
- `smirk_to_flame2023.npz` (izpeljan iz baz FLAME) — **regeneriraj** s `smirk_precompute.py`.
- Naučeni avatarji (`output/…`) in osebni posnetki/fotografije.

## Priznanje avtorstva

Če uporabiš to kodo, citiraj tudi izvorna dela: GaussianAvatars (Qian et al., CVPR 2024),
VHAP (Qian, 2024), 3D Gaussian Splatting (Kerbl et al., 2023), SMIRK (Retsinas et al., CVPR
2024), FLAME (Li et al., 2017), MediaPipe (Lugaresi et al., 2019) in MP\_2\_FLAME (Yan).
