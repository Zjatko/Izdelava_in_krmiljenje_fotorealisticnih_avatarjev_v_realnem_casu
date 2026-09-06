# 05_track_my_video.ps1  --  MILESTONE B, step 1 (VHAP tracking)
# Preprocess -> track -> export your video into a NeRF/3DGS-style dataset.
# Requires: 00 + 02 done, FLAME files placed in VHAP\asset\flame (03).
# Run from an Anaconda PowerShell Prompt.

$ErrorActionPreference = "Stop"
$ROOT = "C:\pot\do\projekta"  # <-- PRILAGODI: koren, kjer so GaussianAvatars, VHAP, smirk
. "$PSScriptRoot\_conda_bootstrap.ps1"
conda activate VHAP
Set-Location "$ROOT\VHAP"

# --- pick the source video ---
# Default: reuse my_capture.mp4. See README §5a -- a frontal talking-head clip with
# expressions tracks far better than a camera-orbit clip. Change $SRC if you record a new one.
$SRC = "$ROOT\videos\my_capture.mp4"
$SEQ = "myseq"

New-Item -ItemType Directory -Force -Path "data\monocular" | Out-Null
Copy-Item $SRC "data\monocular\$SEQ.mp4" -Force
Write-Host "Using video: $SRC  ->  data\monocular\$SEQ.mp4" -ForegroundColor Cyan

$TRACK = "output/monocular/${SEQ}_whiteBg_staticOffset"
$EXPORT = "export/monocular/${SEQ}_whiteBg_staticOffset"

# 1) preprocess: extract frames + foreground matting (GPU)
python vhap/preprocess_video.py --input "data/monocular/$SEQ.mp4" --matting_method robust_video_matting

# 2) track: per-frame FLAME fitting (sequential, then 30 epochs global)
python vhap/track.py --data.root_folder "data/monocular" --exp.output_folder $TRACK --data.sequence $SEQ

# 3) export: build images + transforms.json for GaussianAvatars
python vhap/export_as_nerf_dataset.py --src_folder $TRACK --tgt_folder $EXPORT --background-color white

Write-Host "`nTracking + export done." -ForegroundColor Green
Write-Host "Exported dataset: $ROOT\VHAP\$EXPORT" -ForegroundColor Green
Write-Host "TIP: inspect tracking quality first:" -ForegroundColor Yellow
Write-Host '  python vhap/flame_viewer.py --param_path output/monocular/myseq_whiteBg_staticOffset/<timestamp>/tracked_flame_params_30.npz' -ForegroundColor Yellow
Write-Host "Then run 06_train_avatar.ps1" -ForegroundColor Green
