# 06_train_avatar.ps1  --  MILESTONE B, step 2 (train GaussianAvatars)
# Trains 3D Gaussians rigged to the FLAME mesh on your exported VHAP dataset.
# Requires: 05 finished (an export folder exists). Env: gaussian-avatars.

$ErrorActionPreference = "Stop"
$ROOT = "C:\pot\do\projekta"  # <-- PRILAGODI: koren, kjer so GaussianAvatars, VHAP, smirk
. "$PSScriptRoot\_conda_bootstrap.ps1"
conda activate gaussian-avatars
Set-Location "$ROOT\GaussianAvatars"

$SEQ = "myseq"
$SRC = "$ROOT\VHAP\export\monocular\${SEQ}_whiteBg_staticOffset"
$OUT = "output\${SEQ}_avatar"

if (-not (Test-Path "$SRC\transforms.json") -and -not (Test-Path "$SRC")) {
    Write-Host "Export folder not found: $SRC  -- run 05_track_my_video.ps1 first." -ForegroundColor Red
    exit 1
}

Write-Host "Training avatar from $SRC ..." -ForegroundColor Cyan
Write-Host "(~30k iterations; monitor live in another terminal with: python remote_viewer.py --port 60000)" -ForegroundColor Gray

python train.py `
  -s "$SRC" `
  -m "$OUT" `
  --eval --bind_to_mesh --white_background --port 60000

Write-Host "`nTraining done. Model in $ROOT\GaussianAvatars\$OUT" -ForegroundColor Green
Write-Host "Next: 07_view_avatar.ps1" -ForegroundColor Green
