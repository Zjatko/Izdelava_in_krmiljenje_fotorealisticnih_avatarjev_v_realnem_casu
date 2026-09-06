# 07_view_avatar.ps1  --  MILESTONE B, step 3 (view / animate your avatar)
# Env: gaussian-avatars.

$ErrorActionPreference = "Stop"
$ROOT = "C:\pot\do\projekta"  # <-- PRILAGODI: koren, kjer so GaussianAvatars, VHAP, smirk
. "$PSScriptRoot\_conda_bootstrap.ps1"
conda activate gaussian-avatars
Set-Location "$ROOT\GaussianAvatars"

$SEQ = "myseq"
$ITER = 30000
$PLY = "output\${SEQ}_avatar\point_cloud\iteration_${ITER}\point_cloud.ply"

if (-not (Test-Path $PLY)) {
    Write-Host "Not found: $PLY" -ForegroundColor Red
    Write-Host "Check the iteration number that actually got saved:" -ForegroundColor Yellow
    Get-ChildItem "output\${SEQ}_avatar\point_cloud" -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "  $($_.Name)" }
    exit 1
}

Write-Host "Opening avatar viewer..." -ForegroundColor Cyan
python local_viewer.py --point_path $PLY
# To drive with a different motion sequence add:  --motion_path <some_sequence>.npz
