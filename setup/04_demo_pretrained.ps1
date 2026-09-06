# 04_demo_pretrained.ps1  --  MILESTONE A
# Renders the bundled pretrained avatar. If this works, your install is correct.
# Requires: 00 + 01 done, FLAME files placed (03).

$ErrorActionPreference = "Stop"
$ROOT = "C:\pot\do\projekta"  # <-- PRILAGODI: koren, kjer so GaussianAvatars, VHAP, smirk
. "$PSScriptRoot\_conda_bootstrap.ps1"
conda activate gaussian-avatars
Set-Location "$ROOT\GaussianAvatars"

Write-Host "Launching local viewer on the pretrained subject 306..." -ForegroundColor Cyan
python local_viewer.py --point_path media/306/point_cloud.ply
