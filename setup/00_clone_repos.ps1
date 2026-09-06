# 00_clone_repos.ps1
# Clones GaussianAvatars (recursive, for submodules) and VHAP into the project root.
# Run from an Anaconda PowerShell Prompt. No conda env needed for this step.

$ErrorActionPreference = "Stop"
$ROOT = "C:\pot\do\projekta"  # <-- PRILAGODI: koren, kjer so GaussianAvatars, VHAP, smirk
Set-Location $ROOT

if (-not (Test-Path "$ROOT\GaussianAvatars")) {
    Write-Host "Cloning GaussianAvatars (with submodules)..." -ForegroundColor Cyan
    git clone https://github.com/ShenhanQian/GaussianAvatars.git --recursive
} else {
    Write-Host "GaussianAvatars already present. Ensuring submodules are present..." -ForegroundColor Yellow
    git -C "$ROOT\GaussianAvatars" submodule update --init --recursive
}

if (-not (Test-Path "$ROOT\VHAP")) {
    Write-Host "Cloning VHAP..." -ForegroundColor Cyan
    git clone https://github.com/ShenhanQian/VHAP.git
} else {
    Write-Host "VHAP already present." -ForegroundColor Yellow
}

Write-Host "`nDone. Next: 01_env_gaussianavatars.ps1" -ForegroundColor Green
Write-Host "Submodules that MUST exist (non-empty):" -ForegroundColor Gray
Get-ChildItem "$ROOT\GaussianAvatars\submodules" -ErrorAction SilentlyContinue | ForEach-Object { Write-Host "  $($_.Name)" }
