# 01_env_gaussianavatars.ps1
# Creates the `gaussian-avatars` conda env on Windows using the CUDA 12.1 + torch 2.2 combo
# (the combo that COMPILES on VS2022 -- avoids the 11.7/VS2022 failure).
# Uses the Visual Studio Dev Shell (equivalent of vcvars64.bat) so cl.exe is on PATH.
# Run from an Anaconda PowerShell Prompt.

$ErrorActionPreference = "Stop"
$ROOT = "C:\pot\do\projekta"  # <-- PRILAGODI: koren, kjer so GaussianAvatars, VHAP, smirk

# Make `conda` work in any PowerShell window
. "$PSScriptRoot\_conda_bootstrap.ps1"

# ---------------------------------------------------------------------------
# Helper: enter the VS2022 x64 developer environment (the PowerShell version
# of running vcvars64.bat). Auto-detects the VS install with vswhere.
# ---------------------------------------------------------------------------
function Enter-VS2022x64 {
    $vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $vswhere)) {
        throw "vswhere.exe not found. Is Visual Studio (with the C++ workload) installed?"
    }
    # Constrain to VS2022 (major version 17). CUDA 12.1's nvcc does NOT support
    # VS2026 (v18 / MSVC 14.5x) -- it will refuse to compile the CUDA extensions.
    # vswhere may return MULTIPLE matching installs (one per line) -> take the first.
    $vsPath = & $vswhere -version "[17.0,18.0)" -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1
    if (-not $vsPath) {
        throw "No VS2022 (v17) C++ toolset found. Install 'Build Tools for Visual Studio 2022' with the 'Desktop development with C++' workload. (CUDA 12.1 cannot use VS2026.)"
    }

    $devShell = Join-Path $vsPath "Common7\Tools\Microsoft.VisualStudio.DevShell.dll"
    Import-Module $devShell

    # CUDA 12.1 needs MSVC <= ~14.39. The newest toolset (14.44+) STL even *requires*
    # CUDA >= 12.4 and breaks. If the CUDA-12.1-compatible 14.38 toolset is installed,
    # pin to it. (Install it in the VS Installer: "MSVC v143 ... build tools (v14.38-17.8)".)
    $devArgs = "-arch=x64 -host_arch=x64"
    $t1438 = Get-ChildItem "$vsPath\VC\Tools\MSVC" -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -like "14.38.*" } | Select-Object -First 1
    if ($t1438) {
        $devArgs += " -vcvars_ver=14.38"
        Write-Host "Pinning MSVC toolset 14.38 (compatible with CUDA 12.1)." -ForegroundColor Green
    } else {
        Write-Host "WARNING: MSVC 14.38 toolset not found. Install 'MSVC v143 ... (v14.38-17.8)' in the VS Installer, or the CUDA build will fail (STL1002)." -ForegroundColor Yellow
    }
    Enter-VsDevShell -VsInstallPath $vsPath -DevCmdArguments $devArgs -SkipAutomaticLocation
    # Required by PyTorch's cpp_extension builder when the VC env is active,
    # otherwise it refuses to compile (the "DISTUTILS_USE_SDK is not set" error).
    $env:DISTUTILS_USE_SDK = "1"
    # CUDA 12.1's nvcc rejects MSVC toolsets newer than ~14.39. VS2022 Build Tools
    # ship 14.44+, so allow the (slightly) newer host compiler. nvcc reads this var.
    $env:NVCC_PREPEND_FLAGS = "-allow-unsupported-compiler"
    Write-Host "VS Dev Shell active. Compiler:" -ForegroundColor Green
    (Get-Command cl -ErrorAction SilentlyContinue).Source
}

# --- 1. Create env ---
conda create --name gaussian-avatars -y python=3.10
conda activate gaussian-avatars

# --- 2. CUDA toolkit + ninja (12.1.1 to match torch cu121) ---
conda install -y -c "nvidia/label/cuda-12.1.1" cuda-toolkit ninja

# --- 3. Make CUDA_PATH persistent for this env, then re-activate to load it ---
conda env config vars set CUDA_PATH="$env:CONDA_PREFIX"
conda deactivate
conda activate gaussian-avatars

# --- 4. Enter the VS2022 x64 dev environment (gives us cl.exe) ---
Enter-VS2022x64

# --- 5. PyTorch (cu121) -- pinned to the tested 2.2.0 ---
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

# torch 2.2.0 was built against NumPy 1.x -- keep numpy < 2 or torch fails to import
pip install "numpy<2"

python -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"

# --- 6. Rest of requirements (compiles diff-gaussian-rasterization, simple-knn, nvdiffrast) ---
# --no-build-isolation is REQUIRED: these CUDA extensions need to see the installed
# torch at build time, which pip's default isolated build env hides.
Set-Location "$ROOT\GaussianAvatars"
pip install -r requirements.txt --no-build-isolation

Write-Host "`ngaussian-avatars env ready. Next: place FLAME files (03_flame_checklist.md), then 04_demo_pretrained.ps1" -ForegroundColor Green
