# enter_build_env.ps1
# Activates the VS2022 14.38 build environment in the CURRENT shell so that
# runtime CUDA compilation works (e.g. nvdiffrast JIT-compiles its plugin on
# first use). Dot-source this AFTER `conda activate <env>`:
#
#     conda activate VHAP
#     . ".\setup\enter_build_env.ps1"
#     python vhap/flame_viewer.py --param_path ...
#
# Needed for: VHAP track.py / flame_viewer.py, and GaussianAvatars "show mesh"
# in local_viewer — anything that uses nvdiffrast (which compiles at runtime).

$vswhere = "C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe"
$vsPath = & $vswhere -version "[17.0,18.0)" -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1
if (-not $vsPath) { throw "VS2022 (v17) C++ toolset not found." }

Import-Module (Join-Path $vsPath "Common7\Tools\Microsoft.VisualStudio.DevShell.dll")

# Pin the CUDA-12.1-compatible 14.38 toolset if present
$devArgs = "-arch=x64 -host_arch=x64"
$t1438 = Get-ChildItem "$vsPath\VC\Tools\MSVC" -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -like "14.38.*" } | Select-Object -First 1
if ($t1438) { $devArgs += " -vcvars_ver=14.38"; Write-Host "Pinning MSVC 14.38 (CUDA 12.1 compatible)." -ForegroundColor Green }
else { Write-Host "WARNING: MSVC 14.38 not found; runtime CUDA compile may fail." -ForegroundColor Yellow }

Enter-VsDevShell -VsInstallPath $vsPath -DevCmdArguments $devArgs -SkipAutomaticLocation

# Flags required for PyTorch's runtime extension build
$env:DISTUTILS_USE_SDK = "1"
$env:NVCC_PREPEND_FLAGS = "-allow-unsupported-compiler"
# Use a clean, space-free cache dir so all JIT-built objects share ONE compiler
# (mixing 14.38 + 14.51 objects causes __std_*_1 link errors). Off the C: drive.
$env:TORCH_EXTENSIONS_DIR = "$env:TEMP\torch_ext"

Write-Host "Build env ready. Compiler:" -ForegroundColor Green
(Get-Command cl -ErrorAction SilentlyContinue).Source
