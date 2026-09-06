# _conda_bootstrap.ps1
# Makes `conda` (and `conda activate`) work in ANY PowerShell window, even if it
# isn't the "Anaconda PowerShell Prompt". Dot-sourced at the top of the other scripts.
# If conda is already available, this does nothing.

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    $condaCandidates = @(
        "$env:USERPROFILE\miniconda3",
        "$env:USERPROFILE\anaconda3",
        "$env:LOCALAPPDATA\miniconda3",
        "$env:LOCALAPPDATA\anaconda3",
        "$env:LOCALAPPDATA\Continuum\anaconda3",
        "C:\ProgramData\miniconda3",
        "C:\ProgramData\Anaconda3",
        "C:\Miniconda3",
        "C:\Anaconda3"
    )
    $condaRoot = $condaCandidates | Where-Object { Test-Path "$_\Scripts\conda.exe" } | Select-Object -First 1
    if (-not $condaRoot) {
        throw "Could not locate a conda install. Either run this from the 'Anaconda PowerShell Prompt', or run 'conda init powershell' once in your conda terminal and reopen PowerShell."
    }
    Write-Host "Bootstrapping conda from $condaRoot ..." -ForegroundColor DarkGray
    (& "$condaRoot\Scripts\conda.exe" "shell.powershell" "hook") | Out-String | Invoke-Expression
}
