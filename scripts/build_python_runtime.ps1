<#
.SYNOPSIS
    Builds the portable Python runtime bundled into the Tauri desktop app.

.DESCRIPTION
    Orion's Tauri build ships a self-contained Python interpreter alongside
    the backend/src source (see tauri.conf.json's bundle.resources), so a
    packaged install doesn't depend on a dev .venv existing on the target
    machine. This script downloads the official Python embeddable package,
    enables site-packages, bootstraps pip, and installs requirements.txt
    into it.

    Run this once (or whenever requirements.txt changes) before `npx tauri
    build`. The output directory (python-runtime/) is gitignored -- it's a
    build artifact, not source.

.PARAMETER PythonVersion
    Python version to bundle. Should match (or be compatible with) the
    interpreter used to develop against -- see .venv's version.

.EXAMPLE
    .\scripts\build_python_runtime.ps1
    .\scripts\build_python_runtime.ps1 -PythonVersion 3.12.6
#>

param(
    [string]$PythonVersion = "3.12.6"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$RuntimeDir = Join-Path $RepoRoot "python-runtime"
$ShortVersion = ($PythonVersion -split '\.')[0..1] -join ''  # e.g. "3.12.6" -> "312"

if (Test-Path $RuntimeDir) {
    Write-Host "Removing existing python-runtime/ ..."
    Remove-Item -Recurse -Force $RuntimeDir
}
New-Item -ItemType Directory -Force -Path $RuntimeDir | Out-Null

Write-Host "Downloading Python $PythonVersion embeddable package..."
$EmbedUrl = "https://www.python.org/ftp/python/$PythonVersion/python-$PythonVersion-embed-amd64.zip"
$EmbedZip = Join-Path $RuntimeDir "python-embed.zip"
Invoke-WebRequest -Uri $EmbedUrl -OutFile $EmbedZip

Write-Host "Extracting..."
Expand-Archive -Path $EmbedZip -DestinationPath $RuntimeDir -Force
Remove-Item $EmbedZip

# Enable site-packages. The embeddable package ships in "isolated path" mode
# by default (no site-packages, no CWD/PYTHONPATH on sys.path) -- `..` makes
# the repo root (where this runtime sits as a sibling of backend/ and src/,
# both in dev and in the bundled app resource dir) importable via `-m`.
$PthFile = Join-Path $RuntimeDir "python$ShortVersion._pth"
@"
python$ShortVersion.zip
.
Lib\site-packages
..

import site
"@ | Set-Content -Path $PthFile -Encoding ASCII

Write-Host "Bootstrapping pip..."
$GetPip = Join-Path $RuntimeDir "get-pip.py"
Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $GetPip
& (Join-Path $RuntimeDir "python.exe") $GetPip --no-warn-script-location
Remove-Item $GetPip

Write-Host "Installing requirements.txt (this pulls torch/transformers/chromadb -- expect several minutes)..."
& (Join-Path $RuntimeDir "python.exe") -m pip install -r (Join-Path $RepoRoot "requirements.txt") --no-warn-script-location

$SizeGB = [math]::Round((Get-ChildItem $RuntimeDir -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1GB, 2)
Write-Host "Done. python-runtime/ is $SizeGB GB. Run 'npx tauri build' from frontend/ to package it."
