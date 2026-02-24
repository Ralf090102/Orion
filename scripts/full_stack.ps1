<#
.SYNOPSIS
    Starts the full Orion stack (backend + frontend) simultaneously.

.DESCRIPTION
    This script launches both the FastAPI backend and SvelteKit frontend.
    It can run in local mode (opens browser) or VM mode (headless).

.PARAMETER Mode
    Specify 'local' or 'vm'. If not provided, the script will auto-detect or prompt.

.EXAMPLE
    .\full_stack.ps1
    .\full_stack.ps1 -Mode local
    .\full_stack.ps1 -Mode vm
#>

param(
    [Parameter(Mandatory=$false)]
    [ValidateSet('local', 'vm', '')]
    [string]$Mode = ''
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

# Colors for output
function Write-Info { Write-Host "[INFO] $args" -ForegroundColor Cyan }
function Write-Success { Write-Host "[OK] $args" -ForegroundColor Green }
function Write-Warn { Write-Host "[WARN] $args" -ForegroundColor Yellow }
function Write-Err { Write-Host "[ERROR] $args" -ForegroundColor Red }

# Detect if running on VM or local
function Detect-Environment {
    # Check for common VM indicators
    $isVM = $false
    
    # GCP VM usually has google in the hostname or specific metadata
    $hostname = (Get-WmiObject Win32_ComputerSystem).Name
    if ($hostname -match "gcp|google|vm|instance") {
        $isVM = $true
    }
    
    # Check if running via SSH (no console window width means likely remote)
    if ($env:SSH_CONNECTION -or $env:SSH_CLIENT) {
        $isVM = $true
    }
    
    # Check for display (if no display, likely VM/headless)
    try {
        Add-Type -AssemblyName System.Windows.Forms -ErrorAction SilentlyContinue
        $screens = [System.Windows.Forms.Screen]::AllScreens
        if ($screens.Count -eq 0) { $isVM = $true }
    } catch {
        # If we can't load Windows Forms, might be headless
    }
    
    return $isVM
}

# Prompt user for mode if not specified
function Get-RunMode {
    if ($Mode -ne '') {
        return $Mode
    }
    
    $detected = Detect-Environment
    if ($detected) {
        Write-Info "Detected VM/remote environment"
        return 'vm'
    }
    
    # Ask user
    Write-Host ""
    Write-Host "Select run mode:" -ForegroundColor White
    Write-Host "  [1] Local  - Opens browser, localhost access" -ForegroundColor Gray
    Write-Host "  [2] VM     - Headless, network access (0.0.0.0)" -ForegroundColor Gray
    Write-Host ""
    
    $choice = Read-Host "Enter choice (1 or 2)"
    
    switch ($choice) {
        "1" { return 'local' }
        "2" { return 'vm' }
        default { 
            Write-Warn "Invalid choice, defaulting to local"
            return 'local'
        }
    }
}

# Check prerequisites
function Check-Prerequisites {
    Write-Info "Checking prerequisites..."
    
    # Check Python
    try {
        $pythonVersion = python --version 2>&1
        Write-Success "Python: $pythonVersion"
    } catch {
        Write-Err "Python not found. Please install Python 3.10+"
        exit 1
    }
    
    # Check Node.js
    try {
        $nodeVersion = node --version 2>&1
        Write-Success "Node.js: $nodeVersion"
    } catch {
        Write-Err "Node.js not found. Please install Node.js 18+"
        exit 1
    }
    
    # Check if venv exists
    $venvPath = Join-Path $ProjectRoot ".venv"
    if (-not (Test-Path $venvPath)) {
        Write-Warn "Virtual environment not found at $venvPath"
        Write-Info "Creating virtual environment..."
        python -m venv $venvPath
    }
    
    # Check if node_modules exists
    $nodeModules = Join-Path $ProjectRoot "frontend\node_modules"
    if (-not (Test-Path $nodeModules)) {
        Write-Warn "node_modules not found"
        Write-Info "Installing frontend dependencies..."
        Push-Location (Join-Path $ProjectRoot "frontend")
        npm install
        Pop-Location
    }
    
    Write-Success "Prerequisites OK"
}

# Global process tracking
$script:BackendJob = $null
$script:FrontendJob = $null

# Cleanup function
function Stop-Stack {
    Write-Host ""
    Write-Info "Shutting down Orion stack..."
    
    if ($script:BackendJob) {
        Stop-Job -Job $script:BackendJob -ErrorAction SilentlyContinue
        Remove-Job -Job $script:BackendJob -Force -ErrorAction SilentlyContinue
        Write-Success "Backend stopped"
    }
    
    if ($script:FrontendJob) {
        Stop-Job -Job $script:FrontendJob -ErrorAction SilentlyContinue
        Remove-Job -Job $script:FrontendJob -Force -ErrorAction SilentlyContinue
        Write-Success "Frontend stopped"
    }
    
    # Kill any lingering processes on the ports
    $backendPort = 8000
    $frontendPort = 5173
    
    Get-NetTCPConnection -LocalPort $backendPort -ErrorAction SilentlyContinue | 
        ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
    
    Get-NetTCPConnection -LocalPort $frontendPort -ErrorAction SilentlyContinue | 
        ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
    
    Write-Success "Cleanup complete"
}

# Register cleanup on script exit
Register-EngineEvent -SourceIdentifier PowerShell.Exiting -Action { Stop-Stack } -ErrorAction SilentlyContinue

# Main execution
function Start-Stack {
    param([string]$RunMode)
    
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Magenta
    Write-Host "       ORION FULL STACK LAUNCHER       " -ForegroundColor Magenta
    Write-Host "========================================" -ForegroundColor Magenta
    Write-Host ""
    
    Check-Prerequisites
    
    Write-Info "Starting in $RunMode mode..."
    Write-Host ""
    
    # Determine host binding
    $backendHost = if ($RunMode -eq 'vm') { "0.0.0.0" } else { "127.0.0.1" }
    $frontendHost = if ($RunMode -eq 'vm') { "0.0.0.0" } else { "localhost" }
    
    # Activate venv path
    $venvActivate = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"
    $frontendDir = Join-Path $ProjectRoot "frontend"
    
    # Start Backend
    Write-Info "Starting backend on $backendHost`:8000..."
    $script:BackendJob = Start-Job -ScriptBlock {
        param($venvActivate, $projectRoot, $host_bind)
        Set-Location $projectRoot
        & $venvActivate
        uvicorn backend.app:app --host $host_bind --port 8000 --reload
    } -ArgumentList $venvActivate, $ProjectRoot, $backendHost
    
    # Give backend a moment to start
    Start-Sleep -Seconds 2
    
    # Start Frontend
    Write-Info "Starting frontend on $frontendHost`:5173..."
    $script:FrontendJob = Start-Job -ScriptBlock {
        param($frontendDir, $host_bind)
        Set-Location $frontendDir
        if ($host_bind -eq "0.0.0.0") {
            npm run dev -- --host 0.0.0.0
        } else {
            npm run dev
        }
    } -ArgumentList $frontendDir, $frontendHost
    
    # Wait for services to be ready
    Start-Sleep -Seconds 3
    
    # Display access info
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "         ORION IS RUNNING              " -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    
    if ($RunMode -eq 'vm') {
        $externalIP = (Invoke-WebRequest -Uri "http://checkip.amazonaws.com" -UseBasicParsing -TimeoutSec 5).Content.Trim()
        Write-Host "  Frontend:  http://$externalIP`:5173" -ForegroundColor White
        Write-Host "  Backend:   http://$externalIP`:8000" -ForegroundColor White
        Write-Host "  API Docs:  http://$externalIP`:8000/docs" -ForegroundColor Gray
    } else {
        Write-Host "  Frontend:  http://localhost:5173" -ForegroundColor White
        Write-Host "  Backend:   http://localhost:8000" -ForegroundColor White
        Write-Host "  API Docs:  http://localhost:8000/docs" -ForegroundColor Gray
    }
    
    Write-Host ""
    Write-Host "  Press Ctrl+C to stop all services" -ForegroundColor Yellow
    Write-Host ""
    
    # Open browser in local mode
    if ($RunMode -eq 'local') {
        Start-Process "http://localhost:5173"
    }
    
    # Stream logs from both jobs
    Write-Info "Streaming logs (Backend=Blue, Frontend=Green)..."
    Write-Host ""
    
    try {
        while ($true) {
            # Get backend output
            $backendOutput = Receive-Job -Job $script:BackendJob -ErrorAction SilentlyContinue
            if ($backendOutput) {
                $backendOutput | ForEach-Object { Write-Host "[Backend] $_" -ForegroundColor Blue }
            }
            
            # Get frontend output
            $frontendOutput = Receive-Job -Job $script:FrontendJob -ErrorAction SilentlyContinue
            if ($frontendOutput) {
                $frontendOutput | ForEach-Object { Write-Host "[Frontend] $_" -ForegroundColor Green }
            }
            
            # Check if jobs are still running
            if ($script:BackendJob.State -eq 'Failed') {
                Write-Err "Backend crashed!"
                Receive-Job -Job $script:BackendJob
            }
            if ($script:FrontendJob.State -eq 'Failed') {
                Write-Err "Frontend crashed!"
                Receive-Job -Job $script:FrontendJob
            }
            
            Start-Sleep -Milliseconds 500
        }
    } finally {
        Stop-Stack
    }
}

# Run
try {
    $runMode = Get-RunMode
    Start-Stack -RunMode $runMode
} catch {
    Write-Err $_.Exception.Message
    Stop-Stack
    exit 1
}
