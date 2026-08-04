<#
.SYNOPSIS
    Levanta el microservicio de reranking del proyecto.

.DESCRIPTION
    Sin este servicio el sistema no falla: degrada en silencio al rerank
    heuristico, que cuesta unos 0,20 de hit@1 (un tercio de la calidad de
    recuperacion). Por eso conviene arrancarlo antes de trabajar con el RAG.

    El modelo se carga de forma perezosa: recien arrancado ocupa ~700 MB de RAM
    y nada de VRAM. Los ~2 GB de VRAM se reservan con la primera consulta.

.PARAMETER Stop
    Detiene el servicio en lugar de arrancarlo.

.PARAMETER Status
    Solo consulta el estado y sale.

.EXAMPLE
    .\scripts\start-reranker.ps1
    .\scripts\start-reranker.ps1 -Status
    .\scripts\start-reranker.ps1 -Stop
#>
[CmdletBinding()]
param(
    [switch]$Stop,
    [switch]$Status
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv-reranker\Scripts\python.exe'
$appDir = Join-Path $projectRoot 'apps\reranker'
$logFile = Join-Path $projectRoot '.data\reranker_stdout.log'
$port = 7997
$healthUrl = "http://localhost:$port/health"

function Get-RerankerProcesses {
    Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -like '*uvicorn*' -and $_.CommandLine -like '*reranker*' }
}

function Get-RerankerHealth {
    try {
        return Invoke-RestMethod -Uri $healthUrl -TimeoutSec 4 -ErrorAction Stop
    } catch {
        return $null
    }
}

function Show-Status {
    $health = Get-RerankerHealth
    if ($null -eq $health) {
        Write-Host "reranker: NO responde en el puerto $port" -ForegroundColor Yellow
        Write-Host "  el sistema degradaria al rerank heuristico (~0.20 menos de hit@1)"
        return $false
    }
    $vram = if ($null -ne $health.vram_used_mb) { "$($health.vram_used_mb) MB" } else { 'n/d' }
    $modelo = if ($health.cached_models.Count -gt 0) { 'cargado' } else { 'aun no cargado (perezoso)' }
    Write-Host "reranker: OK" -ForegroundColor Green
    Write-Host "  device      : $($health.device)"
    Write-Host "  modelo      : $($health.default_model) [$modelo]"
    Write-Host "  max_length  : $($health.max_length)"
    Write-Host "  VRAM en uso : $vram"
    return $true
}

if ($Status) {
    if (Show-Status) { exit 0 } else { exit 1 }
}

if ($Stop) {
    $procesos = Get-RerankerProcesses
    if (-not $procesos) {
        Write-Host "no habia ningun reranker corriendo"
        exit 0
    }
    foreach ($proceso in $procesos) {
        Stop-Process -Id $proceso.ProcessId -Force -ErrorAction SilentlyContinue
        Write-Host "detenido PID $($proceso.ProcessId)"
    }
    exit 0
}

# --- arranque ---
if (Get-RerankerHealth) {
    Write-Host "ya estaba corriendo:" -ForegroundColor Green
    Show-Status | Out-Null
    exit 0
}

if (-not (Test-Path $python)) {
    Write-Host "no se encuentra el entorno del reranker: $python" -ForegroundColor Red
    Write-Host "  crealo con: py -3.12 -m venv .venv-reranker; .\.venv-reranker\Scripts\pip install -e .\apps\reranker"
    exit 1
}

# Un proceso puede estar vivo sin responder (arrancando o colgado).
$huerfanos = Get-RerankerProcesses
if ($huerfanos) {
    Write-Host "hay procesos del reranker que no responden; se reinician" -ForegroundColor Yellow
    foreach ($proceso in $huerfanos) {
        Stop-Process -Id $proceso.ProcessId -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 2
}

$logDir = Split-Path -Parent $logFile
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }

Write-Host "arrancando el reranker (importar torch tarda ~1 min la primera vez)..."
Start-Process -FilePath $python `
    -ArgumentList @('-m', 'uvicorn', 'app.main:app', '--host', '0.0.0.0', '--port', "$port", '--app-dir', $appDir) `
    -WorkingDirectory $projectRoot `
    -RedirectStandardOutput $logFile `
    -RedirectStandardError "$logFile.err" `
    -WindowStyle Hidden

for ($i = 1; $i -le 30; $i++) {
    Start-Sleep -Seconds 5
    if (Get-RerankerHealth) {
        Show-Status | Out-Null
        Write-Host ""
        Write-Host "log: $logFile"
        exit 0
    }
    Write-Host "  esperando... ($($i * 5)s)"
}

Write-Host "no respondio en 150s. Revisa el log:" -ForegroundColor Red
Write-Host "  $logFile"
Write-Host "  $logFile.err"
exit 1
