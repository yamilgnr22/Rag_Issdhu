@echo off
REM Arranca el reranker con doble clic, sin abrir PowerShell a mano.
REM Sin este servicio el sistema no falla: degrada en silencio al rerank
REM heuristico y pierde ~0.20 de hit@1.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-reranker.ps1" %*
if errorlevel 1 (
    echo.
    echo El arranque fallo. Revisa el log indicado arriba.
    pause
)
