@echo off
cd /d C:\Users\PC\Desktop\tamc-monitor-web

echo.
echo === Subiendo SOLO event_analysis ===

REM Añadir solo la carpeta
git add event_analysis

REM Commit solo si hay cambios
git diff --cached --quiet
IF %ERRORLEVEL%==0 (
    echo No hay cambios en event_analysis.
    pause
    exit
)

git commit -m "Update event_analysis (real events)"

REM Subir a GitHub
git push origin main

echo.
echo === SUBIDA COMPLETADA ===
pause