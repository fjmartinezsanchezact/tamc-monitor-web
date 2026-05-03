@echo off
chcp 65001 >nul
title TAMC–FRANJAMAR Auto Pipeline + Deploy

echo ==========================================
echo TAMC–FRANJAMAR PIPELINE + DEPLOY
echo ==========================================
echo.

REM ====== CONFIG ======
set PROJECT_DIR=C:\Users\PC\Desktop\tamcsismico
set GIT_EXE="C:\Program Files\Git\bin\git.exe"

cd /d %PROJECT_DIR%

REM ====== ACTIVATE ENV ======
call conda activate tamc

REM ====== STEP 1: PIPELINE ======
echo.
echo [1/4] Ejecutando pipeline principal...
python full_pipeline_franjamarv3_multimonitor_global.py
IF ERRORLEVEL 1 (
    echo.
    echo ERROR: fallo en el pipeline. Abortando.
    pause
    exit /b 1
)

REM ====== STEP 2: GENERATE SUMMARIES ======
echo.
echo [2/4] Generando network summaries (logica conservadora)...
python generate_network_summaries_FINAL_conservative.py
IF ERRORLEVEL 1 (
    echo.
    echo ERROR: fallo generando summaries. Abortando.
    pause
    exit /b 1
)

REM ====== STEP 3: BUILD WEB DATA ======
echo.
echo [3/4] Construyendo web_data...
python build_web_data_from_here.py
IF ERRORLEVEL 1 (
    echo.
    echo ERROR: fallo construyendo web_data. Abortando.
    pause
    exit /b 1
)

REM ====== STEP 4: GIT PUSH ======
echo.
echo [4/4] Subiendo cambios a GitHub...

%GIT_EXE% add -A
%GIT_EXE% commit -m "auto update pipeline + summaries"
%GIT_EXE% push origin main

IF ERRORLEVEL 1 (
    echo.
    echo ERROR: fallo al hacer push. Revisa Git.
    pause
    exit /b 1
)

echo.
echo ==========================================
echo TODO OK - DATOS ACTUALIZADOS EN STREAMLIT
echo ==========================================
echo.

pause