REM ---------------- GIT ----------------
echo Sincronizando web_data con GitHub...


set "GIT_EXE=C:\Program Files\Git\bin\git.exe"
cd /d "%~dp0"

echo FORZANDO reemplazo de web_data en GitHub...

REM Solo añadir web_data
"%GIT_EXE%" add -A web_data

REM Crear commit
for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd_HH-mm-ss"') do set "fecha=%%i"

"%GIT_EXE%" commit -m "force replace web_data !fecha!"

REM ⚠️ CLAVE: push forzado (sobrescribe GitHub)
"%GIT_EXE%" push origin main --force

echo Listo: GitHub ahora tiene EXACTAMENTE tu web_data local.
pause