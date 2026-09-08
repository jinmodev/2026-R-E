@echo off
setlocal
cd /d "%~dp0"

where pyw.exe >nul 2>nul
if %errorlevel%==0 (
    start "Robot Path Lab" pyw.exe app.py
    exit /b 0
)

where pythonw.exe >nul 2>nul
if %errorlevel%==0 (
    start "Robot Path Lab" pythonw.exe app.py
    exit /b 0
)

set "ROBOT_LAB_CODEX_PYTHON=%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\pythonw.exe"
if exist "%ROBOT_LAB_CODEX_PYTHON%" (
    start "Robot Path Lab" "%ROBOT_LAB_CODEX_PYTHON%" app.py
    exit /b 0
)

echo Python 3.11 or newer could not be found.
echo Install Python from https://www.python.org/downloads/ and try again.
pause
exit /b 1

