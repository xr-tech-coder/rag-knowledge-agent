@echo off
cd /d %~dp0

if not exist ".venv\Scripts\python.exe" (
    echo [x] venv not found.
    pause
    exit /b 1
)

if "%API_KEY%"=="" (
    echo.
    echo Paste your API key and press Enter ^(this run only^):
    set /p API_KEY=Key: 
)
if "%API_KEY%"=="" (
    echo [x] no key, exit.
    pause
    exit /b 1
)

if "%BASE_URL%"=="" set BASE_URL=https://api.deepseek.com
if "%MODEL%"=="" set MODEL=deepseek-chat

echo.
echo === stream diagnosis ===  model=%MODEL%
.venv\Scripts\python.exe diag_stream.py
echo ----------------------------------------
pause
