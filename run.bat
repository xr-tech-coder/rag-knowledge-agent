@echo off
cd /d %~dp0

if not exist ".venv\Scripts\python.exe" (
    echo [x] venv not found. Run first:
    echo     python -m venv .venv
    pause
    exit /b 1
)

if "%API_KEY%"=="" if "%DASHSCOPE_API_KEY%"=="" (
    echo.
    echo Paste your API key and press Enter ^(this run only^):
    set /p API_KEY=Key: 
)

if "%BASE_URL%"=="" set BASE_URL=https://api.deepseek.com
if "%MODEL%"=="" set MODEL=deepseek-chat

rem THINK=0 turns thinking off: ~35s -> ~1s first token, 20x cheaper
if "%THINK%"=="" set THINK=0

echo.
echo model=%MODEL%  THINK=%THINK%
echo ----------------------------------------
.venv\Scripts\python.exe main.py
echo ----------------------------------------
echo Set THINK=1 before running this file to enable thinking.
pause
