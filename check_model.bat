@echo off
cd /d %~dp0

if not exist ".venv\Scripts\python.exe" (
    echo [x] venv not found.
    pause
    exit /b 1
)

if "%API_KEY%"=="" if "%DASHSCOPE_API_KEY%"=="" (
    echo.
    echo Paste your API key and press Enter ^(this run only, not saved anywhere^):
    set /p API_KEY=Key:
)
if "%API_KEY%"=="" if "%DASHSCOPE_API_KEY%"=="" (
    echo [x] no key, exit.
    pause
    exit /b 1
)

rem 换模型只改这一行，或先 set MODEL=xxx 再运行
if "%MODEL%"=="" set MODEL=qwen3.6-flash
if "%BASE_URL%"=="" set BASE_URL=https://ws-rgi549ol413mnbmz.cn-beijing.maas.aliyuncs.com/compatible-mode/v1

echo.
echo === model self-check ===  model=%MODEL%
.venv\Scripts\python.exe check_model.py
echo ----------------------------------------
pause
