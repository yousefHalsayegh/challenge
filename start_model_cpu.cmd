@echo off
setlocal
cd /d "%~dp0"
if "%~1"=="" (
  python run.py --serve
) else (
  python run.py --serve --model-path "%~1"
)
if errorlevel 1 pause
