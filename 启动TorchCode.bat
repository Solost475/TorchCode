@echo off
chcp 65001 >nul
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" "torchcode_launcher.py"
) else if exist "venv\Scripts\python.exe" (
    "venv\Scripts\python.exe" "torchcode_launcher.py"
) else (
    python "torchcode_launcher.py"
)

if errorlevel 1 pause
