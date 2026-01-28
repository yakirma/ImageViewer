@echo off
:: Fix for running from UNC paths (network shares)
pushd "%~dp0"

echo Starting ImageViewer Build...

:: Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not found in your PATH.
    echo Please install Python from python.org (ensure "Add Python to PATH" is checked)
    echo or run this script from a terminal where Python is configured.
    pause
    exit /b 1
)

python build.py
if %errorlevel% neq 0 (
    echo Build failed!
    pause
    exit /b %errorlevel%
)
echo Build script finished.
pause
