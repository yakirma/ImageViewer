@echo off
setlocal enableextensions
echo Starting ImageViewer Build...

:: Fix for running from UNC paths (network shares)
pushd "%~dp0"
echo Working Directory: %CD%

:: Check if build.py exists (verify pushd worked)
if not exist "build.py" (
    echo.
    echo Error: build.py not found!
    echo The script failed to switch to the correct directory.
    echo Please move the folder to a local drive (e.g., C:\ImageViewer) and try again.
    pause
    exit /b 1
)

:: Try finding Python
set PYTHON_CMD=python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 'python' command not found, trying 'py' launcher...
    py --version >nul 2>&1
    if %errorlevel% equ 0 (
         set PYTHON_CMD=py
    ) else (
        echo.
        echo Error: Python is not found!
        echo ---------------------------------------------------------
        echo 1. Go to python.org and install Python 3.10+
        echo 2. IMPORTANT: Check "Add Python to PATH" during install
        echo ---------------------------------------------------------
        pause
        exit /b 1
    )
)

echo Using Python command: %PYTHON_CMD%
%PYTHON_CMD% build.py
if %errorlevel% neq 0 (
    echo Build failed!
    pause
    exit /b %errorlevel%
)
echo Build script finished.
pause
popd
