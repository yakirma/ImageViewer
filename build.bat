@echo off
setlocal EnableDelayedExpansion
echo Starting ImageViewer Build...

:: Fix for running from UNC paths (network shares)
pushd "%~dp0"
echo Working Directory: !CD!

:: Check if build.py exists (verify pushd worked)
if not exist "build.py" (
    echo.
    echo Error: build.py not found!
    echo.
    echo Contents of this directory:
    dir /b
    echo.
    echo The script cannot find build.py.
    echo Please try moving the entire folder to a local path like C:\ImageViewer
    pause
    exit /b 1
)

:: Try finding Python
set PYTHON_FOUND=0
set PYTHON_CMD=python

python --version >nul 2>&1
if !errorlevel! equ 0 (
    set PYTHON_FOUND=1
    set PYTHON_CMD=python
) else (
    echo 'python' not found, checking 'py' launcher...
    py --version >nul 2>&1
    if !errorlevel! equ 0 (
        set PYTHON_FOUND=1
        set PYTHON_CMD=py
    )
)

if !PYTHON_FOUND! neq 1 (
    echo.
    echo Error: Python is not found!
    echo ---------------------------------------------------------
    echo 1. Go to python.org and install Python 3.10+
    echo 2. IMPORTANT: Check "Add Python to PATH" during install
    echo ---------------------------------------------------------
    pause
    exit /b 1
)

echo Using Python command: !PYTHON_CMD!
!PYTHON_CMD! build.py
if !errorlevel! neq 0 (
    echo Build failed!
    pause
    exit /b !errorlevel!
)

echo.
echo Build script finished.
pause
popd
