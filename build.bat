@echo off
echo Starting ImageViewer Build...
python build.py
if %errorlevel% neq 0 (
    echo Build failed!
    pause
    exit /b %errorlevel%
)
echo Build script finished.
pause
