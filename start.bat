@echo off
title ISL Recognition System
color 0A
echo.
echo  ============================================
echo    ISL RECOGNITION SYSTEM - Windows Launcher
echo  ============================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
  echo  ERROR: Python not found.
  echo  Download and install Python 3.10+ from https://python.org
  echo  Make sure to check "Add Python to PATH" during install.
  pause
  exit /b 1
)

echo  [1/2] Installing requirements...
pip install flask flask-socketio mediapipe opencv-python scikit-learn numpy pyttsx3 --quiet --upgrade
if errorlevel 1 (
  echo  WARNING: Some packages may have failed to install.
  echo  Try running as Administrator or check your internet connection.
)

echo  [2/2] Starting ISL Recognition System...
echo.
echo  Open your browser at:  http://localhost:5000
echo  Press Ctrl+C to stop.
echo.

python isl_main.py
pause
