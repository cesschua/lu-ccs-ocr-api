@echo off
title Dean OCR Server — EasyOCR + Tesseract
color 0A
cd /d "%~dp0"

echo ============================================
echo   Dean OCR Server  |  EasyOCR + Tesseract
echo ============================================
echo.

:: Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.9+
    pause
    exit /b 1
)

echo [1/2] Checking and installing dependencies...
python -m pip install flask flask-cors opencv-python-headless numpy Pillow reportlab easyocr pytesseract torchvision torch --index-url https://download.pytorch.org/whl/cpu

echo.
echo [2/2] Starting server...
echo.
echo TIP: If Tesseract error occurs, install it from:
echo https://github.com/UB-Mannheim/tesseract/wiki
echo.

python app.py

pause
echo Checking dependencies...
pip install -q easyocr pypdf pdf2image flask flask-cors pillow numpy opencv-python-headless 2>nul
echo Dependencies OK.
echo.

echo Starting OCR Server on http://localhost:5050 ...
echo.
echo  EasyOCR is loading language models — this may take
echo  30-60 seconds on first run. Please wait.
echo.
echo  Keep this window open while using the Dean Scanner.
echo  Press Ctrl+C to stop the server.
echo ============================================
echo.

python app.py

pause
