@echo off
chcp 65001 >nul
title PaddleOCR Web Studio (GPU RTX 3050 Ti)
echo ===============================================================
echo   Dang khoi dong PaddleOCR Web Studio...
echo   Giao dien se tu dong mo tai: http://127.0.0.1:8000
echo ===============================================================
set PYTHONUTF8=1
start "" http://127.0.0.1:8000
"d:\Code làm dữ liệu\PaddleOCR\.venv\Scripts\python.exe" "d:\Code làm dữ liệu\PaddleOCR\web_ui\app.py"
pause
