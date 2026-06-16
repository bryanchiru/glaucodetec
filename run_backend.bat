@echo off
title GlaucoDetec API
echo ==========================================
echo   GlaucoDetec - Iniciando API FastAPI
echo ==========================================
cd /d "%~dp0backend"
set PYTHON="C:\Users\bryan\Downloads\Inteligencia Artificial\python.exe"
%PYTHON% -m pip install -r requirements.txt -q
echo.
echo API disponible en: http://localhost:8000
echo Docs:              http://localhost:8000/docs
echo.
%PYTHON% -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
pause
