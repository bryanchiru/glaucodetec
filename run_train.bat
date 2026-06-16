@echo off
title GlaucoDetec - Entrenamiento
echo ==========================================
echo   GlaucoDetec - Entrenando modelo IA
echo   EfficientNetB0 + EyePACS AIROGS
echo ==========================================
set PYTHON="C:\Users\bryan\Downloads\Inteligencia Artificial\python.exe"
%PYTHON% -m pip install torch torchvision scikit-learn -q
%PYTHON% ml\train.py
pause
