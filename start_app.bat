@echo off
title DataMind AI Launcher
echo ===================================================
echo Starting DataMind AI Backend and Frontend Services...
echo ===================================================

cd /d "%~dp0"

echo Starting FastAPI Backend on port 8000...
start "DataMind Backend" /min "C:\Users\USER\anaconda3\python.exe" -m uvicorn app.main:app --host 0.0.0.0 --port 8000

timeout /t 3 /nobreak >nul

echo Starting Streamlit Frontend on port 8501...
start "DataMind Frontend" "C:\Users\USER\anaconda3\python.exe" -m streamlit run streamlit_app/Home.py --server.port 8501

echo ===================================================
echo Services Started!
echo Streamlit UI: http://localhost:8501 or http://127.0.0.1:8501
echo FastAPI Backend: http://localhost:8000
echo ===================================================
