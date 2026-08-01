@echo off
cd /d "%~dp0"
python --version
pause
python -m streamlit run app.py --server.maxUploadSize 1000
pause
