@echo off
cd /d "%~dp0"
python -m streamlit run app.py --server.maxUploadSize 1000
pause
