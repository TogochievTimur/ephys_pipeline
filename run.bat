@echo off
call C:\anaconda\Scripts\activate.bat
cd /d "C:\Users\Ти\ephys"
streamlit run app.py --server.maxUploadSize 1000
pause