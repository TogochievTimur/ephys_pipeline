@echo off
echo Installing required packages...
echo.
python -m pip install -r requirements.txt
python -m pip install streamlit==1.40.0 --force-reinstall
echo.
echo ============================================
echo Installation complete!
echo Now double-click run.bat to start the app.
echo ============================================
pause
