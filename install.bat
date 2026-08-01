@echo off
echo Installing required packages...
echo.


set PATH=%PATH%;C:\Python313;C:\Python313\Scripts;C:\Python312;C:\Python312\Scripts;C:\Python311;C:\Python311\Scripts


for /f "tokens=*" %%i in ('where python 2^>nul') do set PYTHON_PATH=%%i

if defined PYTHON_PATH (
    echo Python found. Installing...
    pip install -r requirements.txt
) else (
    echo ERROR: Python not found.
    echo.
    echo Please reinstall Python from python.org
    echo and check "Add Python to PATH" during installation.
)

echo.
pause