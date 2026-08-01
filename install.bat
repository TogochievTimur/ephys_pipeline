@echo off
echo Installing required packages...
echo.

for /f "tokens=*" %%i in ('where python 2^>nul') do set PYTHON=%%i

if defined PYTHON (
    echo Using: %PYTHON%
    %PYTHON% --version
    echo.
    %PYTHON% -m pip install -r requirements.txt --upgrade
    echo.
    echo Done! Double-click run.bat to start.
) else (
    echo Python not found.
    echo Install Python 3.12 from python.org
    echo Check "Add Python to PATH"
)

pause
