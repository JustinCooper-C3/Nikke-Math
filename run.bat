@echo off
REM Nikke Math Solver - Run Script
REM Installs dependencies, activates virtual environment, and runs main.py

cd /d "%~dp0"

REM Create venv if it doesn't exist
if not exist "venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Install/update dependencies
echo Checking dependencies...
pip install -r requirements.txt --quiet

REM Run the application with debug mode enabled
python main.py --debug
pause
