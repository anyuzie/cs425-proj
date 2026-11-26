@echo off
cd /d "%~dp0"

REM Create the virtual environment named ".venv" (if it doesn't exist)
if not exist ".venv" (
    python -m venv .venv
)

REM Activate the virtual environment
call .venv\Scripts\activate.bat

REM Install requirements
pip install -r requirements.txt -r requirements-models.txt ^
            -r requirements-ranking.txt -r requirements-utterance.txt

REM Check the NLI checkpoint
python scripts\check_nli_checkpoint.py --smoke

REM Run the app
python run.py

REM Keep the command prompt open inside the activated venv
cmd /k