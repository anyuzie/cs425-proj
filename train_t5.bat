@echo off
REM Reproduce the chosen T5 model (report Section 6.4) and evaluate it.
REM Needs models\t5-stage6-balanced\ (from the T5 model zip, unzipped into this folder).
REM Output: runs\t5\ (model) and runs\t5_eval\metrics.json (results).
cd /d "%~dp0"
REM Use the project virtual environment if there is one (this works even if the folder was moved).
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
"%PY%" -c "import torch, transformers" 2>nul || (
    echo Python at "%PY%" cannot import torch and transformers. Install the requirements first, see README.md.
    goto fail
)

if not exist "models\t5-stage6-balanced\model.safetensors" (
    echo models\t5-stage6-balanced\ not found. Unzip the T5 model files into %CD% first.
    goto fail
)
if exist "runs\t5" (
    echo runs\t5\ already exists. Delete or rename it, then run this file again.
    goto fail
)

echo [1/2] Training: one epoch on data\t5\standard from the stage 6 checkpoint
"%PY%" -m training.t5.train --output runs/t5 --epochs 1 --learning-rate 1e-5 || goto fail

echo [2/2] Evaluating on the 432 development prompts and the app suites
"%PY%" -m training.t5.evaluate --model runs/t5 --output runs/t5_eval --application-suites || goto fail

echo.
echo Done. Model: %CD%\runs\t5\   Results: %CD%\runs\t5_eval\metrics.json
pause
exit /b 0

:fail
echo.
echo Stopped because of the error above.
pause
exit /b 1
