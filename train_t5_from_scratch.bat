@echo off
REM Retrain the T5 question writer from pretrained T5-small: report Section 6.2 stages 2, 4, 5
REM and 6, then the final step from Section 6.4, then evaluate.
REM Needs generator_training\t5-small-pretrained\ (from the T5 model zip, unzipped into this folder).
REM Output: runs\t5_scratch\final\ (model) and runs\t5_scratch\eval\metrics.json (results).
REM Results can differ slightly from the supplied checkpoints (see report Section 6.5).
cd /d "%~dp0"
REM Use the project virtual environment if there is one (this works even if the folder was moved).
set "PY=python"
if exist ".venv\Scripts\python.exe" set "PY=.venv\Scripts\python.exe"
"%PY%" -c "import torch, transformers" 2>nul || (
    echo Python at "%PY%" cannot import torch and transformers. Install the requirements first, see README.md.
    goto fail
)

if not exist "generator_training\t5-small-pretrained\model.safetensors" (
    echo generator_training\t5-small-pretrained\ not found. Unzip the T5 model files into %CD% first.
    goto fail
)
if exist "runs\t5_scratch" (
    echo runs\t5_scratch\ already exists. Delete or rename it, then run this file again.
    goto fail
)
set OUT=runs/t5_scratch

echo [1/6] Stage 2: contextual
"%PY%" -m training.t5.train --model generator_training/t5-small-pretrained --data data/t5/stage2_contextual --output %OUT%/s2 || goto fail
echo [2/6] Stage 4: revised action-focused
"%PY%" -m training.t5.train --model %OUT%/s2 --data data/t5/stage4_action_revised --output %OUT%/s4 --epochs 2 || goto fail
echo [3/6] Stage 5: task and retry refinement
"%PY%" -m training.t5.train --model %OUT%/s4 --data data/t5/stage5_task_retry --output %OUT%/s5 || goto fail
echo [4/6] Stage 6: balanced full replay
"%PY%" -m training.t5.train --model %OUT%/s5 --data data/t5/standard --output %OUT%/s6 --epochs 1 --learning-rate 3e-5 || goto fail
echo [5/6] Final step (Section 6.4)
"%PY%" -m training.t5.train --model %OUT%/s6 --output %OUT%/final --epochs 1 --learning-rate 1e-5 || goto fail
echo [6/6] Evaluating on the 432 development prompts and the app suites
"%PY%" -m training.t5.evaluate --model %OUT%/final --output %OUT%/eval --application-suites || goto fail

echo.
echo Done. Model: %CD%\runs\t5_scratch\final\   Results: %CD%\runs\t5_scratch\eval\metrics.json
pause
exit /b 0

:fail
echo.
echo Stopped because of the error above.
pause
exit /b 1
