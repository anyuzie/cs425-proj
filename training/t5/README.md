# T5 question writer: training and evaluation

T5-small (Raffel et al., 2020) phrases the question for the action the policy chose
(CLARIFY, PROBE, CHALLENGE, or a retry of one). The app inserts the user's quotes and
falls back to a template when a generated question fails its checks
(`pipeline/t5_v2.py`, `pipeline/t5_v3.py`). See report Section 6.

Activate the project virtual environment (see the root README), then run every command
from the repository root. Each command is a single line, so it works the same in
PowerShell, cmd, bash and zsh.

## Files

| Path | What it is |
|---|---|
| `training/t5/train.py` | Full or LoRA fine-tuning; keeps the epoch with the lowest validation loss |
| `training/t5/evaluate.py` | Joint checks, reference-token accuracy and the app suites |
| `training/t5/common.py` | The four checks and an auditing copy of the app's `T5V3Filler` |
| `training/t5/controlled.py` | The 112-case controlled dialogue check |
| `data/t5/` | All training splits, the benchmarks and the decoding settings |

Model folders (not in Git; copy them from the team drive into `models/`):

| Folder | Role |
|---|---|
| `models/t5-small-pretrained/` | Pretrained T5-small, the start of stage 1 |
| `models/t5-stage6-balanced/` | Stage 6 (balanced full replay), the starting checkpoint for model selection |
| `models/t5-questions-v3b/` | **The chosen model, used by the app** (`configs/system.json`) |

## Data (`data/t5/`)

All pairs are synthetic (LLM-drafted from templates, `human_reviewed` is false). Each row
has `input` (`career question v2: {json}` with the action, RIASEC type, activity, a
reference question, context, retry flag, previous question and style) and `target` (the
question only). Activities are assigned to splits before the pairs are expanded, so no
activity appears in two splits.

| Folder | Train / validation / test | Used for |
|---|---|---|
| `stage1_exploratory/` | 420 / 90 / 90 | Stage 1 |
| `stage2_contextual/` | 882 / 189 / 189 | Stage 2 |
| `stage3_action_focused/` | 432 / 108 / 108 | Stage 3 |
| `stage4_action_revised/` | 432 / 108 / 108 | Stage 4 |
| `stage5_task_retry/` | 648 / 144 / 144 | Stage 5 |
| `standard/` | 864 / 216 / 216 | Stage 6, and the **chosen** continuation |
| `standard_plus_varied/` | 1,512 / 378 / - | Comparison: 864 standard + 648 varied |
| `reduced_varied/` | 864 / 378 / - | Comparison: 216 standard + 648 varied |
| `benchmarks/development_432.json` | 432 prompts (144 per action) | Model selection (Section 6.4) |
| `benchmarks/reserved_216.json` | 216 prompts | Held-out check |
| `decode.json` | | Up to 64 new tokens, 2 beams, no repeated 3-grams, input limit 256 tokens |

How the varied pairs were built: `data/t5/varied_construction.md`.

## Reproduce the chosen model

```
# 1. train: one more epoch on the 864 standard pairs, starting from stage 6
python -m training.t5.train --model models/t5-stage6-balanced --data data/t5/standard --output runs/t5-chosen --epochs 1 --learning-rate 1e-5

# 2. evaluate on the 432 development prompts and the app suites
python -m training.t5.evaluate --model runs/t5-chosen --output runs/t5-chosen-eval --application-suites

# 3. deploy: copy runs/t5-chosen/ to models/t5-questions-v3b/, then
python -m unittest discover -s tests -p "test_t5*.py"
python run.py
```

On Windows, `train_t5.bat` in the repository root runs steps 1 and 2 (model in `runs/t5/`,
results in `runs/t5_eval/metrics.json`), and `train_t5_from_scratch.bat` runs the full
rebuild below (model in `runs/t5_scratch/final/`, results in `runs/t5_scratch/eval/metrics.json`).

Defaults: batch size 8, seed 42, 3 epochs and learning rate 1e-4 for training;
`--benchmark data/t5/benchmarks/development_432.json` and `--decode data/t5/decode.json`
for evaluation. Pass `--benchmark data/t5/benchmarks/reserved_216.json` for the held-out set.

`train.py` refuses to write into a non-empty output folder, and `evaluate.py` refuses an
existing one. Pass `--device cpu` or `--device cuda` to choose; the default is CUDA when
available.

Evaluating the supplied `models/t5-questions-v3b` on CPU gives the reported row exactly:
415/432 joint, 0 focus errors, 44/46 tasks, 21/23 retries, 69.95% token accuracy.
Training fixes seed 42, which sets the order of the training examples and the dropout masks.
The reported model, like every run in report Section 6.4, was trained with
PyTorch's default GPU kernels rather than `--deterministic`; the final run kept that setting
so it was trained the same way as the runs it was compared against. Some default kernels add
floating-point values in a varying order, so a retrain gives slightly different weights and
results can differ slightly. Add `--deterministic` for retrains that repeat exactly:

| Retrain with the step-1 command | Joint /432 | Focus errors | Tasks /46 | Retries /23 | Token acc. | Train time |
|---|---|---|---|---|---|---|
| Reported model (RTX 3080) | 415 | 0 | 44 | 21 | 69.95% | |
| RTX 3080, `--device cuda`, two runs | 414, 414 | 0, 0 | 44, 44 | 21, 21 | 70.00%, 70.00% | about 18 s |
| RTX 3080, `--device cuda --deterministic` (identical weights every run) | 415 | 0 | 44 | 21 | 69.96% | about 17 s |
| CPU, `--device cpu` | 410 | 1 | 44 | 21 | 70.12% | about 5 min |

The GPU runs used Python 3.12, torch 2.14.1+cu130 and transformers 5.19.0, the same
versions as the reported run. For CUDA, install a CUDA build of PyTorch
(see https://pytorch.org/get-started/locally/); on Windows, `pip install -r requirements-models.txt`
installs the CPU build.

Shared settings (all runs): AdamW, weight decay 0.01, linear warm-up over 10% of steps
then linear decay, gradient clip 1.0, batch size 8, no label smoothing, inputs up to
256 tokens and targets up to 64. LoRA: rank 8, alpha 16, dropout 0.05 on the q and v
projections (294,912 trainable parameters, against about 60.5 million for full tuning).

## Comparison runs (Section 6.4)

Every comparison starts from `models/t5-stage6-balanced`, runs one epoch with seed 42,
and is evaluated with the step-2 command.

| Report row | `--data` | Other flags |
|---|---|---|
| Full, 864 standard (chosen) | `data/t5/standard` | `--learning-rate 1e-5` |
| Full, 864 standard + 648 varied | `data/t5/standard_plus_varied` | `--learning-rate 1e-5` |
| Full, 216 standard + 648 varied | `data/t5/reduced_varied` | `--learning-rate 1e-5` |
| LoRA, 864 standard + 648 varied | `data/t5/standard_plus_varied` | `--method lora --learning-rate 3e-4` |

To evaluate a LoRA run, pass the base and the adapter:
`--model models/t5-stage6-balanced --adapter runs/<lora run>`.

## Earlier stages (Section 6.2)

These are the recorded settings of the stage checkpoints. Stages 2 to 6 were trained on CPU
with earlier versions of this script.

### Rebuild from pretrained T5-small

Only stages 2, 4, 5 and 6 lead to the chosen model (stages 1 and 3 were not carried forward):

```
python -m training.t5.train --model models/t5-small-pretrained --data data/t5/stage2_contextual --output runs/s2
python -m training.t5.train --model runs/s2 --data data/t5/stage4_action_revised --output runs/s4 --epochs 2
python -m training.t5.train --model runs/s4 --data data/t5/stage5_task_retry --output runs/s5
python -m training.t5.train --model runs/s5 --data data/t5/standard --output runs/s6 --epochs 1 --learning-rate 3e-5
python -m training.t5.train --model runs/s6 --output runs/t5-rebuilt --epochs 1 --learning-rate 1e-5
python -m training.t5.evaluate --model runs/t5-rebuilt --output runs/t5-rebuilt-eval --application-suites
```

The data in each folder matches the hashes saved with the original stage runs. The chain runs
end to end, but retraining from scratch does not always give the same results as the supplied
checkpoints, because small differences add up over the five runs: each stage starts from the
one before, so a difference in one stage carries into every later stage. The differences come
from the default GPU kernels (see above) and from the earlier script versions: even on CPU, a
rerun of stage 2 ends at validation loss 0.1631 against the recorded 0.1626.

| Rebuild | Stage 6 joint /432 | Final joint /432 | Focus errors | Tasks /46 | Retries /23 | Token acc. | Train time |
|---|---|---|---|---|---|---|---|
| Supplied checkpoints | 411 | 415 | 0 | 44 | 21 | 69.95% | |
| RTX 3080, `--device cuda` | 402 | 402 | 0 | 39 | 16 | 68.57% | about 3 min |
| CPU, `--device cpu` | | 401 | 0 | 44 | 21 | 70.15% | about 27 min |

The supplied checkpoints are the reference for the reported results.

| Stage | `--model` (start) | `--data` | Epochs (kept) | LR |
|---|---|---|---|---|
| 1. Exploratory | `models/t5-small-pretrained` | `data/t5/stage1_exploratory` | 3 (3) | 1e-4 |
| 2. Contextual | `models/t5-small-pretrained` | `data/t5/stage2_contextual` | 3 (3) | 1e-4 |
| 3. Action-focused | stage 2 output | `data/t5/stage3_action_focused` | 2 (1) | 1e-4 |
| 4. Revised action-focused | stage 2 output | `data/t5/stage4_action_revised` | 2 (1) | 1e-4 |
| 5. Task and retry refinement | stage 4 output | `data/t5/stage5_task_retry` | 3 (3) | 1e-4 |
| 6. Balanced full replay | stage 5 output | `data/t5/standard` | 1 (1) | 3e-5 |

Stage 1 used the older `career question:` prompt. The app reads only the
`career question v2:` format used from stage 2 on.
