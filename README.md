# Career Matcher

CS425 group project (G2T4). Career Matcher is a task-oriented chatbot that recommends
careers. Over up to ten turns it reads the RIASEC interests a student describes, asks
whether an interest comes from enjoyment or only from familiarity, challenges statements
that contradict earlier ones, and then recommends five O\*NET occupations.

Each turn goes through:

1. a fine-tuned RoBERTa extractor (`pipeline/extractor.py`)
2. a rule-based state tracker
3. a contradiction rule, confirmed by a fine-tuned DeBERTa NLI model (`pipeline/nli.py`)
4. a ranker that fuses RIASEC cosine, a fine-tuned MiniLM encoder and BM25 (`recommender/`)
5. a rule-based policy (`pipeline/policy.py`)
6. a fine-tuned T5-small that writes the next question (`pipeline/t5_v3.py`)

## Setup

Python 3.11 or newer.

### 1. Download the model files

The trained models and ranking artifacts are too large for Git. After pulling the
repository, download **only** the `models` and `artifacts` folders and the
`occupation_work_context.jsonl` file from the restricted
[team Google Drive folder](https://drive.google.com/drive/folders/1r_4NHUjsxD4TsezeASW82okh1d1IKwvy?usp=drive_link).
You must sign in with an authorized SMU account, ask the project team for access
if needed.

Place the two folders directly in the repository root (beside this README), and
put the JSONL file at
`data/work_context/onet_31_0/prepared/occupation_work_context.jsonl`. Do not edit
or re-save the JSONL file. The resulting paths should include:

```
models/
  occupation_semantics_v1/
    finetuned/
    finetuned_index/
    data.json
artifacts/
data/work_context/onet_31_0/prepared/occupation_work_context.jsonl
```

If the downloaded bundle contains the semantic encoder under
`artifacts/ranking/occupation_semantic_v1/`, move that entire folder to
`models/occupation_semantics_v1/`. In `finetuned_index/manifest.json`, set the
top-level `checkpoint` field to `models/occupation_semantics_v1/finetuned`.

### 2. Install dependencies and run the app

On **Windows**, open a terminal in the repository root and run the batch launcher
(`run_pipeline.bat` in Command Prompt, or `.\run_pipeline.bat` in PowerShell). It
creates `.venv` if needed, installs dependencies, checks the NLI checkpoint, and
starts the app:

```bat
.\run_pipeline.bat
```

On **macOS/Linux**, create and activate a virtual environment, install dependencies,
then start the app:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-models.txt \
  -r requirements-ranking.txt -r requirements-utterance.txt
python scripts/check_nli_checkpoint.py --smoke
python run.py
```

The `.bat` launcher is specific to Windows; on macOS/Linux use the commands above.
Once the terminal reports "Application startup complete", open
http://127.0.0.1:8000. Model loading can take a little while on CPU. Keep the
terminal open while using the app.

Only needed to retrain T5 (see `training/t5/README.md`):

```
generator_training/t5-stage6-balanced/   T5 starting checkpoint for the final fine-tune
generator_training/t5-small-pretrained/  pretrained T5-small, for retraining from stage 1
```

To chat in the terminal instead:

```bash
python chat_cli.py
```

Environment variables:

| Variable | Effect |
|---|---|
| `CM_CONFIG=configs/system.question-bank.json` | Use template questions instead of T5 |
| `CM_CONFIG=configs/system.cosine.json` | Rank with RIASEC cosine only |
| `CM_STUDY_MODE=1` | Hide the researcher panel (used for the user study) |

Each conversation is logged to `logs/<participant_code>.jsonl`.

## Tests

```bash
python -m unittest discover -s tests
```

## Training and evaluation

| Component | Training | Evaluation |
|---|---|---|
| Extractor | `model_training_code/Extractor_training_code.ipynb` | `python data/eval/lexicon_baseline.py test` (keyword baseline) |
| Contradiction NLI | `python scripts/finetune_nli.py train --data data/nli_natural_v1.jsonl --output <dir> --epochs 2 --batch-size 8 --learning-rate 1.75e-5 --seed 425 --allow-unreviewed` | `python scripts/finetune_nli.py compare --data data/nli_natural_v1.jsonl --tuned <dir>/checkpoint --output <dir2>` |
| Semantic encoder | Run the four stages below with a fresh experiment config. | The `evaluate` stage writes the held-out results. |
| Ranker | no training (BM25 index is built at load time) | `python -m eval.task_retrieval_fusion` |

### Retrain the semantic encoder

Use the dependencies in `requirements-occupation-semantic.txt` and place the
pretrained MiniLM checkpoint at
`artifacts/ranking/models/all-MiniLM-L6-v2_1110a243/`. Copy
`configs/occupation_semantic_v1.json` to `configs/occupation_semantic_v2.json`
and set `artifacts` to `models/occupation_semantics_v2` and `report` to
`reports/ranking/occupation_semantic_v2` to preserve the supplied v1 outputs.

```bash
python -m training.occupation_semantic prepare --config configs/occupation_semantic_v2.json
python -m training.occupation_semantic train --config configs/occupation_semantic_v2.json
python -m training.occupation_semantic index --config configs/occupation_semantic_v2.json
python -m training.occupation_semantic evaluate --config configs/occupation_semantic_v2.json
```
| T5 question writer | `python -m training.t5.train --model generator_training/t5-stage6-balanced --data data/t5/standard --output runs/t5-chosen --epochs 1 --batch-size 8 --learning-rate 1e-5 --seed 42` | `python -m training.t5.evaluate --model runs/t5-chosen --output runs/t5-chosen-eval --application-suites`. Or run `train_t5.bat` (chosen model) or `train_t5_from_scratch.bat` (from pretrained T5-small). Details: `training/t5/README.md` |
| End-to-end | | `python -m eval.dialogue_fusion --probing both` (simulated students, with and without probing) |

The live ranker is chosen in `configs/ranking.yaml`.

## Repository layout

```
app/          FastAPI server and terminal chat
pipeline/     extractor, tracker, contradiction rule, NLI, policy, T5 question writer
recommender/  ranking models, fusion and ranking evaluation
static/       web chat page
configs/      system and ranking configurations
data/         O*NET catalogue, training data and data-building scripts
training/     training code for the semantic encoder, classifiers and T5 (training/t5/)
eval/         end-to-end and component evaluations
scripts/      NLI fine-tuning and checkpoint checks
tests/        unit and integration tests
```

O\*NET 31.0 data is used under CC BY 4.0 (National Center for O\*NET Development).
