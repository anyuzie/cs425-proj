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

Create and activate a virtual environment, then install the dependencies.

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-models.txt \
            -r requirements-ranking.txt -r requirements-utterance.txt
```

Windows PowerShell:

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt -r requirements-models.txt `
            -r requirements-ranking.txt -r requirements-utterance.txt
```

### Install the model files

The trained models and ranking artifacts are too large for Git. They are kept in
the restricted [team Google Drive folder](https://drive.google.com/drive/folders/1r_4NHUjsxD4TsezeASW82okh1d1IKwvy?usp=drive_link);
ask the project team for access if you cannot open it.

1. Sign in to Google Drive with an account that has access.
2. Download the models and artifacts folde from the Drive. Drive downloads folders as a ZIP archive.
3. Extract the archive into the repository root (the folder containing this README).
   The paths below should exist directly under the repository root, not inside an
   additional nested folder:

```
models/extractor/                        RoBERTa extractor checkpoint and tokenizer
models/nli-probing-v2/checkpoint/         fine-tuned NLI checkpoint
models/t5-questions-v3b/                  fine-tuned T5 question writer
artifacts/ranking/occupation_semantic_v1/ fine-tuned MiniLM encoder and occupation index
data/work_context/onet_31_0/prepared/occupation_work_context.jsonl
```

Check that the NLI checkpoint loads:

```bash
python scripts/check_nli_checkpoint.py --smoke
```

The Drive folder is access-restricted, so users download the files in a browser
after signing in; the project does not store Google credentials or download them
automatically. Keep the virtual environment active for the remaining commands.

Only needed to retrain T5 (see `training/t5/README.md`):

```
generator_training/t5-stage6-balanced/   T5 starting checkpoint for the final fine-tune
generator_training/t5-small-pretrained/  pretrained T5-small, for retraining from stage 1
```

## Running the app

```bash
python run.py
```

Open http://127.0.0.1:8000 once the terminal shows "Application startup complete".
The models load at start-up, which takes a little while on CPU.

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
| Semantic encoder | `python -m training.occupation_semantic prepare\|train\|index --config configs/occupation_semantic_v1.json` | `python -m training.occupation_semantic evaluate --config configs/occupation_semantic_v1.json` |
| Ranker | no training (BM25 index is built at load time) | `python -m eval.task_retrieval_fusion` |
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
