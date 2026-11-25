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

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-models.txt \
            -r requirements-ranking.txt -r requirements-utterance.txt
```

### Model files

The trained models are too large for Git. Download the shared model folder from the
team drive ([link to be added]) and copy it into the repository so that these
folders exist:

```
models/extractor/                        RoBERTa extractor checkpoint and tokenizer
models/nli-probing-v2/checkpoint/        fine-tuned NLI checkpoint
models/t5-questions-v3b/                 fine-tuned T5 question writer (the chosen model)
artifacts/ranking/occupation_semantic_v1/  fine-tuned MiniLM encoder and occupation index
data/work_context/onet_31_0/prepared/occupation_work_context.jsonl   O*NET work context (the index checks its hash)
```

Only needed to retrain T5 (see `training/t5/README.md`):

```
models/t5-stage6-balanced/               T5 starting checkpoint for the final fine-tune
models/t5-small-pretrained/              pretrained T5-small, for retraining from stage 1
```

Check that the NLI checkpoint is the right one:

```bash
python scripts/check_nli_checkpoint.py --smoke
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

To try the interface without the model files (mock extractor and NLI, everything
else real):

```bash
python -m devtools.run_mock serve
```

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
| T5 question writer | `python -m training.t5.train --model models/t5-stage6-balanced --data data/t5/standard --output runs/t5-chosen --epochs 1 --batch-size 8 --learning-rate 1e-5 --seed 42` | `python -m training.t5.evaluate --model runs/t5-chosen --benchmark data/t5/benchmarks/development_432.json --decode data/t5/decode.json --output runs/t5-chosen-eval --application-suites` (details: `training/t5/README.md`) |
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
