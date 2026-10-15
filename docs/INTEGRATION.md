# What's done, what's left

The chatbot already runs end to end. Parts that aren't built yet use a simple **stand-in** so the app still works.
Each unbuilt part has its own file with a **`PUT YOUR CODE HERE`** block waiting for you.
For how every step works and which files are involved, see [ARCHITECTURE.md](ARCHITECTURE.md).

## Done

| Part | File |
|---|---|
| Chat page | `static/` |
| Server (`/api/session`, `/api/chat`, `/api/health`) | `app/` |
| Runs the 7 steps each turn | `pipeline/system.py` |
| Step 3 · contradiction rule (arithmetic, no model) | `pipeline/stubs.py` → `ContradictionRule` |
| Step 5 · structured RIASEC score (one of the three ranking signals) | `recommender/structured.py` |
| Step 7 · template filler (question bank) | `pipeline/templates.py` |

## To build

Suggested order, top to bottom. "Stand-in now" is what runs until you switch it.

| # | Part | Model? | Your file | Stand-in now | Config line |
|---|---|---|---|---|---|
| 1 | Build BM25 + FAISS indexes | MiniLM | `scripts/build_indexes.py` | — | — |
| 2 | BM25 signal | no | `recommender/bm25.py` | — | (part of `rrf`) |
| 3 | Dense task matching | MiniLM + FAISS | `recommender/dense.py` | — | (part of `rrf`) |
| 4 | RRF fusion of all three signals | no | `recommender/fusion.py` | structured score + word-overlap task quotes | `"ranker": "rrf"` |
| 5 | Preference extractor | DeBERTa (trained) | `pipeline/extractor.py` | keyword list | `"extractor": "deberta"` |
| 6 | NLI contradiction check | NLI DeBERTa | `pipeline/nli.py` | says yes to every conflict | `"confirmer": "nli"` |
| 7 | Tracker with recency decay | no | `pipeline/tracker.py` | keeps the latest reading | `"tracker": "decay"` |
| 8 | Full policy (manual section 7) | no | `pipeline/policy.py` | simplified rules | `"policy": "manual"` |
| 9 | Evaluation harness | no | `eval/harness.py` (new) | — | `"no_probing": true` for the control arm |

## How to switch one part on

1. Open its file. Fill in each `PUT YOUR CODE HERE` block. The docstring above it says what comes in and what must go out.
2. Delete the `raise NotImplementedError(...)` line at the top of `__init__`. It stops the server starting with a half-built part.
3. In `configs/system.json`, change that part's line to the new name (see the table).
4. Run `.venv/bin/python -m unittest discover -s tests`, then start the server and open `/api/health`. Your part's name now appears in place of the `-stub` one.

**Optional:** in `tests/contracts.py`, find the test class for your part (for example `ExtractorContract`) and subclass it with a `make()` that returns your object. It checks that your output has the right shape.

## Rules that apply everywhere

- Types are letters: `R I A S E C`. Use full names only when reading `question_bank.json`.
- Keep the user's sentences word for word (`source`). The quote cards and the NLI check use them.
- Load models once, in `__init__`, never inside `read`, `score` or `rank`.
- Don't import torch at the top of `pipeline/registry.py` or `pipeline/stubs.py`, so the app still runs without it.

## Run everything

```
pip install -r requirements-models.txt          # only once you use a real model
.venv/bin/python scripts/build_indexes.py       # once, after filling it in
.venv/bin/uvicorn app.server:app --port 8000    # open http://localhost:8000
CM_STUDY_MODE=1 .venv/bin/uvicorn app.server:app   # for participant sessions: hides the researcher panel
```

Put the trained extractor in `models/extractor/`. Indexes go in `indexes/`. Both are created locally and not committed.

## Still to decide

- Two links in `static/app.js`: `#rating-form` and `#careers-service`.
- Where transcripts are stored and when they are deleted (manual section 10).
- A "why" line specific to each card (right now all five share one sentence).
