# Adding your part

Read this before you add your work. How the whole system fits together, and what every function does, is in [ARCHITECTURE.md](ARCHITECTURE.md).

The app has **no stand-ins**. It won't start until the **tracker and policy** are filled in and the trained extractor checkpoint is in `models/extractor/`. Each missing piece stops startup with a message pointing here. The tests run on small fakes in `tests/fakes.py`, so you can work and test before the others finish.

---

## 1. What's done and what's yours

| Step | Part | File | Status | Config name (already set) |
|---|---|---|---|---|
| 1 | Preference extractor | `pipeline/extractor.py` | **done**; needs the checkpoint in `models/extractor/` | `"extractor": "deberta"` |
| 2 | State tracker | `pipeline/tracker.py` | **done**: confidence-weighted merge; recency decay is `settings.decay` (1.0 = off, 0.9 = on) | `"tracker": "decay"` |
| 3 | Contradiction rule | `pipeline/rule.py` | done | `"rule": "rule"` |
| 4 | NLI check | `pipeline/nli.py` | **done** (threshold tuned, see §7) | `"confirmer": "nli"` |
| 5 | Ranker | `recommender/` (`RankingPipeline`, set up in `configs/ranking.yaml`) | **done**: structured, BM25, MiniLM/FAISS and RRF (M1–M5) | `"ranker": "configured"` |
| 6 | Policy | `pipeline/policy.py` | **teammate**: fill `ManualPolicy._decide()` | `"policy": "manual"` |
| 7 | Template filler | `pipeline/templates.py` | done | `"filler": "question-bank"` |
| — | Evaluation harness | `eval/harness.py` (new) | to write | `"no_probing": true` for the control arm |

You never need to edit `configs/system.json`, `pipeline/system.py`, `pipeline/registry.py`, `app/` or `static/`.

---

## 2. Where your files go

| Kind | Folder | Committed? |
|---|---|---|
| **App code**: what runs in the chatbot | `pipeline/` (your part's file), `recommender/` | yes |
| **Training and research code**: never imported by the app | `training/<part>/`, e.g. `training/extractor/`, `training/nli/`. Create your own subfolder. | yes, code only |
| **Evaluation** | `eval/` | yes |
| **Your tests** | `tests/test_<your part>.py` | yes |
| **Data package** (O\*NET + team data, read-only) | `data/` | yes, already in the repo; don't edit it |
| **Checkpoints, indexes, logs** | `models/<part>/`, `indexes/`, `logs/` | **never** |

- Training code may import from the app (for example `scripts/check_m5_extractor.py` uses `pipeline/`). **The app never imports from `training/`.**
- `.gitignore` blocks model files anywhere in the repo (`*.safetensors`, `*.bin`, `*.pt`, `*.pth`, `*.ckpt`, `*.onnx`, `*.pkl`, `*.zip`, `checkpoints/`, `runs/`, `wandb/`, `outputs/`).

**In your `training/<part>/` folder:**
- Your training script or notebook. Clear notebook outputs before committing.
- `requirements-training.txt`, or an install cell at the top of the notebook.
- A small `results.json` with the seed, a hash of the data you used, and your metrics.

---

## 3. Your part, step by step

### Step 1 · Extractor (`pipeline/extractor.py`)

**How answers to bot questions are read** (`pipeline/answers.py`): the extractor was trained with the bot's question attached only to short CLARIFY replies, so attaching it to anything else makes the question's words leak into the reading. So the answer **speaks for itself first**: it is read alone, and the question is attached only if that finds nothing ("the first one", "either is fine"). Short answers to a CHALLENGE ("the first one" / "the second one") keep the earlier or newer statement, and short answers to a PROBE ("I enjoy it" / "just what I know") keep the stance as enjoyment or set it to 0, by rule.

**Done.** `load()` and `predict()` are implemented for the trained 45-output checkpoint (18 direction, 3 reason, 24 strength). The outputs are picked by their `config.id2label` names and converted to the `labels.json` order, so the model's own class order doesn't matter. It's a standard `AutoModelForSequenceClassification`, so no custom model class is needed. What's left is installing the checkpoint:

1. **Get the checkpoint** from the team Drive folder ([TEAM DRIVE LINK]) and unzip it so the files sit **directly** in `models/extractor/` (`models/extractor/config.json`, not `models/extractor/extractor-3f9c2a/config.json`). It needs `config.json` (with the 45 labels and the `extractor` calibration block), the weights and the tokenizer files.
2. **Check it:** `python scripts/check_m5_extractor.py --checkpoint models/extractor` runs the real extractor with the configured ranking and the remaining parts faked. Details: [extractor wiring and validation](../reports/ranking/m5/EXTRACTOR_INTEGRATION.md).

**Retraining or re-exporting:** train in `training/extractor/` on `data/extractor.jsonl` (use the shipped `split` field and the class weights in `data/labels.json`), keep the 45 labelled outputs and the `extractor` calibration block in `config.json`, zip the checkpoint with a short hash in the name (e.g. `extractor-3f9c2a.zip`), upload it and post the name in the group chat.

### Step 2 · Tracker (`pipeline/tracker.py`)

1. **Write `update(state, reading)`.** It merges this turn's readings into `state.slots`, weighting older evidence down (recency decay), with confidence as the merge weight. Keep `value` in −1..1 and `confidence` in 0..1. It changes `state` and returns nothing.
2. It must also do three things:
   - Append a `HistoryEntry` for each reading, keeping the `source` sentence word for word plus its `reason_confidence`. The quote cards, the NLI check and the PROBE rule all depend on these.
   - Set `state.last_reading_empty = not reading.slots`.
   - If `state.last_action` was a PROBE on a type, add that type to `state.resolved`.
3. **Delete** the `raise NotImplementedError` line in `__init__`.

Copy from: `SimpleTracker` in `tests/fakes.py`, which does everything except the decay.

### Step 4 · NLI check (`pipeline/nli.py`)

**Done.** Real system presets load `models/nli-probing-v2/checkpoint` once. The confirmer returns P(contradiction) and the app keeps candidates whose score is strictly above **0.90**. The checkpoint uses the project probing-conflict policy. See [deployment and checkpoint verification](NLI_DEPLOYMENT.md).

### Step 6 · Policy (`pipeline/policy.py`)

1. **Write `ManualPolicy._decide(state)`** and return `Action(kind, slot, why, payload)`. Check these top to bottom; the first match wins:
   1. **CHALLENGE**: `state.flags` isn't empty. Payload `{old_src, old_turn, new_src}`.
   2. **PROBE**: a slot's `reason == "familiarity"` **and** `reason_confidence > settings.probe_reason_conf` (default 0.60), and it hasn't been probed yet. Payload `{new_src}`.
   3. **ABSTAIN**: the last action was a question and the reply had no stance. Ask **once** more (same question with an example after "what do you mean?"/"not sure", next phrasing after an unreadable answer); if still unsure, move on: the type is marked "no clear preference" (counts as covered; asked once more only while the profile is not ready), and a stuck CHALLENGE keeps the newer statement at half confidence.
   4. **RECOMMEND**: every type filled or settled, the profile has converged, and the conversation is 5 to 10 turns in.
   5. **CLARIFY**: ask the unfilled type with the highest variance. Use `state.ranking`, this turn's top 100 occupations, best first, each with `occupation_z`.
2. **Update the bookkeeping yourself:** `state.challenged`, `probed`, `settled` and `strikes`.
3. **Skip rules 1–2** when `self.no_probing` is True (the evaluation's control arm).
4. **Delete** the `raise NotImplementedError` line in `__init__`.

`PolicyBase` (same file, done) handles these for you:
- It returns RECOMMEND at turn 10.
- It raises an error if you probe while `no_probing` is on.
- It adds the question-rotation counters to the payload.

Copy from: `SimplePolicy` in `tests/fakes.py`, a minimal version of all five rules.

### Step 5 · Ranker

**Done.** `"ranker": "configured"` loads `configs/ranking.yaml` once at startup and builds a `RankingPipeline` (structured, BM25, MiniLM/FAISS, RRF). The default preset is structured (M1) only; the three-way RRF preset is tested separately. It reranks every turn from the current tracker values and the user's own messages, before the policy runs. Cards show up to three matched tasks when a text component (BM25 or dense) is on; with the structured-only default they show none, and the card copy says the ranking comes from the interest profile. `settings.input_mode` must be set explicitly: `actual_tracker` for the app, `mock_tracker` for fixtures. More: [ranking architecture](RANKING_EXPERIMENT_ARCHITECTURE.md), [M5 decision](../reports/ranking/m5/DECISION.md).

---

## 4. Check that your part works

1. **Run all tests:** `.venv/bin/python -m unittest discover -s tests`. Always use `discover -s tests`, or the contract imports fail.
2. **Add a contract test** for your part. It checks the output shape (valid letters and ranges, no crash on empty or odd text):
   ```python
   # tests/test_my_tracker.py
   import unittest
   from contracts import TrackerContract        # ExtractorContract, ConfirmerContract, PolicyContract, …
   from pipeline.tracker import DecayTracker

   class TestDecay(TrackerContract, unittest.TestCase):
       def make(self):
           return DecayTracker()
   ```
3. **Once all four parts are in,** start the server with `.venv/bin/uvicorn app.server:app` and open `http://localhost:8000/api/health`. It should list `deberta`, `decay`, `nli` and `manual`. Then chat with it at `http://localhost:8000`; the researcher panel shows the profile and each decision.

---

## 5. Rules for every part

- Types are letters `R I A S E C`. Use full names (`Realistic`) only when reading `question_bank.json`.
- Keep the user's sentences word for word in `source`.
- Load models once (in `load()` or `__init__`), never inside `read`, `score`, `update` or `decide`.
- In `pipeline/extractor.py` and `pipeline/nli.py`, import torch and transformers inside the method that needs them, so the tests run without them. At the top of anything in `training/` is fine.
- Need another package? Add it to `requirements-models.txt` with an exact version (`package==1.2.3`) if the app needs it, or to `training/<part>/requirements-training.txt` if only training does.
- Same input, same output: the rule, the policy and the ranker must be deterministic.

---

## 6. Before you open a pull request

- [ ] Branched from `main`
- [ ] `unittest discover -s tests` passes
- [ ] No model files, data, logs or zips in the diff
- [ ] Notebook outputs cleared
- [ ] The PR states your seed, your metrics and the checkpoint's hash name

---

## 7. Still to decide (team)

- The two links in `static/app.js`: `#rating-form` and `#careers-service`.
- Where transcripts are stored and when they are deleted (manual section 10).
- A "why" line specific to each card (right now all five share one sentence).
- The team Drive folder link (step 1 above).

## Install and run

### Without the trained models (mock mode)

For trying the whole app on any laptop before the models are shared. **Everything is real except the three
model parts:** a keyword reader stands in for the extractor, every rule-flagged contradiction counts as
confirmed, and the fixed question bank replaces T5. No torch, no checkpoints, nothing to download.

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m devtools.run_mock serve            # browser: http://localhost:8000 (researcher panel on)
.venv/bin/python -m devtools.run_mock chat             # same, in the terminal
.venv/bin/python -m devtools.run_mock eval --limit 20  # full vs no_probing on 20 dev profiles
```

On Windows use `.venv\Scripts\python.exe` instead of `.venv/bin/python`.

Things to try: *"My dad runs a sales company so that's what I know."* → PROBE → *"just what I know"*;
*"I love working with machines and tools."* then *"I hate repairing machines, it is boring."* → CHALLENGE →
*"the second one"*; answer the either-or questions with *"definitely"*, *"no way"* or *"either is fine"*.

**Mock results are flow checks, not results.** The keyword reader is crude, so the `eval` numbers mean
nothing. Mock runs are labelled `mock_tracker`, and the app refuses to label them `actual_tracker`.

### With the real models

The models are not in GitHub (`/models/` is gitignored: each is several hundred MB). Every machine needs:

| Folder | What | Get it from |
|---|---|---|
| `models/extractor/` | Trained DeBERTa extractor | Koon (Drive zip; `config.json` directly in the folder) |
| `models/nli-probing-v2/checkpoint/` | Fine-tuned probing-v2 NLI | Copy shared checkpoint and verify with `scripts/check_nli_checkpoint.py` |
| `models/t5-questions-v3b/` | Trained T5 question wording | Chua. Only needed for T5 wording: the evaluation uses the question bank. |

About 3–4 GB in total with the Python packages.

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-models.txt -r requirements-ranking.txt
# data/ comes with the repo (O*NET 31.0 + the team data package); nothing to download
.venv/bin/uvicorn app.server:app --reload           # http://localhost:8000, researcher panel on
CM_STUDY_MODE=1 .venv/bin/uvicorn app.server:app    # participant sessions: panel hidden
```

Then, with the server running for the scenarios:

```
.venv/bin/python -m eval.scenario_tests                                    # 19 real-model scenarios
.venv/bin/python -m eval.dialogue --config configs/system.question-bank.json --limit 120   # headline experiment
.venv/bin/python training/extractor/context_leak_check.py                  # does the question leak into readings?
```

**Reading the transcripts.** Every turn is logged to `logs/<participant code>.jsonl`, including how the answer was read (alone, with the question, or by a rule), what the extractor found, contradiction checks with their NLI scores, the policy's reason, whether T5 or the question bank wrote the reply, the top 5 and the time per step:

```
.venv/bin/python -m devtools.view_logs                        # list logs and sessions
.venv/bin/python -m devtools.view_logs logs/P-014.jsonl --last  # latest session, turn by turn
.venv/bin/python -m devtools.view_logs logs/P-014.jsonl --brief # one line per turn
```

For the decay comparison, run the headline experiment twice: with `"decay": 1.0` and with `"decay": 0.9` in the config.
