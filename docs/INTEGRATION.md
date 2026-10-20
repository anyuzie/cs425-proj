# Adding your part

Read this before you add your work. How the whole system fits together, and what every function does, is in [ARCHITECTURE.md](ARCHITECTURE.md).

The app has **no stand-ins**. It won't start until the remaining teammate parts (tracker, NLI check, policy) are filled in and the extractor checkpoint is in `models/extractor/`. Each one stops startup with a message pointing here. The tests run on small fakes in `tests/fakes.py`, so you can work and test before the others finish.

---

## 1. What's done and what's yours

| Step | Part | File | Status | Config name (already set) |
|---|---|---|---|---|
| 1 | Preference extractor | `pipeline/extractor.py` | **done** (model integrated; needs the checkpoint in `models/extractor/`) | `"extractor": "deberta"` |
| 2 | State tracker | `pipeline/tracker.py` | **teammate**: fill `update()` | `"tracker": "decay"` |
| 3 | Contradiction rule | `pipeline/rule.py` | done | `"rule": "rule"` |
| 4 | NLI check | `pipeline/nli.py` | **teammate**: fill `__init__()`, `score()` | `"confirmer": "nli"` |
| 5 | Ranker | `recommender/structured.py` | done (baseline); BM25 + FAISS + RRF is the project lead's work | `"ranker": "structured"` |
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

- Training code may import from the app (for example `training/extractor/smoke_test.py` imports `pipeline/extractor.py`). **The app never imports from `training/`.**
- `.gitignore` blocks model files anywhere in the repo (`*.safetensors`, `*.bin`, `*.pt`, `*.pth`, `*.ckpt`, `*.onnx`, `*.pkl`, `*.zip`, `checkpoints/`, `runs/`, `wandb/`, `outputs/`).

**In your `training/<part>/` folder:**
- Your training script or notebook. Clear notebook outputs before committing.
- `requirements-training.txt`, or an install cell at the top of the notebook.
- A small `results.json` with the seed, a hash of the data you used, and your metrics.

---

## 3. Your part, step by step

### Step 1 · Extractor (`pipeline/extractor.py`)

You write **two** methods. `read()` is already done: it splits the message into sentences, calls your `predict()` on each, and builds the readings.

1. **Train** in `training/extractor/`, on `data/extractor.jsonl`:
   - Use the shipped `split` field; don't re-split.
   - Use the class weights in `data/labels.json`.
   - The trained model is a standard `AutoModelForSequenceClassification` with 45 outputs (18 direction, 3 reason, 24 strength) and its labels in `config.id2label`, so no custom model class is needed. If a future version needs one, put it in `pipeline/` and import it from the training code, so both use the same class.
2. **Export and share:**
   - Save the checkpoint folder (weights, `config.json`, tokenizer files) and zip it with a short hash in the name, e.g. `extractor-3f9c2a.zip`.
   - Upload it to the team Drive folder ([TEAM DRIVE LINK]) and post the name in the group chat.
3. **Everyone unzips it** so the files sit **directly** in `models/extractor/` (`models/extractor/config.json`, not `models/extractor/extractor-3f9c2a/config.json`).
4. **Fill `load(checkpoint_dir)`.** It runs once at server start. Load the tokenizer and the model onto `self` (e.g. `self.tok`, `self.model`) and call `.eval()`. Import torch and transformers here, not at the top of the file.
5. **Fill `predict(context, sentence)`.** It runs once per sentence. Tokenize with `tokenizer(context, sentence)` when `context` isn't `None` (it's the bot question being answered), else `tokenizer(sentence)`, with `max_length=128` and `truncation=True`, inside `torch.no_grad()`. Return plain Python floats and lists, already softmaxed. The class order inside each `direction` row can be your model's own: `read()` takes it from the model's `config.id2label` (e.g. `R:negative, R:neutral, R:positive`) and falls back to the `labels.json` order below only if the model has no such labels:
   ```python
   {"direction": [[p_neutral, p_positive, p_negative], ...],   # 6 lists, types in R I A S E C order
    "strength":  [s, ...],                                     # 6 floats 0..1 (sigmoid)
    "reason":    [p_enjoyment, p_familiarity, p_neither]}      # class orders: data/labels.json
   ```
   A wrong shape makes `read()` raise an error naming the key.
6. **Delete** the two `raise NotImplementedError` lines.

### Step 2 · Tracker (`pipeline/tracker.py`)

1. **Write `update(state, reading)`.** It merges this turn's readings into `state.slots`, weighting older evidence down (recency decay), with confidence as the merge weight. Keep `value` in −1..1 and `confidence` in 0..1. It changes `state` and returns nothing.
2. It must also do three things:
   - Append a `HistoryEntry` for each reading, keeping the `source` sentence word for word plus its `reason_confidence`. The quote cards, the NLI check and the PROBE rule all depend on these.
   - Set `state.last_reading_empty = not reading.slots`.
   - If `state.last_action` was a PROBE on a type, add that type to `state.resolved`.
3. **Delete** the `raise NotImplementedError` line in `__init__`.

Copy from: `SimpleTracker` in `tests/fakes.py`, which does everything except the decay.

### Step 4 · NLI check (`pipeline/nli.py`)

1. **Fill `__init__(model_name)`.** Load `cross-encoder/nli-deberta-v3-base` once (e.g. `sentence_transformers.CrossEncoder`) and find which output index is "contradiction". There's no training; the model downloads by name.
2. **Fill `score(old_src, new_src)`.** Return the softmaxed probability of "contradiction" as a plain float.
3. **Tune the threshold** in `training/nli/`:
   - Sweep it on the 75 `split == "dev"` pairs of `data/nli_pairs.jsonl`, using `nli_binary` from `labels.json`.
   - Report precision and recall once on the 64 `test` pairs.
   - Put the chosen number in `settings.confirm_threshold` (currently 0.60).
4. **Delete** the `raise NotImplementedError` lines.

Copy from: `PassthroughConfirmer` in `tests/fakes.py` (it always returns 1.0).

### Step 6 · Policy (`pipeline/policy.py`)

1. **Write `ManualPolicy._decide(state)`** and return `Action(kind, slot, why, payload)`. Check these top to bottom; the first match wins:
   1. **CHALLENGE**: `state.flags` isn't empty. Payload `{old_src, old_turn, new_src}`.
   2. **PROBE**: a slot's `reason == "familiarity"` **and** `reason_confidence > PROBE_REASON_CONF` (0.60), and it hasn't been probed yet. Payload `{new_src}`.
   3. **ABSTAIN**: the last action was a question and the reply had no stance. Two strikes settle the type.
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

### Step 5 · Ranker (project lead)

Write a class with `name`, `eligible_count` and `rank(interests, observed_types, user_sentences, top_k) -> RankingResponse`. Put each card's matched tasks in `evidence`; the cards quote the first two. Register it in `pipeline/registry.py` and set `"ranker"` in the config. Plan: BM25 → MiniLM + FAISS → RRF, per `cs425-ranking-docs/`. BM25 should skip short replies like "the first one".

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

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-models.txt
# data/ comes with the repo (O*NET 31.0 + the team data package); nothing to download
.venv/bin/uvicorn app.server:app --reload           # http://localhost:8000, researcher panel on
CM_STUDY_MODE=1 .venv/bin/uvicorn app.server:app    # participant sessions: panel hidden
```
