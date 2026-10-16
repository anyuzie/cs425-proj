# Career Matcher: architecture

How the system is built, which file does what, and how one message moves through the code.
To add your part (what to fill in, where files go, training hand-off, PR checklist), see [INTEGRATION.md](INTEGRATION.md). Section 11 below lists every function in each important file.

---

## 1. The big picture

A student chats in a web page. Each message goes through **seven steps**. They turn it into an interest profile (six RIASEC numbers), check the profile for contradictions, and choose what the bot says next. After 5–10 turns the bot recommends five of the 378 degree-level O\*NET occupations.

- **One Python process.** FastAPI serves the page and the API. Models load once at startup and run in-process.
- **Models do perception** (reading a message, checking a contradiction). **Plain Python does the deciding** (tracker, rule, policy, templates).
- **No LLM runs inside the system.** Every bot sentence comes from `data/question_bank.json`.
- **No stand-ins.** The app only contains real code. The server starts once the four teammate parts are filled in.

---

## 2. The repository: app vs training vs local files

| Kind | Folder | What goes there | Committed? |
|---|---|---|---|
| **App**: what runs in the chatbot | `static/` | The web page | yes |
| | `app/` | FastAPI server: routes, sessions, logs | yes |
| | `pipeline/` | The seven steps and the code that runs them | yes |
| | `recommender/` | Ranking (step 5) | yes |
| | `configs/` | `system.json`: which version of each part runs, plus settings and paths | yes |
| **Training and research**: never imported by the app | `training/<part>/` | Training scripts, notebooks, results JSON per part (see INTEGRATION.md §2) | yes, code only |
| | `eval/` | The evaluation harness, created when written | yes |
| **Tests** | `tests/` | Unit tests, contract tests, test-only fakes | yes |
| **Docs** | `docs/`, `cs425-ranking-docs/`, `reports/` | Guides, ranking plan, milestone reports | yes |
| **Local only** | `data/` | The data package (unzip `career-matcher-data.zip`) | **no** |
| | `models/` | Trained checkpoints, e.g. `models/extractor/` | **no** |
| | `indexes/` | BM25 / FAISS indexes for the ranker | **no** |
| | `logs/` | Transcripts, one file per participant code | **no** |

Rule of thumb: **training code may import from the app** (for example the model class in `pipeline/extractor_model.py`), but **the app never imports from `training/`.** The only things that cross from training to the app are a checkpoint folder in `models/` and a number in `configs/system.json`.

### Files in the app

| File | Step | Who | What it holds |
|---|---|---|---|
| `pipeline/system.py` | — | done | **The conductor**: `System.step()` runs the seven steps for one message |
| `pipeline/interfaces.py` | — | done | The data passed between steps (`Reading`, `SlotReading`, `State`, `Flag`, `Action`) and what each part must provide |
| `pipeline/registry.py` | — | done | Maps config names (`"deberta"`, `"nli"`, …) to the code that builds each part |
| `pipeline/extractor.py` | 1 | **teammate** fills `load()` + `predict()` | Preference extractor. `read()` is done. |
| `pipeline/extractor_model.py` | 1 | **teammate** creates | Their 3-head DeBERTa model class |
| `pipeline/tracker.py` | 2 | **teammate** fills `update()` | State tracker with recency decay |
| `pipeline/rule.py` | 3 | done | Contradiction rule |
| `pipeline/nli.py` | 4 | **teammate** fills `__init__()` + `score()` | NLI contradiction check |
| `recommender/structured.py` | 5 | done | Structured RIASEC ranker (the baseline); BM25 + FAISS + RRF to add |
| `pipeline/policy.py` | 6 | **teammate** fills `ManualPolicy._decide()` | Dialogue policy. `PolicyBase.decide()` is done. |
| `pipeline/templates.py` | 7 | done | Template filler |
| `app/server.py`, `app/schemas.py`, `app/turn_log.py` | — | done | Server, API shapes, transcript log |
| `static/app.js`, `index.html`, `styles.css` | — | done | Chat page and researcher panel |

---

## 3. Layers: who calls whom

Each layer only calls the one below it, so a part can be replaced without touching anything above it.

```
Page        static/app.js                         talks only to /api/session and /api/chat
   ↓ HTTP JSON
Server      app/server.py · schemas.py · turn_log.py   checks input, sessions, logs. Knows nothing about models.
   ↓ System.step(state, text)
Conductor   pipeline/system.py · registry.py · configs/system.json
   ↓ read · update · candidates · score · decide · fill · rank
Parts       extractor.py · tracker.py · rule.py · nli.py · policy.py · templates.py
   ↓ rank(...)
Ranking     recommender/structured.py (+ your BM25 / FAISS / RRF)
```

---

## 4. The teammate functions, and how they work with `System.step`

Each teammate's code only **receives data and returns data**. It never touches the server, the page, or another teammate's file. `System.step` hands each result to the next part in a fixed order, so the parts fit together as long as each method returns what its docstring says.

### Step 1 · Extractor: `pipeline/extractor.py`

| Function | Who writes it | Called by | When | Does |
|---|---|---|---|---|
| `load(checkpoint_dir)` | **teammate** | `__init__` | **once**, at server start | Loads the tokenizer and trained model from `models/extractor/` into `self.tok`, `self.model` |
| `predict(context, sentence)` | **teammate** | `read()` | once **per sentence** | Runs the model on one sentence and returns raw probabilities (below) |
| `read(text, context, expected_slot)` | done | `System.step` | once **per message** | Splits the message into sentences, calls `predict()` on each, and builds the `Reading` |

`predict()` must return plain Python floats and lists, already softmaxed, with class orders from `data/labels.json`:

```python
{"direction": [[p_neutral, p_positive, p_negative], ...],   # 6 lists, types in R I A S E C order
 "strength":  [s, ...],                                     # 6 floats 0..1
 "reason":    [p_enjoyment, p_familiarity, p_neither]}
```

- **Splitting:** `read()` splits on `. ! ?` and newlines. Pieces of six words or fewer join the sentence before.
- **Context:** for a reply to a question, `context` is the bot's question, so the model reads `tokenizer(question, reply)`.
- **Building each reading:** types predicted neutral are dropped. The rest become `SlotReading(slot, sign, strength, confidence, reason, source, reason_confidence)`, where `confidence` is the probability of the chosen direction and `source` is the sentence word for word.
- **Bad output:** if `predict()` returns the wrong shape, `read()` raises an error naming the bad key.

```
"I love analysing data. My dad runs a sales company so that's what I know."
read() → 2 sentences → predict() × 2
  → SlotReading(I, +1, 0.95, 0.91, enjoyment,   "I love analysing data.",                              0.88)
  → SlotReading(E, +1, 0.55, 0.76, familiarity, "My dad runs a sales company so that's what I know.", 0.82)
```

### Step 2 · Tracker: `pipeline/tracker.py`

| Function | Who | Called by | When | Does |
|---|---|---|---|---|
| `update(state, reading)` | **teammate** | `System.step` | once per message | Merges the readings into `state.slots`, weighting older evidence down. It changes `state` in place and returns nothing. |

It must also do three things:
- Keep every reading's `source` sentence and `reason_confidence` in the type's history.
- Set `state.last_reading_empty`.
- After a PROBE, add that type to `state.resolved`.

A minimal working version is `SimpleTracker` in `tests/fakes.py`.

### Step 3 · Contradiction rule: `pipeline/rule.py` (done)

`candidates(state)` returns a possible conflict when the same type has opposite signs and both readings have confidence above 0.4. It skips types already challenged or resolved. There's no model; it's arithmetic.

### Step 4 · NLI check: `pipeline/nli.py`

| Function | Who | Called by | When | Does |
|---|---|---|---|---|
| `__init__(model_name)` | **teammate** | registry | once, at server start | Loads `cross-encoder/nli-deberta-v3-base` |
| `score(old_src, new_src)` | **teammate** | `System._confirmed` | **only when step 3 found a candidate** | Returns the probability that the two sentences contradict each other |

`System` keeps the conflict only if the score is above `settings.confirm_threshold` (0.60). Tune that threshold on the dev split of `data/nli_pairs.jsonl`.

```
rule: S was +1 "I love teaching kids", now −1 "I hate teaching, it's draining"
score(...) → 0.91 > 0.60 → kept → the policy will CHALLENGE
score(...) → 0.20       → dropped
```

### Step 6 · Policy: `pipeline/policy.py`

| Function | Who | Called by | When | Does |
|---|---|---|---|---|
| `ManualPolicy._decide(state)` | **teammate** | `decide()` | once per message | Picks the move and returns `Action(kind, slot, why, payload)` |
| `PolicyBase.decide(state)` | done, inherited | `System.step` | once per message | Returns RECOMMEND at turn 10 itself, otherwise calls `_decide()`. Rejects CHALLENGE/PROBE when `no_probing` is on. Adds the question-rotation counters to the payload. |

The five rules, checked top to bottom (first match wins):
1. **CHALLENGE** when there's a confirmed contradiction.
2. **PROBE** when a reason is familiarity **and** its `reason_confidence` > 0.60.
3. **ABSTAIN** when the reply had no stance. Two strikes settle that type.
4. **RECOMMEND** when the profile is ready, or at turn 10.
5. **CLARIFY** the unfilled type with the highest variance.

`state.ranking` holds this turn's top 100 occupations (best first, each with its RIASEC z-scores), so `_decide` can choose which type to ask about. `no_probing` (from `settings`) turns rules 1–2 off; that's the evaluation's control arm. `PolicyBase` raises an error if `_decide` probes anyway, so the comparison can't be silently invalid. A minimal working version is `SimplePolicy` in `tests/fakes.py`.

### Steps 5 and 7 (done)

- **Step 5:** `ranker.rank(interests, observed, sentences, top_k)` returns a `RankingResponse`. The default is the structured ranker. Its cards have no task quotes until the dense (MiniLM + FAISS) ranker exists; that's the project lead's work, per `cs425-ranking-docs/`.
- **Step 7:** `filler.fill(action, payload)` picks the reply from the question bank and fills in the user's quoted sentences.

### All together: `System.step` in `pipeline/system.py`

```
System.step(state, text)
 ├─ extractor.read(text, context, slot)          done
 │     └─ predict(context, sentence) × N          TEAMMATE   (model loaded by load() at startup)
 ├─ tracker.update(state, reading)                TEAMMATE   → state.slots
 ├─ rule.candidates(state)                        done       → possible conflicts
 │     └─ confirmer.score(old, new) per candidate TEAMMATE   → kept if > 0.60 → state.flags
 ├─ ranker.rank(..., 100) → state.ranking        done       (the policy reads it)
 ├─ policy.decide(state)                          done       (turn-10 stop, no_probing check)
 │     └─ _decide(state)                          TEAMMATE   → Action
 ├─ filler.fill(action, payload)                  done       → reply text
 └─ at RECOMMEND: state.ranking[:5]               done       → 5 cards
```

---

## 5. One message, file by file

What runs when the user presses Send, in order.

| # | File | Function | What happens |
|---|---|---|---|
| | **The page sends the message** | | |
| 1 | `static/app.js` | `send(text)` | Shows the user's bubble and the typing dots, makes a message id |
| 2 | `static/app.js` | `api('/api/chat', …)` | `fetch` POST `{session_id, text, message_id}` |
| | **The server checks it** | | |
| 3 | `app/schemas.py` | `ChatRequest` | Trims the text, checks 1–2000 characters and the message id. Bad input → 422 |
| 4 | `app/server.py` | `chat()` | Finds the session (404 if unknown). A repeated message id returns the stored reply. Finished → 409. Copies the state. |
| | **The seven steps** | | |
| 5 | `pipeline/system.py` | `System.step()` | Turn + 1, stores the text, picks the context (the bot's last question) |
| 6 | `pipeline/extractor.py` | `read()` → teammate's `predict()` per sentence | Message → readings |
| 7 | `pipeline/tracker.py` | teammate's `update()` | Merges the readings into `state.slots` |
| 8 | `pipeline/rule.py` | `candidates()` | Opposite signs, both confidence > 0.4? |
| 9 | `pipeline/system.py` | `_confirmed()` | Only for candidates: asks the NLI check, keeps the conflict if the score > 0.60 |
| 10 | `pipeline/nli.py` | teammate's `score()` | Contradiction probability |
| 11 | `recommender/structured.py` → `recommender/schemas.py` | `StructuredRanker.rank` (top 100) | Ranks the 378 occupations; stored in `state.ranking` for the policy |
| 12 | `pipeline/policy.py` | `decide()` → teammate's `_decide()` | Picks the action and its payload, using `state.ranking` |
| 13 | `pipeline/templates.py` | `QuestionBankFiller.fill` | The bot's sentence |
| 14 | `pipeline/system.py` | `recommendations()` | Only at RECOMMEND: the five cards, from `state.ranking[:5]` |
| 15 | `pipeline/system.py` | `return TurnResult` | Reply, action, quote cards, cards, done |
| | **Back through the server** | | |
| 16 | `app/server.py` → `pipeline/system.py` | `_response()` → `System.trace()` | Packs the JSON. With the researcher panel on, the live top 5 comes from `state.ranking` (no second ranking) |
| 17 | `app/turn_log.py` | `append()` | Writes one line to `logs/<code>.jsonl`, the only disk access per message |
| 18 | `app/server.py`, `app/schemas.py` | save, `TurnResponse` | Swaps in the new state, remembers the message id, checks the reply shape |
| | **The page shows the reply** | | |
| 19 | `static/app.js` | `addBot`, `renderTrace`, `renderResults` | Bot bubble (quote cards for CHALLENGE and PROBE), researcher panel, five cards when done |

- Nothing in `data/` or `models/` is read during a message; it was all loaded at startup.
- Opening the page first calls `startSession()` → `create_session()` → `System.start()`, which fills the opening question and logs it as turn 0.

### Which files are read when

| File | Read by | When |
|---|---|---|
| `configs/system.json` | `pipeline/system.py` `System.load` | Startup |
| `data/occupations.csv` | `recommender/data_loader.py`, through `StructuredRanker` | Startup |
| `data/question_bank.json`, `data/labels.json` | `pipeline/templates.py`; `labels.json` also by `pipeline/extractor.py` (class orders) | Startup |
| `models/extractor/` | `pipeline/extractor.py` `load()` | Startup |
| `cross-encoder/nli-deberta-v3-base` | `pipeline/nli.py` (Hugging Face cache) | Startup |
| `static/*` | The browser, served by FastAPI | Page load |
| `logs/<code>.jsonl` | Written by `app/turn_log.py` | Every message, and the opening |
| `data/extractor.jsonl`, `data/nli_pairs.jsonl` | `training/` code only | Training |
| `data/profiles.jsonl`, `data/eval/` | `eval/` only, never the server | Evaluation |

---

## 6. From training to the app (the hand-off)

```
training/extractor/   train on Colab (imports pipeline/extractor_model.py for the model class)
        │  export the checkpoint folder
        ▼
extractor-3f9c2a.zip  → team Drive → name posted in the group chat
        │  everyone downloads and unzips
        ▼
models/extractor/config.json, weights, tokenizer files     (files directly inside, gitignored)
        │  loaded once at startup
        ▼
pipeline/extractor.py  load()  →  predict() per sentence  →  read()  →  System.step
```

The NLI check needs no hand-off: the model downloads by name. Its tuned threshold goes into `settings.confirm_threshold`.

---

## 7. The server and the page

| Endpoint | Sends | Gets back |
|---|---|---|
| `GET /api/health` | — | status, eligible (378), the name of each running part |
| `POST /api/session` | `participant_code` (optional: letters, digits, `_`, `-`, up to 32), `debug` | session id, opening question |
| `POST /api/chat` | `session_id`, `text` (1–2000 chars), `message_id` | reply, action, turn, done, quotes, recommendations, trace |

- **Researcher panel:** on by default. Hide it in one tab with `?debug=0`; `CM_STUDY_MODE=1` hides it for everyone.
- **Sessions** live in memory and end with the recommendations. Only the transcript log is kept.
- **Retry is safe.** The same `message_id` never runs a turn twice, and the state is saved only after the log is written. Each session has its own lock.
- **The page** inserts text only, never HTML.

---

## 8. Configuration

`configs/system.json` names the version of each part:

```json
"components": {"extractor": "deberta", "tracker": "decay", "rule": "rule", "confirmer": "nli",
               "ranker": "structured", "policy": "manual", "filler": "question-bank"},
"settings":   {"max_turns": 10, "confirm_threshold": 0.6, "no_probing": false}
```

- These are the final names. Teammates fill in their file, and the config doesn't change.
- The file for a part that isn't filled in yet stops startup with `fill in … (docs/INTEGRATION.md)`.
- Run another config with `CM_CONFIG=path.json`. The other environment variables are `CM_DATA_DIR`, `CM_LOG_DIR` and `CM_STUDY_MODE`.

---

## 9. Tests

| File | Checks |
|---|---|
| `tests/test_pipeline.py` | A full conversation, CHALLENGE, PROBE (including the 0.60 reason-confidence rule), ABSTAIN, the no-signal ending, and that the structured ranker gives cards without task quotes |
| `tests/test_extractor.py` | `read()`: sentence splitting, label order, neutral filter, shape errors (with a faked `predict()`) |
| `tests/test_server.py` | API errors, participant codes, study mode, logging, safe retry, separate sessions |
| `tests/contracts.py` + `test_contracts.py` | One shape-check class per part: `ExtractorContract`, `TrackerContract`, `RuleContract`, `ConfirmerContract`, `RankerContract`, `PolicyContract`, `FillerContract` |
| `tests/fakes.py` | **Test-only** fakes so the tests run before the real models exist. They're also minimal working examples to copy from. |
| `tests/test_config.json` | Points the tests at those fakes |

To check your part, subclass its contract in a new test file:

```python
# tests/test_my_tracker.py
import unittest
from contracts import TrackerContract
from pipeline.tracker import DecayTracker

class TestDecay(TrackerContract, unittest.TestCase):
    def make(self):
        return DecayTracker()
```

Always run `.venv/bin/python -m unittest discover -s tests`.

---

## 10. Run commands

```
.venv/bin/pip install -r requirements.txt -r requirements-models.txt
.venv/bin/uvicorn app.server:app --reload           # http://localhost:8000, researcher panel on
CM_STUDY_MODE=1 .venv/bin/uvicorn app.server:app    # participant sessions, panel off
.venv/bin/python chat_cli.py                        # terminal chat
.venv/bin/python -m unittest discover -s tests      # all tests (run without any model)
```

---

## 11. Function reference: every important file

"Per message" means it runs on every user message; "startup" means once, when the server boots.

### `pipeline/system.py`: the conductor

| Function | When | What it does |
|---|---|---|
| `System.load(data_dir, config_path)` | startup | Reads `configs/system.json`, asks the registry to build all seven parts, reads the type names from `labels.json`, returns a `System` |
| `System.start(state)` | session start | Fills the opening question; sets `last_action = OPENING` and `last_reply` |
| `System.step(state, text)` | per message | **The main loop.** Turn + 1 → extractor → tracker → rule + NLI (`_confirmed`) → ranking into `state.ranking` → policy → filler → quotes or cards → bookkeeping. Returns a `TurnResult`. |
| `System._confirmed(state)` | per message | For each rule candidate, calls `confirmer.score()`. Keeps a flag only if the score > `confirm_threshold`. |
| `System._rank(state, top_k)` | per message | Builds the six-value interest vector (unobserved = 0) and calls `ranker.rank()`. Returns `[]` when there's no signal. |
| `System.recommendations(state)` | at RECOMMEND | The five cards from `state.ranking[:5]`: rank, code, title, why, up to two task quotes |
| `System._why(state)` | at RECOMMEND | One "why" line from the user's two strongest positive types |
| `System.top5(state)` | per message, panel on | Titles of `state.ranking[:5]` for the researcher panel |
| `System.profile(state)` | per message | The six slots (value, confidence, reason, reason confidence) for the panel and the log |
| `System.trace(state)` | per message, panel on | `profile` + action log + top 5 + part names |
| `System.components` | health check | The name of each running part |

### `pipeline/interfaces.py`: the data passed between steps

| Name | What it is |
|---|---|
| `SlotReading(slot, sign, strength, confidence, reason, source, reason_confidence)` | One stance on one type from one sentence. `value = sign × strength`. |
| `Reading(text, slots)` | Everything the extractor found in one message |
| `HistoryEntry`, `SlotState` | One past reading, and the current state of a type with its history |
| `Flag(slot, old, new, score)` | A contradiction: the two history entries and the NLI score |
| `Action(kind, slot, why, payload)` | The policy's decision. `payload` holds what the filler needs. |
| `State` | One conversation: turn, slots, user texts, last action/reply, flags, ranking, what's been challenged/probed/settled/resolved, counters, action log, done |
| `Extractor`, `Tracker`, `ContradictionRule`, `Confirmer`, `Ranker`, `Policy`, `Filler` | The method each part must provide (the contracts) |

### `pipeline/registry.py`

| Function | When | What it does |
|---|---|---|
| `build(config, data_dir)` | startup | For each of the seven kinds, looks up the name in `config["components"]` and calls its factory. An unknown name → error listing the valid names. |
| `_extractor_deberta`, `_tracker_decay`, `_confirmer_nli`, `_policy_manual`, `_rule`, `_ranker_structured`, `_filler_bank` | startup | One factory per part. Each imports its file only when chosen and passes it paths and settings from the config. |

### `pipeline/extractor.py`: step 1

| Function | Who | When | What it does |
|---|---|---|---|
| `DebertaExtractor.__init__(checkpoint_dir, labels_path)` | done | startup | Reads the class orders from `labels.json`, then calls `load()` |
| `load(checkpoint_dir)` | **teammate** | startup | Loads the tokenizer and model |
| `predict(context, sentence)` | **teammate** | per sentence | Model forward pass → `{"direction", "strength", "reason"}` probabilities |
| `sentences(text)` | done | per message | Splits on `. ! ?` and newlines, joining pieces of six words or fewer to the sentence before |
| `_check(out)` | done | per sentence | Raises a clear error if `predict()` returned the wrong shape |
| `read(text, context, expected_slot)` | done | per message | `sentences()` → `predict()` each → drops neutral types → `SlotReading`s, keeping the most confident per type |

### `pipeline/tracker.py`: step 2

| Function | Who | When | What it does |
|---|---|---|---|
| `DecayTracker.update(state, reading)` | **teammate** | per message | Merges readings into `state.slots` with recency decay, appends history, sets `last_reading_empty`, marks a PROBE answer as resolved |

### `pipeline/rule.py`: step 3

| Function | When | What it does |
|---|---|---|
| `ContradictionRule.candidates(state)` | per message | For each type updated this turn (not challenged or resolved): an earlier reading with the opposite sign, both confidence > 0.4 → `Flag` |

### `pipeline/nli.py`: step 4

| Function | Who | When | What it does |
|---|---|---|---|
| `NliConfirmer.__init__(model_name)` | **teammate** | startup | Loads the NLI cross-encoder |
| `score(old_src, new_src)` | **teammate** | only when the rule fires | Contradiction probability 0–1 |

### `pipeline/policy.py`: step 6

| Function | Who | When | What it does |
|---|---|---|---|
| `PolicyBase.decide(state)` | done | per message | Returns RECOMMEND at `max_turns`; otherwise calls `_decide()`, rejects CHALLENGE/PROBE under `no_probing`, then adds `index`, `slot` and `clarify_index` to the payload and updates the rotation counters |
| `ManualPolicy._decide(state)` | **teammate** | per message | The five rules (manual §7) → `Action` |
| `PROBE_REASON_CONF` | — | — | 0.60, the reason-confidence bar for PROBE |

### `pipeline/templates.py`: step 7

| Function | When | What it does |
|---|---|---|
| `QuestionBankFiller.fill(action, payload)` | per message | Calls `fill()` with the loaded question bank |
| `fill(action, payload, question_bank, type_names)` | per message | Picks the phrasing by `index` (rotated, never repeated) and formats `{old_src}`/`{new_src}`. CLARIFY and ABSTAIN look up the type's full name. |

### `recommender/`: step 5

| Function | When | What it does |
|---|---|---|
| `structured.StructuredRanker.__init__(occupations_path)` | startup | Loads the 378 eligible occupations through `data_loader.load_eligible_occupations` |
| `StructuredRanker.rank(interests, observed_types, user_sentences, top_k)` | per message | Score = Σ interest × occupation z-score over observed types, sorted (ties by code). No signal → `status "no_signal"`. |
| `data_loader.load_eligible_occupations(path)` | startup | Validates `occupations.csv` and keeps Job Zone 4–5 occupations with complete RIASEC z-scores |
| `schemas.RankingRequest`, `RankingResponse`, `RankedOccupation`, `TaskEvidence` | — | The ranking input and output shapes; `RankingRequest` validates the inputs |

### `app/server.py`

| Function | When | What it does |
|---|---|---|
| `lifespan(app)` | startup | `System.load(CM_DATA_DIR, CM_CONFIG)`, creates the session dict and its lock |
| `health()` | `GET /api/health` | status, eligible count, part names |
| `create_session(req)` | `POST /api/session` | New `State` and session id (participant code or `anon-xxxxxxxx`); debug off under `CM_STUDY_MODE`; logs the opening |
| `chat(req)` | `POST /api/chat` | Finds the session (404); a repeated `message_id` returns the stored reply; done → 409; runs `System.step` on a **copy** of the state, logs it, then saves the copy (any error → 500, state unchanged) |
| `_response(...)` | per message | Packs the JSON reply (adds `trace` when debug is on) |
| `_log(...)` | per message | One line through `turn_log.append` |

### `app/schemas.py` and `app/turn_log.py`

| Name | What it does |
|---|---|
| `SessionRequest`, `ChatRequest` | Validate input: participant code `A–Z a–z 0–9 _ -` up to 32, text 1–2000 chars after trimming, message id up to 64. Bad input → 422. |
| `TurnResponse`, `Rec`, `Trace`, `QuotesChallenge`, `QuotesNow`, `Health` | The shape of every reply |
| `turn_log.append(log_dir, code, record)` | Re-checks the code, then appends one timestamped JSON line to `logs/<code>.jsonl` |

### `static/app.js`: the page

| Function | What it does |
|---|---|
| `startSession()` | On load: reads `?p=` and `?debug=`, calls `/api/session`, shows the opening question |
| `send(text, isRetry)` | Adds the user's bubble, the typing dots and a message id, calls `/api/chat`, then renders the reply |
| `api(path, body)` | `fetch` wrapper that returns the JSON or throws with the status |
| `addUser(text)` / `addBot(data)` | Draw the chat bubbles; `addBot` draws quote cards for CHALLENGE and PROBE |
| `renderTrace(trace)` | The researcher panel: RIASEC bars, action log, top 5, part names |
| `renderResults(recs)` | The five cards and the "Before you go" section |
| `handleError(err, text, retryable)` | Network/500 → Retry with the same message id; 404 → Start again; 409 → finished |
| `updateHeader`, `syncComposer`, `showTyping`/`hideTyping`, `newMessageId`, `el` | Turn counter, enabling the input box, typing dots, message ids, safe element builder (text only, never HTML) |

### `chat_cli.py`

`main()` runs the same `System` in a terminal loop, which is handy for testing without the browser.

