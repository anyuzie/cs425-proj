# Career Matcher: architecture


**Current status:** M1, M3 BM25/RRF, M4 cached MiniLM/FAISS and M5 ranking integration are implemented. The live app selects `ranker: configured`, loading `configs/ranking.yaml` once at startup; **M1-only remains the default**. The legacy `structured` facade remains available. M5's ranking control is frozen with explicitly mocked upstream acceptance. Production activation still requires the extractor checkpoint, completed tracker/NLI/policy implementations and teammate latency agreement. See [ranking architecture memory](RANKING_EXPERIMENT_ARCHITECTURE.md), [M5 acceptance](../reports/ranking/m5/DECISION.md) and [extractor integration status](../reports/ranking/m5/EXTRACTOR_INTEGRATION.md).

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
| | `configs/` | `system.json`: application components/paths; `ranking.yaml`: ranking selection; `m2.json`: evaluation settings | yes |
| **Training and research**: never imported by the app | `training/<part>/` | Training scripts, notebooks, results JSON per part (see INTEGRATION.md §2) | yes, code only |
| | `recommender/evaluate.py`, `compare.py`, `history.py` | Implemented offline ranking evaluation, comparison and experiment registration | yes |
| **Tests** | `tests/` | Unit tests, contract tests, test-only fakes | yes |
| **Docs** | `docs/`, `cs425-ranking-docs/`, `reports/` | Guides, ranking plan, milestone reports | yes |
| **Data** | `data/` | The data package: O\*NET 31.0 and the team's generated data (read-only) | yes |
| **Local only** | `models/` | Trained checkpoints, e.g. `models/extractor/` | **no** |
| | `artifacts/ranking/` | Pinned MiniLM model and immutable task embedding/FAISS cache; BM25 uses an in-memory index | **no** |
| | `logs/` | Transcripts, one file per participant code | **no** |

Rule of thumb: **training code may import shared application code, but the app never imports from `training/`.** The current extractor uses Hugging Face `AutoModelForSequenceClassification`; a separate `pipeline/extractor_model.py` is not present or required by this adapter. Training supplies the checkpoint, tokenizer, labelled heads and calibration settings; application configs select paths and variants.

### Files in the app

| File | Step | Who | What it holds |
|---|---|---|---|
| `pipeline/system.py` | — | done | **The conductor**: `System.step()` runs the seven steps for one message |
| `pipeline/interfaces.py` | — | done | The data passed between steps (`Reading`, `SlotReading`, `State`, `Flag`, `Action`) and what each part must provide |
| `pipeline/registry.py` | — | done | Maps config names (`"deberta"`, `"nli"`, …) to the code that builds each part |
| `pipeline/extractor.py` | 1 | adapter implemented; checkpoint pending locally | Loads the trained 45-logit DeBERTa checkpoint, maps semantic head labels and produces `Reading` |
| `pipeline/tracker.py` | 2 | done | State tracker: confidence-weighted merge, optional recency decay (`settings.decay`); in the control arm it merges opposite statements instead of holding them |
| `pipeline/rule.py` | 3 | done | Contradiction rule |
| `pipeline/nli.py` | 4 | **teammate** fills `__init__()` + `score()` | NLI contradiction check |
| `recommender/ranker.py` | 5 | done | Configured M1/BM25/MiniLM pipeline with optional occupation-level RRF; M1-only default |
| `recommender/structured.py` | 5 | done | Backward-compatible M1-only facade |
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
Ranking     recommender/ranker.py → enabled M1 / BM25 / MiniLM → optional RRF
```

---

## 4. The teammate functions, and how they work with `System.step`

Each teammate's code only **receives data and returns data**. It never touches the server, the page, or another teammate's file. `System.step` hands each result to the next part in a fixed order, so the parts fit together as long as each method returns what its docstring says.

### Step 1 · Extractor: `pipeline/extractor.py`

**How answers to bot questions are read** (`pipeline/answers.py`): the extractor was trained with the bot's question attached only to short CLARIFY replies, so attaching it to anything else makes the question's words leak into the reading. So the answer **speaks for itself first**: it is read alone, and the question is attached only if that finds nothing ("the first one", "either is fine"). Short answers to a CHALLENGE ("the first one" / "the second one") keep the earlier or newer statement, and short answers to a PROBE ("I enjoy it" / "just what I know") keep the stance as enjoyment or set it to 0, by rule.

| Function | Who writes it | Called by | When | Does |
|---|---|---|---|---|
| `load(checkpoint_dir)` | implemented | `__init__` | **once**, at server start | Loads local tokenizer/model, validates 45 labelled logits and calibration; checkpoint files required |
| `predict(context, sentence)` | implemented | `read()` | once **per sentence** | Runs inference, maps checkpoint class ordering to the shared labels and returns probabilities/mapped strengths |
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

The tracker is ranking's immediate upstream component. It updates `state.slots` in place; `System._rank` adapts that state into the ranking request:

| Ranking field | Source and contract |
|---|---|
| `interests` | Exactly six RIASEC values from current tracker slots, finite and within `[-1,1]`; missing slots become zero |
| `observed_types` | Keys present in `state.slots`; an absent slot remains distinct from an observed zero |
| `user_sentences` | All chronological user-only messages accumulated by `System`, including earlier contradictory statements under the frozen raw-text policy |
| `top_k` | 100 for policy consumption; fewer results or `no_signal` are permitted |

Confidence merging and recency decay belong to the tracker. Ranking recomputes from each current snapshot without merging confidence again or accumulating occupation scores. The default oracle evaluator instead uses synthetic `hidden_slots` on an unrestricted z-scale through `OracleInput`; those profiles are not actual extractor/tracker outputs and must remain a separate evaluation condition.

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
2. **PROBE** when a reason is familiarity **and** its `reason_confidence` > `settings.probe_reason_conf` (default 0.60).
3. **ABSTAIN** when the reply had no stance: ask **once** more, then move on. "What do you mean?" / "not sure" keeps the same question with a concrete example; an unreadable answer gets the next phrasing. A second unsure answer marks the type "no clear preference" (counts as covered); it is asked once more later **only if the profile is not ready to recommend yet**. A stuck CHALLENGE keeps the newer statement at half confidence, is logged in `state.unresolved_conflicts`, and the bot re-ranks and does not recommend on that turn.
4. **RECOMMEND** when the profile is ready, or at turn 10.
5. **CLARIFY** the unfilled type with the highest variance.

`state.ranking` holds this turn's top 100 occupations (best first, each with its RIASEC z-scores), so `_decide` can choose which type to ask about. `no_probing` (from `settings`) turns rules 1–2 off; that's the evaluation's control arm. `PolicyBase` raises an error if `_decide` probes anyway, so the comparison can't be silently invalid. A minimal working version is `SimplePolicy` in `tests/fakes.py`.

### Steps 5 and 7 (done)

- **Step 5:** `RankingPipeline.rank(interests, observed_types, user_sentences, top_k)` returns a `RankingResponse`. `configs/ranking.yaml` independently selects M1, BM25 and MiniLM variants. One usable component bypasses fusion; two or more use configured equal-weight RRF at `k=60`; none returns `no_signal`. M1-only remains default and supplies no task quotes. Enabled MiniLM retrieval attaches provisional task matches; cards expose up to three per occupation without claiming that similarity establishes enjoyment.
- **Step 7:** `filler.fill(action, payload)` picks the reply from the question bank and fills in the user's quoted sentences.

### All together: `System.step` in `pipeline/system.py`

```
System.step(state, text)
 ├─ extractor.read(text, context, slot)          done
 │     └─ predict(context, sentence) × N          implemented (local checkpoint required at startup)
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
| 6 | `pipeline/answers.py` → `pipeline/extractor.py` | `read_answer()` → `read()` → `predict()` per sentence | Short CHALLENGE/PROBE answers by rule; otherwise the answer alone first, the question only if needed. Message → readings |
| 7 | `pipeline/tracker.py` | teammate's `update()` | Merges the readings into `state.slots` |
| 8 | `pipeline/rule.py` | `candidates()` | Opposite signs, both confidence > 0.4? |
| 9 | `pipeline/system.py` | `_confirmed()` | Only for candidates: asks the NLI check, keeps the conflict if the score > 0.60 |
| 10 | `pipeline/nli.py` | teammate's `score()` | Contradiction probability |
| 11 | `pipeline/system.py` → `recommender/ranker.py` | `_rank()` → `RankingPipeline.rank` (top 100) | Reranks current tracker values and accumulated user text; stores up to 100 occupations in `state.ranking` and captures input/status/timing in `ranking_trace` |
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
| `configs/ranking.yaml` | `pipeline/registry.py`, through `RankingPipeline` | Startup; legacy `structured` factory retains fixed M1-only selection |
| `data/occupations.csv` | `recommender/data_loader.py`, through `RankingPipeline` | Startup |
| `artifacts/ranking/` | MiniLM component: pinned model and task cache/index | Startup, only when M4 is enabled; no task corpus embedding per turn |
| `data/question_bank.json`, `data/labels.json` | `pipeline/templates.py`; `labels.json` also by `pipeline/extractor.py` (class orders) | Startup |
| `models/extractor/` | `pipeline/extractor.py` `load()` | Startup |
| `cross-encoder/nli-deberta-v3-base` | `pipeline/nli.py` (Hugging Face cache) | Startup |
| `static/*` | The browser, served by FastAPI | Page load |
| `logs/<code>.jsonl` | Written by `app/turn_log.py` | Every message, and the opening |
| `data/extractor.jsonl`, `data/nli_pairs.jsonl` | `training/` code only | Training |
| `data/dev_profiles.jsonl` or configured tracker-session JSONL | `recommender/evaluate.py`, never the live server | Offline development evaluation |

---

## 6. From training to the app (the hand-off)

```
training/extractor/   external training/export workflow (not implemented in this checkout)
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
               "ranker": "configured", "policy": "manual", "filler": "question-bank"},
"settings":   {"max_turns": 10, "confirm_threshold": 0.6, "no_probing": false, "decay": 1.0,
               "probe_reason_conf": 0.6, "input_mode": "actual_tracker"},
"paths":      {"ranking_config": "configs/ranking.yaml", "extractor_checkpoint": "models/extractor"}
```

- `no_probing`: the control arm. No CHALLENGE or PROBE, and the tracker merges opposite statements instead of holding them, so it "listens to everything".
- `decay`: how much one turn of age shrinks older evidence. `1.0` = off (all turns count equally), `0.9` = on. Test both.
- `probe_reason_conf`: PROBE only fires when the reason head is more sure than this that the reason is familiarity.

**Both arms treat evidence the same way.** An opposite statement is held for a CHALLENGE only when the rule and the NLI check confirm a contradiction. If they don't (low confidence, or NLI says it is an elaboration), it is merged like any other evidence, which is exactly what the control arm does with every opposite statement. So full vs `no_probing` differs only in asking.

**Known limits of the simulated evaluation** (`eval/dialogue.py`):
- The simulated student answers a CHALLENGE or PROBE retry with the same resolution sentence, so those retries cannot succeed in simulation.
- The short-answer rules in `pipeline/answers.py` rarely fire there: the simulated resolutions and reveals are full sentences. How the rules behave with real people is only tested by the unit tests and the pilot.

- The snippet shows component names and relevant settings/path entries; the complete file also contains NLI and legacy artifact paths. `configured` loads ranking selections from YAML; `structured` remains available for compatibility.
- The file for a part that isn't filled in yet stops startup with `fill in … (docs/INTEGRATION.md)`.
- Run another config with `CM_CONFIG=path.json`. The other environment variables are `CM_DATA_DIR`, `CM_LOG_DIR` and `CM_STUDY_MODE`.

Ranking configuration and evaluation configuration stay separate. `python -m recommender.evaluate` automatically loads `configs/ranking.yaml` and `configs/m2.json`. The default evaluation uses synthetic oracle development profiles. Explicit `actual_tracker` and `mock_tracker` modes replay recorded sessions, retain per-turn rankings and compute the unchanged M2 metrics on each session's final turn. Reports preserve exact configurations, input mode/provenance, hashes and component variants; successful runs append to the centralized evaluation history. See the [tracker replay workflow](../cs425-ranking-docs/README.md#multi-turn-tracker-evaluation).

---

## 9. Tests

| File | Checks |
|---|---|
| `tests/test_pipeline.py` | A full conversation, CHALLENGE, PROBE (including the 0.60 reason-confidence rule), ABSTAIN, the no-signal ending, and that the structured ranker gives cards without task quotes |
| `tests/test_extractor.py` | `read()`: sentence splitting, label order, neutral filter, shape errors (with a faked `predict()`) |
| `tests/test_extractor_checkpoint.py` | Checkpoint label mapping, signed strength propagation into M5, inference context, lazy imports and missing-checkpoint errors; tensor fixtures, not model accuracy |
| `tests/test_integration.py` | M5 multi-turn/conflict fixtures, runtime provenance, short/empty rankings, history registration and cached real three-way retrieval |
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
.venv/bin/python -m unittest discover -s tests      # full suite; see local resource requirements below
```

Production launch requires the installed extractor checkpoint and completed tracker/NLI/policy. Most tests use fixtures; full discovery currently includes a startup test requiring the missing extractor checkpoint, while real dense tests need the prepared M4 model/cache. `python scripts/check_m5_extractor.py --checkpoint <directory>` runs real extraction with configured ranking and explicitly mocked remaining upstream components; it does not establish actual-tracker quality.

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
| `System.recommendations(state)` | at RECOMMEND | Up to five cards from `state.ranking[:5]`: rank, code, title, why, up to three task texts and structured evidence metadata/status |
| `System._why(state)` | at RECOMMEND | One "why" line from the user's two strongest positive types |
| `System.top5(state)` | per message, panel on | Titles of `state.ranking[:5]` for the researcher panel |
| `System.profile(state)` | per message | The six slots (value, confidence, reason, reason confidence) for the panel and the log |
| `System.trace(state)` | per message, panel on | Profile, action log, top 5, part names, input mode, per-turn ranking trace and resolved ranking config |
| `System.tracker_record(state, profile_id=..., revision=...)` | explicit export | Chronological tracker inputs and upstream provenance for evaluation; rejects unverified mode; independent annotations add relevance labels |
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
| `_extractor_deberta`, `_tracker_decay`, `_confirmer_nli`, `_policy_manual`, `_rule`, `_ranker_configured`, `_ranker_structured`, `_filler_bank` | startup | Factories import selected components and pass configured paths/settings. All component names are validated before construction. |

### `pipeline/answers.py`: reading answers to questions

| Function | When | What it does |
|---|---|---|
| `read_answer(extractor, state, last, text)` | per message | After a question: short-answer rules first, then the answer alone, then with the question only if the answer alone has no stance |
| `short_answer(kind, text, state, slot)` | per message | ≤ 8 words: CHALLENGE "first/earlier" → earlier statement, "second/latest" → newer; PROBE "enjoy" → keep as enjoyment, "just what I know" / "don't enjoy" → 0. Ambiguous answers return `None` and are read normally |

### `pipeline/extractor.py`: step 1

| Function | Who | When | What it does |
|---|---|---|---|
| `DebertaExtractor.__init__(checkpoint_dir, labels_path)` | done | startup | Reads the class orders from `labels.json`, then calls `load()` |
| `load(checkpoint_dir)` | implemented | startup | Loads local tokenizer/model, validates head count/calibration and configures semantic logit indices |
| `_configure_heads(config)` | implemented | startup | Maps checkpoint `id2label` into shared direction/reason/strength ordering; rejects incompatible heads |
| `predict(context, sentence)` | implemented | per sentence | Inference-only forward pass → shared-order direction/reason probabilities and discrete strengths |
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
| `StructuredRanker.rank_oracle(inputs)` | offline evaluation | Ranks the whole pool from an `OracleInput` (a profile's hidden z-scale scores), for the M2 evaluator; never used by the app |
| `StructuredRanker.eligible_codes` | offline evaluation | The set of eligible occupation codes |
| `data_loader.load_eligible_occupations(path)` | startup | Validates `occupations.csv` and keeps Job Zone 4–5 occupations with complete RIASEC z-scores |
| `ranker.RankingPipeline` | startup/per message | Resolves enabled registered variants once; validates runtime inputs, invokes independent components and assembles standalone or RRF rankings |
| `bm25_ranker.BM25Ranker`, `dense_ranker.DenseRanker`, `fusion.ReciprocalRankFusion` | enabled components | Implemented lexical ranking, cached semantic task retrieval and occupation-level RRF; M1 remains in `m1.py` |
| `evaluation.relevance_labels`, `evaluate_ranking`, `aggregate` | offline evaluation | Unchanged Hit@1/5/10, full-pool MRR and NDCG@5/10 per evaluation unit and on average |
| `evaluate.run(config_path, output)` / `main()` | offline evaluation | Loads separate default configs; evaluates oracle profiles or explicit tracker-session snapshots; exports fresh reports under `reports/ranking/experiments/` by default |
| `history.append_history(report)` | completed evaluation | Appends one result per unique run ID to `reports/ranking/evaluation_history.csv` |
| `compare.paired_comparison(control, treatment)` | offline evaluation | Paired bootstrap comparison of two evaluated runs |
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

