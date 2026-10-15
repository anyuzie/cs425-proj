# Career Matcher: how the system works

This document is the whole architecture in one place: what runs, in what order, which file does each job, what is real today and what is still a stand-in. For the short "what's left to build" checklist, see [INTEGRATION.md](INTEGRATION.md).

---

## 1. The big picture

A student chats with the bot in a web page. Each message goes through **seven steps**. They turn the message into an interest profile (six RIASEC numbers), check it for contradictions, rank 378 degree-level occupations, and choose what the bot says next. After 5–10 turns the bot recommends five occupations and quotes the O\*NET task statements that matched.

Everything runs in **one Python process**:

```
 Browser (static/)                        FastAPI server (app/)
 ┌────────────────────┐   POST /api/chat  ┌───────────────────────────────────────────┐
 │ chat page          │ ────────────────► │ server.py  → sessions → System.step(...)   │
 │ researcher panel   │ ◄──────────────── │                         │                  │
 │ 5 recommendation   │   reply + action  │   pipeline/system.py runs steps 1–7:       │
 │ cards              │   + cards + trace │   extract → track → rule → NLI → rank      │
 └────────────────────┘                   │   → policy → template                      │
                                          │   models + indexes loaded once at startup  │
                                          └───────────────────────────────────────────┘
```

- **Models do perception** (reading the message, checking contradictions, matching tasks).
- **Plain Python does the deciding** (the tracker, the rule, the policy, the templates). Every recommendation can be traced back to the profile that produced it.
- **No LLM runs inside the system.** Every sentence the bot says comes from `data/question_bank.json`.

### Layers: who calls whom

Each layer only calls the one below it, so a part can be replaced without touching anything above.

```
Page        static/index.html, app.js, styles.css        browser; talks only to /api/session and /api/chat
   ↓ HTTP JSON
Server      app/server.py, schemas.py, turn_log.py         checks input, sessions, logs; knows nothing about models
   ↓ System.step(state, text)
Conductor   pipeline/system.py, registry.py, configs/system.json   calls the seven parts in order
   ↓ read · update · candidates · score · decide · fill · rank
Parts       pipeline/stubs.py (stand-ins), templates.py, extractor.py, tracker.py, nli.py, policy.py
   ↓ ranker.rank(...)
Ranking     recommender/structured.py, schemas.py, data_loader.py, fusion.py, bm25.py, dense.py
```

### One message, file by file

What runs when the user presses Send, in order. `[stand-in]` marks code that will be replaced, with the file that replaces it.

| # | File | Function | What happens |
|---|---|---|---|
| | **The page sends the message** | | |
| 1 | `static/app.js` | `send(text)` | Shows the user's bubble, the typing dots, makes a message id |
| 2 | `static/app.js` | `api('/api/chat', …)` | `fetch` POST `{session_id, text, message_id}` |
| | **The server checks it** | | |
| 3 | `app/schemas.py` | `ChatRequest` | Trims the text, checks 1–2000 characters and the message id. Bad input → 422 |
| 4 | `app/server.py` | `chat()` | Finds the session (404 if unknown). Same message id as last time → returns the stored reply. Finished → 409. Copies the state. |
| | **The seven steps** | | |
| 5 | `pipeline/system.py` | `System.step()` | Turn + 1, stores the text, picks the context (the bot's last question) |
| 6 | `pipeline/stubs.py` | `LexiconExtractor.read` [stand-in → `pipeline/extractor.py`] | Message → readings per type |
| 7 | `pipeline/stubs.py` | `SimpleTracker.update` [stand-in → `pipeline/tracker.py`] | Merges readings into `state.slots`, keeps the sentence |
| 8 | `pipeline/stubs.py` | `ContradictionRule.candidates` | Opposite signs, both confidence > 0.4? |
| 9 | `pipeline/system.py` | `_confirmed()` | Only for candidates: asks the confirmer, keeps a flag if the score > 0.60 |
| 10 | `pipeline/stubs.py` | `PassthroughConfirmer.score` [stand-in → `pipeline/nli.py`] | Contradiction probability for the two sentences |
| 11 | `pipeline/stubs.py` | `RulePolicy.decide` [stand-in → `pipeline/policy.py`] | Picks the action and its payload |
| 12 | `pipeline/templates.py` | `QuestionBankFiller.fill` | Picks the sentence, fills in the quotes |
| 13 | `pipeline/system.py` | `recommendations()` | Only at RECOMMEND: builds the five cards |
| 14 | `pipeline/stubs.py` → `recommender/structured.py` → `recommender/schemas.py` | `StructuredKeywordRanker.rank` [stand-in → `recommender/fusion.py` + `bm25.py` + `dense.py`] | Ranks the 378 occupations and attaches the matched tasks |
| 15 | `pipeline/system.py` | `return TurnResult` | Reply, action, quote cards, cards, done |
| | **Back through the server** | | |
| 16 | `app/server.py` → `pipeline/system.py` | `_response()` → `System.trace()` | Packs the JSON. Researcher panel on → `profile()` plus `top5()`, which runs `ranker.rank` again |
| 17 | `app/turn_log.py` | `append()` | Writes one line to `logs/<code>.jsonl`, the only disk access per message |
| 18 | `app/server.py`, `app/schemas.py` | save, `TurnResponse` | Swaps in the new state, remembers the message id, checks the reply shape |
| | **The page shows the reply** | | |
| 19 | `static/app.js` | `addBot`, `renderTrace`, `renderResults` | Bot bubble (quote cards for CHALLENGE and PROBE), researcher panel, five cards when done |

Two things this shows:
- Nothing in `data/`, `models/` or `indexes/` is read during a message; it's all loaded into memory at startup. The only disk access is the log line.
- The ranker isn't on a normal turn's own path. It runs for the researcher panel's live top 5 (row 16) and for the cards at RECOMMEND (row 14). The stand-in policy doesn't use the ranking.

The first message works the same way, except that opening the page first calls `startSession()` in `app.js`, which calls `create_session()` in `server.py`, then `System.start()`. That fills the opening question and logs it as turn 0.

### Which files are read when

| File | Read by | When |
|---|---|---|
| `configs/system.json` | `pipeline/system.py` `System.load` | Startup |
| `data/occupations.csv` | `recommender/data_loader.py`, through `StructuredRanker` | Startup |
| `data/tasks.csv` | `pipeline/stubs.py` `StructuredKeywordRanker` (task quotes) | Startup |
| `data/riasec_lexicon.json` | `pipeline/stubs.py` `LexiconExtractor` | Startup |
| `data/question_bank.json` | `pipeline/templates.py` `QuestionBankFiller` | Startup |
| `data/labels.json` | `pipeline/system.py` and the filler (type names) | Startup |
| `models/extractor/` | `pipeline/extractor.py` | Startup, once switched on |
| `cross-encoder/nli-deberta-v3-base` | `pipeline/nli.py` (Hugging Face cache) | Startup, once switched on |
| `indexes/bm25.pkl` | `recommender/bm25.py` | Startup, once switched on |
| `indexes/faiss.idx`, `faiss_owners.json`, MiniLM | `recommender/dense.py` | Startup, once switched on |
| `static/*` | The browser, served by FastAPI | Page load |
| `logs/<code>.jsonl` | Written by `app/turn_log.py` | Every message, and the opening |
| `data/profiles.jsonl`, `data/eval/` | The evaluation harness only, never the server | Evaluation runs |

---

## 2. Folder map

| Folder / file | What it is | Runs when |
|---|---|---|
| `static/` | The web page: `index.html`, `styles.css`, `app.js` | In the browser |
| `app/server.py` | FastAPI app: routes, sessions, startup | Server |
| `app/schemas.py` | Shapes of every API request and response | Server |
| `app/turn_log.py` | Writes one JSON line per turn to `logs/<code>.jsonl` | Server |
| `pipeline/system.py` | **The conductor.** Runs the seven steps in order for one turn | Every turn |
| `pipeline/interfaces.py` | The shapes passed between steps (`Reading`, `State`, `Action`, …) and what each step must provide | Everywhere |
| `pipeline/registry.py` | Maps config names (e.g. `"deberta"`) to the code that builds that part | Startup |
| `pipeline/stubs.py` | All the **stand-ins**, plus the real contradiction rule | Every turn, until replaced |
| `pipeline/templates.py` | Step 7, the template filler (real) | Every turn |
| `pipeline/extractor.py`, `tracker.py`, `nli.py`, `policy.py` | **Skeletons** for steps 1, 2, 4, 6, with `PUT YOUR CODE HERE` blocks | When switched on |
| `recommender/` | Step 5, scoring and ranking. `structured.py` is real; `bm25.py`, `dense.py`, `fusion.py` are skeletons | Every turn |
| `scripts/build_indexes.py` | Skeleton: builds the BM25 and FAISS indexes once, offline | By hand, once |
| `configs/system.json` | **Which version of each part runs**, plus settings and model paths | Startup |
| `data/` | The data package: occupations, tasks, question bank, labels, profiles (gitignored) | Startup |
| `models/` | The trained extractor checkpoint (gitignored) | Startup |
| `indexes/` | `bm25.pkl`, `faiss.idx`, `faiss_owners.json` (gitignored) | Startup |
| `logs/` | Transcripts, one file per participant code (gitignored) | Every turn |
| `tests/` | Unit tests, contract tests, `stub_config.json` | `unittest` |
| `chat_cli.py` | The same pipeline in a terminal loop | By hand |
| `reports/ranking/` | Milestone write-ups (M0, M1, …) | Documentation |

---

## 3. Startup: what happens before anyone types

```
.venv/bin/uvicorn app.server:app
  └─ app/server.py  lifespan()
      └─ System.load(data/, configs/system.json)              pipeline/system.py
          └─ registry.build(config)                            pipeline/registry.py
              for each part, read its name in "components" and build it:
                extractor  "lexicon-stub" → stubs.LexiconExtractor(riasec_lexicon.json)
                           "deberta"      → extractor.DebertaExtractor(models/extractor)   ← loads model
                tracker    "simple-stub"  → stubs.SimpleTracker()
                           "decay"        → tracker.DecayTracker()
                rule       "rule"         → stubs.ContradictionRule()
                confirmer  "passthrough-stub" → stubs.PassthroughConfirmer()
                           "nli"          → nli.NliConfirmer(cross-encoder/nli-deberta-v3-base) ← loads model
                ranker     "structured+keyword-stub" → stubs.StructuredKeywordRanker(occupations, tasks)
                           "rrf"          → recommender.fusion.RRFRanker(...)  ← loads MiniLM, BM25, FAISS
                policy     "rule-stub"    → stubs.RulePolicy(max_turns)
                           "manual"       → policy.ManualPolicy(max_turns, no_probing)
                filler     "question-bank"→ templates.QuestionBankFiller(question_bank.json)
      └─ sessions = {}   (in memory; lost on restart, which is fine for study sessions)
```

- A model's file is **only imported when its name is chosen**, so the app runs without torch while everything is a stand-in.
- An **unknown name** stops startup and lists the valid names.
- A **skeleton that isn't filled in yet** stops startup with a pointer to `INTEGRATION.md`, so a half-built part can't run quietly.
- With all real models this takes roughly 15 seconds, once.

---

## 4. One turn, step by step

The page sends `POST /api/chat {session_id, text, message_id}`. `app/server.py` finds the session, runs `System.step(state, text)` on a **copy** of the state, writes the log line, and only then saves the new state. If anything fails, the state is unchanged and the page's Retry sends the same `message_id`, so the turn is never counted twice.

Inside `System.step` (`pipeline/system.py`):

```
text ─► 1 extractor.read ─► 2 tracker.update ─► 3 rule.candidates ─► 4 confirmer.score (only if 3 found something)
     ─► 5 ranker.rank ─► 6 policy.decide ─► 7 filler.fill ─► reply (+ quotes, + 5 cards at the end)
```

### Step 1 · Preference extractor

| | |
|---|---|
| **Job** | Read one user message. Say which RIASEC types it takes a stance on, which direction, how strongly, how sure the model is, and whether the reason is enjoyment or familiarity. |
| **Gets** | `text` (the message), `context` (the bot question being answered, or `None`), `expected_slot` (the letter that question was about, e.g. `"C"`) |
| **Returns** | `Reading(text, slots=[SlotReading(slot, sign, strength, confidence, reason, source), …])` |
| **Example** | "I love figuring out why things go wrong" → `SlotReading("I", +1, 1.0, 0.88, "enjoyment", "I love figuring out why things go wrong")` |
| **Model** | `microsoft/deberta-v3-base`, fine-tuned on `data/extractor.jsonl` with three heads: direction (positive/neutral/negative per type), strength (0–1 per type), reason (enjoyment/familiarity/neither). Replies are read as `tokenizer(question, reply)`. |
| **Now** | Stand-in `LexiconExtractor` (`pipeline/stubs.py`): keyword list from `data/riasec_lexicon.json`, fixed confidence 0.7 |
| **To build** | `pipeline/extractor.py` → `"extractor": "deberta"`. Checkpoint in `models/extractor/`. |

`value = sign × strength`. Types read as neutral are left out of the reading.

### Step 2 · State tracker

| | |
|---|---|
| **Job** | Merge this turn's readings into the running profile, keeping a history for each type |
| **Gets** | `state` and the step 1 `Reading` |
| **Returns** | Nothing. It updates `state.slots[letter] = SlotState(value, confidence, reason, source, turn, history)` |
| **Must also** | Set `state.last_reading_empty`. Keep the **source sentence** of every reading (the quote cards and the NLI check need it). If the last bot action was a PROBE on a type, add that type to `state.resolved`. |
| **Model** | None. Plain Python, with recency decay (older evidence weighted down, merge weight = confidence). |
| **Now** | Stand-in `SimpleTracker`: the latest reading replaces the old one |
| **To build** | `pipeline/tracker.py` → `"tracker": "decay"` |

### Step 3 · Contradiction rule

| | |
|---|---|
| **Job** | A cheap test for a *possible* conflict: the same type has readings with opposite signs, both with confidence > 0.4 |
| **Gets** | `state` |
| **Returns** | Candidate `Flag(slot, old, new)` objects, each holding the two history entries with their sentences |
| **Skips** | Types already challenged, and types in `state.resolved` (a probe answer is a resolution, not a contradiction) |
| **Model** | None. It's arithmetic on two numbers. |
| **Now** | **Real**: `ContradictionRule` in `pipeline/stubs.py` |
| **Note** | The 0.4 gate only does something once the real extractor produces real confidences. The stand-in always says 0.7, so the gate always passes. |

### Step 4 · NLI check

| | |
|---|---|
| **Job** | Decide whether the two flagged sentences really contradict each other. **Only runs when step 3 found a candidate.** |
| **Gets** | `old_src`, `new_src` (the two sentences) |
| **Returns** | Probability of contradiction, 0–1. `System` keeps the flag only if it is above `settings.confirm_threshold` (0.60). |
| **Example** | "Leading a team is fine for me" vs "I'd rather be a strong contributor than the team lead" sound opposite but aren't, so the score should be low and no challenge follows. |
| **Model** | `cross-encoder/nli-deberta-v3-base`, used as is (no training) |
| **Tune** | The threshold on the 75 `dev` pairs in `data/nli_pairs.jsonl`. Report once on the 64 `test` pairs. |
| **Now** | Stand-in `PassthroughConfirmer`: always 1.0, so every candidate becomes a CHALLENGE |
| **To build** | `pipeline/nli.py` → `"confirmer": "nli"` |

### Step 5 · Scorer / ranker

| | |
|---|---|
| **Job** | Rank the 378 degree-level occupations (Job Zones 4–5) for this user, and say which task statements matched |
| **Gets** | `interests` (six values, unobserved types = 0), `observed_types`, `user_sentences`, `top_k` |
| **Returns** | `RankingResponse` with `RankedOccupation`s. Each carries `evidence` (matched tasks); the cards quote the first two. |
| **Three signals** | **Structured**: interest vector × each occupation's z-scored RIASEC row (`recommender/structured.py`, real). **BM25**: keyword match of the user's text against each occupation's text (`recommender/bm25.py`). **Dense**: MiniLM embeddings of the user's sentences against every task statement, through FAISS (`recommender/dense.py`). The dense task matches are the quoted evidence. |
| **Fusion** | RRF, k = 60: `score = Σ 1 / (60 + rank in that signal)`, in `recommender/fusion.py` |
| **Indexes** | Built once by `scripts/build_indexes.py` into `indexes/` |
| **Model** | `all-MiniLM-L6-v2` (no training) |
| **Now** | Stand-in `StructuredKeywordRanker`: real structured score, plus task quotes picked by word overlap |
| **To build** | `scripts/build_indexes.py`, `bm25.py`, `dense.py`, `fusion.py` → `"ranker": "rrf"`. Milestones M2–M5 in `cs425-ranking-docs/`. |
| **Gotcha** | `user_sentences` currently includes short replies like "the first one". The manual says to query BM25 with free text only, so filter those out inside BM25. |

The ranker runs to fill the researcher panel's live top 5 every turn, and again at RECOMMEND to build the cards.

### Step 6 · Policy

| | |
|---|---|
| **Job** | Choose the bot's next move. Five rules, checked top to bottom; the first match wins. |
| **Gets** | `state` (profile, confirmed flags, turn count, what has been asked) |
| **Returns** | `Action(kind, slot, why, payload)`. The payload holds what step 7 needs. |
| **Rules** | **1 CHALLENGE**: a confirmed contradiction exists, so quote both sentences and ask which is closer. **2 PROBE**: a stance was justified by familiarity, so ask whether they'd enjoy it or just know it. **3 ABSTAIN**: the reply had no stance, so rephrase; after two strikes the type is settled. **4 RECOMMEND**: ready (every type filled or settled, profile converged, 5–10 turns), or turn 10 reached. **5 CLARIFY**: ask about the unfilled type with the highest variance. |
| **Model** | None. Same input, same output. |
| **Now** | Stand-in `RulePolicy`: all five rules, simplified (fixed question order, no 0.60 reason-confidence check, no re-challenge, no convergence check) |
| **To build** | `pipeline/policy.py`, method `_decide` only → `"policy": "manual"`. It also reads `settings.no_probing`, the evaluation's control arm (no CHALLENGE or PROBE). |

The question rotation (so the bot never repeats a phrasing) is already handled by `RulePolicy.decide`, which `ManualPolicy` inherits.

### Step 7 · Template filler

| | |
|---|---|
| **Job** | Turn the action into the bot's sentence |
| **Gets** | `action`, `payload` |
| **Returns** | Reply text, picked from `data/question_bank.json`: 1 opening, 18 CLARIFY (3 per type), 4 CHALLENGE, 4 PROBE, 3 ABSTAIN prefixes, 2 RECOMMEND headers. `{old_src}` and `{new_src}` are filled with the user's own sentences. |
| **Model** | None. Nothing is generated. |
| **Now** | **Real**: `pipeline/templates.py` → `QuestionBankFiller` |

### After the seven steps

`System.step` packs the turn result:
- **Quotes for the page:** both sentences for CHALLENGE, one for PROBE.
- **At RECOMMEND:** five cards (title, O\*NET code, a "why" line, up to two task quotes) and `done = true`.
- **Bookkeeping:** turn count, last action, last reply, and the action log used by the researcher panel.

If the conversation ends with no usable signal, the bot apologises instead of recommending.

---

## 5. What a session's data looks like

`State` (`pipeline/interfaces.py`), one per conversation, keyed by letters `R I A S E C`:

| Field | Meaning |
|---|---|
| `turn` | User turns so far (the opening is turn 0) |
| `slots` | Per type: current value, confidence, reason, source sentence, turn, history |
| `user_texts` | Everything the user said |
| `last_action`, `last_reply` | What the bot did and said last (context for step 1) |
| `flags` | Confirmed contradictions this turn |
| `challenged`, `probed`, `settled`, `resolved`, `strikes` | What the policy has already done |
| `clarify_count`, `kind_count` | Question rotation counters |
| `actions` | The log shown in the researcher panel |

---

## 6. The server and the page

**API** (`app/server.py`, shapes in `app/schemas.py`):

| Endpoint | In | Out |
|---|---|---|
| `GET /api/health` | — | `status`, `eligible` (378), `components` (the name of each part that's running) |
| `POST /api/session` | `participant_code` (optional, `A–Z a–z 0–9 _ -`, ≤32), `debug` | Session id, opening question, `turn 0` |
| `POST /api/chat` | `session_id`, `text` (1–2000 chars), `message_id` | `reply`, `action`, `turn`, `max_turns`, `done`, `quotes`, `recommendations`, `trace` |

Errors: 404 unknown session, 409 conversation finished, 422 bad input.

- **Researcher panel** (`trace`): six RIASEC bars, policy log, live top 5, component names. On by default for demos. Hide it for one tab with `?debug=0`. For participant sessions, start the server with `CM_STUDY_MODE=1`, which hides it server-side whatever the URL says.
- **Logs:** one JSON line per turn in `logs/<participant code>.jsonl`, with the user text, reply, action, profile and cards. The code is checked before it becomes a filename. Without `?p=CODE`, the server makes `anon-xxxxxxxx`.
- **Locks:** each session has its own lock, so two people chatting don't wait for each other.
- **The page** (`static/app.js`) only ever inserts text, never HTML, so user text can't inject anything. It renders CHALLENGE and PROBE as quote cards and the RECOMMEND turn as five cards plus a "Before you go" section. It works at phone width.

---

## 7. Configuration: switching parts

`configs/system.json`:

```json
{
  "components": {
    "extractor": "lexicon-stub",           // → "deberta"
    "tracker":   "simple-stub",            // → "decay"
    "rule":      "rule",
    "confirmer": "passthrough-stub",       // → "nli"
    "ranker":    "structured+keyword-stub",// → "rrf"
    "policy":    "rule-stub",              // → "manual"
    "filler":    "question-bank"
  },
  "settings": {"max_turns": 10, "confirm_threshold": 0.6, "no_probing": false},
  "paths": {"extractor_checkpoint": "models/extractor", "nli_model": "cross-encoder/nli-deberta-v3-base",
            "embedder": "all-MiniLM-L6-v2", "bm25": "indexes/bm25.pkl",
            "faiss": "indexes/faiss.idx", "faiss_owners": "indexes/faiss_owners.json"}
}
```

**To switch a part:**
1. Fill in its `PUT YOUR CODE HERE` blocks.
2. Delete the `raise NotImplementedError(...)` guard line.
3. Change its name in `components`.
4. Restart and check `/api/health`.

Parts can be switched one at a time, in any order; the rest stay stand-ins. Run a different config with `CM_CONFIG=path/to/other.json`. Other env vars: `CM_DATA_DIR`, `CM_LOG_DIR`, `CM_STUDY_MODE`.

---

## 8. Tests

| File | Checks |
|---|---|
| `tests/test_pipeline.py` | Turn behaviour: a full conversation, CHALLENGE, PROBE, ABSTAIN, the probe-answer rule, the no-signal ending |
| `tests/test_server.py` | API: errors, participant codes, study mode, logging, retry with the same `message_id`, a log failure leaving state unchanged, separate sessions |
| `tests/contracts.py` | **One test class per part** (`ExtractorContract`, `TrackerContract`, `RuleContract`, `ConfirmerContract`, `RankerContract`, `PolicyContract`, `FillerContract`). They check output shape only, never accuracy: valid letters and ranges, no crash on empty or emoji or very long text, deterministic where required. |
| `tests/test_contracts.py` | Runs those classes against the stand-ins |
| `tests/stub_config.json` | Pins the behaviour tests to the stand-ins, so switching `configs/system.json` to real models never breaks them |
| `tests/test_m0.py`, `test_structured.py` | The ranking milestones M0 and M1 |

To check a new part, subclass its contract:

```python
# tests/test_my_extractor.py
import unittest
from contracts import ExtractorContract
from pipeline.extractor import DebertaExtractor

class TestDeberta(ExtractorContract, unittest.TestCase):
    def make(self):
        return DebertaExtractor("models/extractor")
```

Always run with `.venv/bin/python -m unittest discover -s tests`. The contract import only works through `discover -s tests`.

**Dry run (verified):** every skeleton was filled with fake code in a scratch copy and every config line was switched to the real names. The full chat, all five actions and the five cards worked, and 92/92 tests passed. The wiring is ready; only the model code itself is missing.

---

## 9. Offline work (not part of the server)

| Job | Where | Output |
|---|---|---|
| Train the extractor | Anywhere with a GPU (e.g. Colab), on `data/extractor.jsonl` with its shipped split and the class weights in `data/labels.json` | A checkpoint folder plus the model class file → `models/extractor/` |
| Build indexes | `scripts/build_indexes.py` | `indexes/bm25.pkl`, `faiss.idx`, `faiss_owners.json` |
| Tune the NLI threshold | `data/nli_pairs.jsonl`, dev split | A number for `confirm_threshold` |
| Evaluate | `eval/harness.py` (to write): the 100 profiles in `data/profiles.jsonl` run through `System.step` with `data/eval/simulator.py`, scored with `data/eval/metrics.py` | hit@5, NDCG@10, full vs `no_probing` |

The evaluation calls `System.step` directly with no server, so it measures exactly what the chat page runs.

---

## 10. Run commands

```
.venv/bin/uvicorn app.server:app --reload           # dev: http://localhost:8000 (panel on)
CM_STUDY_MODE=1 .venv/bin/uvicorn app.server:app    # participant sessions (panel off)
.venv/bin/python chat_cli.py                        # terminal chat
.venv/bin/python -m unittest discover -s tests      # all tests
pip install -r requirements-models.txt              # once a real model is switched on
.venv/bin/python scripts/build_indexes.py           # once, after filling it in
```

---

## 11. Status at a glance

| Step | Part | Model | Status |
|---|---|---|---|
| — | Page, server, sessions, logs, conductor | — | Real |
| 1 | Extractor | DeBERTa (trained) | Stand-in → `pipeline/extractor.py` |
| 2 | Tracker | — | Stand-in → `pipeline/tracker.py` |
| 3 | Contradiction rule | — | **Real** |
| 4 | NLI check | NLI DeBERTa | Stand-in → `pipeline/nli.py` |
| 5 | Ranker | MiniLM + FAISS + BM25 | Structured signal real; BM25, dense and RRF → `recommender/` |
| 6 | Policy | — | Stand-in → `pipeline/policy.py` |
| 7 | Template filler | — | **Real** |
| — | Evaluation harness | — | To write → `eval/harness.py` |

Every stand-in's name ends in `-stub`. Once `/api/health` shows no `-stub` names, the whole system is real. **Don't report any results from stand-ins.**
