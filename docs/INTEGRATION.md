# Integration guide: replacing stand-ins with real components

Every pipeline step sits behind an interface (`pipeline/interfaces.py`) and is chosen by name in `configs/system.json`.
Swapping a step = write your file, replace its `_todo` placeholder in `pipeline/registry.py` with a lazy factory, flip one config line, run the contract test.

## 1. Where things stand

| Step | What it is | Model or plain Python | Status | Component name now | File |
|---|---|---|---|---|---|
| 1 | Extractor: text to signed RIASEC readings | Model (DeBERTa) | stand-in | `lexicon-stub` | `pipeline/stubs.py` |
| 2 | Tracker: merge readings into state/history | Plain Python (decay/blend) | stand-in | `simple-stub` | `pipeline/stubs.py` |
| 3 | Contradiction rule: opposite signs, both confidence > 0.4 | Plain arithmetic, no model | **real** | `rule` | `pipeline/stubs.py` (`ContradictionRule`) |
| 4 | NLI confirmer (`cross-encoder/nli-deberta-v3-base`) | Model | **missing, not built**: `passthrough-stub` scores every candidate 1.0, so every rule candidate becomes a CHALLENGE | `passthrough-stub` | `pipeline/stubs.py` |
| 5 | Scorer/ranker: structured + BM25 + FAISS, fused by RRF | Models + indexes | stand-in (structured scoring is real, text signals missing) | `structured+keyword-stub` | `pipeline/stubs.py`, `recommender/structured.py` |
| 6 | Policy: choose the next action | Plain Python | stand-in | `rule-stub` | `pipeline/stubs.py` |
| 7 | Template filler: question bank text | Plain Python | **real** | `question-bank` | `pipeline/templates.py` |

Caveat: the rule's 0.4 confidence gate is meaningless until the real extractor produces real confidences; the stub uses a fixed 0.7 for every reading.

`System.step` only orchestrates: extract, track, rule, confirm (keep flag if score > `confirm_threshold`, 0.60), score, policy, fill, quotes/recommendations, bookkeeping. Per-turn rules live in the components:

- Tracker records `state.resolved` when the last action was a PROBE on the slot; the rule skips resolved and challenged slots.
- Policy returns a complete filler payload and owns the rotation counters (`state.clarify_count`, `state.kind_count`).
- Extractor receives `context` (the bot's previous question, `state.last_reply`, only when the last action was CLARIFY/ABSTAIN/PROBE/CHALLENGE) and `expected_slot`.

Common gotchas for every step:

- Slots are the letters `R I A S E C`, never full type names ("Realistic").
- Source sentences must be kept verbatim (`source` in readings/history): the CHALLENGE and PROBE quote cards and the NLI input depend on them.
- Load models once, inside the factory (never per call).
- Never import torch/transformers at the top of `pipeline/registry.py` or `pipeline/stubs.py`; import inside the factory function.
- Standard library only in `pipeline/` and `app/` except in your own model module.

Registry pattern (replace the `_todo` entry; example for step 1):

```python
def _extractor_deberta(cfg, d):
    from .extractor import DebertaExtractor      # lazy: torch is imported only when chosen
    return DebertaExtractor(ROOT / cfg["paths"]["extractor_checkpoint"])
REGISTRY["extractor"]["deberta"] = _extractor_deberta
```

Contract tests: `tests/contracts.py` has one mixin per interface. Subclass it in a new `tests/test_<yourthing>.py` with your own `make()`.

## 2. Steps that still need work

### 2.1 Step 5: scorer (BM25, then MiniLM + FAISS, then RRF)

1. Create `scripts/build_indexes.py` (not written yet). It reads `data/occupations.csv` and `data/tasks.csv` and writes `indexes/bm25.pkl`, `indexes/faiss.idx` and `indexes/faiss_owners.json` (row index to occupation code), paths from `configs/system.json` `paths`.
2. Create `recommender/fusion.py` with class `RRFRanker`, `name = "rrf"`, attribute/property `eligible_count: int`, and
   ```python
   def rank(self, interests: dict, observed_types: set, user_sentences: list[str], top_k: int) -> RankingResponse
   ```
   `RankingResponse(status, occupations, reason)` and `RankedOccupation(code, title, rank, occupation_z, rrf_score, signals, evidence)` come from `recommender/schemas.py` (do not change them). Fill `signals` per signal and `rrf_score`. Return `status="no_signal"` with empty list and a reason when nothing usable. Ranks are 1..n, codes unique and eligible, deterministic for the same input.
3. Fill `evidence: list[TaskEvidence]` for every returned occupation (`task_id`, `occupation_code`, `text`, `match_score`, `status`). The server shows the first two `text` values as the card quotes.
4. In the factory, load once: `StructuredRanker`, BM25 pickle, FAISS index, owners json, the embedder (`paths.embedder`). Reuse `StructuredRanker` for the structured signal; ranking milestones M3-M4 are in `cs425-ranking-docs/`.
5. Registry: replace the `ranker` / `rrf` placeholder with a lazy factory. Config: `"ranker": "rrf"`.
6. Contract test:
   ```python
   import unittest
   from contracts import RankerContract
   from recommender.fusion import RRFRanker

   class TestRRF(RankerContract, unittest.TestCase):
       def make(self):
           return RRFRanker(...)   # load from indexes/
   ```
7. `/api/health` then shows `"ranker": "rrf"`.

### 2.2 Step 1: extractor (DeBERTa)

1. Create `pipeline/extractor.py` with class `DebertaExtractor`, `name = "deberta"`.
   ```python
   def read(self, text: str, context: str | None = None, expected_slot: str | None = None) -> Reading
   ```
   Returns `Reading(text, slots)`; each `SlotReading(slot, sign, strength, confidence, reason, source)`: slot in `RIASEC` letters, sign in {-1, 1}, strength and confidence in [0, 1], reason in `enjoyment | familiarity | neither`, source a verbatim sentence from `text`, at most one reading per slot.
2. Use `tokenizer(context, text)` when `context` is not None (it is the bot question being answered). `expected_slot` is the slot that question was about.
3. Load tokenizer and checkpoint (`paths.extractor_checkpoint`, default `models/extractor`) once in the factory. Never crash on empty, whitespace, very long, emoji or non-English text.
4. Registry: lazy factory for `extractor` / `deberta`. Config: `"extractor": "deberta"`.
5. Contract test:
   ```python
   import unittest
   from contracts import ExtractorContract
   from pipeline.extractor import DebertaExtractor

   class TestDeberta(ExtractorContract, unittest.TestCase):
       def make(self):
           return DebertaExtractor("models/extractor")
   ```
6. `/api/health` shows `"extractor": "deberta"`. After this, check that the rule's 0.4 gate and the policy's familiarity PROBE still behave with real confidences.

### 2.3 Step 4: NLI confirmer

1. Create `pipeline/nli.py` with class `NliConfirmer`, `name = "nli"`.
   ```python
   def score(self, old_src: str, new_src: str) -> float   # contradiction probability, 0..1
   ```
2. Load `paths.nli_model` (`cross-encoder/nli-deberta-v3-base`) once in the factory; return the softmax probability of the contradiction class as a plain `float`. Handle empty and very long strings.
3. The system keeps a rule candidate only if `score > settings.confirm_threshold` (0.60) and stores the score on `Flag.score`.
4. Tune/validate the threshold on the 75 `split == "dev"` pairs in `data/nli_pairs.jsonl`; report precision/recall once on the 64 `split == "test"` pairs in the same file, using `nli_binary` from `labels.json`. Do not tune on the test pairs. (`data/nli_pairs_blind.jsonl` is the 60 hand-made pairs without labels, for a teammate's blind sort, not the test set.)
5. Registry: lazy factory for `confirmer` / `nli`. Config: `"confirmer": "nli"` (and `settings.confirm_threshold` if changed).
6. Contract test:
   ```python
   import unittest
   from contracts import ConfirmerContract
   from pipeline.nli import NliConfirmer

   class TestNli(ConfirmerContract, unittest.TestCase):
       def make(self):
           return NliConfirmer("cross-encoder/nli-deberta-v3-base")
   ```
7. `/api/health` shows `"confirmer": "nli"`.

### 2.4 Step 2: tracker

1. Create `pipeline/tracker.py` with class `DecayTracker`, `name = "decay"`.
   ```python
   def update(self, state: State, reading: Reading) -> None
   ```
   Mutates `state` in place. Must: set `state.last_reading_empty = not reading.slots`; for each reading write `state.slots[slot] = SlotState(value, confidence, reason, source, turn, history)` where `history` gains a `HistoryEntry(value, confidence, reason, source, state.turn)` per reading (the rule and the quote cards read it); and if `state.last_action` is a PROBE with a slot, add that slot to `state.resolved`.
2. `value` in [-1, 1] with the sign of the latest stance, `confidence` in [0, 1]. Keep the raw `source` of the latest reading.
3. Registry: lazy factory for `tracker` / `decay`. Config: `"tracker": "decay"`.
4. Contract test:
   ```python
   import unittest
   from contracts import TrackerContract
   from pipeline.tracker import DecayTracker

   class TestDecay(TrackerContract, unittest.TestCase):
       def make(self):
           return DecayTracker()
   ```
5. `/api/health` shows `"tracker": "decay"`.

### 2.5 Step 6: policy

1. Create `pipeline/policy.py` with class `ManualPolicy`, `name = "manual"`, constructor taking `max_turns` (and the `no_probing` flag, see section 4).
   ```python
   def decide(self, state: State) -> Action
   ```
   `Action(kind, slot, why, payload)`; kind in `CLARIFY PROBE CHALLENGE ABSTAIN RECOMMEND`. The payload must already be complete for the filler: `slot`, `index` (= `state.kind_count[kind]`), `clarify_index` (= `state.clarify_count[slot]`) when there is a slot; CHALLENGE adds `old_src`, `old_turn`, `new_src`; PROBE adds `new_src`. The policy owns the counters: increment `state.kind_count[kind]` and, for CLARIFY/ABSTAIN with a slot, `state.clarify_count[slot]`, and update `state.challenged / probed / settled / strikes`. Same state in, same action out. Hard stop: RECOMMEND at `state.turn >= max_turns`. Rules: manual section 7.
2. Registry: lazy factory for `policy` / `manual`, passing `settings.max_turns`. Config: `"policy": "manual"`.
3. Contract test:
   ```python
   import unittest
   from contracts import PolicyContract
   from pipeline.policy import ManualPolicy

   class TestManual(PolicyContract, unittest.TestCase):
       def make(self):
           return ManualPolicy(10)
   ```
4. `/api/health` shows `"policy": "manual"`. Re-run `tests/test_pipeline.py` (full conversation, probe, abstain tests) as the regression check.

## 3. Spin it all up

1. `pip install -r requirements-models.txt` (only needed once a real model is selected).
2. Put the extractor checkpoint in `models/extractor/`.
3. Build the indexes: `scripts/build_indexes.py` still has to be written (step 5); it produces `indexes/bm25.pkl`, `indexes/faiss.idx`, `indexes/faiss_owners.json`.
4. Edit `configs/system.json` `components` to the real names (`deberta`, `decay`, `nli`, `rrf`, `manual`). An unknown name fails at startup and lists the valid names; a registered but unwritten one raises `NotImplementedError` pointing here. Use `CM_CONFIG=/path/to/other.json` to run another config.
5. `.venv/bin/uvicorn app.server:app --port 8000`
6. `curl localhost:8000/api/health`: no component name should end in `-stub`.
7. For human study sessions set `CM_STUDY_MODE=1` (hides the trace/profile from participants). `CM_LOG_DIR` and `CM_DATA_DIR` also exist.
8. `.venv/bin/python -m unittest discover -s tests -v` must pass.

## 4. Also to add

- Evaluation harness `eval/harness.py`: run the 100 profiles through `System.step`, compare full vs no-probing. Needs a `no_probing` setting in `configs/system.json` `settings`, read by the policy factory.
- Fill the 2 link placeholders in `static/app.js` (`#rating-form`, `#careers-service`), and decide where transcripts are stored and when they are deleted (manual section 10; consent happens outside the app since there is no consent screen).
- A per-card "why" that is specific to each occupation (currently one sentence for all five).
- Optional idle-session timeout (sessions are in memory and never expire).
