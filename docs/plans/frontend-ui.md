# Plan: chat UI + FastAPI backend (branch `feat/frontend-ui`)

Design: https://claude.ai/artifact/DbSkEk6Zext8JtgmBZtZTr (screens 2–4; no welcome/consent screen).
Manual: section 11 (interface), section 3 (turn flow), section 7 (policy).

## Scope

- One FastAPI process serving a JSON API and a static page.
- The page opens straight into the conversation.
- The real `StructuredRanker` produces the ranking.
- The extractor, tracker, contradiction check and policy are **stand-ins** behind interfaces, so teammates' real components drop in without changing the server or page.
- No BM25/FAISS/NLI in this branch (ranking milestones M3–M4 and Koon's NLI are separate work).

## Decisions taken in review

1. **No consent screen.** The participant code comes from the URL (`/?p=P-014`); when missing, the server generates `anon-<8 hex>`. A persistent one-line banner in the chat says "Research prototype — not career advice" (manual section 10 requires this before any output).
2. **API under `/api/`** so the static mount at `/` cannot shadow it.
3. **Participant code is validated** as `^[A-Za-z0-9_-]{1,32}$` because it becomes a log filename (path traversal otherwise).
4. **Stand-ins are labelled.** Every trace and `/api/health` reports component names (`lexicon-stub`, `rule-stub`, …) so nobody mistakes stub output for results.
5. **Dependencies pinned** in `requirements.txt` (fastapi, uvicorn, httpx for TestClient). Pipeline code stays standard library.
6. **Tests use `unittest`**, matching the repo.

## API contract (both sides build against this)

```
GET  /api/health
  → {"status":"ok","eligible":378,"components":{"extractor":"lexicon-stub","tracker":"simple-stub",
     "checker":"rule-stub","policy":"rule-stub","evidence":"keyword-stub","ranker":"structured"}}

POST /api/session   {"participant_code": str | null, "debug": bool = false}
  → {"session_id": str, "participant_code": str, "reply": str, "action": "OPENING",
     "turn": 0, "max_turns": 10, "done": false, "recommendations": null, "trace": Trace | null}

POST /api/chat      {"session_id": str, "text": str (1..2000 chars after strip)}
  → {"session_id", "participant_code", "reply", "action": "CLARIFY"|"PROBE"|"CHALLENGE"|"ABSTAIN"|"RECOMMEND",
     "turn": int, "max_turns": 10, "done": bool, "recommendations": [Rec] | null, "trace": Trace | null,
     "quotes": {"earlier": str, "earlier_turn": int, "now": str} | {"now": str} | null}

Rec   = {"rank": 1..5, "code": str, "title": str, "why": str, "tasks": [str, ...]}   (0–2 tasks)
Trace = {"profile": {"R": {"value": float|null, "confidence": float, "reason": str|null}, ... six letters},
         "actions": [{"turn": int, "action": str, "why": str}],
         "top5": [{"code": str, "title": str}],
         "components": {...same as health...}}

Errors: 404 unknown session · 409 chat after done · 422 bad input (FastAPI default body).
`quotes` is set for CHALLENGE (earlier + now) and PROBE (now) so the page can render the quote cards.
`trace` is non-null only when the session was created with debug=true.
```

## Layout

```
pipeline/interfaces.py   Protocols: Extractor, Tracker, ContradictionChecker, Policy, EvidenceProvider; dataclasses SlotReading, Reading, State, Action
pipeline/stubs.py        LexiconExtractor (port of data/eval/lexicon_baseline.py, no sklearn), SimpleTracker,
                         RuleChecker, RulePolicy, KeywordEvidence (top-2 task statements by word overlap)
pipeline/templates.py    fill(action, payload, question_bank) — question_bank.json only, nothing generated
pipeline/system.py       System.load(data_dir), System.start(state), System.step(state, text) -> TurnResult, top5
app/server.py            FastAPI app, lifespan loads System once, sessions dict + lock, routes, StaticFiles at "/"
app/schemas.py           pydantic request/response models matching the contract
app/turn_log.py          append one JSON line per turn to logs/<participant_code>.jsonl
chat_cli.py              terminal loop over the same System
static/index.html, static/app.js, static/styles.css
tests/test_pipeline.py, tests/test_server.py
requirements.txt
```

## Stand-in policy (rungs from manual section 7, first match wins)

1. CHALLENGE: the checker flagged a slot (same slot, opposite signs, both confidence > 0.4) not yet challenged.
2. PROBE: a slot whose latest reading has reason `familiarity` (cue phrases: "that's what I know", "my dad/mum/parents", "always been around", "did the … track", "internship", "grew up") and that has not been probed.
3. ABSTAIN: last action was a question and the reply had no stance on any type; two strikes on the same slot mark it settled.
4. RECOMMEND: turn ≥ 5 and every slot filled or settled, or turn == 10.
5. CLARIFY: the unfilled slot in fixed order C, E, A, R, S, I that is not settled; rotate the 3 phrasings.

## Done when

- `python -m unittest discover -s tests -v` passes (existing 28 tests plus new ones).
- `uvicorn app.server:app` serves the page; a full conversation reaches RECOMMEND and shows five cards.
- `/?p=P-014&debug=1` shows the researcher panel; the transcript lands in `logs/P-014.jsonl`.
- The page works at 390 px wide.
