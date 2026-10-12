# Experiment Record — COPY THIS FILE

> Save one copy per experiment under `reports/ranking/experiments/<run_id>/DECISION.md`. Complete **before** implementation where possible, and fill outcomes after running. Do not overwrite earlier experiment records.

## 1. Identity

- Run ID: `TBD`
- Milestone: `M?`
- Date / owner: `TBD`
- Status: `PLANNED / RUNNING / DONE`
- Control run ID: `TBD`
- Treatment run ID: `TBD`

## 2. Hypothesis and controlled change

- **Hypothesis:** If we change `__________`, then `__________` should improve because `__________`.
- **Exactly one material changed variable:** `TBD`
- **What remains identical:** eligible pool, evaluated snapshots, data split, input mode, structured state, tokenizer, indices, fusion parameters, metrics (note any legitimate exceptions).
- **Expected failure mode / risk:** `TBD`
- **Decision to be made:** `KEEP / REVISE / REVERT`

## 3. Reproducibility manifest

- Git commit: `TBD`
- Development / held-out split: `TBD`
- Dataset fingerprint: `TBD`
- Evaluated input snapshot fingerprint: `TBD`
- Candidate rule: `TBD`
- Input mode: `oracle / mock_tracker / actual_tracker`
- Config path + hash: `TBD`
- Embedding model and revision: `TBD / not applicable`
- FAISS index fingerprint: `TBD / not applicable`
- Evaluator version: `TBD`
- Python and package versions: `TBD`
- Random seed(s): `TBD`
- Exact test command: `TBD`
- Exact evaluation command: `TBD`

## 4. Correctness gate

- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Outputs in eligible pool, no duplicates, no non-finite scores
- [ ] Control and treatment use equivalent inputs
- [ ] Target/relevance labels never passed to ranker

**Known failures:** `None / describe`.

## 5. Results

| Metric | Control | Treatment | Paired delta | 95% CI if available |
|---|---:|---:|---:|---|
| Hit@5 | TBD | TBD | TBD | TBD |
| MRR | TBD | TBD | TBD | TBD |
| NDCG@10 | TBD | TBD | TBD | TBD |
| p95 latency (ms) | TBD | TBD | TBD | — |

- Cases evaluated / missing / ineligible: `TBD`
- Cases actually affected by treatment: `TBD`
- Recovery / stale-evidence / genuine-familiarity diagnostics (if applicable): `TBD`
- Prediction file: `TBD`
- Paired difference file: `TBD`

## 6. Failure review

| Profile/snapshot ID | Target | Rank control → treatment | Helped/harmed? | Mechanism / suspected issue |
|---|---|---|---|---|
| TBD | TBD | TBD | TBD | TBD |

Record at least five relevant cases where available. Include harmed examples, not just successes.

## 7. Decision and next action

- **Decision:** `KEEP / REVISE / REVERT / STOP`
- **Why:** `TBD`
- **Limitations / uncertainty:** `TBD`
- **Follow-up experiment (if any):** `TBD`
- **New frozen baseline ID (if kept):** `TBD`
