# Decision: RIASEC cosine + fine-tuned semantic fusion becomes the live ranker (9 November 2026)

**Change.** `configs/ranking.yaml` now selects M1 `cosine_similarity@baseline_v1` **and** M4 `occupation_dual_encoder@finetuned_v1`, combined by equal-weight occupation-level RRF (k=60); M3 null. This is identical to `configs/ranking/occupation_semantic_fusion_v1.json`. The website (`app/server.py` without `CM_CONFIG`), `app/terminal.py`, `app/ranking_terminal.py` and the zero-argument evaluator all read it through `configs/system.json`. It is the new baseline against which later ranking increments are compared.

How the two run in parallel on every turn:

```
user text ──> DeBERTa ──> RIASEC tracker ──> M1 cosine ranking ───┐
          └─────────────> fine-tuned dual encoder ──> M4 ranking ─┴─> RRF k=60 ──> recommendations
```

- RRF fuses **ranks**, `1/(60+rank_cosine) + 1/(60+rank_semantic)`; raw cosine and embedding scores are never added.
- If only one component is usable, RRF is bypassed and that ranking is used (`composition: single_usable_component`). With no observed RIASEC type (e.g. "I am not sure what I want.") the semantic ranking alone is returned instead of `no_signal`.
- RIASEC probing answers change the fused ranking as types are observed.

This is an **explicit owner decision**, not an automated promotion. Earlier the same day the default was briefly semantic-only; the owner replaced it with fusion so RIASEC and semantic matching both contribute.

**Unchanged.**
- `cosine_similarity@baseline_v1` alone stays selectable: `configs/system.cosine.json` / `configs/ranking/m1_cosine.json`. Semantic-only: `configs/system.occupation-semantic-finetuned-v1.json`.
- The frozen dot-product control stays the programmatic `BASELINE_CONFIG` (`configs/ranking/baseline_v1.json`).
- The experiment workflow (`configs/ranking/active.json`, `candidate.json`, promotion policy) keeps its cosine selection; it no longer mirrors the app YAML.
- `configs/system.mock.json` (app without model weights) pins `configs/ranking/m1_cosine.json`, because M4 needs the encoder.
- Extractor, tracker, NLI, probing policy and question generation are unchanged.

## Evidence

| Benchmark | Fusion | Cosine | Semantic only | Source |
|---|---|---|---|---|
| 120 development profiles, real extractor, opening turn (MRR) | .096 | .117 | .040 | `reports/ranking/occupation_semantic_v1/dev_extractor_inputs/` |
| Same, Hit@5 | .142 | .175 | .042 | same |
| Held-out task retrieval, 1,133 test groups (MRR) | .201 | .032 | .518 | `reports/ranking/occupation_semantic_v1/evaluation.json` |
| 38 held-out occupations (MRR) | .174 | .055 | .418 | same |

Paired bootstrap (95% CI): on development profiles fusion − cosine MRR −.021 [−.059, +.014], Hit@5 −.033 [−.108, +.042], NDCG@10 −.008 [−.040, +.021], Recall@10 +.001 [−.020, +.021] — **no significant difference**; fusion − semantic-only +.056 [+.023, +.092]. On task retrieval fusion − cosine +.169 [+.155, +.182].

The benchmarks favour their own construction: development-profile targets come from hidden RIASEC profiles; task-retrieval labels come from O*NET task ownership. Neither measures conversational career relevance. Both evaluate a single turn, where RIASEC is weakest; multi-turn simulated dialogues (`eval/dialogue*.py`) were not rerun. Equal RRF weighting is the existing frozen fusion, not tuned; a weighted variant would need a new version and data not already inspected.

## Verification

- Full suite: 476 tests, the same 5 pre-existing errors as before any default change (stale old-M4 MiniLM cache ×2, history fixture fingerprint ×2, `test_occupation_classifier` keyword argument); no new failures.
- Tests asserting the old default were updated: `test_configuration` (fusion default plus cosine still selectable; history row identifiers), `test_history` (M1/M4/fusion identifiers), `test_ranking_app_integration` default-startup test, `test_modular_ranking` (workflow compares with `m1_cosine.json`). `tests/m5_fixture.py` pins `m1_cosine.json`: its M5 integration tests check RIASEC-conflict behaviour of the structured ranker.
- `recommender/experiments/evaluate.py` provenance records the dual encoder's checkpoint/split hashes (it previously assumed the legacy MiniLM identity and raised `KeyError`).
- Live, no flags/no `CM_CONFIG`:
  - `python -m app.ranking_terminal --text "I am an information systems student specialised in cloud computing with a second major in AI" --json` → M1 and M4 `usable`, `rrf_k60`; top 5 Computer and Information Systems Managers (cosine rank 3, semantic 1), Computer and Information Research Scientists, Computer Systems Analysts, Education Administrators, Postsecondary (cosine 7, semantic 37), Information Security Analysts.
  - "I am not sure what I want." → M1 `unusable`, semantic ranking used alone.
  - 4-turn website conversation: observed RIASEC grew I → I,S → C,I,S and the fused top 3 changed accordingly; M4 `usable` on every turn.

## Consequences and caveats

- **Artifacts are local.** `artifacts/ranking/occupation_semantic_v1/` (checkpoint + index) is gitignored. Without it M4 is `unavailable`; fusion then silently degrades to cosine only (diagnostics show the M4 status/reason). Share the folder (e.g. via `/share/`) or rebuild with the commands in the [experiment report](../occupation_semantic_v1/README.md).
- **Negation is not handled** by M4: "I do not want to teach mathematics" boosts mathematics teachers in the semantic half. The preference extractor records it as an avoid task, but nothing scores task records yet (planned next increment).
- With one observed RIASEC type, cosine ties hundreds of occupations in code order; through RRF those ties can still pull in weakly related occupations (e.g. Education Administrators above).
- Vague or nonsense text yields a full semantic ranking when RIASEC is empty.
- The final reply wording ("based on the preferences you described") is unchanged.

## Reversal

Restore the cosine selection in `configs/ranking.yaml` (identical to `configs/ranking/m1_cosine.json`), revert the test expectations listed above, and optionally point `tests/m5_fixture.py`/`configs/system.mock.json` back at `configs/ranking.yaml`.
