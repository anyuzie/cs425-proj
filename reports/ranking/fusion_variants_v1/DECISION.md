# Fusion variants v1 (#1 in docs/SEMANTIC_LEDGER_PLAN.md §12.3): decision report

Date: 9 November 2026. Branch `feat/ranker-v2`, base commit `9fd317b` plus the uncommitted changes listed under Provenance.

**Outcome:** two new fusion variants are implemented and evaluated. `zscore_sum@equal_v1` meets the conflict-persona goal in part and keeps agree-persona quality. It does worse than the control on opening-turn development profiles. The default was unchanged by this report; the owner then promoted z-score (option B) the same day, see [fusion_zscore_default/DECISION.md](../fusion_zscore_default/DECISION.md).

## What was built

Only the fusion component changed. There was no retraining and no index rebuild, and no component or accepted variant changed.

| Selection | Identifier | Rule |
|---|---|---|
| `{"algorithm": "rrf", "k": 60}` | `rrf_k60` | Control and live default, unchanged. Equal-weight `Σ 1/(60 + rank)`. |
| `{"algorithm": "rrf_tie_aware", "variant": "k60_v1"}` | `rrf_tie_aware@k60_v1` | Same as the control, but occupations with exactly equal component scores share the average of the ranks they span (#1a). |
| `{"algorithm": "zscore_sum", "variant": "equal_v1"}` | `zscore_sum@equal_v1` | Per query, z-score each usable component's raw scores (population SD) over the occupations it ranked, then sum with equal weights. An occupation a component did not rank gets that component's minimum z. A component with all-equal scores contributes 0 (#1b). |

- **Code:** `recommender/models/fusion.py` (`FUSION_VARIANTS`, `build_fusion`, `fusion_identifier`). Unknown fusion settings fail with an error.
- **Pipeline labels:** `RankingPipeline.fusion_identifier` and the `occupation_semantic` diagnostics `composition` now name the fusion actually used. For non-RRF variants, `RankedOccupation.rrf_score` holds the fused score, so the response schema doesn't change.
- **Configs:** `configs/ranking/occupation_semantic_fusion_tieavg_v1.json`, `configs/ranking/occupation_semantic_fusion_zscore_v1.json`.
- **Not tuned:** both variants were fixed from first principles before any run (equal weights, no thresholds). Variant (c) was dropped, see §12.5.

## Evidence

### 1. Persona dialogues (decision benchmark; never tuned on)

`python -m eval.dialogue_fusion` → `reports/ranking/dialogue_fusion_v2/run_20261108T184532Z/`. Same protocol, personas and metric as §12.4–12.5, 10 arms. The control reproduces the v1 run exactly (MRR .391 / .114 / .743).

| Ranker (probing on) | All MRR | Conflict MRR | Agree MRR | Conflict Hit@5 | Agree Hit@5 |
|---|---|---|---|---|---|
| fusion, `rrf_k60` (control) | .391 | .114 | .743 | .21 | .91 |
| `rrf_tie_aware@k60_v1` | .391 | .114 | .743 | .21 | .91 |
| `zscore_sum@equal_v1` | **.680** | **.639** | .733 | .71 | .91 |
| semantic only | .758 | .881 | .603 | 1.00 | .73 |
| cosine only | .198 | .007 | .440 | .00 | .55 |

Paired differences in MRR with 95% CIs (10,000 persona resamples, seed 425):

| Contrast | All | Conflict (14) | Agree (11) |
|---|---|---|---|
| z-score − control, probing on | +.289 [+.121, +.463] | **+.525 [+.328, +.722]** | −.010 [−.192, +.136] |
| z-score − control, probing off | +.283 [+.101, +.464] | +.556 [+.368, +.744] | −.066 [−.255, +.109] |
| z-score − semantic, probing on | −.078 [−.218, +.047] | −.242 [−.426, −.084] | **+.130 [+.030, +.249]** |
| tie-aware − control, both | +.000 | +.000 | +.000 |

Best acceptable rank per persona (probing on), control → z-score (semantic): conflict F01 3→1 (1), F02 24→1 (1), F03 3→1, F04 11→1, F05 19→1, F06 3→1, F07 15→5 (2), F08 20→7 (2), F09 19→1, F10 26→1, F11 37→9 (1), F12 13→3 (1), F13 29→13 (3), F14 15→12 (1). Agree F15–F25: unchanged at 1 except F17 2→1, F20 3→2, F21 geology 1→4, F25 sales 7→9. Accounting, the case semantic alone fails (rank 33), stays at 5.

Per-turn diagnostic (best acceptable rank under each fusion for the same inputs, fusion-probing dialogues; mean reciprocal rank): with 2–3 types observed, control .242, z-score .727, semantic .862; with 4–6 types, .389, .734, .809.

**Tie-aware has no effect on the personas.** By their first ranked turn, every persona except one has 2 or more RIASEC types observed (1 of 199 per-turn entries has a single type), so cosine ties are rare. This benchmark cannot test it.

### 2. Held-out task retrieval (single sentence; formal proxy, not career relevance)

`python -m eval.task_retrieval_fusion` → `reports/ranking/fusion_variants_v1/task_retrieval/run_20261108T184604Z/`. Same protocol as `reports/ranking/occupation_semantic_v1/evaluation.json`. Semantic only reproduces the saved MRR exactly. The control gives .2000 against the saved .2010: only arms that read RIASEC through the extractor can differ, and the local extractor checkpoint changed after that run (§12.5, Checkpoints).

| MRR | control | tie-aware | z-score | semantic |
|---|---|---|---|---|
| test (1,133 groups) | .200 | .402 | .507 | .518 |
| occupation held-out (903) | .174 | .332 | .417 | .418 |

- **z-score:** − control +.307 [+.284, +.331] (test) and +.243 [+.218, +.269] (held-out); − semantic −.010 [−.020, −.002] and −.000 [−.008, +.007].
- **Tie-aware:** − control +.202 [+.182, +.221] and +.157 [+.135, +.180]. A single sentence usually yields one observed type, so cosine ties are common here, and this is where tie-aware helps.

### 3. Development profiles, opening turn, real extractor (labels follow the RIASEC profile)

`python -m recommender.evaluate` with `reports/ranking/occupation_semantic_v1/dev_extractor_inputs/fusion/evaluation_config.json` → `reports/ranking/fusion_variants_v1/dev_extractor_inputs/`. The control reproduces the saved run exactly (MRR .096). Paired profile bootstrap (2,000).

| MRR / Hit@5 | control | tie-aware | z-score | semantic | cosine |
|---|---|---|---|---|---|
| 120 profiles | .096 / .142 | .097 / .142 | .064 / .075 | .040 / .042 | .117 / .175 |

- **Tie-aware − control:** +.001 [−.003, +.005].
- **z-score − control:** **−.032 [−.059, −.011]**, Hit@5 −.067 [−.133, .000].
- **z-score − semantic:** +.024 [+.004, +.043].

## Interpretation

1. **Score-aware fusion fixes most of the §12.1 failure.** On conflict personas, z-score moves the first acceptable occupation from a median rank of about 17 to rank 1 in 8 of 14. On agree personas it matches the control (−.010, CI spans 0) and beats semantic only (+.130). This is the combination §12.5 asked for. It does not fully reach semantic's conflict results (.639 vs .881); the remaining gap is F07, F08, F11, F13 and F14, where cosine strongly favours a different field.
2. **The cost is on opening-turn development profiles (−.032).** There the labels follow RIASEC and M4 is weak (.040), so giving M4 more weight hurts. It is the same mechanism working in the other direction: z-score trusts the more peaked component, and on short single openings M4 can be peaked and wrong. These profiles are single-turn; the persona and task results are multi-turn and topic-driven respectively.
3. **Tie-aware is safe and only helps where ties exist.** It gives a large gain on single-sentence task retrieval, no change on the personas or development profiles, and never does worse. It does not address the conflict failure.

## Decision options (owner)

- **A. Keep `rrf_k60` as the default.** No user-visible change; the conflict failure remains (MRR .114).
- **B. Promote `zscore_sum@equal_v1`** (recommended for consideration). This means a large gain where profile and topic disagree, unchanged agree quality, and task retrieval close to semantic only. The trade-off is the −.032 opening-turn regression, and the persona openings name the field outright, so the .639 is likely an upper bound for real conflict cases (§12.5 caveat). To promote, set `fusion: {algorithm: zscore_sum, variant: equal_v1}` in `configs/ranking.yaml`. Roll back by restoring `rrf` k=60.
- **C. Promote `rrf_tie_aware@k60_v1` only.** It never does worse, but it doesn't fix the conflict failure.

Combining (a) and (b) is not needed: with z-score, a one-type cosine ranking becomes a two-level step function whose z is bounded, so ties no longer decide the order.

## Not done / caveats

- No fusion weights or thresholds were tuned; equal weights are a first-principles choice. Any tuned variant must use data other than the 25 personas.
- The extractor's calibration metadata is still unfinished (§12.5), so tracker confidences are uncalibrated.
- Only 11 agree personas, so their CIs are wide.
- Browser check of the app with z-score fusion: not done. After promotion, `configs/system.json` runs z-score; `configs/system.rrf.json` runs the previous RRF default for comparison.

## Provenance

- **Code:** `recommender/models/fusion.py`, `recommender/models/ranker.py`, `eval/dialogue_fusion.py` (new arms, `--rankers`/`--probing`, output `dialogue_fusion_v2`), new `eval/task_retrieval_fusion.py`, tests in `tests/test_fusion.py` and `tests/test_dialogue_fusion.py`.
- **Source hashes:** each run's manifest/evaluation file records them. The persona run was made with the working tree dirty (uncommitted variants), and its manifest records source hashes for `fusion.py` and `ranker.py`.
- **Tests:** `python -m unittest discover -s tests`: 457 run, the 8 new tests pass. The same 23 pre-existing failures occur on clean `9fd317b` (HuggingFace offline, `dev_profiles.jsonl` fingerprint, stale MiniLM cache, NLI preset version, terminal display signature).
