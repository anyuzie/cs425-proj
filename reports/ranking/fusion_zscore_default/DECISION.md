# Decision: z-score fusion as the live default

Date: 9 November 2026. Branch `feat/ranker-v2`. Decision by the project owner (option B in [fusion_variants_v1/DECISION.md](../fusion_variants_v1/DECISION.md)).

## Change

| | Before | After |
|---|---|---|
| `configs/ranking.yaml` `fusion` | `{algorithm: rrf, k: 60}` (`rrf_k60`) | `{algorithm: zscore_sum, variant: equal_v1}` (`zscore_sum@equal_v1`) |
| Components | M1 `cosine_similarity@baseline_v1`, M4 `occupation_dual_encoder@finetuned_v1` | unchanged |

- **Where it applies:** everywhere that reads `configs/ranking.yaml`: the website (`configs/system.json`), the terminals and the zero-argument evaluator.
- **Unchanged selections:** the workflow (`configs/ranking/active.json`, `candidate.json`, fusion null), the programmatic `BASELINE_CONFIG`, the mock mode and every saved preset.
- **Single usable component:** when M1 or M4 is unusable (e.g. no RIASEC type observed yet, or the M4 files are missing), fusion is still bypassed and the remaining ranking is used as before.
- **Response schema:** `RankedOccupation.rrf_score` now holds the z-score sum, keeping its field name for compatibility. The diagnostics `composition` and the run manifests' `components.fusion` read `zscore_sum@equal_v1`.

## Evidence (from the variant report; nothing new was run for this decision)

| MRR | RRF k=60 | z-score | difference [95% CI] |
|---|---|---|---|
| Conflict personas (14), probing on | .114 | .639 | +.525 [+.328, +.722] |
| Agree personas (11), probing on | .743 | .733 | −.010 [−.192, +.136] |
| Held-out task retrieval (1,133) | .200 | .507 | +.307 [+.284, +.331] |
| Development profiles (120), opening turn | .096 | .064 | **−.032 [−.059, −.011]** |

**Accepted costs:**
- The opening-turn development regression.
- The persona caveat: conflict openings name the field outright, so the gain on real conflict cases is probably smaller.

## Rollback and comparison

- **Roll back:** restore `fusion: {algorithm: rrf, k: 60}` in `configs/ranking.yaml`. This gives exactly `configs/ranking/occupation_semantic_fusion_v1.json`.
- **Compare without editing:** run the website with the previous behaviour via `$env:CM_CONFIG = 'configs/system.rrf.json'` (identical to `configs/system.json` except the ranking selection).

## Verification

- `tests/test_configuration.py` pins the YAML to the z-score preset and keeps RRF and `system.rrf.json` selectable.
- `tests/test_ranking_app_integration.py` checks that the web app's ranking equals the z-score preset's.
- The full suite's pre-existing failures are unchanged; see the commit or PR for the run.

## Still to do

- A browser check of the live site.
- A qualitative replay of real sessions (e.g. `logs/anon-c509a1d7.jsonl`) under both fusions.
- Rerun the persona evaluation once the extractor is calibrated (#2).
