# Decision: signed lexical + signed semantic + D2 title v3 as the live default

Date: 9 November 2026. Branch `feat/ranker-v2` (uncommitted working tree). Decision by the project owner. Design and history: [docs/LEXICAL_MATCHING_PLAN.md](../../../docs/LEXICAL_MATCHING_PLAN.md).

## Change

| | Before | After |
|---|---|---|
| M1 | `cosine_similarity@baseline_v1` | unchanged |
| M3 (lexical) | none | `bm25_signed@ignore_negative_v3` |
| M4 (semantic) | `occupation_dual_encoder@finetuned_v1` | `occupation_dual_encoder_signed@finetuned_v2` (same checkpoint and index; negative and "don't mind" clauses deleted from the query) |
| Fusion | `zscore_sum@equal_v1` | unchanged |
| Post-fusion | none | `dislike_demotion@title_v3` (strong dislikes move title-matched jobs below all others; mild dislikes lower their fused score by 1.5 SD) |

- **Where it applies:** everything that reads `configs/ranking.yaml`: the website (`configs/system.json`), the terminals and the zero-argument evaluator. `configs/ranking.yaml` equals `configs/ranking/occupation_semantic_signed_v2_fusion_zscore_bm25_signed_ignore_v3_dislike_title_v3.json`.
- **Unchanged:** the workflow selection (`configs/ranking/active.json`, `candidate.json`), the programmatic `BASELINE_CONFIG`, mock mode, and every saved preset and variant.
- **New config key:** `post_fusion` (optional; `RankingPipeline` rejects it together with description reranking). The history CSV has no column for it; run manifests and `resolved_ranking.json` record it.
- **Degraded mode:** without the local M4 files, ranking falls back to cosine + signed lexical + the dislike stage (previously cosine only).

## Evidence

Full tables: [lexical_ablation_d2/README.md](../lexical_ablation_d2/README.md). Live default = before.

| Benchmark | Metric | Before | After | After − before [95% CI] |
|---|---|---|---|---|
| Personas (25), probing on | MRR | .680 | .770 | +.090 [−.037, +.226] |
| | Hit@5 / NDCG@10 | .80 / .539 | .96 / .620 | |
| | MRR conflict / agree | .639 / .733 | .780 / .758 | conflict +.141 [−.019, +.311]; agree +.025 [−.190, +.244] |
| Personas, probing off | MRR | .653 | .777 | +.123 [−.003, +.258] |
| Negation probes, owner-reviewed (24) | avoided jobs in top 10 | 2.79 | 0.08 | test split 1.67 → 0.00 |
| | acceptable MRR / Hit@5 | .317 / .50 | .676 / .91 | |
| Development profiles (120), opening turn | MRR | .064 | .057 | −.007 [−.031, +.019] |
| | Hit@5 / NDCG@10 | .075 / .094 | .042 / .077 | Hit@5 −.033 [−.083, +.008] |
| | Recall@50 | .391 | .355 | **−.036 [−.067, −.008]** |
| Ranking latency, 200 persona turns | median / p95 | 19.4 / 29.3 ms | 20.8 / 31.2 ms | |

## Accepted costs

1. **Development-profile regression from lexical matching:** Recall@50 −.036 (significant); against the same pipeline without lexical: Hit@5 −.050 [−.092, −.017], NDCG@10 −.025 [−.045, −.006], Recall@50 −.047 [−.069, −.026]. These labels follow the RIASEC profile, so they favour cosine.
2. **Lexical's persona gain is probably an upper bound:** conflict persona openings name the field outright, which exact title matching rewards; agree personas trend down with lexical (−.116 [−.318, +.086] probing on).
3. **Mild dislikes keep a large lead:** "teaching, not as much" lowers Psychology Teachers by 1.5 SD; early in a conversation its lead is larger, so it can stay #1 (seen in a two-turn live check; #2 after six turns). Strong wording removes it.
4. **D2 title matching only acts on dislikes that name a job word;** "I'd rather not work with patients" relies on the semantic and lexical queries instead.
5. **Evidence caveats:** probe set reviewed by one person (the owner); no browser review of this exact configuration by others; tracker input-quality issues (#2) still affect the cosine half.

## Rollback and comparison

- **Run the previous default without editing:** `$env:CM_CONFIG = 'configs/system.zscore.json'`.
- **Roll back the default:** replace `configs/ranking.yaml` with the content of `configs/ranking/occupation_semantic_fusion_zscore_v1.json` (M3 null, M4 `occupation_dual_encoder@finetuned_v1`, no `post_fusion`).
- Lexical-free variant of the new default: `configs/system.signed-v3-dislike-title-v3-nolex.json`.

## Verification

- `tests/test_configuration.py` pins `configs/ranking.yaml` to the new preset and keeps the previous z-score default (`configs/system.zscore.json`), RRF and cosine selectable.
- `tests/test_ranking_app_integration.py` checks that the web app's ranking equals the new preset's.
- Live server (default config, no `CM_CONFIG`): resolves `cosine_similarity@baseline_v1`, `bm25_signed@ignore_negative_v3`, `occupation_dual_encoder_signed@finetuned_v2`, `zscore_sum@equal_v1`, `dislike_demotion@title_v3`; diagnostics list the dislike target and strength; `rank_snapshot` present.
- `tests/test_history.py` pins the default's history row (`bm25_signed@ignore_negative_v3`, `occupation_dual_encoder_signed@finetuned_v2`).
- Full suite: 507 tests; 23 failures/errors, all the pre-existing ones (extractor checkpoint 17, history fingerprints 2, stale legacy M4 cache 2, NLI preset 1, terminal signature 1). The one new failure found during promotion was `test_history` still pinning the old component names; updated.
