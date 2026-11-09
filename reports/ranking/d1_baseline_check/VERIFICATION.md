# Verification: D1 + signed lexical as the live default

Date: 9 November 2026. Branch `feat/ranker-v2` (uncommitted working tree). Candidate requested by the owner; this report checks it before any promotion. Plan: [docs/LEXICAL_MATCHING_PLAN.md](../../../docs/LEXICAL_MATCHING_PLAN.md).

## Candidate

| | Live default (`configs/ranking.yaml`) | Candidate |
|---|---|---|
| M1 | `cosine_similarity@baseline_v1` | unchanged |
| M3 | none | `bm25_signed@ignore_negative_v2` |
| M4 | `occupation_dual_encoder@finetuned_v1` | `occupation_dual_encoder_signed@finetuned_v1` (same checkpoint and index; negative and skipped clauses deleted from the query) |
| Fusion | `zscore_sum@equal_v1` | unchanged |

Preset: `configs/ranking/occupation_semantic_signed_fusion_zscore_bm25_signed_ignore_v2.json`; website config `configs/system.semantic-signed-bm25-signed-ignore-v2.json`.

## Checks

| Check | Result | Verdict |
|---|---|---|
| Personas, probing on (`../dislike_d1/run_20261109_personas/`) | MRR .770 vs .680: all +.090 [−.037, +.226], conflict +.141 [−.019, +.311], agree +.025 [−.190, +.244]; Hit@5 .96 vs .80 | pass (no agree loss) |
| Personas, probing off (`personas_no_probing/`) | MRR .777 vs .653: all +.123 [−.003, +.258], conflict +.149 [−.014, +.323], agree +.091 [−.104, +.307]; Hit@5 1.00 vs .80 | pass, consistent |
| Negation probes, draft (`negation_probes/`) | avoided in top 10 2.79 → 0.67; any avoided 79% → 29%; acceptable MRR .317 → .614; control and collateral probes unchanged | pass, but narrow kind worse (below) |
| Development profiles, 120, opening turn (`dev_extractor_inputs/`; control reproduces .064) | MRR −.007 [−.031, +.020] (76 profiles worse, 43 better); Hit@5 −.033 [−.083, +.008]; **Recall@50 −.036 [−.067, −.008]**. D1 alone: MRR +.008 [−.004, +.024], no metric worse | small regression from the lexical part |
| Held-out task retrieval | not rerun. D1 changes 5 of 1,133 test sentences (0.4%, all fully deleted; bound ≈ .004 MRR). BM25 is leaked on this benchmark (task text is in `bm25_doc`), so the fused candidate cannot be judged here | not applicable |
| Ranking latency, 200 persona turns | median 28.8 ms vs 27.4 ms; p95 40.3 vs 43.0 ms | pass |
| Full test suite | 485 tests, 23 failures/errors, same count as the 23 pre-existing (extractor checkpoint, NLI preset, stale M4 cache, history fingerprints, terminal signature). None in changed code; `pipeline/` untouched | pass |
| Live server, HTTP session | server boots with the candidate config; resolved `cosine_similarity@baseline_v1`, `bm25_signed@ignore_negative_v2`, `occupation_dual_encoder_signed@finetuned_v1`, `zscore_sum@equal_v1`; diagnostics list the removed clause "teaching, not as much" | pass |

## Costs and limitations to accept

1. **Development-profile regression** from adding lexical: Recall@50 −.036 (significant), MRR −.007 (not). Labels follow the RIASEC profile, so this benchmark favours cosine; compare the −.032 MRR accepted for z-score fusion.
2. **Routing bug, "X, just not Y"**: one clause, routed negative, deleted with its positive half. "I love biology, just not lab work" loses "biology"; narrow-kind probe MRR .333 → .189 (N02, N10, N20). For this phrasing the candidate is worse than the live default.
3. **Routing bug, "anything but X"**: split on "but", so "X" stays in the query (no worse than today).
4. **Descriptive "avoid"/"no"**: "I want to help people avoid debt" is routed negative and deleted (seen in 5 O*NET task sentences).
5. **Dislikes are still not enforced in the final list** (D2 pending): Psychology Teachers stays #1 for "i study psychology … but teaching, not as much".
6. **Evidence caveats:** probe set is a draft pending team review; persona openings name the field outright; no browser review of this exact candidate yet.
7. **Degraded mode changes:** without the local M4 files the site falls back to cosine + signed lexical (previously cosine only).

## Rollback

Restore the previous `components` in `configs/ranking.yaml` (identical to `configs/ranking/occupation_semantic_fusion_zscore_v1.json`), or run the old default without editing via a system config pointing at that preset.

## Decision (owner, 9 November 2026)

**Not promoted yet.** The live default stays unchanged. The candidate stays selectable for browser review via `configs/system.semantic-signed-bm25-signed-ignore-v2.json`.
