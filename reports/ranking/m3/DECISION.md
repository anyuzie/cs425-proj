# M3 — BM25 and two-signal fusion

**Decision: M3 baseline implemented and evaluated; retain it as an experimental control. Keep structured-only as the default because this hybrid regresses. Stop before M4.**

The applicable acceptance checklist is [DEVELOPMENT.md, M3](../../../cs425-ranking-docs/DEVELOPMENT.md#m3--bm25-and-two-signal-fusion). The [pre-evaluation protocol](EXPERIMENT.md) states the frozen hypothesis/configuration; [failure review](failure_review.md) records dislike, misleading-familiarity, genuine-familiarity and gain cases.

## Implementation

`recommender/bm25_ranker.py` provides independent `bm25@baseline_v1`. It indexes only eligible occupations' supplied `bm25_doc`, using an in-memory inverted index. `Occupation` carries optional prepared text; M1 still works on fixtures without text, while enabling BM25 requires a nonempty, tokenizable document for every eligible occupation. No dataset values, candidate eligibility, M1 formula, or M2 metric definitions changed.

BM25 parameters are frozen at `k1=1.5`, `b=0.75`, with positive smoothed IDF `log(1+(N-df+0.5)/(df+0.5))`. Both documents and queries use lowercase `[a-z0-9]+(?:'[a-z]+)?` tokens and the explicit stopword list stored in the resolved configuration. No stemming, synonym expansion, sentence/evidence filtering, or polarity/familiarity interpretation is applied. Negators are retained. Query counts are linear across all supplied user utterances; no bot messages are assembled by the ranker. Only strictly positive matches receive lexical ranks; unmatched documents never get artificial zero-score ranks.

`recommender/fusion.py` implements equal-weight occupation-level RRF: `sum(1/(60+rank))`, using one-based component ranks and zero contribution for absent occupations. It sorts fused score descending then occupation code ascending. No usable signal returns `no_signal`; one usable signal bypasses fusion with raw scores; multiple usable signals use RRF. Raw component scores/ranks remain in `signals`; fused scores populate `rrf_score`. M4/task evidence remains absent.

The registry resolves both accepted variants. `configs/ranking/m3_structured.json`, `m3_bm25.json`, and `m3_hybrid.json` preserve independently selectable configurations. `configs/ranking.yaml` deliberately retains the M1-only default. Enabling its M3 selection uses the same shared pipeline/evaluator without a new metric implementation.

## Saved experiment results

All runs use the identical 60 development/oracle request snapshots and current byte-identical data fingerprints, with 378 eligible occupations. Oracle vectors are privileged synthetic information; BM25 uses the opening user texts from those same profiles, not tracker-derived evidence. There is no held-out ranking or tuning.

| Configuration | Hit@1 | Hit@5 | Hit@10 | MRR | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| [Structured](structured/metrics.json) | .033333 | .166667 | .433333 | .135636 | .151459 | .203471 |
| [BM25](bm25/metrics.json) | .000000 | .016667 | .033333 | .016099 | .014219 | .020544 |
| [Hybrid](hybrid/metrics.json) | .016667 | .133333 | .200000 | .084887 | .098101 | .116610 |

The structured control exactly reproduces the original M2 metric values. Hybrid versus structured has Hit@5 difference −.033333 (5 gains, 7 losses; paired bootstrap 95% interval [−.15, .083333]); NDCG@10 difference −.086861 (interval [−.139415, −.037834]). These are 2,000-resample seeded profile bootstrap intervals, not end-to-end dialogue estimates. See [BM25 comparison](comparison_bm25/comparison.json) and [hybrid comparison](comparison_hybrid/comparison.json) for all metrics and paired profile differences.

| Configuration | Load/index time | Preloaded median | Preloaded p95 |
|---|---:|---:|---:|
| Structured | 44.21 ms | 1.201 ms | 1.405 ms |
| BM25 | 95.96 ms | .962 ms | 1.418 ms |
| Hybrid | 84.07 ms | 2.263 ms | 2.495 ms |

These are local one-call-per-profile observations, excluding metric/export time; they are not a service benchmark. All positive lexical matches are evaluated without truncation; unretrieved targets get MRR zero. Hybrid retains the structured full-pool ranking. Component index metadata, resolved parameters, data/config/source hashes, source configuration snapshots, git revision/dirty state, predictions and per-profile metrics are saved in each run. History records three new successful runs, with `bm25@baseline_v1` and `rrf_k60` where selected; original rows are preserved. Run-ID prefixes remain the evaluation configuration's prefix and are not algorithm identifiers.

[Reproducibility checks](reproducibility_check.json) confirm identical inputs, metrics and predictions excluding unique run IDs for all three repeated runs. Repeats used temporary report/history directories; no fabricated rows were appended.

## Acceptance mapping

| Instruction requirement / acceptance criterion | Evidence | Result |
|---|---|---|
| Index eligible prepared `bm25_doc` with fixed tokenizer | Registered BM25 implementation, frozen resolved parameters/index metadata; unchanged 378-pool | Pass |
| Query accumulated user-only statements, no evidence filtering | Shared `ComponentInput.user_sentences`; repetition/negation fixture; frozen raw-text protocol | Pass |
| Exclude zero-relevance BM25 results from fusion | Positive-only lexical output; empty, punctuation, stopword and OOV fixtures; standalone fallback | Pass |
| One-based RRF at k=60, missing occupations contribute zero | Hand-calculated exact RRF fixtures, disjoint/overlapping lists, raw-score independence and stable ties | Pass |
| Compare structured-only, BM25-only, hybrid on identical development cases | Three saved runs, compatible paired comparisons, matching input/data hashes and pool | Pass |
| Exact-term fixture retrieves intended occupation | `test_exact_term_and_hand_computed_score` | Pass |
| Every returned occupation unique and eligible | Shared output validation, fixture exclusion of Job Zone 3, positive-only lexical output, real-pool evaluation validation | Pass |
| Failure review includes explicit dislikes and familiarity | D002/D046/D047 dislikes; D047 misleading familiarity; D048/D058 genuine familiarity; D005 gain | Pass |
| Reproducibility and prior controls preserved | Original 22,680 M1 oracle results unchanged; repeat checks; original artifacts/history retained | Pass |

## Tests and integration limits

**72 ranking/evaluation tests pass**, including all M0–M2 suites and 11 M3/history compatibility tests. These cover exact BM25 scores, tokenizer/no-signal cases, accumulated text, deterministic ordering, eligible membership, runtime text-only and structured-only fallback, post-fusion `top_k`, RRF arithmetic, manifest/history identities, and content-drift rejection.

Restoring the ranking refactor exposed two compatibility issues: the team's test helper accessed a private occupation container, now replaced with public `eligible_codes`; the legacy importer now permits CRLF→LF normalization only if it exactly recovers the original saved dataset hash. Actual content drift still fails. Current experiment fingerprints retain actual raw bytes. No data file was rewritten.

The broader team suite is not fully runnable here: API/model dependencies (`fastapi`, `transformers`, and model artifacts) are absent. The team pipeline test that attempts to load the real extractor also fails on the missing `transformers` import. This is not a passed end-to-end acceptance claim. The live app factory remains wired to `StructuredRanker`; configurable hybrid runtime is available via `RankingPipeline`, with full app wiring/evidence integration still pending. Production tracker, NLI, and policy scaffolds are unchanged.

## Reproduce

Activate the prepared environment, then use fresh output directories. Focused M3 fixtures can be rerun with `python -m unittest discover -s tests -p test_bm25.py -v` and `python -m unittest discover -s tests -p test_fusion.py -v`:

```powershell
.\.venv\Scripts\Activate.ps1
python -m recommender.evaluate --ranking-config configs/ranking/m3_structured.json --output reports/ranking/m3/structured_repeat
python -m recommender.evaluate --ranking-config configs/ranking/m3_bm25.json --output reports/ranking/m3/bm25_repeat
python -m recommender.evaluate --ranking-config configs/ranking/m3_hybrid.json --output reports/ranking/m3/hybrid_repeat
python -m recommender.compare reports/ranking/m3/structured_repeat reports/ranking/m3/hybrid_repeat --output reports/ranking/m3/comparison_repeat
```

Keep this baseline and its negative result. Next planned milestone is M4 semantic task retrieval, requiring a subsequent instruction. Any lexical tuning should introduce a new named variant and its own preregistered comparison; do not overwrite `baseline_v1` or tune on held-out profiles.
