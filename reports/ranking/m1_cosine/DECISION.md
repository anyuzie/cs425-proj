# M1 cosine similarity comparison

Implemented `cosine_similarity@baseline_v1` as a separate M1 algorithm in `recommender/cosine.py`, registered through the existing factory. The accepted dot product is preserved. Select cosine with `configs/ranking/m1_cosine.json`; the default `configs/ranking.yaml` remains dot product pending actual-tracker validation.

Cosine divides the signed dot product by both vectors' L2 norms over observed dimensions only. Observed zero dimensions contribute to the occupation norm; unknown dimensions are excluded from both norms. Missing structured input is unavailable, zero user norm is unusable, and zero occupation norm is omitted. Negative scores remain valid, with ascending-code tie-breaking. No occupation rescaling, oracle clamping, retrieval, fusion or metric change was introduced.

## Controlled development results

Both runs use the same 60 development/oracle profiles, all six observed RIASEC dimensions, supplied occupation z-scores and 378 eligible occupations. All targets were ranked. Evaluation used full ranking depth and identical metric definitions. The new dot-product control exactly reproduces the historical aggregate metrics. Both runs were appended to `reports/ranking/evaluation_history.csv` with distinct algorithm identities; existing rows were preserved.

| Metric | Dot product | Cosine | Difference |
|---|---:|---:|---:|
| Hit@1 | .033333 | .583333 | +.550000 |
| Hit@5 | .166667 | .883333 | +.716667 |
| Hit@10 | .433333 | .966667 | +.533333 |
| MRR | .135636 | .713695 | +.578059 |
| NDCG@5 | .151459 | .537513 | +.386054 |
| NDCG@10 | .203471 | .528027 | +.324556 |

[Paired comparison](comparison/comparison.json): 2,000 profile-bootstrap samples, seed 425. Hit@5 difference has 95% interval [.6000, .8333]; MRR [.4815, .6721]; NDCG@10 [.2718, .3766]. Target rank improves for 57 profiles, regresses for one, and stays equal for two. D029 regresses from rank 2 to 5. Graded NDCG@10 improves for 57 and regresses for two, illustrating that exact-target and graded metrics can disagree. See [per-profile differences](comparison/paired_diffs.csv).

Local preloaded full-pool median/p95: dot product 1.347/1.695 ms; cosine 1.526/1.839 ms. These single-run workstation measurements are descriptive, without a dedicated latency benchmark.

## Decision and limits

Keep cosine as a supported alternative and preferred candidate for the next tracker comparison. This is a strong improvement on this synthetic oracle benchmark. The dataset was constructed from target occupation RIASEC rows plus noise, so direction matching is particularly aligned with its construction; the result does not establish real-user suitability, reliable upstream extraction, partial-profile quality, or an end-to-end probing benefit. No held-out data was used. Hybrid cosine configurations have not been evaluated in this experiment.

All 172 tests pass, including six new cosine tests. Correctness coverage includes hand-computed positive/negative/orthogonal similarities, ties, partial observations versus observed neutrality, missing/zero vectors, zero-norm occupations, scale invariance, sign reversal, immutable pool snapshots, rejected parameters, real-pool registry selection, deterministic ranking, top-k and oracle/runtime separation. The registry rejection test now uses an unregistered name because cosine is supported. Full-suite results are in [test_results.txt](test_results.txt).

The subsequent [fusion experiment](FUSION_DECISION.md) evaluates all three cosine M1 + M3/M4 combinations; all regress versus cosine alone on these same development/oracle cases.

## Reproduction

From the repository root with the project environment:

```powershell
python -m recommender.evaluate --ranking-config configs/ranking/baseline_v1.json --output <fresh-control-directory>
python -m recommender.evaluate --ranking-config configs/ranking/m1_cosine.json --output <fresh-cosine-directory>
python -m recommender.compare <fresh-control-directory> <fresh-cosine-directory> --output <fresh-comparison-directory>
```

Each run retains exact configs, resolved parameters, inputs, data/dependency/source fingerprints, predictions, metrics and latency. The accompanying `source_snapshot.zip` preserves the uncommitted implementation used for these runs. For runtime selection, change only the M1 algorithm in `configs/ranking.yaml` to `cosine_similarity` and keep `variant: baseline_v1`; the saved preset is standalone, while the existing registry also supports selecting cosine within a separately evaluated fusion configuration.
