# M2 — Evaluation foundation

**Decision: M2 accepted. Retain the structured-only oracle development run as the control; stop before M3.** All 38 M0–M2 tests pass (10 new M2 tests). Correctness and reproducibility are established; recommendation quality remains limited. No held-out ranking or tuning was performed.

## Implementation

- `recommender/evaluation.py`: exact-target Hit@5, full-pool MRR, and graded NDCG@10. Grades are target=4, Primary-Short=3, Primary-Long=2, Supplemental=1; duplicates use the maximum grade. Ineligible labels are excluded from both DCG and IDCG. Zero IDCG yields zero and is counted. Missing/ineligible targets remain in the denominator and are reported. Duplicate or out-of-pool recommendations raise errors.
- `StructuredRanker.rank_oracle(OracleInput)`: explicit offline full-pool adapter, preserving synthetic z-scale values without clamping or weakening runtime validation. Uses the same scoring implementation as M1. All six oracle dimensions are observed. Labels never enter the ranker.
- `recommender/evaluate.py`: fixed development/oracle baseline runner, saved input snapshots, full predictions with component scores/ranks, per-profile metrics, aggregate metrics, timings, configuration, and manifest. Existing run directories are protected from overwrite. M2 intentionally supports only the frozen development/oracle configuration; mock/actual tracker evaluation needs separately labeled future adapters.
- `recommender/compare.py`: paired per-profile differences, improved/regressed counts, and seeded profile bootstrap percentile 95% intervals (2,000 resamples). Rejects mismatched modes, splits, input snapshots, datasets, pools, eligibility, evaluator definitions, and profile IDs. This does not automatically establish that a future experiment changes only one variable: reviewers must still check its declared treatment/configuration.

## Frozen control

Configuration: `configs/m2.json`. Saved run: `reports/ranking/m2/baseline/`.

| Measure | Result |
|---|---:|
| Input condition | Synthetic oracle, all six dimensions |
| Development profiles | 60 |
| Eligible occupations per full ranking | 378 |
| Hit@5 | 0.166667 (10/60) |
| MRR | 0.135636 |
| NDCG@10 | 0.203471 |
| Missing / ineligible targets | 0 / 0 |
| Zero-IDCG / unranked-target cases | 0 / 0 |
| Load time | 40.1 ms |
| Preloaded ranking median / p95 | 0.641 / 1.511 ms |

Timing is one call per development profile, with nearest-rank p95 and metric/export time excluded. It is a small local timing sample, not a production service benchmark.

The manifest records dataset/config/input/source hashes, candidate codes, parent git revision and dirty state, software version, input mode, metric definitions, and reproduction command. Source fingerprints identify uncommitted evaluated code. No embedding model or index exists at M2. Saved predictions contain evaluator-side target labels and must not be fed back into retrieval.

## Acceptance and reproducibility

Hand-calculated tests cover ranks 1, 6, and absent; exact Hit@5 and NDCG@10 cutoffs; full-ranking MRR; graded DCG; perfect/irrelevant/empty-ideal rankings; maximum-grade deduplication; pool filtering; invalid recommendations; and target diagnostics. Integration tests verify oracle values above 1 remain unchanged while the runtime interface rejects them, identical baseline predictions/config/inputs/metrics across runs, zero self-comparison differences, known nonzero paired differences, incompatible-mode rejection, held-out configuration rejection, and overwrite protection.

From repository root (Python 3.10+):

```powershell
python -m unittest discover -s tests -v
python -m recommender.evaluate --config configs/m2.json --output reports/ranking/m2/reproduction_01
python -m recommender.compare reports/ranking/m2/baseline reports/ranking/m2/reproduction_01 --output reports/ranking/m2/comparison_01
```

On this machine substitute `& 'C:/Users/aanny/AppData/Local/Programs/Python/Python313/python.exe'` for `python`. Use a fresh output directory on every run. Predictions, input snapshots, config and metrics reproduce exactly; timestamps, timings, and git/source metadata can differ. No third-party dependencies are required.

See `failure_review.md` for five inspected misses. There is no treatment at M2, so there are no claimed gains or regressions versus a prior measured model. Bootstrap self-comparison validates plumbing; its zero interval is not an uncertainty estimate for baseline accuracy.

## Next gate

M3 adds BM25-only and structured+BM25 RRF comparisons using the same development cases and explicitly matching input conditions. Preserve this control. M1's potential scoring alternatives remain KIV; these baseline results alone do not justify selecting a replacement. Actual-tracker and end-to-end probing effectiveness are untested. Existing teammate integration decisions remain pending.
