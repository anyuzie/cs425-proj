# MPNet versus MiniLM: description reranking in groups of 10

31 October 2026. Fresh paired experiment on the same 120 synthetic development/oracle profiles and 378 eligible occupations. Weighted dot product selects the top 100. Descriptions only are encoded, and cosine similarity reranks independently within baseline ranks 1–10, 11–20, …, 91–100. Original RIASEC scores/ranks and the full-pool tail are preserved.

## Model replacement

The existing pipeline now accepts mpnet_descriptions@mpnet_blocks10_v1. MiniLM code, frozen parameters, task rankers, configurations and historical reports are preserved. The new configs/ranking/workflow.mpnet.json selects MPNet through the existing workflow; normal active/default selections remain unchanged. Each pipeline has its own model and vector cache.

MiniLM: sentence-transformers/all-MiniLM-L6-v2, revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41, 384 normalized dimensions. MPNet: sentence-transformers/all-mpnet-base-v2, revision e8c3b32edf5434bc2275fc9bab85f82640a19130, 768 normalized dimensions. Both use CPU, deterministic settings, the same 256-token limit, and newline-joined user-only history. Tokenizers differ, so equal token limits do not guarantee equal character spans.

## Nine permanent metrics

| Metric | MiniLM, groups of 10 | MPNet, groups of 10 | MPNet minus MiniLM |
|---|---:|---:|---:|
| hit_at_1 | 0.033333 | 0.041667 | +0.008333 |
| hit_at_5 | 0.208333 | 0.175000 | -0.033333 |
| hit_at_10 | 0.383333 | 0.383333 | +0.000000 |
| mrr | 0.134000 | 0.133113 | -0.000886 |
| ndcg_at_5 | 0.143141 | 0.139903 | -0.003238 |
| ndcg_at_10 | 0.186716 | 0.187431 | +0.000715 |
| recall_at_5 | 0.080453 | 0.081946 | +0.001493 |
| recall_at_10 | 0.167393 | 0.167393 | +0.000000 |
| recall_at_20 | 0.303874 | 0.303874 | +0.000000 |

Recall is macro-averaged coverage of all distinct eligible positive-grade labels, including the target and related occupations. Empty relevance sets score zero. No profiles have missing/ineligible targets or zero ideal relevance. Hit@10 and Recall@10/20 cannot change under this block constraint; their ties are structural.

## Paired interpretation and decision

MPNet raises Hit@1 from 4/120 to 5/120, while lowering Hit@5 from 25/120 to 21/120. Four profiles gain a top-five target and eight lose one. The Hit@5 change is -3.33 percentage points with paired bootstrap 95% interval [-8.33, +1.67] percentage points. MRR changes by -.000886, interval [-.022739, +.022478]. All nonconstant metric intervals include zero. Bootstrap uses 2,000 paired profile resamples, seed 425. These development results do not establish an overall quality improvement.

MPNet slightly improves Recall@5 and NDCG@10, but slightly reduces MRR and NDCG@5. Against the previously matched dot-product-only baseline (Hit@5=.166667, MRR=.141508), MPNet has higher Hit@5=.175 but lower MRR=.133113. These synthetic targets emphasize RIASEC fit; independently labeled semantic intent is still needed for a broader user-quality claim.

Decision: retain both models as selectable variants. Do not automatically promote MPNet: its lower Hit@5 and other regressions do not meet the existing no-regression promotion rule. The normal active workflow retains weighted dot product.

## Runtime

| Recorded request timing | MiniLM | MPNet |
|---|---:|---:|
| Median | 36.9 ms | 197.8 ms |
| p95 | 251.6 ms | 1662.5 ms |

These are observed timings over the evaluator sequence, including lazy model loading/downloading and incremental description cache population. They are not a controlled warm-cache benchmark. In this run MPNet had greater request cost.

## Verification and artifacts

29 targeted tests pass: 5 encoder/reranking contracts, 12 evaluator/history checks, 5 modular architecture checks and 7 configuration checks. The encoder test verifies selected model ID, revision, dimensionality, token limit and model reuse; block contracts cover MPNet and MiniLM. All 120 paired predictions preserve identical block membership, original structured scores/ranks, description fingerprints and tail order. Both runs have identical implementation source hashes. All 45,360 MiniLM prediction records exactly reproduce the preceding MiniLM groups-of-10 experiment. See verification.json.

Each run contains full predictions, nine aggregate/per-profile metrics, frozen inputs/configurations, model/dependency metadata, dataset/source fingerprints and latency. comparison.json contains the paired bootstrap results; paired_diffs.json contains all profile-level changes; metrics_summary.json provides both aggregate tables. Both runs are registered in local and central histories. See failure_review.md for inspected cases.

## Reproduce

```powershell
python -m recommender.experiments.mpnet_comparison --output reports/ranking/mpnet_blocks10_new_run
python -m recommender.workflow chat --config configs/ranking/workflow.mpnet.json
```

Use a new output directory and install requirements-description-ranking.txt. On this workstation use the Python313 interpreter with .runtime on PYTHONPATH. The first MPNet request needs the pinned model download; subsequent requests reuse the project cache. Chat requires the existing upstream extractor resources. See ../../../../docs/MPNET_RERANKING.md for architecture and usage.
