# Updated development data: dot product, cosine and combinations

Eight fresh development/oracle evaluations use current data/dev_profiles.jsonl (120 profiles), 378 eligible occupations, unchanged baseline variants, raw text and equal-weight RRF k=60 for combinations. Dataset bytes are preserved in dataset_snapshot.jsonl. No held-out test set or extractor inference is used. Ranking defaults remain unchanged.

| Configuration | Hit@1 | Hit@5 | Hit@10 | MRR | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| dot | 0.050000 | 0.166667 | 0.383333 | 0.141508 | 0.149974 | 0.195757 |
| cosine | 0.416667 | 0.683333 | 0.783333 | 0.539974 | 0.413876 | 0.421261 |
| dot_bm25 | 0.008333 | 0.125000 | 0.208333 | 0.078370 | 0.096153 | 0.121253 |
| cosine_bm25 | 0.075000 | 0.233333 | 0.408333 | 0.174876 | 0.158430 | 0.189981 |
| dot_dense | 0.016667 | 0.108333 | 0.191667 | 0.077858 | 0.096793 | 0.120825 |
| cosine_dense | 0.066667 | 0.216667 | 0.366667 | 0.165777 | 0.145171 | 0.175661 |
| dot_three_way | 0.016667 | 0.075000 | 0.150000 | 0.062427 | 0.073497 | 0.091857 |
| cosine_three_way | 0.041667 | 0.108333 | 0.225000 | 0.101906 | 0.097452 | 0.119284 |

Standalone cosine remains strongest; every combination regresses against its matching standalone M1 on all six metrics. All eight runs were appended exactly once to reports/ranking/evaluation_history.csv and notebooks/ranking_evaluation_history_clean.csv. Full rankings, per-profile scores and aggregate metrics were verified; dataset/input/pool/evaluator conditions match across all eight runs. Six vs_standalone.json files and cosine_vs_dot.json preserve compatible paired bootstrap comparisons (2,000 samples, seed 425).

Compared with the previous 120-profile experiment inputs, opening text changed for 17 profiles: D001, D011, D021, D031, D041, D051, D061, D071, D075, D081, D085, D091, D095, D101, D105, D111, D115. Structured vectors changed for 0 profiles. Standalone M1 results remain identical.

## Descriptive change from previous 120-profile runs

The datasets differ; these differences are descriptive, not outputs of the compatible-run paired comparison.

| Configuration | Previous Hit@5 | Current Hit@5 | Difference |
|---|---:|---:|---:|
| dot | 0.166667 | 0.166667 | +0.000000 |
| cosine | 0.683333 | 0.683333 | +0.000000 |
| dot_bm25 | 0.108333 | 0.125000 | +0.016667 |
| cosine_bm25 | 0.241667 | 0.233333 | -0.008333 |
| dot_dense | 0.108333 | 0.108333 | +0.000000 |
| cosine_dense | 0.216667 | 0.216667 | +0.000000 |
| dot_three_way | 0.075000 | 0.075000 | +0.000000 |
| cosine_three_way | 0.125000 | 0.108333 | -0.016667 |

Reproduce a run using its evaluation_config.json and ranking_config.json with python -m recommender.evaluate and a fresh output directory. Restore the preserved dataset snapshot to the configured input path and use fingerprint-matching occupation data and semantic cache for exact reproduction. These synthetic oracle results do not establish end-to-end conversation or student recommendation quality.
