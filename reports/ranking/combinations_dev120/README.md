# M1 combinations on 120 development profiles

All six combinations use D001-D120, the same 378 eligible occupations, oracle hidden_slots, raw opening text, unchanged baseline variants and equal-weight RRF k=60. M3 is BM25; M4 is pinned MiniLM task retrieval. M1 receives the same six observed dimensions. No held-out data or default changes.

| Configuration | Hit@1 | Hit@5 | Hit@10 | MRR | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| dot_standalone | 0.050000 | 0.166667 | 0.383333 | 0.141508 | 0.149974 | 0.195757 |
| dot_bm25 | 0.016667 | 0.108333 | 0.208333 | 0.081666 | 0.095631 | 0.123509 |
| dot_dense | 0.016667 | 0.108333 | 0.191667 | 0.079808 | 0.101217 | 0.124163 |
| dot_three_way | 0.016667 | 0.075000 | 0.158333 | 0.063418 | 0.072884 | 0.092786 |
| cosine_standalone | 0.416667 | 0.683333 | 0.783333 | 0.539974 | 0.413876 | 0.421261 |
| cosine_bm25 | 0.083333 | 0.241667 | 0.400000 | 0.181617 | 0.162507 | 0.190854 |
| cosine_dense | 0.075000 | 0.216667 | 0.391667 | 0.170684 | 0.148325 | 0.183130 |
| cosine_three_way | 0.041667 | 0.125000 | 0.241667 | 0.110372 | 0.101186 | 0.123148 |

Every combination regresses against its matching standalone M1 on all six metrics. Cosine remains stronger than dot product within each matching combination. Standalone cosine remains the best of these eight configurations. These are synthetic oracle findings, not production tracker validation.

All six runs are registered exactly once in both central and clean notebook histories. Each directory preserves predictions, per-profile metrics, configs, manifest, and vs_standalone.json with paired 2,000-sample bootstrap intervals. Dataset/input hashes, pool and evaluator compatibility were checked against the matching standalone control. All aggregate metrics were independently recomputed from exported full rankings; all 120 profiles and 378 candidates were present in each run. summary.csv contains the eight configurations.

Cache provenance: the previous cache failed only tasks_sha256; parsed task mapping, pool, model, configuration and dependency identities matched. The old cache was preserved at artifacts/ranking/minilm_mean_tasks_v1_preserved_ac844274a309. The active cache was rebuilt using the unchanged pinned local model and baseline parameters against current data/tasks.csv (8,243 tasks); no download or metadata bypass was used. New manifests record the rebuilt artifacts.

Reproduction: use each run directory evaluation_config.json and ranking_config.json with python -m recommender.evaluate and a fresh output directory. Rebuild or restore the fingerprint-matching cache for semantic runs. Historical runs and accepted variants are preserved.
