# Weighted dot product versus cosine: 120 development profiles

Both baseline_v1 algorithms were rerun using configs/m2.json on the same D001-D120 oracle hidden_slots, all six observed dimensions, the same 378 eligible occupations and supplied occupation z-scores, full-pool rankings, seed 425, and unchanged m2-v2 metrics. No lexical/dense retrieval or fusion contributes. These are synthetic oracle results.

| Metric | Weighted dot product | Cosine | Cosine minus dot |
|---|---:|---:|---:|
| hit_at_1 | 0.050000 | 0.416667 | +0.366667 |
| hit_at_5 | 0.166667 | 0.683333 | +0.516667 |
| hit_at_10 | 0.383333 | 0.783333 | +0.400000 |
| mrr | 0.141508 | 0.539974 | +0.398466 |
| ndcg_at_5 | 0.149974 | 0.413876 | +0.263903 |
| ndcg_at_10 | 0.195757 | 0.421261 | +0.225504 |

Cosine improves all six overall metrics. The paired 2,000-resample bootstrap gives a Hit@5 difference of +0.516667 (95% CI +0.416667 to +0.616667). Compatibility checks verified identical dataset/input fingerprints, pool, split, mode, eligibility, ranking depth and metric definitions. No profiles were excluded; no targets were missing or ineligible.

Each algorithm directory preserves configurations, resolved parameters, manifests, full predictions, input snapshots and per-profile metrics. comparison/ contains comparison.json and paired_diffs.csv. Both runs are appended to reports/ranking/evaluation_history.csv and notebooks/ranking_evaluation_history_clean.csv; previous rows are preserved.

Input repair before evaluation: four malformed trailing fragments after D120 were removed from data/dev_profiles.jsonl. All 120 valid profile records were preserved byte-for-byte. The first attempted evaluation failed before producing a registered run.

Reproduce with a fresh output directory for each command:

```powershell
.\.venv\Scripts\python.exe -m recommender.evaluate --config configs/m2.json --ranking-config configs/ranking/baseline_v1.json --output <fresh-dot-directory>
.\.venv\Scripts\python.exe -m recommender.evaluate --config configs/m2.json --ranking-config configs/ranking/m1_cosine.json --output <fresh-cosine-directory>
.\.venv\Scripts\python.exe -m recommender.compare <fresh-dot-directory> <fresh-cosine-directory> --output <fresh-comparison-directory>
```
