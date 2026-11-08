# Recall@50 for all semantic configurations

Evaluator m2-v4 reports ten permanent metrics: Hit@1/5/10, MRR, NDCG@5/10,
and Recall@5/10/20/50. Recall uses all distinct eligible positive-grade labels,
including the exact target and related occupations; the result is macro-averaged
across profiles. Empty relevance sets receive zero.

All eight configurations in semantic_experiments_consolidated.csv now contain
Recall@50. The adjacent executed notebook has eight rows and ten metric columns,
including per-column colouring and tied maximum highlights for Recall@50.

## Frozen experiments

Original metrics.json, per_profile.json and manifest.json files are unchanged.
Each run has recall50_extension.json with aggregate and per-profile Recall@50,
its definition, original manifest and prediction hashes and dataset fingerprints.
Extensions were computed from existing full predictions; no model inference was
rerun. The exporter and paired comparator validate these hashes before merging
extensions. Recall@50-aware paired results are saved separately in
reports/ranking/recall50_paired_comparisons.json.

The two duplicate verification/control runs were also extended. Central history
now has a Recall@50 column, populated only for known matching run IDs. Unknown
older values remain blank; evaluation_history.csv.pre_recall50.bak preserves
the original history. Future appends support both six- and nine-metric historical
schemas while requiring all ten values for evaluator m2-v4.

New evaluations emit Recall@50 directly in aggregate and per-profile metric JSON,
history and comparisons. Promotion no-regression checks include it. Historical
and current evaluator versions remain distinct, so incompatible runs cannot be
silently compared.

Recall@50 equals the baseline for groups of 5 and 10 because 50 is a complete
group boundary. With groups of 20, rank 50 cuts a group in half, so either
sequential variant can change membership of the first 50 occupations.

See reports/ranking/recall50_backfill_manifest.json for all values and provenance.
