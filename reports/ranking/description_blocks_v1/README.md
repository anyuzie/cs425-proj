# Description reranking within baseline rank blocks

31 October 2026. Four fresh matched runs on 120 synthetic development/oracle profiles and 378 eligible occupations. Model, descriptions, input snapshots and labels are held fixed. No actual-tracker accuracy or held-out claim is made.

## Design

Weighted dot product is the sole RIASEC scorer and candidate selector. MiniLM uses occupation descriptions only and the same newline-joined user history. Blocks of 5 rerank baseline positions 1–5, 6–10, …, 96–100 independently; blocks of 10 rerank 1–10, 11–20, …, 91–100. Scores are not blended. Baseline structured scores and original ranks remain in predictions; semantic ranks describe the unrestricted cosine ranking of the shortlist. Candidates never cross blocks and the tail after rank 100 remains unchanged. Rank adjacency does not guarantee that raw dot-product scores are close.

## Metrics

| Metric | Dot product | Full top-100 rerank | Blocks of 5 | Blocks of 10 |
|---|---:|---:|---:|---:|
| hit_at_1 | 0.050000 | 0.025000 | 0.016667 | 0.033333 |
| hit_at_5 | 0.166667 | 0.075000 | 0.166667 | 0.208333 |
| hit_at_10 | 0.383333 | 0.141667 | 0.383333 | 0.383333 |
| mrr | 0.141508 | 0.073640 | 0.116317 | 0.134000 |
| ndcg_at_5 | 0.149974 | 0.075118 | 0.137773 | 0.143141 |
| ndcg_at_10 | 0.195757 | 0.099685 | 0.184246 | 0.186716 |
| recall_at_5 | 0.089856 | 0.047520 | 0.089856 | 0.080453 |
| recall_at_10 | 0.167393 | 0.100079 | 0.167393 | 0.167393 |
| recall_at_20 | 0.303874 | 0.203944 | 0.303874 | 0.303874 |

Recall is macro-averaged coverage of all distinct eligible positive-grade relevance labels, including target and related occupations; empty relevance sets receive zero. It is now permanent in evaluator m2-v3, not a supplementary-only export. All profiles have eligible targets and nonzero ideal relevance.

## Interpretation and decision

Block-10 Hit@5 increases from 20/120 to 25/120: 14 profiles gain a top-five target and 9 lose one. Its +4.17 percentage-point paired change has a 95% bootstrap interval of [-3.33, +12.50] percentage points (2,000 resamples; seed 425). MRR falls by .007508, interval [-.050927, .033973]. Recall@5 also falls, from .089856 to .080453. This is a mixed development result with substantial uncertainty, not an established improvement.

Block-5 has lower Hit@1, MRR and NDCG than the control. Its MRR change is -.025191, interval [-.055432, .000391]. Both constrained variants substantially improve over unrestricted top-100 semantic reranking on these data.

At a block boundary the retrieved set is invariant: block-5 cannot change Hit@5, Hit@10 or Recall@5/10/20; block-10 cannot change Hit@10 or Recall@10/20. Those ties are structural and do not demonstrate improved retrieval. Within-block ordering is assessed by Hit@1, MRR and NDCG; block-10 also permits Hit@5 and Recall@5 to change.

Decision: preserve the weighted-dot-product active default. Both named variants are available for experiments. Neither satisfies the current higher-Hit@5-with-no-other-metric-regression promotion rule. Do not choose or retune on a held-out set.

## Verification and limitations

38 targeted tests pass: 4 description contracts, 12 evaluation/history migration, 10 promotion, 7 configuration, 5 modular architecture. A broader dialogue evaluation test cannot import because the recovered workspace lacks the existing eval package; that check remains unavailable.

All 120 profiles in each block variant were checked against the saved control for exact block membership, unchanged tail order, and identical original structured scores/ranks. Paired comparisons verify matching dataset/input fingerprints and evaluation definitions. Model cold loading and incremental description caching affect reported latency; timing is not a clean warm-cache benchmark.

## Reproduce

From the implementation directory, install requirements-description-ranking.txt, then run:

```powershell
python -m recommender.experiments.block_comparison --output reports/ranking/description_blocks_new_run
```

Use a new output directory. The local workstation interpreter requires .runtime on PYTHONPATH. Each run saves complete predictions, input/configuration snapshots, nine metrics, per-profile results, latency, source/data fingerprints and model metadata. Root comparison files contain all nine paired metric differences and bootstrap intervals. Runs are registered in both local and central histories. Historical artifacts are preserved.

Select configs/ranking/description_blocks5.json or configs/ranking/description_blocks10.json for direct evaluation. See failure_review.md for inspected gains and losses.
