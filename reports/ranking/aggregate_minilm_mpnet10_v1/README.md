# Equal-weight semantic aggregation in groups of 10

31 October 2026. Both models receive the same weighted-dot-product top 100, descriptions only and identical user history. Each returns scores without reordering candidates. Their raw cosine scores are averaged (50% MiniLM, 50% MPNet) and one final sort is performed within consecutive original baseline groups of 10. RIASEC matching remains separate.

This is a new frozen aggregated_descriptions@equal_weights_blocks10_v1 variant. It preserves both sequential orders, single-model variants, upstream RIASEC processing, weighted dot product and active defaults.

| Metric | Dot product | MiniLM 10 | MPNet 10 | Aggregated 10 |
|---|---:|---:|---:|---:|
| hit_at_1 | 0.050000 | 0.033333 | 0.041667 | 0.033333 |
| hit_at_5 | 0.166667 | 0.208333 | 0.175000 | 0.216667 |
| hit_at_10 | 0.383333 | 0.383333 | 0.383333 | 0.383333 |
| mrr | 0.141508 | 0.134000 | 0.133113 | 0.133342 |
| ndcg_at_5 | 0.149974 | 0.143141 | 0.139903 | 0.145068 |
| ndcg_at_10 | 0.195757 | 0.186716 | 0.187431 | 0.186359 |
| recall_at_5 | 0.089856 | 0.080453 | 0.081946 | 0.081798 |
| recall_at_10 | 0.167393 | 0.167393 | 0.167393 | 0.167393 |
| recall_at_20 | 0.303874 | 0.303874 | 0.303874 | 0.303874 |

120 synthetic development/oracle profiles, 378 eligible occupations. All nine permanent metrics use identical labels and input snapshots. Recall is macro-averaged coverage of distinct eligible positive-grade labels. Hit@10 and Recall@10/20 remain structurally equal to the dot-product baseline under these groups of 10.

## Paired uncertainty

Each interval is a 2,000-resample paired profile bootstrap, seed 425; change is aggregation minus comparator.

- Versus dot_product, hit_at_5: +0.050000, 95% interval [-0.025, 0.125]; improved 14, regressed 8 profiles.
- Versus dot_product, mrr: -0.008166, 95% interval [-0.054573971376255644, 0.034818486693361315]; improved 51, regressed 56 profiles.
- Versus minilm10, hit_at_5: +0.008333, 95% interval [-0.041666666666666664, 0.058333333333333334]; improved 5, regressed 4 profiles.
- Versus minilm10, mrr: -0.000657, 95% interval [-0.014122223775659727, 0.012601839900495319]; improved 34, regressed 38 profiles.
- Versus mpnet10, hit_at_5: +0.041667, 95% interval [0.0, 0.08333333333333333]; improved 6, regressed 1 profiles.
- Versus mpnet10, mrr: +0.000229, 95% interval [-0.018444034560333493, 0.0165593429256973]; improved 37, regressed 35 profiles.
- Versus reverse_sequence, hit_at_5: +0.041667, 95% interval [-0.016666666666666666, 0.1]; improved 9, regressed 4 profiles.
- Versus reverse_sequence, mrr: +0.006153, 95% interval [-0.0165339668805844, 0.025331372026592032]; improved 64, regressed 41 profiles.

## Interpretation and implementation

The averaging is on raw cosine scores without calibration. Although both are cosine values, their score distributions can differ, so equal numerical weights do not guarantee equal influence. No embedding vectors are averaged (their dimensions differ). No RIASEC score is blended into the semantic score. Weights are declared in recommender/semantic/aggregation_config.py and captured in the ranking preset. Future weight choices should be new frozen variants, with matching presets, rather than edits to past experiments.

DescriptionReranker.score(codes, utterances) is independent of sorting and RIASEC. SemanticMatchingPipeline owns aggregation. Signals preserve semantic_minilm, semantic_mpnet and semantic_aggregate; semantic aliases the aggregate and stage_aggregate records final position. The tail after 100 remains unchanged.

All 36 targeted tests pass, including model-order invariance, same-shortlist inputs, signed scores, ties, partial groups, display limits and compatibility with both sequential variants. Every saved aggregate formula and block order was checked across all 120 profiles. Existing protected files remain byte-for-byte unchanged. See verification.json.

```powershell
python -m recommender.workflow evaluate --config configs/ranking/workflow.aggregate.json --output reports/ranking/aggregate_new_run
python -m recommender.workflow chat --config configs/ranking/workflow.aggregate.json
```

Consolidated results: ../semantic_experiments_consolidated.csv. No active/default change or automatic promotion is performed. Synthetic RIASEC-derived targets do not independently validate semantic user intent.

Decision: retain aggregation as an experimental option; keep active defaults. Versus MiniLM-only, the net Hit@5 gain is one profile (five gained, four lost), its interval includes zero, and MRR is slightly lower. See failure_review.md for inspected cases.
