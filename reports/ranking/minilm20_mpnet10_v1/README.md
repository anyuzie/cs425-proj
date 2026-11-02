# Reversed semantic sequence: MiniLM 20 → MPNet 10

31 October 2026. Same 120 synthetic development/oracle profiles, 378 eligible occupations, descriptions, query policy, model revisions, token limits and nine permanent metrics as the original sequence.

Weighted dot product selects the top 100. MiniLM reranks within baseline groups of 20; MPNet then reranks groups of 10 formed from MiniLM output. The RIASEC/upstream pipeline is unchanged and semantic matching remains separate. All existing variants remain available.

| Metric | MPNet 20 → MiniLM 10 | MiniLM 20 → MPNet 10 | Change |
|---|---:|---:|---:|
| hit_at_1 | 0.016667 | 0.041667 | +0.025000 |
| hit_at_5 | 0.125000 | 0.175000 | +0.050000 |
| hit_at_10 | 0.250000 | 0.325000 | +0.075000 |
| mrr | 0.101771 | 0.127189 | +0.025418 |
| ndcg_at_5 | 0.122811 | 0.139112 | +0.016301 |
| ndcg_at_10 | 0.160019 | 0.175182 | +0.015163 |
| recall_at_5 | 0.079658 | 0.081282 | +0.001625 |
| recall_at_10 | 0.157609 | 0.156661 | -0.000949 |
| recall_at_20 | 0.303874 | 0.303874 | +0.000000 |

## Paired uncertainty

Change is reverse minus original. Bootstrap: 2,000 paired profile resamples, seed 425. These intervals describe the fixed synthetic development sample, not real-user intent accuracy.

- hit_at_5: change +0.050000; 95% interval [0.0, 0.10833333333333334]; improved 9, regressed 3 profiles.
- mrr: change +0.025418; 95% interval [0.004932068526555386, 0.049964302831799666]; improved 57, regressed 42 profiles.
- ndcg_at_10: change +0.015163; 95% interval [-0.0005578419414101015, 0.032181325482818095]; improved 54, regressed 52 profiles.
- recall_at_5: change +0.001625; 95% interval [-0.006253315720227485, 0.009208796700089268]; improved 20, regressed 16 profiles.
- recall_at_10: change -0.000949; 95% interval [-0.01718112558379586, 0.013745258045761143]; improved 36, regressed 38 profiles.

Recall@20 is structurally invariant under both sequences. The first model determines top-ten membership inside each original group of 20, so Hit@10 and Recall@10 can differ between the two sequences. The second model determines the ordering within those top-ten groups.

## Verification and use

33 targeted tests pass. The saved scores reconstruct both stage orders exactly for all 120 profiles. Original tail, structured implementation, upstream pipeline, active defaults and local terminal edit are preserved. The consolidated CSV now includes seven experiments. No automatic promotion is performed.

```powershell
python -m recommender.workflow evaluate --config configs/ranking/workflow.reverse.json --output reports/ranking/reverse_new_run
python -m recommender.workflow chat --config configs/ranking/workflow.reverse.json
```

Chat still requires existing upstream checkpoints. Saved stage_minilm ranks are intermediate; stage_mpnet ranks are final. The semantic compatibility alias now points to MPNet for this variant. Separate model caches are reused.

See sequence_comparison.json, sequence_paired_diffs.json, verification.json and the consolidated CSV for exact results. Retain the existing active default pending review; a better result than the original sequence alone does not establish superiority to standalone dot product or single-model variants.

The updated seven-experiment CSV is `../semantic_experiments_consolidated_v2.csv`; the earlier CSV was locked and remains preserved. See failure_review.md for five inspected losses and three gains.
