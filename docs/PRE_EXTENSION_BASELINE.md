# Pre-extension ranking and classifier baseline

Recorded 2 November 2026, before integrating clause classification, preference direction, or task/context matching into ranking. This is a documentation snapshot of saved evaluations; no evaluation, training, configuration promotion or scoring change was performed.

## Ranking performance before the extension

The table below records all eight existing weighted-dot-product and description-reranking variants. Values are proportions, rounded to six decimals. MiniLM/MPNet here compare occupation descriptions with user history; these are not the proposed clause-to-task/context matchers.

| Variant | hit_at_1 | hit_at_5 | hit_at_10 | mrr | ndcg_at_5 | ndcg_at_10 | recall_at_5 | recall_at_10 | recall_at_20 | recall_at_50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| weighted_dot_product | 0.050000 | 0.166667 | 0.383333 | 0.141508 | 0.149974 | 0.195757 | 0.089856 | 0.167393 | 0.303874 | 0.589844 |
| minilm_top100 | 0.025000 | 0.075000 | 0.141667 | 0.073640 | 0.075118 | 0.099685 | 0.047520 | 0.100079 | 0.203944 | 0.459094 |
| minilm_groups5 | 0.016667 | 0.166667 | 0.383333 | 0.116317 | 0.137773 | 0.184246 | 0.089856 | 0.167393 | 0.303874 | 0.589844 |
| minilm_groups10 | 0.033333 | 0.208333 | 0.383333 | 0.134000 | 0.143141 | 0.186716 | 0.080453 | 0.167393 | 0.303874 | 0.589844 |
| mpnet_groups10 | 0.041667 | 0.175000 | 0.383333 | 0.133113 | 0.139903 | 0.187431 | 0.081946 | 0.167393 | 0.303874 | 0.589844 |
| mpnet20_then_minilm10 | 0.016667 | 0.125000 | 0.250000 | 0.101771 | 0.122811 | 0.160019 | 0.079658 | 0.157609 | 0.303874 | 0.584180 |
| minilm20_then_mpnet10 | 0.041667 | 0.175000 | 0.325000 | 0.127189 | 0.139112 | 0.175182 | 0.081282 | 0.156661 | 0.303874 | 0.590485 |
| minilm_mpnet_equal_weights_groups10 | 0.033333 | 0.216667 | 0.383333 | 0.133342 | 0.145068 | 0.186359 | 0.081798 | 0.167393 | 0.303874 | 0.589844 |

The combined variant (`minilm_mpnet_equal_weights_groups10`) sends the same dot-product top 100 to MiniLM and MPNet, averages their cosine scores equally, and reorders within original groups of 10. The two sequential orders remain distinct variants. The active central-workflow configuration still selects weighted dot product alone; the combined variant is an experimental comparator, not the active default. The terminal display cutoff of 10 is not an evaluation or model candidate cutoff.

Evaluation conditions: 120 synthetic development profiles, oracle RIASEC inputs, 378 eligible occupations (job zones 4/5 with complete supplied RIASEC z-scores), full-pool ranking. All rows share labels and input snapshots. Hit@k means at least one relevant result in the first k; MRR measures the reciprocal rank of the first relevant result; NDCG measures graded relevance and ordering; Recall@k is macro-averaged coverage of eligible positive-grade labels. These are development/oracle results, not independent real-user or end-to-end conversational validation.

Aggregation improves Hit@5 from 0.166667 to 0.216667, but MRR drops from 0.141508 to 0.133342. The saved paired-bootstrap 95% interval for the Hit@5 difference is [-0.025, 0.125], including zero; this does not establish a reliable overall improvement. See the [aggregation report](../reports/ranking/aggregate_minilm_mpnet10_v1/README.md) for paired comparisons.

Provenance: [full-precision consolidated results](../reports/ranking/semantic_experiments_consolidated.csv) preserve run IDs, report paths and pinned encoder revisions. Historical rows report evaluator m2-v3 with the recall50-v1 extension; the current evaluator documents ten permanent metrics. Dataset fingerprint: `5b3da558ba76296478974ec54d86153b101acc1b07dca13bf3b36e9fb9055af7`. Input SHA-256: `0fcf76d877d8f886a44b7e04ea4c7b15adb0fdf2457caaed59841a75b4bfd8fb`.

## Current BERT classifier performance (separate from ranking)

Checkpoint: `models/task_context_bert`, pretrained `google-bert/bert-base-uncased`, three fine-tuning epochs, clause text only. Two independent sigmoid outputs (Task and Context), each routed at 0.5. These are not calibrated probabilities or a preference-polarity model. Weight SHA-256: `9b32c4c6526b609af5f6b8d77b7399d7170c34c1425fc4b9364c866f8f58a5b2`.

| Evaluation | Examples | Task F1 | Context F1 | Macro F1 | Exact match |
|---|---:|---:|---:|---:|---:|
| Held-out source validation | 3,435 | 1.000000 | 1.000000 | 1.000000 | 100.00% (3,435/3,435) |
| User-style paraphrase diagnostic | 80 | 0.564103 | 0.457831 | 0.510967 | 35.00% (28/80) |

Source training comprises 14,065 examples (13,973 task statements and 92 context texts). Validation comprises 3,413 tasks from held-out occupation groups and 22 context texts from 11 held-out context families. Labels derive from source membership; they are not independently reviewed clause-level ground truth. Training contains only [1,0] and [0,1] targets. Perfect source validation does not demonstrate conversational generalisation, simultaneous routing or rejection.

The frozen-checkpoint paraphrase diagnostic contains 20 semantic families with four wording variants each: 32 task-only, 24 context-only, 12 mixed and 12 irrelevant clauses. Expected labels were assistant-authored before inference, not independently reviewed; variants are correlated and this is not a representative independent benchmark. No retraining or threshold tuning occurred, and no exact normalized training-text matches were found.

| Output, paraphrase diagnostic | Precision | Recall | F1 | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|
| Task | 0.647059 | 0.500000 | 0.564103 | 22 | 12 | 22 | 24 |
| Context | 0.404255 | 0.527778 | 0.457831 | 19 | 28 | 17 | 16 |

Exact match requires both routing decisions to agree with the expected two-bit label. Task-only: 16/32; context-only: 12/24; mixed: 0/12; irrelevant: 0/12. Routing changes within 17/20 paraphrase families. First-person preference phrasing often flips task content to Context; all 12 irrelevant cases route to Context. Wording shortcuts are a plausible explanation, not a proven causal mechanism. Current status: available for experimental integration, but unreliable for conversational routing.

Evidence: [source validation and smoke results](TASK_CONTEXT_CLASSIFIER_RESULTS.md), [paraphrase report](../reports/task_context/paraphrase_v1/REPORT.md), [machine-readable metrics](../reports/task_context/paraphrase_v1/summary.json), [per-clause scores](../reports/task_context/paraphrase_v1/scores.csv). Earlier occupation-pilot Activity/Job-nature scores use different labels/data/checkpoints and are historical, not this classifier's current performance.

## How to assess the later extension

Preserve the existing configs, reports, checkpoint and input/label fingerprints. Evaluate a separately named extension against the exact matching pre-extension variant: dot-product-only against dot-product-only, or the combined semantic variant against that same combined variant. Compare all ten ranking metrics on identical profiles, labels, eligible occupations and candidate depths; report paired per-profile gains/losses and uncertainty, not only a headline Hit@5 change. Retain raw per-occupation and per-clause traces.

Report classifier metrics separately from ranking metrics: higher classifier F1 does not by itself show improved occupation relevance. Record the checkpoint hash, splitting/normalisation revision, thresholds, polarity method, matching and fusion settings. Include classification-only (ranking unchanged), task-only, context-only and combined ablations to locate any gains or regressions. Use independently reviewed conversational content and preference labels to assess routing and polarity; use task/context-sensitive occupation relevance labels for the extension. Existing synthetic RIASEC-derived targets alone cannot independently establish that task/context matching improved user-intent alignment. Any changed benchmark must have a new version and matching fresh control; do not compare unmatched scores directly with this table.
