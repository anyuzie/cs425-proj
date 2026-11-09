# Fusion dialogue evaluation

Primary metric: mrr of first acceptable occupation (§12.4).

| Arm | Group | n | MRR | Hit@5 | Hit@10 | NDCG@10 |
|---|---|---|---|---|---|---|
| fusion_zscore_probing | all | 25 | 0.680 | 0.800 | 0.920 | 0.539 |
| fusion_zscore_probing | conflict | 14 | 0.639 | 0.714 | 0.857 | 0.535 |
| fusion_zscore_probing | agree | 11 | 0.733 | 0.909 | 1.000 | 0.544 |
| fusion_zscore_bm25_ignore_probing | all | 25 | 0.683 | 0.920 | 0.960 | 0.602 |
| fusion_zscore_bm25_ignore_probing | conflict | 14 | 0.780 | 1.000 | 1.000 | 0.701 |
| fusion_zscore_bm25_ignore_probing | agree | 11 | 0.560 | 0.818 | 0.909 | 0.477 |
| fusion_zscore_bm25_filter_probing | all | 25 | 0.685 | 0.960 | 0.960 | 0.602 |
| fusion_zscore_bm25_filter_probing | conflict | 14 | 0.780 | 1.000 | 1.000 | 0.700 |
| fusion_zscore_bm25_filter_probing | agree | 11 | 0.564 | 0.909 | 0.909 | 0.477 |
| fusion_zscore_bm25_ignore_v2_probing | all | 25 | 0.683 | 0.920 | 0.960 | 0.610 |
| fusion_zscore_bm25_ignore_v2_probing | conflict | 14 | 0.780 | 1.000 | 1.000 | 0.714 |
| fusion_zscore_bm25_ignore_v2_probing | agree | 11 | 0.560 | 0.818 | 0.909 | 0.477 |
| fusion_zscore_bm25_filter_v2_probing | all | 25 | 0.685 | 0.960 | 0.960 | 0.609 |
| fusion_zscore_bm25_filter_v2_probing | conflict | 14 | 0.780 | 1.000 | 1.000 | 0.713 |
| fusion_zscore_bm25_filter_v2_probing | agree | 11 | 0.564 | 0.909 | 0.909 | 0.477 |

## Paired differences (MRR, 95% bootstrap CI)

| Contrast | Group | Mean | CI |
|---|---|---|---|
| fusion_zscore_bm25_ignore_probing_minus_fusion_zscore_probing | all | +0.003 | [-0.133, +0.142] |
| fusion_zscore_bm25_ignore_probing_minus_fusion_zscore_probing | conflict | +0.141 | [-0.019, +0.311] |
| fusion_zscore_bm25_ignore_probing_minus_fusion_zscore_probing | agree | -0.173 | [-0.378, +0.000] |
| fusion_zscore_bm25_filter_probing_minus_fusion_zscore_probing | all | +0.005 | [-0.130, +0.143] |
| fusion_zscore_bm25_filter_probing_minus_fusion_zscore_probing | conflict | +0.141 | [-0.019, +0.311] |
| fusion_zscore_bm25_filter_probing_minus_fusion_zscore_probing | agree | -0.169 | [-0.368, +0.000] |
| fusion_zscore_bm25_filter_probing_minus_fusion_zscore_bm25_ignore_probing | all | +0.002 | [+0.000, +0.005] |
| fusion_zscore_bm25_filter_probing_minus_fusion_zscore_bm25_ignore_probing | conflict | +0.000 | [+0.000, +0.000] |
| fusion_zscore_bm25_filter_probing_minus_fusion_zscore_bm25_ignore_probing | agree | +0.004 | [+0.000, +0.012] |
| fusion_zscore_bm25_ignore_v2_probing_minus_fusion_zscore_bm25_ignore_probing | all | +0.000 | [+0.000, +0.000] |
| fusion_zscore_bm25_ignore_v2_probing_minus_fusion_zscore_bm25_ignore_probing | conflict | +0.000 | [+0.000, +0.000] |
| fusion_zscore_bm25_ignore_v2_probing_minus_fusion_zscore_bm25_ignore_probing | agree | +0.000 | [+0.000, +0.000] |
| fusion_zscore_bm25_ignore_v2_probing_minus_fusion_zscore_probing | all | +0.003 | [-0.133, +0.142] |
| fusion_zscore_bm25_ignore_v2_probing_minus_fusion_zscore_probing | conflict | +0.141 | [-0.019, +0.311] |
| fusion_zscore_bm25_ignore_v2_probing_minus_fusion_zscore_probing | agree | -0.173 | [-0.378, +0.000] |
| fusion_zscore_bm25_filter_v2_probing_minus_fusion_zscore_bm25_filter_probing | all | +0.000 | [+0.000, +0.000] |
| fusion_zscore_bm25_filter_v2_probing_minus_fusion_zscore_bm25_filter_probing | conflict | +0.000 | [+0.000, +0.000] |
| fusion_zscore_bm25_filter_v2_probing_minus_fusion_zscore_bm25_filter_probing | agree | +0.000 | [+0.000, +0.000] |
| fusion_zscore_bm25_filter_v2_probing_minus_fusion_zscore_probing | all | +0.005 | [-0.130, +0.143] |
| fusion_zscore_bm25_filter_v2_probing_minus_fusion_zscore_probing | conflict | +0.141 | [-0.019, +0.311] |
| fusion_zscore_bm25_filter_v2_probing_minus_fusion_zscore_probing | agree | -0.169 | [-0.368, +0.000] |
| fusion_zscore_bm25_filter_v2_probing_minus_fusion_zscore_bm25_ignore_v2_probing | all | +0.002 | [+0.000, +0.005] |
| fusion_zscore_bm25_filter_v2_probing_minus_fusion_zscore_bm25_ignore_v2_probing | conflict | +0.000 | [+0.000, +0.000] |
| fusion_zscore_bm25_filter_v2_probing_minus_fusion_zscore_bm25_ignore_v2_probing | agree | +0.004 | [+0.000, +0.012] |

## Per persona (best acceptable rank, final turn)

| Persona | Field | Conflict | fusion_zscore_probing | fusion_zscore_bm25_ignore_probing | fusion_zscore_bm25_filter_probing | fusion_zscore_bm25_ignore_v2_probing | fusion_zscore_bm25_filter_v2_probing |
|---|---|---|---|---|---|---|---|
| F01 | music | True | 1 | 1 | 1 | 1 | 1 |
| F02 | veterinary | True | 1 | 1 | 1 | 1 | 1 |
| F03 | history_heritage | True | 1 | 2 | 2 | 2 | 2 |
| F04 | law | True | 1 | 1 | 1 | 1 | 1 |
| F05 | video_games | True | 1 | 1 | 1 | 1 | 1 |
| F06 | climate_policy | True | 1 | 1 | 1 | 1 | 1 |
| F07 | surgery | True | 5 | 3 | 3 | 3 | 3 |
| F08 | food | True | 7 | 1 | 1 | 1 | 1 |
| F09 | journalism | True | 1 | 1 | 1 | 1 | 1 |
| F10 | architecture | True | 1 | 1 | 1 | 1 | 1 |
| F11 | aviation | True | 9 | 2 | 2 | 2 | 2 |
| F12 | religion | True | 3 | 1 | 1 | 1 | 1 |
| F13 | finance | True | 13 | 4 | 4 | 4 | 4 |
| F14 | sport | True | 12 | 3 | 3 | 3 | 3 |
| F15 | nursing | False | 1 | 1 | 1 | 1 | 1 |
| F16 | software | False | 1 | 2 | 2 | 2 | 2 |
| F17 | counselling | False | 1 | 1 | 1 | 1 | 1 |
| F18 | accounting | False | 5 | 13 | 11 | 13 | 11 |
| F19 | marketing | False | 1 | 1 | 1 | 1 | 1 |
| F20 | primary_teaching | False | 2 | 2 | 2 | 2 | 2 |
| F21 | geology | False | 4 | 4 | 4 | 4 | 4 |
| F22 | graphic_design | False | 1 | 3 | 3 | 3 | 3 |
| F23 | robotics | False | 1 | 1 | 1 | 1 | 1 |
| F24 | statistics | False | 1 | 6 | 5 | 6 | 5 |
| F25 | sales_property | False | 9 | 3 | 3 | 3 | 3 |
