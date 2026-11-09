# Fusion dialogue evaluation

Primary metric: mrr of first acceptable occupation (§12.4).

| Arm | Group | n | MRR | Hit@5 | Hit@10 | NDCG@10 |
|---|---|---|---|---|---|---|
| fusion_zscore_probing | all | 25 | 0.680 | 0.800 | 0.920 | 0.539 |
| fusion_zscore_probing | conflict | 14 | 0.639 | 0.714 | 0.857 | 0.535 |
| fusion_zscore_probing | agree | 11 | 0.733 | 0.909 | 1.000 | 0.544 |
| fusion_zscore_no_probing | all | 25 | 0.653 | 0.800 | 0.920 | 0.538 |
| fusion_zscore_no_probing | conflict | 14 | 0.666 | 0.786 | 0.857 | 0.564 |
| fusion_zscore_no_probing | agree | 11 | 0.637 | 0.818 | 1.000 | 0.507 |
| d2_title_v3_nolex_probing | all | 25 | 0.728 | 0.800 | 0.960 | 0.591 |
| d2_title_v3_nolex_probing | conflict | 14 | 0.613 | 0.714 | 0.929 | 0.535 |
| d2_title_v3_nolex_probing | agree | 11 | 0.874 | 0.909 | 1.000 | 0.662 |
| d2_title_v3_nolex_no_probing | all | 25 | 0.749 | 0.840 | 0.920 | 0.586 |
| d2_title_v3_nolex_no_probing | conflict | 14 | 0.622 | 0.714 | 0.857 | 0.513 |
| d2_title_v3_nolex_no_probing | agree | 11 | 0.909 | 1.000 | 1.000 | 0.680 |
| routing_v3_d2_title_v3_probing | all | 25 | 0.770 | 0.960 | 0.960 | 0.620 |
| routing_v3_d2_title_v3_probing | conflict | 14 | 0.780 | 1.000 | 1.000 | 0.661 |
| routing_v3_d2_title_v3_probing | agree | 11 | 0.758 | 0.909 | 0.909 | 0.567 |
| routing_v3_d2_title_v3_no_probing | all | 25 | 0.777 | 1.000 | 1.000 | 0.643 |
| routing_v3_d2_title_v3_no_probing | conflict | 14 | 0.815 | 1.000 | 1.000 | 0.700 |
| routing_v3_d2_title_v3_no_probing | agree | 11 | 0.727 | 1.000 | 1.000 | 0.570 |

## Paired differences (MRR, 95% bootstrap CI)

| Contrast | Group | Mean | CI |
|---|---|---|---|
| routing_v3_d2_title_v3_probing_minus_d2_title_v3_nolex_probing | all | +0.042 | [-0.096, +0.185] |
| routing_v3_d2_title_v3_probing_minus_d2_title_v3_nolex_probing | conflict | +0.167 | [-0.005, +0.350] |
| routing_v3_d2_title_v3_probing_minus_d2_title_v3_nolex_probing | agree | -0.116 | [-0.318, +0.086] |
| routing_v3_d2_title_v3_no_probing_minus_d2_title_v3_nolex_no_probing | all | +0.028 | [-0.120, +0.183] |
| routing_v3_d2_title_v3_no_probing_minus_d2_title_v3_nolex_no_probing | conflict | +0.193 | [+0.018, +0.375] |
| routing_v3_d2_title_v3_no_probing_minus_d2_title_v3_nolex_no_probing | agree | -0.182 | [-0.386, +0.023] |
| d2_title_v3_nolex_probing_minus_fusion_zscore_probing | all | +0.047 | [-0.034, +0.139] |
| d2_title_v3_nolex_probing_minus_fusion_zscore_probing | conflict | -0.026 | [-0.116, +0.042] |
| d2_title_v3_nolex_probing_minus_fusion_zscore_probing | agree | +0.141 | [+0.000, +0.310] |
| d2_title_v3_nolex_no_probing_minus_fusion_zscore_no_probing | all | +0.095 | [-0.002, +0.198] |
| d2_title_v3_nolex_no_probing_minus_fusion_zscore_no_probing | conflict | -0.044 | [-0.121, -0.000] |
| d2_title_v3_nolex_no_probing_minus_fusion_zscore_no_probing | agree | +0.272 | [+0.114, +0.432] |
| fusion_zscore_probing_minus_fusion_zscore_no_probing | all | +0.027 | [-0.031, +0.090] |
| fusion_zscore_probing_minus_fusion_zscore_no_probing | conflict | -0.027 | [-0.089, +0.007] |
| fusion_zscore_probing_minus_fusion_zscore_no_probing | agree | +0.096 | [+0.000, +0.227] |
| d2_title_v3_nolex_probing_minus_d2_title_v3_nolex_no_probing | all | -0.021 | [-0.059, +0.005] |
| d2_title_v3_nolex_probing_minus_d2_title_v3_nolex_no_probing | conflict | -0.009 | [-0.039, +0.014] |
| d2_title_v3_nolex_probing_minus_d2_title_v3_nolex_no_probing | agree | -0.035 | [-0.106, +0.000] |
| routing_v3_d2_title_v3_probing_minus_routing_v3_d2_title_v3_no_probing | all | -0.007 | [-0.080, +0.060] |
| routing_v3_d2_title_v3_probing_minus_routing_v3_d2_title_v3_no_probing | conflict | -0.036 | [-0.143, +0.036] |
| routing_v3_d2_title_v3_probing_minus_routing_v3_d2_title_v3_no_probing | agree | +0.030 | [-0.045, +0.136] |

## Per persona (best acceptable rank, final turn)

| Persona | Field | Conflict | fusion_zscore_probing | fusion_zscore_no_probing | d2_title_v3_nolex_probing | d2_title_v3_nolex_no_probing | routing_v3_d2_title_v3_probing | routing_v3_d2_title_v3_no_probing |
|---|---|---|---|---|---|---|---|---|
| F01 | music | True | 1 | 1 | 1 | 1 | 1 | 1 |
| F02 | veterinary | True | 1 | 1 | 1 | 1 | 1 | 1 |
| F03 | history_heritage | True | 1 | 1 | 1 | 1 | 2 | 2 |
| F04 | law | True | 1 | 1 | 1 | 1 | 1 | 1 |
| F05 | video_games | True | 1 | 1 | 1 | 1 | 1 | 1 |
| F06 | climate_policy | True | 1 | 1 | 1 | 1 | 1 | 1 |
| F07 | surgery | True | 5 | 5 | 8 | 6 | 3 | 3 |
| F08 | food | True | 7 | 7 | 15 | 15 | 1 | 1 |
| F09 | journalism | True | 1 | 1 | 2 | 2 | 1 | 1 |
| F10 | architecture | True | 1 | 1 | 1 | 1 | 1 | 1 |
| F11 | aviation | True | 9 | 13 | 8 | 21 | 2 | 3 |
| F12 | religion | True | 3 | 3 | 3 | 3 | 1 | 1 |
| F13 | finance | True | 13 | 13 | 10 | 10 | 4 | 4 |
| F14 | sport | True | 12 | 2 | 3 | 2 | 3 | 1 |
| F15 | nursing | False | 1 | 1 | 1 | 1 | 1 | 2 |
| F16 | software | False | 1 | 1 | 1 | 1 | 1 | 1 |
| F17 | counselling | False | 1 | 2 | 1 | 1 | 1 | 1 |
| F18 | accounting | False | 5 | 7 | 9 | 2 | 12 | 4 |
| F19 | marketing | False | 1 | 1 | 1 | 1 | 1 | 1 |
| F20 | primary_teaching | False | 2 | 2 | 1 | 1 | 2 | 2 |
| F21 | geology | False | 4 | 4 | 1 | 1 | 2 | 2 |
| F22 | graphic_design | False | 1 | 1 | 1 | 1 | 1 | 1 |
| F23 | robotics | False | 1 | 1 | 1 | 1 | 1 | 1 |
| F24 | statistics | False | 1 | 2 | 1 | 1 | 4 | 4 |
| F25 | sales_property | False | 9 | 9 | 2 | 2 | 1 | 1 |
