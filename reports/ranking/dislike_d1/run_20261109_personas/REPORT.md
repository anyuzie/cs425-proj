# Fusion dialogue evaluation

Primary metric: mrr of first acceptable occupation (§12.4).

| Arm | Group | n | MRR | Hit@5 | Hit@10 | NDCG@10 |
|---|---|---|---|---|---|---|
| semantic_probing | all | 25 | 0.758 | 0.880 | 0.880 | 0.643 |
| semantic_probing | conflict | 14 | 0.881 | 1.000 | 1.000 | 0.827 |
| semantic_probing | agree | 11 | 0.603 | 0.727 | 0.727 | 0.408 |
| fusion_zscore_probing | all | 25 | 0.680 | 0.800 | 0.920 | 0.539 |
| fusion_zscore_probing | conflict | 14 | 0.639 | 0.714 | 0.857 | 0.535 |
| fusion_zscore_probing | agree | 11 | 0.733 | 0.909 | 1.000 | 0.544 |
| fusion_zscore_bm25_ignore_v2_probing | all | 25 | 0.683 | 0.920 | 0.960 | 0.610 |
| fusion_zscore_bm25_ignore_v2_probing | conflict | 14 | 0.780 | 1.000 | 1.000 | 0.714 |
| fusion_zscore_bm25_ignore_v2_probing | agree | 11 | 0.560 | 0.818 | 0.909 | 0.477 |
| semantic_signed_probing | all | 25 | 0.855 | 0.960 | 0.960 | 0.718 |
| semantic_signed_probing | conflict | 14 | 0.881 | 1.000 | 1.000 | 0.823 |
| semantic_signed_probing | agree | 11 | 0.821 | 0.909 | 0.909 | 0.585 |
| fusion_zscore_d1_probing | all | 25 | 0.731 | 0.800 | 1.000 | 0.598 |
| fusion_zscore_d1_probing | conflict | 14 | 0.619 | 0.714 | 1.000 | 0.546 |
| fusion_zscore_d1_probing | agree | 11 | 0.874 | 0.909 | 1.000 | 0.663 |
| fusion_zscore_d1_bm25_ignore_v2_probing | all | 25 | 0.770 | 0.960 | 0.960 | 0.636 |
| fusion_zscore_d1_bm25_ignore_v2_probing | conflict | 14 | 0.780 | 1.000 | 1.000 | 0.690 |
| fusion_zscore_d1_bm25_ignore_v2_probing | agree | 11 | 0.758 | 0.909 | 0.909 | 0.568 |

## Paired differences (MRR, 95% bootstrap CI)

| Contrast | Group | Mean | CI |
|---|---|---|---|
| fusion_zscore_probing_minus_semantic_probing | all | -0.078 | [-0.218, +0.047] |
| fusion_zscore_probing_minus_semantic_probing | conflict | -0.242 | [-0.426, -0.084] |
| fusion_zscore_probing_minus_semantic_probing | agree | +0.130 | [+0.030, +0.249] |
| fusion_zscore_bm25_ignore_v2_probing_minus_fusion_zscore_probing | all | +0.003 | [-0.133, +0.142] |
| fusion_zscore_bm25_ignore_v2_probing_minus_fusion_zscore_probing | conflict | +0.141 | [-0.019, +0.311] |
| fusion_zscore_bm25_ignore_v2_probing_minus_fusion_zscore_probing | agree | -0.173 | [-0.378, +0.000] |
| semantic_signed_probing_minus_semantic_probing | all | +0.096 | [+0.018, +0.194] |
| semantic_signed_probing_minus_semantic_probing | conflict | +0.000 | [+0.000, +0.000] |
| semantic_signed_probing_minus_semantic_probing | agree | +0.219 | [+0.045, +0.428] |
| fusion_zscore_d1_probing_minus_fusion_zscore_probing | all | +0.050 | [-0.030, +0.142] |
| fusion_zscore_d1_probing_minus_fusion_zscore_probing | conflict | -0.021 | [-0.111, +0.048] |
| fusion_zscore_d1_probing_minus_fusion_zscore_probing | agree | +0.141 | [+0.000, +0.310] |
| fusion_zscore_d1_bm25_ignore_v2_probing_minus_fusion_zscore_bm25_ignore_v2_probing | all | +0.087 | [+0.017, +0.174] |
| fusion_zscore_d1_bm25_ignore_v2_probing_minus_fusion_zscore_bm25_ignore_v2_probing | conflict | +0.000 | [+0.000, +0.000] |
| fusion_zscore_d1_bm25_ignore_v2_probing_minus_fusion_zscore_bm25_ignore_v2_probing | agree | +0.198 | [+0.054, +0.364] |
| fusion_zscore_d1_bm25_ignore_v2_probing_minus_fusion_zscore_d1_probing | all | +0.039 | [-0.098, +0.178] |
| fusion_zscore_d1_bm25_ignore_v2_probing_minus_fusion_zscore_d1_probing | conflict | +0.161 | [-0.007, +0.339] |
| fusion_zscore_d1_bm25_ignore_v2_probing_minus_fusion_zscore_d1_probing | agree | -0.116 | [-0.318, +0.086] |

## Per persona (best acceptable rank, final turn)

| Persona | Field | Conflict | semantic_probing | fusion_zscore_probing | fusion_zscore_bm25_ignore_v2_probing | semantic_signed_probing | fusion_zscore_d1_probing | fusion_zscore_d1_bm25_ignore_v2_probing |
|---|---|---|---|---|---|---|---|---|
| F01 | music | True | 1 | 1 | 1 | 1 | 1 | 1 |
| F02 | veterinary | True | 1 | 1 | 1 | 1 | 1 | 1 |
| F03 | history_heritage | True | 1 | 1 | 2 | 1 | 1 | 2 |
| F04 | law | True | 1 | 1 | 1 | 1 | 1 | 1 |
| F05 | video_games | True | 1 | 1 | 1 | 1 | 1 | 1 |
| F06 | climate_policy | True | 1 | 1 | 1 | 1 | 1 | 1 |
| F07 | surgery | True | 2 | 5 | 3 | 2 | 8 | 3 |
| F08 | food | True | 2 | 7 | 1 | 2 | 7 | 1 |
| F09 | journalism | True | 1 | 1 | 1 | 1 | 2 | 1 |
| F10 | architecture | True | 1 | 1 | 1 | 1 | 1 | 1 |
| F11 | aviation | True | 1 | 9 | 2 | 1 | 8 | 2 |
| F12 | religion | True | 1 | 3 | 1 | 1 | 3 | 1 |
| F13 | finance | True | 3 | 13 | 4 | 3 | 10 | 4 |
| F14 | sport | True | 1 | 12 | 3 | 1 | 3 | 3 |
| F15 | nursing | False | 2 | 1 | 1 | 1 | 1 | 1 |
| F16 | software | False | 1 | 1 | 2 | 1 | 1 | 1 |
| F17 | counselling | False | 1 | 1 | 1 | 1 | 1 | 1 |
| F18 | accounting | False | 33 | 5 | 13 | 31 | 9 | 12 |
| F19 | marketing | False | 2 | 1 | 1 | 1 | 1 | 1 |
| F20 | primary_teaching | False | 2 | 2 | 2 | 2 | 1 | 2 |
| F21 | geology | False | 19 | 4 | 4 | 1 | 1 | 2 |
| F22 | graphic_design | False | 1 | 1 | 3 | 1 | 1 | 1 |
| F23 | robotics | False | 1 | 1 | 1 | 1 | 1 | 1 |
| F24 | statistics | False | 1 | 1 | 6 | 1 | 1 | 4 |
| F25 | sales_property | False | 22 | 9 | 3 | 2 | 2 | 1 |
