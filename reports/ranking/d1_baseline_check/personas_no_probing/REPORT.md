# Fusion dialogue evaluation

Primary metric: mrr of first acceptable occupation (§12.4).

| Arm | Group | n | MRR | Hit@5 | Hit@10 | NDCG@10 |
|---|---|---|---|---|---|---|
| fusion_zscore_no_probing | all | 25 | 0.653 | 0.800 | 0.920 | 0.538 |
| fusion_zscore_no_probing | conflict | 14 | 0.666 | 0.786 | 0.857 | 0.564 |
| fusion_zscore_no_probing | agree | 11 | 0.637 | 0.818 | 1.000 | 0.507 |
| fusion_zscore_bm25_ignore_v2_no_probing | all | 25 | 0.684 | 0.920 | 1.000 | 0.619 |
| fusion_zscore_bm25_ignore_v2_no_probing | conflict | 14 | 0.815 | 1.000 | 1.000 | 0.741 |
| fusion_zscore_bm25_ignore_v2_no_probing | agree | 11 | 0.518 | 0.818 | 1.000 | 0.462 |
| fusion_zscore_d1_no_probing | all | 25 | 0.752 | 0.840 | 0.960 | 0.593 |
| fusion_zscore_d1_no_probing | conflict | 14 | 0.628 | 0.714 | 0.929 | 0.524 |
| fusion_zscore_d1_no_probing | agree | 11 | 0.909 | 1.000 | 1.000 | 0.680 |
| fusion_zscore_d1_bm25_ignore_v2_no_probing | all | 25 | 0.777 | 1.000 | 1.000 | 0.659 |
| fusion_zscore_d1_bm25_ignore_v2_no_probing | conflict | 14 | 0.815 | 1.000 | 1.000 | 0.728 |
| fusion_zscore_d1_bm25_ignore_v2_no_probing | agree | 11 | 0.727 | 1.000 | 1.000 | 0.572 |

## Paired differences (MRR, 95% bootstrap CI)

| Contrast | Group | Mean | CI |
|---|---|---|---|
| fusion_zscore_bm25_ignore_v2_no_probing_minus_fusion_zscore_no_probing | all | +0.031 | [-0.105, +0.172] |
| fusion_zscore_bm25_ignore_v2_no_probing_minus_fusion_zscore_no_probing | conflict | +0.149 | [-0.014, +0.323] |
| fusion_zscore_bm25_ignore_v2_no_probing_minus_fusion_zscore_no_probing | agree | -0.119 | [-0.316, +0.073] |
| fusion_zscore_d1_no_probing_minus_fusion_zscore_no_probing | all | +0.098 | [+0.002, +0.201] |
| fusion_zscore_d1_no_probing_minus_fusion_zscore_no_probing | conflict | -0.039 | [-0.113, +0.002] |
| fusion_zscore_d1_no_probing_minus_fusion_zscore_no_probing | agree | +0.272 | [+0.114, +0.432] |
| fusion_zscore_d1_bm25_ignore_v2_no_probing_minus_fusion_zscore_bm25_ignore_v2_no_probing | all | +0.092 | [+0.022, +0.179] |
| fusion_zscore_d1_bm25_ignore_v2_no_probing_minus_fusion_zscore_bm25_ignore_v2_no_probing | conflict | +0.000 | [+0.000, +0.000] |
| fusion_zscore_d1_bm25_ignore_v2_no_probing_minus_fusion_zscore_bm25_ignore_v2_no_probing | agree | +0.210 | [+0.068, +0.371] |
| fusion_zscore_d1_bm25_ignore_v2_no_probing_minus_fusion_zscore_d1_no_probing | all | +0.025 | [-0.122, +0.176] |
| fusion_zscore_d1_bm25_ignore_v2_no_probing_minus_fusion_zscore_d1_no_probing | conflict | +0.188 | [+0.017, +0.364] |
| fusion_zscore_d1_bm25_ignore_v2_no_probing_minus_fusion_zscore_d1_no_probing | agree | -0.182 | [-0.386, +0.023] |

## Per persona (best acceptable rank, final turn)

| Persona | Field | Conflict | fusion_zscore_no_probing | fusion_zscore_bm25_ignore_v2_no_probing | fusion_zscore_d1_no_probing | fusion_zscore_d1_bm25_ignore_v2_no_probing |
|---|---|---|---|---|---|---|
| F01 | music | True | 1 | 1 | 1 | 1 |
| F02 | veterinary | True | 1 | 1 | 1 | 1 |
| F03 | history_heritage | True | 1 | 2 | 1 | 2 |
| F04 | law | True | 1 | 1 | 1 | 1 |
| F05 | video_games | True | 1 | 1 | 1 | 1 |
| F06 | climate_policy | True | 1 | 1 | 1 | 1 |
| F07 | surgery | True | 5 | 3 | 6 | 3 |
| F08 | food | True | 7 | 1 | 7 | 1 |
| F09 | journalism | True | 1 | 1 | 2 | 1 |
| F10 | architecture | True | 1 | 1 | 1 | 1 |
| F11 | aviation | True | 13 | 3 | 21 | 3 |
| F12 | religion | True | 3 | 1 | 3 | 1 |
| F13 | finance | True | 13 | 4 | 10 | 4 |
| F14 | sport | True | 2 | 1 | 2 | 1 |
| F15 | nursing | False | 1 | 2 | 1 | 2 |
| F16 | software | False | 1 | 2 | 1 | 1 |
| F17 | counselling | False | 2 | 1 | 1 | 1 |
| F18 | accounting | False | 7 | 9 | 2 | 4 |
| F19 | marketing | False | 1 | 1 | 1 | 1 |
| F20 | primary_teaching | False | 2 | 2 | 1 | 2 |
| F21 | geology | False | 4 | 4 | 1 | 2 |
| F22 | graphic_design | False | 1 | 3 | 1 | 1 |
| F23 | robotics | False | 1 | 1 | 1 | 1 |
| F24 | statistics | False | 2 | 6 | 1 | 4 |
| F25 | sales_property | False | 9 | 3 | 2 | 1 |
