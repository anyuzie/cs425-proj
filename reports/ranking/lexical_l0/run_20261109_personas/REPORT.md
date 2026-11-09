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
| lexical_probing | all | 25 | 0.518 | 0.600 | 0.680 | 0.338 |
| lexical_probing | conflict | 14 | 0.669 | 0.786 | 0.857 | 0.482 |
| lexical_probing | agree | 11 | 0.326 | 0.364 | 0.455 | 0.154 |
| fusion_zscore_bm25_probing | all | 25 | 0.658 | 0.920 | 1.000 | 0.599 |
| fusion_zscore_bm25_probing | conflict | 14 | 0.744 | 1.000 | 1.000 | 0.700 |
| fusion_zscore_bm25_probing | agree | 11 | 0.548 | 0.818 | 1.000 | 0.470 |

## Paired differences (MRR, 95% bootstrap CI)

| Contrast | Group | Mean | CI |
|---|---|---|---|
| fusion_zscore_probing_minus_semantic_probing | all | -0.078 | [-0.218, +0.047] |
| fusion_zscore_probing_minus_semantic_probing | conflict | -0.242 | [-0.426, -0.084] |
| fusion_zscore_probing_minus_semantic_probing | agree | +0.130 | [+0.030, +0.249] |
| fusion_zscore_bm25_probing_minus_fusion_zscore_probing | all | -0.023 | [-0.163, +0.121] |
| fusion_zscore_bm25_probing_minus_fusion_zscore_probing | conflict | +0.105 | [-0.078, +0.293] |
| fusion_zscore_bm25_probing_minus_fusion_zscore_probing | agree | -0.185 | [-0.386, -0.013] |
| fusion_zscore_bm25_probing_minus_semantic_probing | all | -0.101 | [-0.242, +0.038] |
| fusion_zscore_bm25_probing_minus_semantic_probing | conflict | -0.137 | [-0.292, +0.018] |
| fusion_zscore_bm25_probing_minus_semantic_probing | agree | -0.055 | [-0.320, +0.181] |

## Per persona (best acceptable rank, final turn)

| Persona | Field | Conflict | semantic_probing | fusion_zscore_probing | lexical_probing | fusion_zscore_bm25_probing |
|---|---|---|---|---|---|---|
| F01 | music | True | 1 | 1 | 1 | 1 |
| F02 | veterinary | True | 1 | 1 | 1 | 1 |
| F03 | history_heritage | True | 1 | 1 | 13 | 2 |
| F04 | law | True | 1 | 1 | 1 | 1 |
| F05 | video_games | True | 1 | 1 | 1 | 1 |
| F06 | climate_policy | True | 1 | 1 | 4 | 2 |
| F07 | surgery | True | 2 | 5 | 11 | 3 |
| F08 | food | True | 2 | 7 | 3 | 1 |
| F09 | journalism | True | 1 | 1 | 1 | 1 |
| F10 | architecture | True | 1 | 1 | 1 | 1 |
| F11 | aviation | True | 1 | 9 | 2 | 2 |
| F12 | religion | True | 1 | 3 | 1 | 1 |
| F13 | finance | True | 3 | 13 | 1 | 4 |
| F14 | sport | True | 1 | 12 | 9 | 3 |
| F15 | nursing | False | 2 | 1 | 9 | 1 |
| F16 | software | False | 1 | 1 | 23 | 2 |
| F17 | counselling | False | 1 | 1 | 1 | 1 |
| F18 | accounting | False | 33 | 5 | 37 | 10 |
| F19 | marketing | False | 2 | 1 | 1 | 1 |
| F20 | primary_teaching | False | 2 | 2 | 4 | 2 |
| F21 | geology | False | 19 | 4 | 20 | 4 |
| F22 | graphic_design | False | 1 | 1 | 28 | 3 |
| F23 | robotics | False | 1 | 1 | 1 | 1 |
| F24 | statistics | False | 1 | 1 | 173 | 7 |
| F25 | sales_property | False | 22 | 9 | 15 | 5 |
