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
| d2_title_v3_plain_semantic_probing | all | 25 | 0.680 | 0.800 | 0.920 | 0.551 |
| d2_title_v3_plain_semantic_probing | conflict | 14 | 0.639 | 0.714 | 0.857 | 0.557 |
| d2_title_v3_plain_semantic_probing | agree | 11 | 0.733 | 0.909 | 1.000 | 0.544 |
| d2_title_v3_plain_semantic_no_probing | all | 25 | 0.653 | 0.800 | 0.920 | 0.548 |
| d2_title_v3_plain_semantic_no_probing | conflict | 14 | 0.666 | 0.786 | 0.857 | 0.581 |
| d2_title_v3_plain_semantic_no_probing | agree | 11 | 0.637 | 0.818 | 1.000 | 0.507 |

## Paired differences (MRR, 95% bootstrap CI)

| Contrast | Group | Mean | CI |
|---|---|---|---|
| d2_title_v3_plain_semantic_probing_minus_fusion_zscore_probing | all | +0.000 | [+0.000, +0.000] |
| d2_title_v3_plain_semantic_probing_minus_fusion_zscore_probing | conflict | +0.000 | [+0.000, +0.000] |
| d2_title_v3_plain_semantic_probing_minus_fusion_zscore_probing | agree | +0.000 | [+0.000, +0.000] |
| d2_title_v3_plain_semantic_no_probing_minus_fusion_zscore_no_probing | all | +0.000 | [+0.000, +0.000] |
| d2_title_v3_plain_semantic_no_probing_minus_fusion_zscore_no_probing | conflict | +0.000 | [+0.000, +0.000] |
| d2_title_v3_plain_semantic_no_probing_minus_fusion_zscore_no_probing | agree | +0.000 | [+0.000, +0.000] |
| fusion_zscore_probing_minus_fusion_zscore_no_probing | all | +0.027 | [-0.031, +0.090] |
| fusion_zscore_probing_minus_fusion_zscore_no_probing | conflict | -0.027 | [-0.089, +0.007] |
| fusion_zscore_probing_minus_fusion_zscore_no_probing | agree | +0.096 | [+0.000, +0.227] |
| d2_title_v3_plain_semantic_probing_minus_d2_title_v3_plain_semantic_no_probing | all | +0.027 | [-0.031, +0.090] |
| d2_title_v3_plain_semantic_probing_minus_d2_title_v3_plain_semantic_no_probing | conflict | -0.027 | [-0.089, +0.007] |
| d2_title_v3_plain_semantic_probing_minus_d2_title_v3_plain_semantic_no_probing | agree | +0.096 | [+0.000, +0.227] |

## Per persona (best acceptable rank, final turn)

| Persona | Field | Conflict | fusion_zscore_probing | fusion_zscore_no_probing | d2_title_v3_plain_semantic_probing | d2_title_v3_plain_semantic_no_probing |
|---|---|---|---|---|---|---|
| F01 | music | True | 1 | 1 | 1 | 1 |
| F02 | veterinary | True | 1 | 1 | 1 | 1 |
| F03 | history_heritage | True | 1 | 1 | 1 | 1 |
| F04 | law | True | 1 | 1 | 1 | 1 |
| F05 | video_games | True | 1 | 1 | 1 | 1 |
| F06 | climate_policy | True | 1 | 1 | 1 | 1 |
| F07 | surgery | True | 5 | 5 | 5 | 5 |
| F08 | food | True | 7 | 7 | 7 | 7 |
| F09 | journalism | True | 1 | 1 | 1 | 1 |
| F10 | architecture | True | 1 | 1 | 1 | 1 |
| F11 | aviation | True | 9 | 13 | 9 | 13 |
| F12 | religion | True | 3 | 3 | 3 | 3 |
| F13 | finance | True | 13 | 13 | 13 | 13 |
| F14 | sport | True | 12 | 2 | 12 | 2 |
| F15 | nursing | False | 1 | 1 | 1 | 1 |
| F16 | software | False | 1 | 1 | 1 | 1 |
| F17 | counselling | False | 1 | 2 | 1 | 2 |
| F18 | accounting | False | 5 | 7 | 5 | 7 |
| F19 | marketing | False | 1 | 1 | 1 | 1 |
| F20 | primary_teaching | False | 2 | 2 | 2 | 2 |
| F21 | geology | False | 4 | 4 | 4 | 4 |
| F22 | graphic_design | False | 1 | 1 | 1 | 1 |
| F23 | robotics | False | 1 | 1 | 1 | 1 |
| F24 | statistics | False | 1 | 2 | 1 | 2 |
| F25 | sales_property | False | 9 | 9 | 9 | 9 |
