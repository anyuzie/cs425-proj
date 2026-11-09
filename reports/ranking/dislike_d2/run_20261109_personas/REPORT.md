# Fusion dialogue evaluation

Primary metric: mrr of first acceptable occupation (§12.4).

| Arm | Group | n | MRR | Hit@5 | Hit@10 | NDCG@10 |
|---|---|---|---|---|---|---|
| fusion_zscore_probing | all | 25 | 0.680 | 0.800 | 0.920 | 0.539 |
| fusion_zscore_probing | conflict | 14 | 0.639 | 0.714 | 0.857 | 0.535 |
| fusion_zscore_probing | agree | 11 | 0.733 | 0.909 | 1.000 | 0.544 |
| fusion_zscore_d1_bm25_ignore_v2_probing | all | 25 | 0.770 | 0.960 | 0.960 | 0.636 |
| fusion_zscore_d1_bm25_ignore_v2_probing | conflict | 14 | 0.780 | 1.000 | 1.000 | 0.690 |
| fusion_zscore_d1_bm25_ignore_v2_probing | agree | 11 | 0.758 | 0.909 | 0.909 | 0.568 |
| routing_v3_probing | all | 25 | 0.770 | 0.960 | 0.960 | 0.620 |
| routing_v3_probing | conflict | 14 | 0.780 | 1.000 | 1.000 | 0.661 |
| routing_v3_probing | agree | 11 | 0.758 | 0.909 | 0.909 | 0.567 |
| routing_v3_d2_title_v1_probing | all | 25 | 0.770 | 0.960 | 0.960 | 0.620 |
| routing_v3_d2_title_v1_probing | conflict | 14 | 0.780 | 1.000 | 1.000 | 0.661 |
| routing_v3_d2_title_v1_probing | agree | 11 | 0.758 | 0.909 | 0.909 | 0.567 |
| routing_v3_d2_semantic_v1_probing | all | 25 | 0.770 | 0.960 | 0.960 | 0.595 |
| routing_v3_d2_semantic_v1_probing | conflict | 14 | 0.780 | 1.000 | 1.000 | 0.617 |
| routing_v3_d2_semantic_v1_probing | agree | 11 | 0.758 | 0.909 | 0.909 | 0.567 |
| routing_v3_d2_title_semantic_v1_probing | all | 25 | 0.770 | 0.960 | 0.960 | 0.620 |
| routing_v3_d2_title_semantic_v1_probing | conflict | 14 | 0.780 | 1.000 | 1.000 | 0.661 |
| routing_v3_d2_title_semantic_v1_probing | agree | 11 | 0.758 | 0.909 | 0.909 | 0.567 |

## Paired differences (MRR, 95% bootstrap CI)

| Contrast | Group | Mean | CI |
|---|---|---|---|
| routing_v3_probing_minus_fusion_zscore_d1_bm25_ignore_v2_probing | all | +0.000 | [+0.000, +0.000] |
| routing_v3_probing_minus_fusion_zscore_d1_bm25_ignore_v2_probing | conflict | +0.000 | [+0.000, +0.000] |
| routing_v3_probing_minus_fusion_zscore_d1_bm25_ignore_v2_probing | agree | +0.000 | [+0.000, +0.000] |
| routing_v3_probing_minus_fusion_zscore_probing | all | +0.090 | [-0.037, +0.226] |
| routing_v3_probing_minus_fusion_zscore_probing | conflict | +0.141 | [-0.019, +0.311] |
| routing_v3_probing_minus_fusion_zscore_probing | agree | +0.025 | [-0.190, +0.244] |
| routing_v3_d2_title_v1_probing_minus_routing_v3_probing | all | +0.000 | [+0.000, +0.000] |
| routing_v3_d2_title_v1_probing_minus_routing_v3_probing | conflict | +0.000 | [+0.000, +0.000] |
| routing_v3_d2_title_v1_probing_minus_routing_v3_probing | agree | +0.000 | [+0.000, +0.000] |
| routing_v3_d2_title_v1_probing_minus_fusion_zscore_probing | all | +0.090 | [-0.037, +0.226] |
| routing_v3_d2_title_v1_probing_minus_fusion_zscore_probing | conflict | +0.141 | [-0.019, +0.311] |
| routing_v3_d2_title_v1_probing_minus_fusion_zscore_probing | agree | +0.025 | [-0.190, +0.244] |
| routing_v3_d2_semantic_v1_probing_minus_routing_v3_probing | all | +0.000 | [+0.000, +0.000] |
| routing_v3_d2_semantic_v1_probing_minus_routing_v3_probing | conflict | +0.000 | [+0.000, +0.000] |
| routing_v3_d2_semantic_v1_probing_minus_routing_v3_probing | agree | +0.000 | [+0.000, +0.000] |
| routing_v3_d2_semantic_v1_probing_minus_fusion_zscore_probing | all | +0.090 | [-0.037, +0.226] |
| routing_v3_d2_semantic_v1_probing_minus_fusion_zscore_probing | conflict | +0.141 | [-0.019, +0.311] |
| routing_v3_d2_semantic_v1_probing_minus_fusion_zscore_probing | agree | +0.025 | [-0.190, +0.244] |
| routing_v3_d2_title_semantic_v1_probing_minus_routing_v3_probing | all | +0.000 | [+0.000, +0.000] |
| routing_v3_d2_title_semantic_v1_probing_minus_routing_v3_probing | conflict | +0.000 | [+0.000, +0.000] |
| routing_v3_d2_title_semantic_v1_probing_minus_routing_v3_probing | agree | +0.000 | [+0.000, +0.000] |
| routing_v3_d2_title_semantic_v1_probing_minus_fusion_zscore_probing | all | +0.090 | [-0.037, +0.226] |
| routing_v3_d2_title_semantic_v1_probing_minus_fusion_zscore_probing | conflict | +0.141 | [-0.019, +0.311] |
| routing_v3_d2_title_semantic_v1_probing_minus_fusion_zscore_probing | agree | +0.025 | [-0.190, +0.244] |

## Per persona (best acceptable rank, final turn)

| Persona | Field | Conflict | fusion_zscore_probing | fusion_zscore_d1_bm25_ignore_v2_probing | routing_v3_probing | routing_v3_d2_title_v1_probing | routing_v3_d2_semantic_v1_probing | routing_v3_d2_title_semantic_v1_probing |
|---|---|---|---|---|---|---|---|---|
| F01 | music | True | 1 | 1 | 1 | 1 | 1 | 1 |
| F02 | veterinary | True | 1 | 1 | 1 | 1 | 1 | 1 |
| F03 | history_heritage | True | 1 | 2 | 2 | 2 | 2 | 2 |
| F04 | law | True | 1 | 1 | 1 | 1 | 1 | 1 |
| F05 | video_games | True | 1 | 1 | 1 | 1 | 1 | 1 |
| F06 | climate_policy | True | 1 | 1 | 1 | 1 | 1 | 1 |
| F07 | surgery | True | 5 | 3 | 3 | 3 | 3 | 3 |
| F08 | food | True | 7 | 1 | 1 | 1 | 1 | 1 |
| F09 | journalism | True | 1 | 1 | 1 | 1 | 1 | 1 |
| F10 | architecture | True | 1 | 1 | 1 | 1 | 1 | 1 |
| F11 | aviation | True | 9 | 2 | 2 | 2 | 2 | 2 |
| F12 | religion | True | 3 | 1 | 1 | 1 | 1 | 1 |
| F13 | finance | True | 13 | 4 | 4 | 4 | 4 | 4 |
| F14 | sport | True | 12 | 3 | 3 | 3 | 3 | 3 |
| F15 | nursing | False | 1 | 1 | 1 | 1 | 1 | 1 |
| F16 | software | False | 1 | 1 | 1 | 1 | 1 | 1 |
| F17 | counselling | False | 1 | 1 | 1 | 1 | 1 | 1 |
| F18 | accounting | False | 5 | 12 | 12 | 12 | 12 | 12 |
| F19 | marketing | False | 1 | 1 | 1 | 1 | 1 | 1 |
| F20 | primary_teaching | False | 2 | 2 | 2 | 2 | 2 | 2 |
| F21 | geology | False | 4 | 2 | 2 | 2 | 2 | 2 |
| F22 | graphic_design | False | 1 | 1 | 1 | 1 | 1 | 1 |
| F23 | robotics | False | 1 | 1 | 1 | 1 | 1 | 1 |
| F24 | statistics | False | 1 | 4 | 4 | 4 | 4 | 4 |
| F25 | sales_property | False | 9 | 1 | 1 | 1 | 1 | 1 |
