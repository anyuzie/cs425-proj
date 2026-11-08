# Fusion dialogue evaluation

Primary metric: mrr of first acceptable occupation (§12.4).

| Arm | Group | n | MRR | Hit@5 | Hit@10 | NDCG@10 |
|---|---|---|---|---|---|---|
| fusion_probing | all | 25 | 0.391 | 0.520 | 0.560 | 0.282 |
| fusion_probing | conflict | 14 | 0.114 | 0.214 | 0.214 | 0.050 |
| fusion_probing | agree | 11 | 0.743 | 0.909 | 1.000 | 0.577 |
| fusion_no_probing | all | 25 | 0.371 | 0.520 | 0.600 | 0.274 |
| fusion_no_probing | conflict | 14 | 0.110 | 0.214 | 0.286 | 0.062 |
| fusion_no_probing | agree | 11 | 0.702 | 0.909 | 1.000 | 0.543 |
| cosine_probing | all | 25 | 0.198 | 0.240 | 0.280 | 0.118 |
| cosine_probing | conflict | 14 | 0.007 | 0.000 | 0.000 | 0.000 |
| cosine_probing | agree | 11 | 0.440 | 0.545 | 0.636 | 0.267 |
| cosine_no_probing | all | 25 | 0.141 | 0.200 | 0.240 | 0.092 |
| cosine_no_probing | conflict | 14 | 0.008 | 0.000 | 0.000 | 0.000 |
| cosine_no_probing | agree | 11 | 0.310 | 0.455 | 0.545 | 0.209 |
| semantic_probing | all | 25 | 0.758 | 0.880 | 0.880 | 0.643 |
| semantic_probing | conflict | 14 | 0.881 | 1.000 | 1.000 | 0.827 |
| semantic_probing | agree | 11 | 0.603 | 0.727 | 0.727 | 0.408 |
| semantic_no_probing | all | 25 | 0.739 | 0.880 | 0.880 | 0.642 |
| semantic_no_probing | conflict | 14 | 0.881 | 1.000 | 1.000 | 0.832 |
| semantic_no_probing | agree | 11 | 0.559 | 0.727 | 0.727 | 0.399 |

## Paired differences (MRR, 95% bootstrap CI)

| Contrast | Group | Mean | CI |
|---|---|---|---|
| fusion_probing_minus_cosine_probing | all | +0.193 | [+0.087, +0.316] |
| fusion_probing_minus_cosine_probing | conflict | +0.107 | [+0.052, +0.168] |
| fusion_probing_minus_cosine_probing | agree | +0.303 | [+0.077, +0.554] |
| fusion_no_probing_minus_cosine_no_probing | all | +0.230 | [+0.112, +0.366] |
| fusion_no_probing_minus_cosine_no_probing | conflict | +0.102 | [+0.057, +0.157] |
| fusion_no_probing_minus_cosine_no_probing | agree | +0.393 | [+0.147, +0.650] |
| fusion_probing_minus_semantic_probing | all | -0.367 | [-0.578, -0.149] |
| fusion_probing_minus_semantic_probing | conflict | -0.767 | [-0.877, -0.646] |
| fusion_probing_minus_semantic_probing | agree | +0.141 | [-0.073, +0.366] |
| fusion_no_probing_minus_semantic_no_probing | all | -0.368 | [-0.576, -0.150] |
| fusion_no_probing_minus_semantic_no_probing | conflict | -0.771 | [-0.878, -0.651] |
| fusion_no_probing_minus_semantic_no_probing | agree | +0.144 | [-0.063, +0.368] |
| semantic_probing_minus_cosine_probing | all | +0.561 | [+0.350, +0.760] |
| semantic_probing_minus_cosine_probing | conflict | +0.873 | [+0.744, +0.991] |
| semantic_probing_minus_cosine_probing | agree | +0.162 | [-0.133, +0.478] |
| semantic_no_probing_minus_cosine_no_probing | all | +0.598 | [+0.402, +0.782] |
| semantic_no_probing_minus_cosine_no_probing | conflict | +0.873 | [+0.743, +0.990] |
| semantic_no_probing_minus_cosine_no_probing | agree | +0.249 | [-0.032, +0.538] |
| fusion_probing_minus_fusion_no_probing | all | +0.020 | [-0.046, +0.094] |
| fusion_probing_minus_fusion_no_probing | conflict | +0.004 | [-0.010, +0.026] |
| fusion_probing_minus_fusion_no_probing | agree | +0.041 | [-0.100, +0.182] |
| cosine_probing_minus_cosine_no_probing | all | +0.057 | [-0.004, +0.156] |
| cosine_probing_minus_cosine_no_probing | conflict | -0.001 | [-0.002, -0.000] |
| cosine_probing_minus_cosine_no_probing | agree | +0.131 | [-0.009, +0.354] |
| semantic_probing_minus_semantic_no_probing | all | +0.019 | [-0.041, +0.080] |
| semantic_probing_minus_semantic_no_probing | conflict | +0.000 | [+0.000, +0.000] |
| semantic_probing_minus_semantic_no_probing | agree | +0.044 | [-0.094, +0.182] |

## Per persona (best acceptable rank, final turn)

| Persona | Field | Conflict | fusion_probing | fusion_no_probing | cosine_probing | cosine_no_probing | semantic_probing | semantic_no_probing |
|---|---|---|---|---|---|---|---|---|
| F01 | music | True | 3 | 3 | 317 | 317 | 1 | 1 |
| F02 | veterinary | True | 24 | 24 | 201 | 201 | 1 | 1 |
| F03 | history_heritage | True | 3 | 3 | 39 | 39 | 1 | 1 |
| F04 | law | True | 11 | 8 | 159 | 106 | 1 | 1 |
| F05 | video_games | True | 19 | 16 | 200 | 180 | 1 | 1 |
| F06 | climate_policy | True | 3 | 5 | 57 | 44 | 1 | 1 |
| F07 | surgery | True | 15 | 14 | 137 | 137 | 2 | 2 |
| F08 | food | True | 20 | 20 | 95 | 95 | 2 | 2 |
| F09 | journalism | True | 19 | 19 | 343 | 343 | 1 | 1 |
| F10 | architecture | True | 26 | 18 | 293 | 290 | 1 | 1 |
| F11 | aviation | True | 37 | 33 | 285 | 288 | 1 | 1 |
| F12 | religion | True | 13 | 13 | 139 | 139 | 1 | 1 |
| F13 | finance | True | 29 | 29 | 239 | 239 | 3 | 3 |
| F14 | sport | True | 15 | 13 | 355 | 340 | 1 | 1 |
| F15 | nursing | False | 1 | 2 | 1 | 74 | 2 | 1 |
| F16 | software | False | 1 | 1 | 11 | 11 | 1 | 1 |
| F17 | counselling | False | 2 | 1 | 14 | 14 | 1 | 2 |
| F18 | accounting | False | 5 | 4 | 5 | 4 | 33 | 22 |
| F19 | marketing | False | 1 | 1 | 1 | 1 | 2 | 2 |
| F20 | primary_teaching | False | 3 | 3 | 16 | 16 | 2 | 2 |
| F21 | geology | False | 1 | 1 | 15 | 15 | 19 | 19 |
| F22 | graphic_design | False | 1 | 1 | 10 | 10 | 1 | 1 |
| F23 | robotics | False | 1 | 1 | 1 | 1 | 1 | 1 |
| F24 | statistics | False | 1 | 2 | 1 | 2 | 1 | 2 |
| F25 | sales_property | False | 7 | 7 | 4 | 4 | 22 | 22 |
