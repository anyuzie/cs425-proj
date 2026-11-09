| Arm | Group | n | avoided in top 10 | any avoided in top 10 | MRR | acceptable in top 10 |
|---|---|---|---|---|---|---|
| fusion_zscore | all | 24 | 2.79 | 0.79 | 0.317 | 0.40 |
| fusion_zscore | split=calibration | 12 | 3.92 | 0.83 | 0.351 | 0.41 |
| fusion_zscore | split=test | 12 | 1.67 | 0.75 | 0.283 | 0.39 |
| fusion_zscore | kind=casual | 4 | 2.25 | 1.00 | 0.171 | 0.08 |
| fusion_zscore | kind=collateral | 2 | 0.00 | 0.00 | 0.750 | 1.00 |
| fusion_zscore | kind=control | 2 | 0.00 | 0.00 | 0.750 | 1.00 |
| fusion_zscore | kind=explicit | 4 | 3.75 | 1.00 | 0.154 | 0.17 |
| fusion_zscore | kind=extractor_miss | 5 | 4.40 | 0.80 | 0.200 | 0.33 |
| fusion_zscore | kind=narrow | 5 | 2.20 | 1.00 | 0.333 | 0.43 |
| fusion_zscore | kind=only_negative | 2 | 5.00 | 1.00 | 0.000 | 0.00 |
| fusion_zscore_d1 | all | 24 | 0.79 | 0.29 | 0.485 | 0.55 |
| fusion_zscore_d1 | split=calibration | 12 | 1.08 | 0.33 | 0.511 | 0.52 |
| fusion_zscore_d1 | split=test | 12 | 0.50 | 0.25 | 0.459 | 0.58 |
| fusion_zscore_d1 | kind=casual | 4 | 1.00 | 0.25 | 0.472 | 0.58 |
| fusion_zscore_d1 | kind=collateral | 2 | 0.00 | 0.00 | 0.750 | 1.00 |
| fusion_zscore_d1 | kind=control | 2 | 0.00 | 0.00 | 0.750 | 1.00 |
| fusion_zscore_d1 | kind=explicit | 4 | 0.75 | 0.50 | 0.398 | 0.40 |
| fusion_zscore_d1 | kind=extractor_miss | 5 | 2.20 | 0.60 | 0.517 | 0.53 |
| fusion_zscore_d1 | kind=narrow | 5 | 0.20 | 0.20 | 0.322 | 0.30 |
| fusion_zscore_d1 | kind=only_negative | 2 | 0.00 | 0.00 | 0.000 | 0.00 |
| semantic | all | 24 | 3.33 | 0.83 | 0.350 | 0.44 |
| semantic | split=calibration | 12 | 4.00 | 0.83 | 0.361 | 0.39 |
| semantic | split=test | 12 | 2.67 | 0.83 | 0.338 | 0.48 |
| semantic | kind=casual | 4 | 2.25 | 1.00 | 0.302 | 0.17 |
| semantic | kind=collateral | 2 | 0.00 | 0.00 | 1.000 | 1.00 |
| semantic | kind=control | 2 | 0.00 | 0.00 | 0.750 | 1.00 |
| semantic | kind=explicit | 4 | 4.25 | 1.00 | 0.165 | 0.17 |
| semantic | kind=extractor_miss | 5 | 5.20 | 1.00 | 0.131 | 0.27 |
| semantic | kind=narrow | 5 | 2.20 | 1.00 | 0.333 | 0.60 |
| semantic | kind=only_negative | 2 | 8.50 | 1.00 | 0.000 | 0.00 |
| semantic_signed | all | 24 | 0.92 | 0.42 | 0.546 | 0.59 |
| semantic_signed | split=calibration | 12 | 1.00 | 0.42 | 0.552 | 0.49 |
| semantic_signed | split=test | 12 | 0.83 | 0.42 | 0.541 | 0.70 |
| semantic_signed | kind=casual | 4 | 2.00 | 0.75 | 0.619 | 0.67 |
| semantic_signed | kind=collateral | 2 | 0.00 | 0.00 | 1.000 | 1.00 |
| semantic_signed | kind=control | 2 | 0.00 | 0.00 | 0.750 | 1.00 |
| semantic_signed | kind=explicit | 4 | 0.50 | 0.50 | 0.458 | 0.48 |
| semantic_signed | kind=extractor_miss | 5 | 2.00 | 0.80 | 0.542 | 0.67 |
| semantic_signed | kind=narrow | 5 | 0.40 | 0.20 | 0.300 | 0.23 |
| semantic_signed | kind=only_negative | 2 | 0.00 | 0.00 | 0.000 | 0.00 |

| Probe | kind | fusion_zscore avoided/best | fusion_zscore_d1 avoided/best | semantic avoided/best | semantic_signed avoided/best |
|---|---|---|---|---|---|
| N01 | explicit | 8/15 | 2/2 | 8/20 | 1/2 |
| N02 | narrow | 4/20 | 0/117 | 4/21 | 0/None |
| N03 | explicit | 5/36 | 0/2 | 5/13 | 0/2 |
| N04 | casual | 1/2 | 0/2 | 1/1 | 1/1 |
| N05 | extractor_miss | 7/2 | 1/1 | 10/32 | 2/1 |
| N06 | extractor_miss | 9/71 | 9/71 | 6/15 | 6/15 |
| N07 | extractor_miss | 1/4 | 0/2 | 4/4 | 0/2 |
| N08 | collateral | 0/1 | 0/1 | 0/1 | 0/1 |
| N09 | narrow | 2/4 | 1/2 | 2/4 | 2/2 |
| N10 | narrow | 1/5 | 0/10 | 1/5 | 0/None |
| N11 | only_negative | 9/- | 0/- | 7/- | 0/- |
| N12 | control | 0/1 | 0/1 | 0/1 | 0/1 |
| N13 | casual | 3/15 | 4/1 | 3/13 | 5/1 |
| N14 | explicit | 1/2 | 1/2 | 1/2 | 1/2 |
| N15 | extractor_miss | 5/6 | 1/1 | 5/6 | 1/1 |
| N16 | explicit | 1/50 | 0/11 | 3/29 | 0/3 |
| N17 | casual | 4/13 | 0/18 | 4/9 | 2/7 |
| N18 | extractor_miss | 0/14 | 0/14 | 1/7 | 1/7 |
| N19 | narrow | 1/1 | 0/1 | 1/1 | 0/1 |
| N20 | narrow | 3/6 | 0/338 | 3/6 | 0/None |
| N21 | casual | 1/24 | 0/3 | 1/48 | 0/3 |
| N22 | collateral | 0/2 | 0/2 | 0/1 | 0/1 |
| N23 | only_negative | 1/- | 0/- | 10/- | 0/- |
| N24 | control | 0/2 | 0/2 | 0/2 | 0/2 |
