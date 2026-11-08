# M1 cosine development experiment

Hypothesis (before evaluating): occupation magnitude influences the dot product; normalizing both vectors may improve alignment with the profile's RIASEC direction.

Control: `weighted_dot_product@baseline_v1`. Treatment: separate algorithm `cosine_similarity@baseline_v1`. Preserve the existing algorithm and default selection. Use identical development/oracle inputs, 378 eligible occupations, supplied occupation z-scores, full ranking depth, seed and all six M2 metrics. No held-out data, retrieval or fusion tuning.

Formula: restrict both vectors to observed dimensions D, then compute sum(u[t] z[t]) / (L2(u[D]) L2(z[D])). Unknown dimensions affect neither numerator nor norms; observed neutral dimensions remain in the occupation norm. No observed nonzero interest gives an unusable signal. Occupations with zero projected norm are omitted because cosine is undefined; if all are omitted the signal is unusable. Keep negative scores, clamp numerical roundoff to [-1,1], break exact ties by ascending occupation code. All six dimensions are observed in oracle evaluation; do not clamp oracle values to runtime bounds.

Measure Hit@1/5/10, full-pool MRR, NDCG@5/10, paired changes, 2,000-resample bootstrap intervals, per-profile rank shifts and latency. Keep cosine selectable regardless of whether it wins. Changing the production default requires a separate decision informed by tracker evaluation; this oracle comparison does not establish actual-tracker quality.
