# M6 preregistered development stress test

Hypothesis: removing stale or disputed query sources changes text retrieval while preserving the structured signal; effects can include regressions because active text still contains negation and ambiguous interests.

Control: accumulated raw user text. Treatment: active-only source selection. Run paired M1+BM25, M1+dense, and three-way RRF with frozen baseline variants, pool, model/cache and metrics. Default remains M1-only/raw.

Use all 60 development profiles once, assigned by source order modulo four to superseded, disputed, all-active and empty-active conditions. Prepend another development opening at fixed offset 30 as synthetic distractor. Preserve original text. Final lifecycle labels are simulated, not inferred contradictions. Keep both policies' vectors identical: development hidden slots divided by three and clipped to runtime bounds. These are mock inputs, not calibrated tracker predictions. Original labels remain evaluator-only. No held-out data is read.

Measure six M2 metrics, paired bootstrap intervals, target-rank shifts and subgroup effects. All-active rankings must be identical; empty-active must safely fall back to M1. Persist complete per-turn sources, query diagnostics, manifests and source fingerprints. Do not select cases based on outcomes. Bootstrap intervals describe this artificial mixture only; shared distractor texts limit independence.

Decision: retain filtering as an optional research capability if correctness gates pass. Only independently annotated, representative dialogue evidence can justify production activation or a real-world accuracy claim. Synthetic gains alone are insufficient. Reproduce with `python scripts/run_m6_experiment.py --output <fresh-directory>`.
