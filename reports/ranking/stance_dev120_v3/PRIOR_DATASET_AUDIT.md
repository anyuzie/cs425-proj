# Previous dataset verification

Restored canonical file: 120 profiles; SHA-256 `a593af7309f0aa99d049ae4e211fccca7172db9ae5ab16eebe83ca79b54a421a`.

All 16 previous non-stance runs audited used **120**, not 60 profiles. For every run, metrics.profile_count=120, inputs contain exactly D001-D120 once, per_profile has 120 records, and exported predictions cover exactly D001-D120.

- reports/ranking/dot_vs_cosine_dev120: both runs match the restored dataset byte fingerprint exactly.
- reports/ranking/combinations_dev120: all six runs match the restored dataset byte fingerprint exactly.
- reports/ranking/updated_dev120_20261018T171347Z: all eight runs used 120 profiles with fingerprint eec12cc041f7d82c54989fbfe34a7d0628e74d80200cb40d8b5e2b1197cd893b. This was a different opening-text revision, not a 60-profile dataset. Do not treat it as the same canonical input for paired comparison.

The original historical M2/M3/M4 reports intentionally retain their earlier 60-profile evaluations. Their existence does not change which input the later runs used. Detailed evidence is in prior_dataset_audit.csv; no old artifacts or history rows were rewritten.
