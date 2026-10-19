# Stance experiment results

| Condition | Hit@5 | MRR | NDCG@10 |
|---|---:|---:|---:|
| frozen_m4 | 0.041667 | 0.033864 | 0.040534 |
| positive_only | 0.033333 | 0.025191 | 0.039811 |
| signed_stance | 0.033333 | 0.028489 | 0.046595 |
| cosine | 0.683333 | 0.539974 | 0.421261 |
| cosine_stance | 0.308333 | 0.271708 | 0.239108 |

Keep production defaults unchanged. Stance is inferred inside the independent M4 component, never supplied by a dataset property. All five conditions use unchanged canonical dev data and matching oracle inputs/pool/metrics. New history rows were appended; historical results and the frozen baseline are preserved.

Polarity quality is UNVERIFIED without independent labels. Review the blank independent_review_template.csv before accuracy claims or further tuning. Exact activity contradictions are withheld, paraphrases can be missed, and scoped rules have limited coverage. This is static synthetic evaluation, not demonstrated skills, career suitability, novel occupation discovery or multi-turn improvement.

Full six metrics are in summary.csv; comparisons.json contains paired profile bootstrap intervals and per-profile wins/losses. Status counts include unusable queries; subgroup_per_profile.csv permits observed-polarity breakdowns. Task inventory association and top-ten signed task traces are diagnostic artifacts.

Observed decision: signed stance does not improve exact Hit@5 or MRR over frozen M4, though NDCG@10 is slightly higher (.046595 vs .040534). Adding stance to cosine reduces Hit@5 from .683333 to .308333; do not enable it by default. Coverage and scope limitations warrant independent annotation before further tuning. Frozen M4 semantic scores/ranks exactly match 45,360 earlier canonical-data component outputs. Both histories contain each new run exactly once.
