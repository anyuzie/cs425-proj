# Inspected target-rank changes

All listed targets are eligible. These observations compare synthetic utterances
with target descriptions; they do not independently validate the relevance labels.
No stance/lifecycle filtering is applied. Every move stays within its original
rank block; no arbitrary zero-score ranks or RRF are involved.

| Profile | Target | Control | Block 5 | Block 10 | Observation |
|---|---|---:|---:|---:|---|
| D072 | Transportation Inspectors | 1 | 5 | 6 | Site equipment preferences match the practical dimension, but the history also mentions teaching and opposing administrative preferences. These topics can distract a single raw embedding from the narrow transport-safety description. |
| D089 | Music Directors and Composers | 1 | 5 | 7 | The user discusses murals, short films and selling prints, without music. A RIASEC-compatible artistic target has weak domain-specific semantic support; demotion need not imply a worse user recommendation. |
| D061 | Art Directors | 1 | 3 | 4 | Posters, scripts and pitching ideas plausibly support this target. The raw query also includes social, analytical and practical interests plus a qualification about helping others. The observed demotion is a potentially harmful within-block reordering, without enough evidence to assign a specific cause. |
| D071 | Soil and Plant Scientists | 1 | 3 | 5 | Engine repair and materials testing express investigative/practical interests, but the description specifically concerns crops and soil. This again exposes interest-vector target versus domain-intent mismatch. |
| D029 | Urologists | 2 | 5 | 10 | Biomedical research and helping people support broad scientific/social interests, without identifying urology. Semantic demotion is understandable given the description's genitourinary specialty. |
| D073 | Accountants and Auditors | 2 | 1 | — | Record keeping and reconciling accounts explicitly match the description, making the block-5 improvement plausible. The profile also contains a contrary statement about checking figures, which the raw embedding does not resolve. |
| D013 | Nanotechnology Engineering Technologists and Technicians | 8 | — | 1 | Analyzing and tinkering overlap with testing devices; however nanoscale work is not stated and the profile contradicts itself about enjoying research. The improved benchmark rank is not independent evidence of domain intent. |
| D019 | Agents and Business Managers of Artists, Performers, and Athletes | 8 | — | 1 | Selling, leading and creative media plausibly support promotion/representation work. Contradictory mechanical preferences remain in the raw history. |
| D042 | Software Quality Assurance Analysts and Testers | 7 | — | 1 | Debugging code directly matches identifying software problems and their causes. This is a plausible semantic gain with strong domain overlap. |

Em dashes indicate an unreported rank, not an absent recommendation. Exact ranks,
scores and all other occupations are available in each run's predictions.jsonl.
No causal attribution to a specific phrase has been experimentally established.
