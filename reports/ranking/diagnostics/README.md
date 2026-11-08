# Ranking diagnostic analysis outputs

The primary deliverable is the executed [notebook](../../../notebooks/ranking_diagnostic_analysis.ipynb). [Execution instructions](../../../notebooks/README.md) describe fresh-kernel execution and the text-first review protocol. [Computed findings](findings_summary.md) contain the five consequential findings, all section interpretations/limitations, and the top three recommended experiments.

Sources: `data/dev_profiles.jsonl`, `data/occupations.csv`, `data/tasks.csv`, `data/labels.json`, all seven saved M4 configuration directories, their inputs/manifests/predictions/metrics, the history CSV, and the existing pinned MiniLM task/model cache. The M4 structured/lexical controls reproduce the earlier M1/M3 controls. Only the development split is analyzed.

## Acceptance mapping

| Instruction criterion | Evidence |
|---|---|
| Inspect schemas/evaluator/saved artifacts first | Shared loader validates actual files, manifests, schemas, request snapshots and catalog joins |
| Reuse BM25, MiniLM, RRF and M2 metrics | Live BM25/ranker/cache replay and exact evaluator/fusion assertions; no alternative production formulas |
| Validate profile sets, pools and configurations | All seven runs match development/oracle inputs, pool, hashes, evaluator definitions and baseline variant IDs |
| Sequential fresh-kernel execution without errors | `execution_check.json` records successful execution and fresh-kernel reproducibility |
| Derive charts and narratives from computed values | 36 cells, 18 executed code cells, 48 rendered charts; findings generated at runtime |
| Report missing sources/dependencies honestly | All live sources/cache are available locally; 18 specificity proposals have zero verified human reviews; no verified-specificity chart is fabricated |
| Save executed notebook | Embedded outputs and charts in the `.ipynb`; standalone PNGs under `figures/` |
| Minimal helper / no production refactor | One read-only notebook helper; production ranking/evaluation code unchanged |
| Preserve models/data/old artifacts/history | `analysis_manifest.json` hashes 202 protected source/report/model/cache files and verifies unchanged bytes |
| Do not change algorithms, metrics, fusion or eligibility | No production/config changes or experiment-history additions; all suggested improvements remain proposals |

## Tables and review artifacts

- `profile_diagnostics.csv`: 60 joined profiles with numerical, text, metadata, lexical, semantic and per-method metric fields. Missing ranks remain missing; zero scores remain zero.
- `per_profile_metrics.csv`: all six existing metrics, target diagnostics and relevant top-five/top-ten counts for seven methods.
- `subgroup_metrics.csv`, `paired_differences.csv`: sizes and exploratory seeded profile-bootstrap intervals.
- `fusion_regressions.csv`: per-profile successes lost/gained, all-component misses, ranks and NDCG differences.
- `failure_cases.csv`: overlapping provisional indicators with supporting observations, not confirmed causal labels.
- `semantic_sentence_matches.csv`: actual query-sentence winning tasks for targets and best competing occupations; no task corpus re-embedding.
- `riasec_distributions.csv`: per-dimension numerical summaries and polarity proportions.
- `improvement_priorities.csv`: explicit planning rubric, affected indicator counts, confidence, complexity and suggested controlled experiments.
- `specificity_blind_inputs.csv`, `specificity_blind_proposals.csv`: deterministic 18-profile sample and label-free automated stage.
- `specificity_human_reviews.csv`: preserved empty human-review template; no automatic verified judgments.
- `specificity_annotations.csv`: target-revealed review table, generated after blind proposals are frozen.

The notebook regenerates its diagnostic outputs. Human review data is preserved. This is an exploratory analysis of a small synthetic benchmark: oracle/text information differs, related labels are not human relevance judgments, subgroup comparisons are small and correlated, and metadata associations do not establish mechanisms. No held-out exploration, M5 implementation or ranking optimization was performed.
