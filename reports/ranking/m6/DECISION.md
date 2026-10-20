# M6 decision and next work

M6 engineering and the controlled fixture ablation are complete. Keep `active_only` optional; retain M1-only/raw as the default. Representative annotated dialogue validation remains pending. This report does not claim that contradiction detection or probing works end to end.

The [protocol](EXPERIMENT.md) was written before execution. Six evaluations use the same 60 synthetic development sessions (15 per lifecycle condition), 378 candidates, fixed baseline variants and tracker snapshots. Runs and paired bootstrap intervals are in [run_v1/comparisons.json](run_v1/comparisons.json); all six runs were appended to the central history. Never compare these mock-tracker metrics directly to the historical oracle rows.

| Stack | Hit@5 raw → active | MRR raw → active | NDCG@10 raw → active |
|---|---:|---:|---:|
| M1 + BM25 | .1167 → .1500 | .0810 → .1134 | .1096 → .1394 |
| M1 + MiniLM | .0833 → .1333 | .0953 → .1112 | .1169 → .1573 |
| Three-way | .0833 → .1000 | .0898 → .0903 | .0878 → .1144 |

Overall averages hide a substantial fallback effect. In empty-active cases all three treatments revert to M1, improving Hit@5 by .1333. On superseded cases, three-way MRR decreases .0163 and NDCG@10 decreases .0158. On disputed cases, three-way MRR decreases .0613; M1+dense target ranks improve for six cases and regress for seven. Status filtering is correct but does not consistently improve relevance. Active utterances still contain dislikes and familiarity; removing whole utterances can discard useful mixed evidence. The prepended distractor is simulated stale content, not a genuine user correction, so this establishes sensitivity rather than representative efficacy.

Validation: all 166 repository tests pass ([output](test_results.txt)). Saved-run checks cover 360 paired turn snapshots: identical input bytes, exact structured signals against full-pool M1, identical all-active occupation outputs, and exact empty-active top-100 M1 fallback ([validation](run_v1/validation.json)). Reproduce with `python scripts/run_m6_experiment.py --output <fresh-directory>` followed by `python scripts/validate_m6_experiment.py <directory>`. Each report includes input, config, model/cache, dependency and source fingerprints. The local source snapshot accompanies this report because the worktree already contained uncommitted M6 changes.

## Recommendation: enrich evaluation evidence first

Add a separately versioned dialogue dataset and annotation sidecar before a broad search for replacement rankers. Preserve the existing development and held-out files. The current profiles were generated from occupation RIASEC rows plus noise (see `data/README.md`), which structurally favors M1 and does not independently establish occupational suitability. Generic interests may support several occupations; a single planted target can penalize reasonable recommendations.

Prioritize these fields and coverage:

1. Original user turns with sentence/span-level IDs and cumulative lifecycle status, the statement superseded by each correction, and annotation provenance. Include resolved/unresolved contradictions, elaborations that are not contradictions, unchanged controls, mixed-content utterances and empty-active cases. Record real tracker snapshots separately from manually corrected ones.
2. Keep stance, reason (`enjoyment`, `familiarity`, `neither`), confidence and lifecycle separate. Include misleading familiarity and genuine enjoyment after exposure, explicit dislikes, neutral statements, and confirmations. These support the M7 four-way ablation: raw, M6, familiarity-only, M6+M7.
3. Add independently written activity/task details, domain preferences and relevant constraints where needed. Include short/messy and detailed descriptions across all six RIASEC dimensions. Record which details were elicited, rather than injecting target occupation titles or copied O*NET tasks into queries.
4. Independently judge multiple plausible occupations with graded labels, document disagreements, and report per-stratum counts and uncertainty. Split by person/scenario family before writing paraphrases; keep existing held-out data untouched. Choose sample size from a preregistered precision/power target rather than multiplying near-duplicate cases.

Adding static occupation features (skills, work values, work context, education requirements) is justified only when corresponding user evidence and independent relevance judgments exist. Merely adding columns cannot improve the current rankers, which do not consume them. Treat each new feature family as a separate experiment, with missingness and provenance audits. Do not give rankers hidden targets, hidden slots as text, or evaluator relevance labels.

## Then test small alternatives, one at a time

Retain every accepted variant and compare on the same expanded development cases. First establish actual-tracker and manually corrected controls to separate upstream errors from ranking errors. Use the existing M3/M4 failure review and diagnostics as hypotheses, not causal conclusions.

- M1: a separately registered cosine-similarity algorithm is a useful small comparison to the dot product, especially with partial observed dimensions; define zero-vector behavior and normalization explicitly. It will not resolve task/domain ambiguity by itself.
- M3: test a task-focused document variant against the current broad occupation document, then inspect vocabulary mismatch before changing retrieval algorithms.
- M4: test a new top-k task aggregation variant against mean-of-sentence-maxima, holding MiniLM and query selection fixed. Investigate irrelevant sentence dilution and dislike matching before changing embedding models.
- Fusion: revisit equal-weight RRF only after standalone signal quality is understood; avoid simultaneously tuning evidence policy, model, aggregation and fusion.

The next milestone should be the annotated development set and M7 controlled evidence experiment. A limited M1/M3/M4 variant study can follow on that fixed set. A large model search now would risk optimizing around synthetic target construction while leaving the main evidence-aware research question untested.
