# Evaluation Protocol — Controls, Iterations, and Decisions

Return to [README](README.md) · See [Development](DEVELOPMENT.md) · Copy [Experiment template](EXPERIMENT_TEMPLATE.md)

**Current status:** M2 implements the structured-only oracle development baseline (R0), primary metrics, exports, and paired comparisons. Other ladder entries and end-to-end evaluation remain planned. See the [canonical status table](README.md#current-implementation-status) and [M2 decision](../reports/ranking/m2/DECISION.md).

## The four questions we evaluate separately

| Layer | Question | Data / method |
|---|---|---|
| 0. Correctness | Is the implementation correct? | Hand-calculated unit fixtures and integration tests |
| 1. Offline ranking | Does adding BM25/dense/RRF improve rankings for fixed user inputs? | Development profile evaluation; identical input mode and pool |
| 2. Evidence policy | Does filtering superseded/familiarity evidence improve ranking *with fixed RIASEC state*? | Paired counterfactual replay of same conversation snapshots |
| 3. End-to-end | Does probing yield better final user recommendations? | Complete simulator/real tracker, policy on vs off |

**Never conflate these:** a fixed-vector evidence improvement does not prove that the actual extractor/NLI/policy successfully detects and resolves contradictions.

## Dataset and leakage rules

1. **Development (verified 60):** evaluate experiments, tune only here, inspect failures.
2. **Held-out (verified 100):** run once configuration is frozen. Log all cases, including missing/out-of-pool targets.
3. **Conversation subset:** needed for M6–M8. Check whether existing records include text, ordered turns, evidence status, familiarity/reason, and pre/post profile snapshots. If not, create an independently reviewed supplementary set with profile-disjoint development/held-out splits.
4. **Input modes:** `oracle`, `mock_tracker`, `actual_tracker` are separate result categories. Development `hidden_slots` are z-scale synthetic values, *not* runtime `[-1,1]` interests. Never silently mix or clamp.
5. Labels (`target_occupation`, relevance grades, target profile) are read by the **evaluator only**, never by the ranker. No synthetic target answer should leak into retrieval query construction.
6. Use one frozen candidate eligibility rule for each paired comparison. The current implementation plan specifies Job Zones 4–5 (378 occupations); the proposal's 1,016-occupation scope is an outstanding policy decision.

## Metrics

**Primary:**

- **Hit@5** = mean of `1` when exact target occupation appears in ranks 1–5, else `0`.
- **MRR** = mean reciprocal exact-target rank; missing target gives `0`. Report whether rank is searched through the whole 378-pool ranking or only returned top-K, and keep this definition fixed.
- **NDCG@10**: use maximum relevance grade per occupation (`target=4`, `Primary-Short=3`, `Primary-Long=2`, `Supplemental=1`, other=0); `DCG@10 = sum((2**grade - 1) / log2(rank+1))`; divide by ideal DCG over eligible candidates; if ideal DCG=0, define result as 0 and separately count these cases.

**Diagnostic, not replacements for primary metrics:** per-profile exact target rank change, top-5 set change frequency, stale-evidence sensitivity, recovery after probe/clarification, performance on genuine-familiarity cases, runtime (median/p95), error rate, number of affected examples.

**Additional implemented cutoffs:** exact-target Hit@1 and Hit@10, plus graded NDCG@5 using the same labels, gain function, and pool filtering as NDCG@10. New runs save all six metrics and register them in the [evaluation history](README.md#persistent-evaluation-history). The original three metric definitions remain unchanged.

Report quality *and* uncertainty. For paired runs, compute per-profile metric differences and a bootstrap 95% confidence interval resampling profiles (not turns) to preserve dependence. A paired McNemar test may complement binary Hit@5. Report sample sizes; avoid grand claims with a tiny affected subset.

## Baseline ladder (fixed evidence; same development inputs)

| ID | Structured | BM25 | Dense | Purpose |
|---|---|---|---|---|
| R0 | ✓ | | | MVP baseline |
| R1 | | ✓ | | Lexical standalone |
| R2 | | | ✓ | Dense standalone |
| R3 | ✓ | ✓ | | Two-way fusion |
| R4 | ✓ | | ✓ | Structured + semantic |
| R5 | | ✓ | ✓ | Text-only fusion |
| **R6** | ✓ | ✓ | ✓ | **Proposal-compliant equal-weight RRF baseline** |

R0 was completed at M2; add the remaining models as later milestones deliver them. Do not compare runs using different candidate pools or oracle-vs-tracker profiles as though they were controlled ablations.

## Evidence-aware experiment ladder

Use R6 (or a declared frozen best baseline) with identical structured inputs. The **only** changes across rows should be the evidence policies indicated below.

| ID | Superseded / disputed filtering | Unresolved familiarity weighting | Negative channel | Main contrast |
|---|---|---|---|---|
| A | Off | Off (raw) | Off | Control |
| B | On | Off | Off | B − A: contradiction lifecycle |
| C | Off | On | Off | C − A: familiarity policy |
| D | On | On | Off | D − A: combined policy |
| E | Off | Off | On | E − A: signed matching alone |
| F | On | On | On | F − D: incremental signed matching |

For the two binary factors (filtering and familiarity), the interaction on a per-profile metric is `D - B - C + A`, averaged over the same profiles. Interpret this only when cases actually expose the factors.

**Important implementation warning:** a simple statement weight does not magically propagate through a combined BM25 document query or a single averaged embedding. If testing familiarity weights, explicitly define sentence-level score computation and aggregation; freeze that algorithm across control/treatment except the intended weight parameter.

## Controlled conversation replay

Save snapshots *before and after* clarification/probing. For a clean stale-evidence experiment:

```text
                       SAME profile snapshot / SAME occupation corpus
                                          │
                   ┌──────────────────────┴──────────────────────┐
                   ▼                                             ▼
       Control: all user utterances                Treatment: eligible active evidence
                   │                                             │
         same BM25 + dense + RRF                       same BM25 + dense + RRF
                   └──────────────────────┬──────────────────────┘
                                          ▼
                             Paired ranking differences
```

- Keep the **structured RIASEC vector fixed** across the two branches.
- Keep user history, model/index revision, candidate set, tokenization, aggregation, and RRF parameters fixed.
- Change **only the intended evidence selection/weighting policy**.
- Track each evidence record ID and status so a removed statement can be audited.
- For before-vs-after *conversational* improvement, state that both profile and evidence may have changed; that is a different estimand.

## Failure review

For every material ranking change, inspect at least five relevant failures or all failures when fewer than five; include examples where the treatment helped and harmed. Annotate:

- Was the target eligible and the relevance label reasonable?
- Was the RIASEC state unknown, wrong, or stale?
- Did BM25 reward an explicit dislike because of word overlap?
- Did dense task retrieval confuse one matching task for whole-occupation fit?
- Did unprobed familiarity dominate search terms?
- Did a rejected statement remain in the query?
- Did RRF promote zero-score or arbitrary ranks?

Record evidence IDs and occupation codes; don't copy private user text into public reports without authorization. Simulation outputs should be labeled synthetic.

## Required saved artifacts per experiment

Suggested folder: `reports/ranking/experiments/<run_id>/`

```text
manifest.json        # change, control ID, commit, split, input mode, versions, seeds
config.json          # all ranker/query/fusion/evidence settings
predictions.csv      # one row per profile and output occupation (or JSONL)
metrics.json         # aggregate metrics and valid/invalid-case counts
paired_diffs.csv     # one row per paired profile (if comparison)
latency.json         # build time + warm median/p95 query timings
failure_review.md    # concrete helped/harmed cases and explanation
DECISION.md          # keep/revise/revert + rationale + next experiment
```

Minimum prediction fields: `run_id`, `profile_id`, `turn_id/snapshot_id` (if relevant), `input_mode`, `occupation_code`, `rank`, `rrf_score`, component scores/ranks, `target_code` in evaluator-only output, and optional supporting task/evidence IDs.

Minimum manifest fields: timestamp, `run_id`, `control_run_id`, hypothesis, treatment, git commit, full config hash, dataset fingerprint and split, eligibility rule, model ID/revision, task-index fingerprint, Python/package versions, seed, test summary, evaluator version, and exact reproducibility command. Freeze the *evaluated input snapshots* too, so changes to upstream code do not silently modify the control.

## Iteration decision rubric

1. **FAIL correctness?** Fix code or fixtures; **do not interpret ranking metrics**.
2. **Different evaluation inputs or labels?** Repair comparison; **do not infer treatment effect**.
3. **Quality improves with acceptable runtime and explainable failure profile?** **KEEP**, document exact scope and uncertainty.
4. **Mixed/uncertain results?** **REVISE** one assumption (query construction, task aggregation, or weighting); create a new experiment ID, preserve old run.
5. **Persistent regression or unjustified complexity?** **REVERT** to last frozen working baseline; record negative finding.
6. **Milestone is optional and value is unclear?** **STOP** and move to final evaluation rather than accumulate complexity.

No arbitrary global Hit@5 threshold is required. Decide by reproducibility, magnitude and consistency of paired changes, harms, latency, and project goals. Do not select a config on test-set metrics.

## Report table template

| Run | Control | Changed variable | Cases | Hit@5 | MRR | NDCG@10 | ΔNDCG@10 vs control | p95 (ms) | Decision |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| baseline_v1 | — | three-way RRF, raw query | TBD | TBD | TBD | TBD | — | TBD | frozen |
| exp_m6_filter | baseline_v1 | active evidence only | TBD | TBD | TBD | TBD | TBD | TBD | pending |

Use `TBD`, not fabricated performance figures.

## Final claims checklist

- [ ] Each claim cites its experiment ID and frozen control.
- [ ] Correctness tests pass; missing/ineligible cases disclosed.
- [ ] Ranking improvements distinguished from probing / NLI improvements.
- [ ] Affected-case analysis and regressions included.
- [ ] Held-out split evaluated after freeze, without retuning.
- [ ] Model/data revisions, commands, and run artifacts are reproducible.


## Permanent semantic evaluation metrics (31 October 2026)

Evaluator m2-v3 includes Hit@1/5/10, MRR, NDCG@5/10 and macro Recall@5/10/20.
Recall counts all distinct eligible positive-grade labels, including the target
and related occupations; empty relevance sets receive zero. New comparisons,
histories and promotion guards include all nine metrics. Old runs keep their
original evaluator version; unknown historical recall remains blank.
See [integration and consolidated CSV](../docs/SEMANTIC_INTEGRATION.md).
