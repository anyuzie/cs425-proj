# CS425 â€” Occupation Scoring & Ranking

Fusion live default (9 November 2026): `configs/ranking.yaml` now runs M1 `cosine_similarity@baseline_v1` and M4 `occupation_dual_encoder@finetuned_v1` in parallel, combined by equal-weight RRF k=60 (identical to `configs/ranking/occupation_semantic_fusion_v1.json`), for the website, terminals and zero-argument evaluator, by explicit owner decision; it is the new baseline. With no observed RIASEC type the semantic ranking is used alone. On 120 development profiles with real extractor inputs fusion is tied with cosine (MRR .096 vs .117, -.021 [-.059, +.014]) and beats semantic-only (+.056 [+.023, +.092]); on held-out task retrieval .201 vs cosine .032. Single-turn evidence only; negation is not handled by M4. Requires the gitignored local checkpoint/index, otherwise fusion degrades to cosine. Cosine alone remains selectable (`configs/system.cosine.json`); the workflow selection and weight-free mock mode stay on cosine. See [decision](../reports/ranking/occupation_semantic_default/DECISION.md).

Occupation semantic retrieval (9 November 2026): opt-in M4 `occupation_dual_encoder@pretrained_v1` / `@finetuned_v1` (`recommender/occupation_semantic.py`, `training/occupation_semantic.py`) embeds chronological user-only text with MiniLM-L6 and scores all 378 eligible occupations against a validated title/description/task/context index; it runs without extracted preferences or RIASEC and is topic/background association, not preference (negation is not handled). Fine-tuned on source-grouped task/context statements (no dev_profile). Frozen test (1,133 task groups, binary ownership proxy): MRR control .032, pretrained .483, fine-tuned .518 (+.035 [+.020, +.049]), fine-tuned+cosine RRF .201; 38-occupation held-out: .364 to .418. RRF with single-sentence RIASEC hurts on this proxy; conversational relevance is unmeasured and XGBoost is deferred. Terminal/HTTP/restart acceptance pass; browser review pending. Presets `configs/system.occupation-semantic-*-v1.json` and `configs/ranking_terminal/occupation_semantic_*_v1.json`; defaults unchanged. See [report](../reports/ranking/occupation_semantic_v1/README.md).

Context/task text reranking experiment (2 November 2026): six separate frozen treatments add all raw task text to description/context prose, using runtime chunk encoding and unchanged weighted-dot-product first-stage groups. All ten metrics are evaluated on the same 120 development/oracle profiles and 378 occupations; six rows are appended to the preserved eight historical rows. No rated Work Context features or classifier scores are inputs. See [experiment report](../reports/ranking/context_tasks_v1/README.md) for actual metrics, chunk aggregation, grouping and limitations. Active defaults and frozen controls remain unchanged.

> **Project objective:** Build and evaluate a conversational occupation-ranking component that uses a user's evolving RIASEC interest profile and relevant conversation evidence to rank O*NET occupations after each turn.
>
> **Research question (whole project):** Can probing contradictions and challenging familiarity-based justifications give a better occupation ranking than accepting the user's stated profile as given?

This README is the **source of truth for objectives, scope, implementation status, and milestone priorities**. See [DEVELOPMENT.md](DEVELOPMENT.md) for acceptance criteria and implementation steps, [EVALUATION.md](EVALUATION.md) for controls and metrics, and [EXPERIMENT_TEMPLATE.md](EXPERIMENT_TEMPLATE.md) for recording each experiment. The [ranking architecture memory](../docs/RANKING_EXPERIMENT_ARCHITECTURE.md) is the authoritative guide to module boundaries, configuration variants, and future implementation workflows; read it before architectural changes.

## Current implementation status

Fusion dialogue evaluation (9 November 2026, #0 in docs/SEMANTIC_LEDGER_PLAN.md §12.3): `python -m eval.dialogue_fusion` runs 25 team-reviewed personas (`data/eval/fusion_personas_v1.jsonl`, 14 where RIASEC profile and stated topic conflict) through the real System in six arms (RRF fusion / cosine / semantic × probing on/off). Primary metric, fixed in advance: MRR of the first acceptable occupation. Fusion MRR .391 (conflict .114, agree .743), cosine .198, semantic .758 (conflict .881, agree .603); fusion − semantic on conflict −.767 [−.877, −.646], on agree +.141 [−.073, +.366]; probing effects span 0. Equal-weight RRF overrules a confident semantic match in conflict cases, while semantic alone is weaker on agree cases. Defaults unchanged; evaluation only, never tune on the personas. See [SEMANTIC_LEDGER_PLAN.md §12.5](../docs/SEMANTIC_LEDGER_PLAN.md#125-results-of-0-fusion-dialogue-evaluation-9-november-2026).

Confidence-weighted distance (8 November 2026): new M1 algorithm `weighted_distance@kappa3_v1` (`recommender/models/weighted_distance.py`, preset `configs/ranking/m1_weighted_distance.json`) scores -sum over observed t of w_t (3 u_t - z_t)^2 with the tracker slot weight w_t; it is `unavailable` without weights (oracle inputs). Optional `RankingRequest.slot_weights` / `ComponentInput.weights` carry the weights (here from format-3 replay sessions; System and terminal wiring land with the separate tracker/policy change); dot product and cosine ignore them and stay frozen. Tier 2 (`data/eval/extractor_sessions_dev_v3`, v2 readings replayed with weights; vectors identical): MRR .143 vs cosine .117 and dot .112; on the 51 profiles with three or fewer observed types distance minus cosine is MRR +.056 [+.014, +.105] and minus dot +.065 [+.025, +.114]; on four or more types differences are near zero with CIs spanning it. Defaults unchanged. See [report](../reports/ranking/weighted_distance_v1/README.md). Tier 3 (`python -m eval.dialogue_distance`, 120 simulated dialogues per set; dot/cosine dialogues reproduce dialogue_probing_v1 exactly): the primary hypothesis is rejected. Full arm, distance in loop + final versus cosine in loop + final: MRR .158 vs .234 (-.076 [-.122, -.036]); distance still beats the dot product (+.066 [+.019, +.111]). Every final profile has four or more observed types, so the sparse regime where distance won at Tier 2 does not occur at the final turn; probing helps distance less than cosine (+.018 vs +.035 MRR). The in-loop ranker barely changes the dialogues. Cosine stays the default; see [Tier 3 report](../reports/ranking/dialogue_distance_v1/README.md).

Cosine live default (8 November 2026): `configs/ranking.yaml` now selects `cosine_similarity@baseline_v1` (fusion null; identical to `configs/ranking/m1_cosine.json`) for the live app, terminal and zero-argument evaluator. Evidence: simulated multi-turn dialogues with the real System and NLI give final-turn MRR .234 vs .092 (+.142 [+.087, +.206]); probing helps only with cosine (+.035 [+.010, +.066]). `weighted_dot_product@baseline_v1` stays frozen as the programmatic `BASELINE_CONFIG` and `configs/ranking/baseline_v1.json`; the workflow's `configs/ranking/active.json` and initial `candidate.json` now also select cosine with fusion disabled, aligning workflow chat/evaluation with the web app and ordinary terminals. This is an explicit configuration alignment using the existing cosine variant and saved dialogue evidence, not a new automated promotion result. Later workflow promotions can change the workflow selection independently of the app YAML. See [decision](../reports/ranking/cosine_default/DECISION.md) and [dialogue report](../reports/ranking/dialogue_probing_v1/FINDINGS.md).

Familiarity-aware profile policies (8 November 2026): a ranking-owned Stage A `profile_policy` (`recommender/profile_policy.py`; config key alongside `evidence_filter`) transforms only the tracker vector before unchanged components, using each slot's tracker `reason`. Frozen `familiarity_excluded_v1` (familiarity slots become unobserved) and `familiarity_half_v1` (weight .5) are evaluated on extractor-derived v2 sessions, which add `slot_reasons`. Cosine + exclusion gives MRR .135 vs .117, Hit@5 .217 vs .175, NDCG@10 .149 vs .134 (MRR CI [-.015, +.058]). It hurts the dot product (MRR -.013, CI excludes zero), and halving hurts both. By subgroup, cosine + exclusion moves MRR +.216 and Hit@5 +.368 for the 19 profiles with a misleading familiarity slot, and MRR -.086 for the 22 whose familiarity slots are genuine. Two profiles lose all observed types. Distinguishing the two cases is the job of probing. Defaults unchanged. See the [report](../reports/ranking/familiarity_policy_v1/README.md).

Extractor-derived inputs (8 November 2026): `scripts/build_extractor_sessions.py` runs each development profile's opening text through the configured DeBERTa extractor and DecayTracker (one turn, no policy) and saves `actual_tracker` sessions in `data/eval/extractor_sessions_dev_v1/`. On these inputs cosine versus dot product is Hit@5 .175 vs .117 (+.058, 95% CI [-.025, +.142]), MRR .117 vs .112, NDCG@10 .134 vs .142. The oracle gap of +.517 Hit@5 shrinks to a non-significant difference, so most of cosine's oracle advantage reflects profiles built from target z-rows. Extraction: 3.61 of 6 types observed on average, 91–99% sign agreement on salient types, top-type agreement .658. Oracle cosine (.683) remains the information ceiling that better profiling could reach; oracle dot product reaches only .167. Defaults unchanged. See the [report](../reports/ranking/extractor_inputs_v1/README.md).

Sentence-to-task reranking (8 November 2026): four frozen `clause_tasks_v1` treatments (MiniLM/MPNet, unsigned/explicit-dislike-signed) apply M4's per-sentence max-task cosine, averaged over sentences, inside the dot-product groups of 10. All four regress versus the dot product on MRR and NDCG@10 (Hit@5 .133–.158 vs .167) and versus the matching `context_tasks_v1` chunk-mean variants. The new reranker diagnostics give a mean within-group target position of 5.83–6.06, worse than random ordering (5.5); the chunk-mean text variants reach about 5.4–5.5. Four rows appended to the consolidated CSV. `app.ranking_terminal --reranking clause-minilm|clause-mpnet[-signed]` shows matched tasks per occupation. Defaults unchanged. See [docs/CLAUSE_TASK_RERANKING.md](../docs/CLAUSE_TASK_RERANKING.md) and the [report](../reports/ranking/clause_tasks_v1/README.md).

Context acceptance revision (2 November 2026): experimental context matching now abstains on unresolved references/preferences, uses explicit setting evidence, and allows no attribute match. Other wording uses a provisional uncalibrated cosine cutoff of .5 instead of forced top-three matches. Informal negation is recognized. Terminal layout, classifier, ranking and fusion are unchanged; see docs/CLAUSE_CONTEXT_PIPELINE.md for limits.

Per-clause context occupation associations (2 November 2026): context-routed clauses display a separate top ten across all local occupations using similarity-weighted available ratings for the three closest context attributes. Coverage and supporting evidence are shown; negative preferences label associated occupations as potential conflicts. These are context associations, not calibrated preference compatibility; RIASEC ranking and task routing remain unchanged. Details: docs/CLAUSE_CONTEXT_PIPELINE.md.

Context clause diagnostics (2 November 2026): terminal occupation-clause details and historical occupation-classifier loading removed. Context-routed user clauses now independently produce a lexical preference sign (unknown for unresolved cases) and 57 ordered MPNet attribute similarities. This is display-only semantic projection, not occupation ratings or ranking integration. Task routing remains independent; its matcher is deferred. See docs/CLAUSE_CONTEXT_PIPELINE.md for semantics and limits.

Stanza terminal baseline (2 November 2026): user clauses now default to offline Stanza dependency parsing with conservative explicit-subject coordination rules (`stanza_conservative_v1`). Embedded complements and shared-subject negation stay together; spaCy remains selectable with `--clause-splitter spacy`. Occupation annotation parsing, shared System, classifier weights and ranking calculations are unchanged. Example-based regressions do not establish general conversational accuracy.

Terminal clause scoring (2 November 2026): `app.ranking_terminal` now displays independent BERT Task/Context sigmoid scores and routes for each exact spaCy clause. Model loads once; this is display-only and does not change extraction, splitting, ranking, fusion or reviewed annotations. Known paraphrase performance remains Task F1 .564103, Context F1 .457831, exact match 35%. The pre-extension ranking snapshot below remains the control.

Pre-extension snapshot (2 November 2026): [ranking and classifier baseline](../docs/PRE_EXTENSION_BASELINE.md) records all ten ranking metrics for all eight existing variants. Dot-product-only Hit@5 = .166667; equal-weight MiniLM/MPNet groups-of-10 Hit@5 = .216667 (MRR .133342 versus .141508 for dot product). Active workflow remains dot-product-only. Current source-labelled task/context BERT has source-validation F1 1.000 for both outputs, but the 80-case assistant-authored paraphrase diagnostic gives Task F1 .564103, Context F1 .457831 and exact match 35% (28/80). These are separate evaluations; conversational reliability and ranking benefit are not established. No new classifier/matching integration or scoring change is included in this snapshot.


Historical occupation clause classifier pilot (1 November 2026; different checkpoint and label definitions from the current task/context BERT): BERT is trained on 123 human-reviewed clauses from 24 occupations and validated on 32 clauses from six disjoint occupations. Ten-epoch pilot selected by validation macro F1: Activity .625, Job nature .893, exact match .500. No Neither training examples or independent test set. Trained checkpoint installed; real terminal smoke confirms ten displayed occupations and 50 underlying candidates. Occupation prose uses conservative sentence boundaries; user-utterance splitting and ranking scores are unchanged. See docs/OCCUPATION_CLASSIFIER_TRAINING.md for datasets, corpus prediction outputs and limitations.

Utterance preprocessing: optional independent spaCy clause splitting is available through
`settings.utterance_processing`. Results are recorded in the turn trace; original text,
RIASEC scoring and reranking are preserved. Pattern classification is deferred.
See [utterance processing](../docs/UTTERANCE_PROCESSING.md).

Recall@50 extension (31 October 2026): evaluator m2-v4 now reports ten permanent
metrics, including macro Recall@5/10/20/50. All eight configurations have verified
Recall@50 extensions derived from saved predictions and fingerprint-matching labels.
Original historical metrics/manifests are preserved. CSV, notebook, history and
new paired comparisons include Recall@50. See [metric extension](../docs/RECALL50.md).


Semantic aggregation (31 October 2026): a separate frozen equal-weight variant
scores the same dot-product top 100 with MiniLM and MPNet, averages raw cosine
scores and sorts once within baseline groups of 10. Both sequential orders and
all previous variants remain available. RIASEC extraction/scoring is unchanged.
All 36 targeted tests pass; the consolidated comparison now has eight experiments.
See the [aggregation report](../reports/ranking/aggregate_minilm_mpnet10_v1/README.md).


Reverse-sequence follow-up (31 October 2026): frozen `minilm20_mpnet10_v1`
adds MiniLM groups of 20 followed by MPNet groups of 10 formed from MiniLM output.
Both stage orders are supported in the separate semantic pipeline. The consolidated
CSV includes seven experiments; all 33 targeted tests pass. See the
[reverse-sequence report](../reports/ranking/minilm20_mpnet10_v1/README.md). Existing RIASEC processing and defaults are preserved.


Semantic integration (31 October 2026): weighted-dot-product RIASEC matching
remains unchanged and separate from `recommender/semantic/SemanticMatchingPipeline`.
The semantic pipeline supports description-only MiniLM and MPNet independently,
or sequential MPNet groups of 20 followed by MiniLM groups of 10 formed from
MPNet's output. Existing extractor/tracker logic and active defaults are preserved.
Evaluator `m2-v3` permanently includes Recall@5/10/20 alongside the six existing
metrics. See [integration guide](../docs/SEMANTIC_INTEGRATION.md) and the consolidated experiment CSV.


Central workflow automatic promotion (26 October 2026): `active.json` starts with
weighted dot product and records the current best central-workflow selection.
Each `python -m recommender.workflow evaluate` evaluates `candidate.json` against
the active incumbent on matching inputs, promoting only higher Hit@5 without any
regression in the other nine permanent metrics. Both reports, comparison and previous
selection are saved; publication is atomic and locked. Workflow chat uses the
winner. Frozen algorithm variants and the separate application's ranking YAML
are preserved. Direct evaluation commands remain report-only. See the
[workflow and recovery guide](../docs/RANKING_MODULES.md#run-the-recommended-configuration).

Application integration audit: real cosine and cosine+BM25 now have HTTP-level
acceptance with explicitly mocked upstream components, covering recommendation
order, component reuse and participant-mode ranking logs. Live System rejects
replay-only `active_only` selection at startup until upstream lifecycle evidence
is available. Defaults and frozen variants are unchanged. See the
[integration audit](../docs/RANKING_INTEGRATION_AUDIT.md).

The ranking implementation is organized into `recommender/models/`, `data/`,
`preparation/`, and `experiments/`, with compatible legacy imports. Central
`configs/ranking/workflow.json` selects the current active configuration
for evaluation and terminal use; legacy production defaults remain
unchanged. Algorithm parameters and additional named variants are centralized in
`configs/ranking/`. Terminal presentation lives in `app/terminal.py` and calls the
shared conversation `System`. Ranking has resource preparation, not model fitting.
See the [module map and iteration workflow](../docs/RANKING_MODULES.md).

This section is the canonical current status; milestone reports preserve their own scope and recorded acceptance results. The architecture and milestone requirements below include planned work.

| Milestone | Current status | Evidence |
|---|---|---|
| M0 â€” Data + contracts | Implemented and accepted locally; teammate integration sign-off pending | [M0 report](../reports/ranking/m0.md) |
| M1 â€” Structured ranking | Dot-product baseline preserved as the frozen control; cosine is the live default since 8 November 2026 ([decision](../reports/ranking/cosine_default/DECISION.md)) | [M1 report](../reports/ranking/m1.md), [cosine comparison](../reports/ranking/m1_cosine/DECISION.md) |
| M2 â€” Evaluation | Implemented and accepted; frozen structured-only oracle development baseline, metrics, exports, and paired comparison | [M2 decision](../reports/ranking/m2/DECISION.md) |
| M3 - Lexical + fusion | Implemented and evaluated; experimental baseline, M1-only default retained | [M3 decision](../reports/ranking/m3/DECISION.md) |
| M4 - Semantic task retrieval | Implemented and evaluated; cached MiniLM/FAISS and three-signal baseline | [M4 decision](../reports/ranking/m4/DECISION.md) |
| M5 | Ranking integration implemented and fixture-tested; baseline frozen; real upstream and latency sign-off pending | [M5 acceptance](../reports/ranking/m5/DECISION.md) |
| M6 | Query filtering and controlled synthetic ablation complete; optional only, representative dialogue validation pending | [M6 decision](../reports/ranking/m6/DECISION.md) |
| M7-M9 | Not implemented | [Development roadmap](DEVELOPMENT.md) |

**Current gate: M5 ranking integration frozen; production upstream completion and teammate latency agreement remain pending.** The M2 acceptance report records 38 passing M0â€“M2 tests. Its 60-profile oracle development baseline reports Hit@5 = 0.166667, MRR = 0.135636, and NDCG@10 = 0.203471. These are offline synthetic-oracle results, not actual-tracker or end-to-end probing results. BM25 and occupation-level RRF are now implemented at M3. M4 now provides cached semantic task retrieval and provisional task matches. Live configurable ranking is wired through the shared System. Evidence stays provisional; production upstream components and stance-aware support remain pending.

M6 currently filters evaluator-supplied, cumulative evidence snapshots before BM25/MiniLM query construction. The default raw-text ranking behavior is unchanged. A separate 60-session mock-tracker stress set now supplies synthetic lifecycle snapshots: six evaluations show mixed effects, including gains from M1 fallback. Keep filtering optional; representative dialogue validation remains pending. See the [M6 decision and next-work recommendation](../reports/ranking/m6/DECISION.md).

The earlier modular refactor originally registered only `weighted_dot_product@baseline_v1`, preserved the `StructuredRanker` compatibility API, and added a shared component interface, registry, configurable pipeline, and fusion boundary subsequently implemented at M3. The refactor originally passed 54 tests; the default-configuration workflow originally passed 61 tests. At M4, the ranking/evaluation suite passed 85 tests. At the initial M5 freeze, full discovery ran 155 tests: 154 passed, with the same pre-existing missing extractor-checkpoint error. All six M5 integration tests pass; see the M5 acceptance report. See the [refactor acceptance report](../reports/ranking/refactor/DECISION.md).

Cosine M1 has also been evaluated with BM25, MiniLM and both using equal-weight RRF k=60. Development/oracle Hit@5 is .266667, .233333 and .116667 respectively, versus .883333 for cosine alone. All three runs are registered in evaluation history; see the [fusion decision](../reports/ranking/m1_cosine/FUSION_DECISION.md). Defaults remain unchanged.

120-profile follow-up: all six M1+M3, M1+M4 and three-way combinations were rerun for dot-product and cosine M1 on identical development/oracle inputs. Every combination regresses against its matching standalone M1 on all six metrics; standalone cosine Hit@5 is .683333. Both histories contain the six new runs. The stale semantic cache was preserved and rebuilt with the same pinned model and parameters. Defaults remain unchanged. See [120-profile combination report](../reports/ranking/combinations_dev120/README.md).

Stance-aware M4 follow-up: independent `minilm_stance@stance_internal_v1` and positive-only `positive_internal_v1` are implemented with conservative runtime text-only extraction, pinned MiniLM resources, and fixed signed task matching. The frozen M4 and defaults are preserved. All 38 targeted tests pass. Five conditions are evaluated on the restored canonical 120-profile dataset and registered in both histories. Hit@5: frozen M4 .041667, positive-only stance .033333, signed stance .033333, cosine .683333, cosine+stance .308333. Signed stance covers 97/120 profiles; defaults remain unchanged. All 45,360 frozen semantic scores/ranks match the earlier canonical-data run. Polarity quality remains unverified without independent review labels. See [stance experiment status](../reports/ranking/stance_dev120_v3/README.md).

## Component configuration

The live app now selects `ranker: configured` in `configs/system.json`, loading the same ranking YAML once at startup. The legacy `structured` factory stays available. M1-only (cosine since 8 November 2026) remains the default. Full app startup still requires the team's production extractor checkpoint, tracker, NLI and policy implementations.

The DeBERTa adapter now maps checkpoint head labels to the shared RIASEC interface and correctly retains positive strengths. Run `python scripts/check_m5_extractor.py --checkpoint <directory>` once trained weights are installed. The real-extractor smoke harness explicitly labels its remaining mock tracker/policy components; see [extractor integration status](../reports/ranking/m5/EXTRACTOR_INTEGRATION.md).

Extractor follow-up verification: 159 tests run, 158 pass; the sole error still requires the absent production extractor checkpoint. All four new checkpoint-adapter tests pass. Actual trained-model inference remains pending installation of the weights.

M5 freezes the ranking control as [`baseline_v1`](../configs/ranking/baseline_v1.json), alongside an independently tested optional three-way RRF preset. See the [acceptance mapping, latency and limitations](../reports/ranking/m5/DECISION.md).

The evaluator automatically loads [`configs/ranking.yaml`](../configs/ranking.yaml) for M1/M3/M4 algorithms, variants, and fusion, and [`configs/m2.json`](../configs/m2.json) for evaluation settings. Edit these files independently, then run from the repository root:

```powershell
python -m recommender.evaluate
```

Reports are created under `reports/ranking/experiments/<UTC timestamp>_<UUID>/`; the command prints the path and metrics. Successful runs append to the central history CSV with selected `algorithm@variant` identifiers. No flags are required. Missing files, malformed configurations, unknown algorithms, and invalid variants fail clearly before evaluation.

YAML support uses pinned PyYAML. One-time setup and activation:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The development workstation's local `.venv` is already prepared; activate it in a new shell. Explicit `--config`, `--ranking-config`, `--output`, and `--history` overrides remain supported. The old `configs/ranking.json` is retained for compatibility, but is not loaded by default.

Each report saves separate `evaluation_config.json`, `ranking_config.json`, and `resolved_ranking.json` snapshots, byte-for-byte copies of original config sources, and individual SHA-256 hashes in `manifest.json`. The existing `config.json` reproduction bundle and its hash remain available; CSV `config_hash` covers that bundle. To reuse the separate snapshots:

```powershell
python -m recommender.evaluate --config <report>/evaluation_config.json --ranking-config <report>/ranking_config.json
```

The registered variants are `weighted_dot_product@baseline_v1`, `bm25@baseline_v1`, and `minilm@mean_tasks_v1`. `null` disables a component; RRF `k=60` is implemented. One usable signal passes through, no usable signal returns `no_signal`, and multiple usable signals use equal-weight occupation-level RRF. M1 scoring and all six M2 metrics are unchanged. Evaluation defaults to development/oracle with the established candidate pool; explicit actual_tracker and mock_tracker snapshot replay is also supported.

In Python, use `RankingPipeline(config=Path("configs/ranking.yaml"))`; existing `StructuredRanker()` callers remain valid. See the [architecture memory](../docs/RANKING_EXPERIMENT_ARCHITECTURE.md) for configuration provenance, variant preservation, and reproduction rules.

### Multi-turn tracker evaluation

The default evaluator remains oracle-only. To evaluate recorded conversations, copy `configs/m2.json`, set `input_mode` to `actual_tracker` and `profiles` to your annotated session JSONL, then pass that evaluation file with `--config`. Keep algorithm selection in the ranking YAML. Use `mock_tracker` for scripted or simulated tracker snapshots; never relabel fixtures as actual tracker runs.

`System.tracker_record(state, profile_id=..., revision=...)` exports chronological user messages and the tracker snapshot actually used each turn, with extractor/tracker identity and caller-supplied code/model revision. Declare the System's input mode explicitly; unverified exports are rejected. Independently add `target` and the usual `related` relevance labels, then write one JSON object per session. Scoring never receives those labels. Runtime values must be in `[-1,1]`; unknown dimensions are distinguished by `observed_types`.

```json
{"id":"session_1","input_mode":"actual_tracker","upstream":{"extractor":"deberta","tracker":"decay","revision":"<code-and-model-revisions>"},"turns":[{"text":"I enjoy repairing machines.","interests":{"R":0.8,"I":0,"A":0,"S":0,"E":0,"C":0},"observed_types":["R"]},{"text":"Actually, I prefer analysing data.","interests":{"R":-0.2,"I":0.9,"A":0,"S":0,"E":0,"C":0},"observed_types":["R","I"]}],"target":"15-2051.00","related":{"short":[],"long":[],"supplemental":[]}}
```

This illustrates the format, not a real tracker capture. The evaluator replays all turns, writes `turn_predictions.json`, and computes unchanged M2 metrics on the final turn of each session. Input mode and upstream provenance are fingerprinted in saved inputs; incompatible conditions are rejected by paired comparison. Central history descriptions identify the mode. Match inputs, split, datasets and upstream revisions before comparing configurations.

### M6 contradiction-aware query experiment

For M6 replay, each turn may include a cumulative `source_evidence` list with one stable record per user utterance: `id`, one-based `turn`, immutable original `text`, and lifecycle `status` (`active`, `disputed`, or `superseded`). A later snapshot may resolve disputed evidence, but cannot rewrite an utterance or reactivate superseded evidence. Select `evidence_filter: raw` or `evidence_filter: active_only` at the top level of the ranking configuration. `active_only` excludes both disputed and superseded utterances and requires BM25 or MiniLM to be enabled; the structured vector remains unchanged. For a controlled comparison, run both filters against the same annotated session file, ranking stack, and tracker snapshots. The report records source evidence and selected/excluded IDs in query diagnostics. A reproducible synthetic development stress set and six saved evaluations are included in `reports/ranking/m6/run_v1/`; these test sensitivity and fallback, while independently annotated real dialogue quality remains untested.

To reproduce the frozen **mock-upstream** integration baseline and optional real three-way ranking benchmark:

```powershell
python scripts/freeze_ranking_baseline.py --output reports/ranking/m5/reproduction
```

This requires the prepared M4 model/cache and writes fresh reports, traces, source snapshots, latency measurements and history rows. It does not evaluate production upstream behavior.

## M3 baseline and measured decision

M3 adds a frozen BM25 baseline over eligible `bm25_doc` and RRF `sum(1/(60+rank))`. Only positive lexical matches enter fusion; empty/stopword/unmatched queries fall back to other available signals. Queries use all user utterances with no evidence, dislike, or familiarity filtering. Parameters/tokenization are recorded in each resolved configuration.

| Same 60 development/oracle cases | Hit@5 | MRR | NDCG@10 |
|---|---:|---:|---:|
| Structured | .166667 | .135636 | .203471 |
| BM25 | .016667 | .016099 | .020544 |
| Structured + BM25 RRF | .133333 | .084887 | .116610 |

Keep the experimental baseline, but **retain M1-only as the default**. See the [M3 acceptance mapping, comparisons and limitations](../reports/ranking/m3/DECISION.md) and [failure review](../reports/ranking/m3/failure_review.md). To evaluate hybrid explicitly:

```powershell
python -m recommender.evaluate --ranking-config configs/ranking/m3_hybrid.json
```

Sibling presets `m3_structured.json` and `m3_bm25.json` support standalone comparisons. You can also enable `m3: {algorithm: bm25, variant: baseline_v1}` in `configs/ranking.yaml`; M4 stays `null`. This evaluation uses oracle vectors and raw profile texts, not the actual extractor/tracker. The live app's registry still chooses the structured facade.

## M4 semantic baseline

M4 is implemented and evaluated; **structured-only remains the default**. On the same 60 development/oracle cases, Hit@5 is 16.67% for structured, 5.00% for dense, 11.67% for structured+dense, 3.33% for BM25+dense, and 5.00% for three-way RRF. These results do not justify default activation or a probing-effectiveness claim.

The frozen `minilm@mean_tasks_v1` variant uses a pinned MiniLM revision, normalized task embeddings, and exhaustive FAISS IndexFlatIP search. It takes the best task match per occupation per natural-language sentence, then averages across sentences. Task matches are provisional, including when a user expresses a dislike. The cache contains 8,243 eligible tasks across 378 occupations. Tasks are embedded once, not during turns.

```powershell
# Fresh environment / checkout only:
python -m pip install -r requirements-ranking.txt
python -m recommender.build_dense --download-model
# Local environment/model/cache are already prepared:
python -m recommender.evaluate --ranking-config configs/ranking/m4_three_way.json
```

Other presets are `m4_dense.json`, `m4_structured_dense.json`, and `m4_bm25_dense.json`. For the default no-argument workflow, set `m4: {algorithm: minilm, variant: mean_tasks_v1}` in `configs/ranking.yaml`, enabling M1/M3 as desired. Caches and local model files are gitignored; build refuses overwriting an existing cache, and runtime rejects stale/corrupt artifacts.

See the [M4 acceptance mapping and measured decision](../reports/ranking/m4/DECISION.md), [failure review](../reports/ranking/m4/failure_review.md), and [reload reproducibility checks](../reports/ranking/m4/reproducibility_check.json). Model/task/index hashes and dependency versions are retained in detailed manifests; original history rows and baselines remain intact. Next is M5 integration, after a subsequent instruction.

## Persistent evaluation history

[`reports/ranking/evaluation_history.csv`](../reports/ranking/evaluation_history.csv) is an append-only registry: one row per successful evaluation run, with duplicate `run_id` values safely skipped. The first row is the original M2 structured-only oracle development baseline, imported from its saved full-pool predictions. Its original artifacts, timestamp, configuration hash, and git revision are preserved. Added cutoffs were recomputed after verifying dataset/config/input hashes and exact agreement with the original three aggregate metrics; no ranking was rerun for this import.

| CSV columns | Meaning |
|---|---|
| `run_id`, `timestamp` | Unique run identity and evaluation time in UTC |
| `m1`, `m3`, `m4` | Structured, lexical, and semantic `algorithm@variant` identifiers in new runs; unused components are empty fields |
| `fusion` | Fusion identifier; empty for standalone ranking |
| `description` | Brief experiment description |
| `candidate_count`, `profile_count` | Eligible pool size and evaluated profile count |
| `hit_at_1`, `hit_at_5`, `hit_at_10` | Exact-target hit rates at each cutoff |
| `mrr` | Full-pool mean reciprocal exact-target rank |
| `ndcg_at_5`, `ndcg_at_10` | Graded NDCG using unchanged relevance grades and eligibility rules |
| `config_hash` | SHA-256 of the saved configuration |
| `dataset_fingerprint` | SHA-256 of the sorted dataset path-to-SHA256 mapping, serialized as compact JSON |
| `git_commit` | Recorded source revision; manifest source hashes also identify uncommitted code |
| `report_path` | Detailed run directory containing configuration, inputs, manifest, predictions, and metrics |

Run from the repository root:

```powershell
python -m recommender.evaluate
```

Every invocation assigns a unique run ID and snapshots both loaded configurations. After all artifacts are saved, it automatically appends to the central history. `--history <path>` selects another registry, for example in isolated tests. Failed ranking/export runs never reach the append step. A registry error leaves completed artifacts available for registration retry:

```powershell
python -m recommender.history <report-path>
python -m recommender.history reports/ranking/m2/baseline --legacy-baseline
```

The second command verifies and registers the original baseline, or safely skips its existing ID. It requires the original fingerprint-matching local datasets. History writers use an exclusive sibling `.lock` file; a busy registry fails without changing rows. If a process is interrupted, verify no writer is running before removing a stale lock and retrying.

Future rankers can call `recommender.history.append_history(report_path)` after exporting a complete run. Set manifest `components` to stable algorithm/variant IDs, e.g. `{"m1": "weighted_dot_product@baseline_v1", "m3": "bm25@baseline_v1", "m4": "minilm@mean_tasks_v1", "fusion": "rrf_k60"}`. M1/M3/M4/RRF are implemented; configurations select which signals participate. The original M2 row retains its unversioned `weighted_dot_product` identifier unchanged; no CSV schema migration is needed. Keep exact parameters, input mode, split, eligibility, fingerprints, and evaluator version in the detailed manifest/configuration. The evaluator uses the common pipeline, defaults to development/oracle, and supports explicit development tracker-snapshot replay; M1, BM25, and MiniLM task retrieval are its implemented scoring components today.

**Comparison guidance:** the CSV is a registry, not a leaderboard. Compare only matching input modes, datasets, candidate pools, and metric definitions. Use `python -m recommender.compare <control-directory> <treatment-directory> --output <new-directory>` for compatible paired runs. Original M2 artifacts retain evaluator `m2-v1` and three metrics; new runs use `m2-v2` and all six. Legacy-to-legacy comparisons still work. Cross-version comparisons are rejected: reproduce the control and treatment with the same evaluator before comparing them. Prediction scores/order reproduce, while run IDs, timing, timestamps, and source metadata vary.

## 1. Three levels of objectives

| Level | Objective | Definition of success | Ownership |
|---|---|---|---|
| **1 â€” Required engineering** | Implement a deterministic, modular, per-turn occupation ranker using **RIASEC fit, BM25, MiniLM task matching, and RRF (k=60)**. | Correct, runnable, reproducible ranking pipeline with occupation-level output, top 100 for policy, and task evidence for recommendations. | Scoring & Ranking |
| **2 â€” Component research** | Test whether **evidence-aware retrieval** improves rankings compared with retrieval over accumulated raw user text. | Controlled comparisons of stale/contradictory evidence handling and familiarity-aware evidence selection, including regressions. | Scoring & Ranking, with upstream evidence metadata |
| **3 â€” End-to-end research** | Test whether **contradiction clarification and familiarity probing** improve final recommendations over accepting the initial user profile. | Evaluated dialogue runs with/without interventions, using the real extractor, state tracker, contradiction check, and policy. | Whole team |

**Do not confuse these claims.** An improvement using manually corrected evidence (Level 2) does not prove the chatbot can reliably elicit that correction (Level 3). Likewise, improving BM25 or fusion alone does not prove probing helps.

## 2. What the ranking component does â€” and does not do

**Inputs (dynamic):** the tracked RIASEC vector (six signed values in `[-1,+1]`), which dimensions have been observed, and user utterances. Later milestones may also consume sentence-level evidence metadata such as stance, reason, confidence, lifecycle status, and source turn.

**Inputs (static):** eligible O*NET occupation identifiers, occupation RIASEC z-scores, text fields used for BM25, task statements, and precomputed MiniLM task embeddings/index.

**Outputs:** a ranked list of eligible O*NET occupation codes, component rankings/scores, fused ranking, and linked supporting task statements. The dialogue policy consumes the top 100; displayed recommendations can use the top five and their supporting tasks.

**Boundary:** this component **does not** train the RIASEC extractor, decide whether two statements contradict, maintain the canonical user profile, or choose the next dialogue question. It consumes upstream outputs. The ranking code may select/filter eligible evidence, but the underlying classification and evidence-resolution authority must be agreed with upstream teammates.

**Important:** RRF itself consumes *occupation rankings*, not raw O*NET texts. O*NET data is used by the three individual rankers and later to attach metadata/evidence.

## 3. Pipeline at a glance

```text
                        USER CONVERSATION
                                |
                  Extractor + State Tracker
                     Contradiction Check
                                |
           updated RIASEC profile + user utterances
            (+ evidence labels/statuses when ready)
                                |
             [Evidence selection: OPTIONAL M6+]
                                |
               +----------------+----------------+
               |                |                |
         Structured fit      BM25          MiniLM + FAISS
       user RIASEC vs     user text vs     user sentences vs
       occupation scores occupation text  O*NET task embeddings
               |                |                |
         occupation       occupation       matched tasks ->
         ranking          ranking          occupation ranking
               |                |                |
               +----------------+----------------+
                                |
                   RRF fusion (k = 60)
                                |
            ranked O*NET occupations + task evidence
                                |
             Dialogue Policy (top 100) / UI (top 5)
                                |
                    next question or results
```

**Required baseline (M1â€“M5):** RIASEC â†’ BM25 â†’ dense task retrieval â†’ occupation-level aggregation â†’ equal-weight RRF â†’ integration. This baseline must remain runnable and evaluable even after optional extensions are introduced.

**Experimental extensions (M6â€“M8):** Use lifecycle/stance/reason metadata to choose *which* sentences influence text retrieval, optionally separating positive and negative channels. Keep the original rankers and fusion available as controls.

## 4. Non-negotiable research guardrails

1. **Every feature must answer one of three questions:** Does it fulfill a required component capability? Does it improve a measured ranking outcome? Does it directly test the stated research question? If none apply, defer it.
2. **Build before optimizing.** Start with RIASEC-only MVP, then complete the proposal's three-signal baseline before testing evidence-aware variants. Do not make novel extensions prerequisites for core functionality.
3. **One material change per experiment.** Specify the hypothesis, control, treatment, and expected effect *before* looking at results.
4. **Keep comparisons fair.** Hold the candidate pool, test cases, tracker values, embedding model, and evaluation rules fixed when isolating evidence-filter changes. Document any intentional exceptions.
5. **Separate correctness, relevance, and latency.** Passing tests is mandatory; a feature need not improve accuracy to be a valid experiment. Record negative results and regressions.
6. **Do not overclaim.** Oracle/manual evidence results, actual-tracker results, and full dialogue results are distinct evaluation conditions.
7. **Reproduce every run.** Record code revision, configuration, data fingerprint/split, model revision, metrics, per-profile predictions, and timing.
8. **Protect held-out evaluation.** Tune on development cases; freeze configurations before final held-out results.
9. **Keep controls executable.** Do not delete baseline retrieval or overwrite previous experiment reports when introducing extensions.

## 5. Milestones and decision gates

| Milestone | What must exist | Research purpose / go-no-go question |
|---|---|---|
| **M0 â€” Data + contracts** | Verified datasets, eligibility rule, typed I/O contract | Can upstream and downstream components exchange the needed data without ambiguity? |
| **M1 â€” MVP** | Runnable RIASEC-only ranker | Does structured fit produce valid, deterministic occupation rankings? |
| **M2 â€” Evaluator** | Verified Hit@5, MRR, NDCG@10 and saved baseline | Can we trust the comparisons before introducing retrieval complexity? |
| **M3 â€” BM25** | Lexical ranker + two-signal RRF | Does lexical matching add useful retrieval signal, and in which cases does it hurt? |
| **M4 â€” MiniLM + FAISS** | Cached task embeddings/index, task-to-occupation aggregation, three-signal RRF | Is semantic retrieval useful beyond structured/lexical ranking at acceptable cost? |
| **M5 â€” Integration** | Per-turn API, top 100 for policy, task evidence and trace | Can the *required proposal baseline* operate in a realistic conversation? |
| **M6 â€” Contradiction-aware evidence** | Status-aware selection/filtering, controlled ablation | Do superseded or disputed statements distort retrieval after the profile is corrected? |
| **M7 â€” Familiarity-aware evidence** | Reason-aware selection/weighting, controlled ablation | Does reducing unconfirmed familiarity improve retrieval without suppressing genuine interests? |
| **M8 â€” Optional signed channels** | Positive/negative text retrieval variant | Does explicitly handling disliked activities add value beyond M6â€“M7? |
| **M9 â€” Freeze + final evaluation** | Frozen configurations, held-out report, end-to-end results if available | Which changes are justified, and what can we actually claim about probing? |

See [DEVELOPMENT.md](DEVELOPMENT.md) for each milestone's implementation requirements, tests, and exit gate.

**Advance / revise / revert:** Advance when acceptance tests pass and the previous change has been evaluated and documented. Revise when results expose a diagnosable problem. Revert or defer a feature when its extra complexity has no demonstrated benefit, or when its effects cannot be evaluated reliably. **Accuracy improvement is not mandatory to complete an experiment.**

### What counts as a complete project?

- **M5:** A complete engineering deliverable for the proposed scoring-and-ranking subsystem, with baselines and metrics.
- **M7:** The main *component-level* evidence-aware research extensions have been implemented and compared with controls (whether or not they win).
- **M9 + team integration:** Evidence sufficient to address the **overall project research question**. Without complete dialogue integration, explicitly mark the end-to-end claim as untested.

Do not spend time on adaptive fusion, ranking-impact question selection, or elaborate explanations unless core milestones are stable and a specific hypothesis warrants the extra effort.

## 6. Evaluation philosophy â€” always retain a control

Every iteration must preserve a **control** and a **treatment**:

| Field | Record |
|---|---|
| Hypothesis | What behavior is wrong or missing, and why would the change help? |
| Control | Exact runnable configuration of the previous/baseline method |
| Treatment | Exact configuration with one intended change |
| Constants | Same profiles, targets, candidate pool, RIASEC inputs, metric definitions, model/index versions where appropriate |
| Measurements | Hit@5, MRR, NDCG@10, affected-case counts, latency, and per-profile rank shifts |
| Decision | Keep / revise / revert, with evidence and known tradeoffs |

For example, M6 tests accumulated raw user text against status-filtered evidence **using the same final RIASEC vector**. Otherwise a gain might come from changing the tracker state rather than filtering the text.

Use [EVALUATION.md](EVALUATION.md) for precise metric definitions, ablation controls and dataset separation, and [EXPERIMENT_TEMPLATE.md](EXPERIMENT_TEMPLATE.md) for a reusable experiment record.

## 7. Dataset and interface assumptions to resolve at M0

The implementation plan anticipates `data/occupations.csv`, `data/tasks.csv`, `data/dev_profiles.jsonl`, `data/profiles.jsonl`, `data/related_occupations.csv`, and `data/labels.json`. Verify what actually exists; this documentation does not create those datasets.

- The project proposal describes **1,016 occupations with RIASEC scores**; the working implementation plan restricts eligibility to **378 Job Zone 4â€“5 occupations** with complete RIASEC data. **Confirm and record** the candidate pool before running benchmarks. Preserve supplied z-scores rather than recomputing them on a restricted pool.
- The plan anticipates **60 development profiles** and **100 held-out profiles**. Verify that each split exists and that conversation histories, target labels, and evidence states are available for evidence-aware experiments. A final RIASEC vector alone cannot test contradiction/familiarity filtering.
- Runtime tracker scores in `[-1,+1]` are **not interchangeable** with synthetic oracle `hidden_slots` expressed on a z-scale. Use a separate oracle adapter and report results separately.
- Unknown dimensions must remain distinguishable from observed neutral values (e.g., `observed_types`).
- Choose and document task-to-occupation score aggregation before interpreting dense retrieval results.

## 8. Suggested modular code layout

```text
recommender/
  data_loader.py           # O*NET validation and joins
  schemas.py               # input/output contracts
  structured.py            # signed RIASEC scoring
  bm25_ranker.py           # lexical occupation ranking
  dense_ranker.py          # MiniLM / FAISS task matching
  fusion.py                # occupation-level RRF
  evidence.py              # OPTIONAL M6+ evidence selection
  ranker.py                # orchestration / public API
  evaluate.py              # metric and benchmark entry point
  cli.py                   # local reproducible demos
configs/                   # versioned configurations
artifacts/ranking/          # index/cache metadata, gitignored as appropriate
reports/ranking/
  experiments/             # experiment manifests and per-profile results
  final/
tests/
  fixtures/
  test_structured.py
  test_metrics.py
  test_bm25.py
  test_dense.py
  test_fusion.py
  test_evidence.py
```

These names are suggestions, **not confirmation that the modules already exist**. Implement only the modules needed for the active milestone.

## 9. Quick start

M0â€“M2 are implemented. Contracts and validation live in `recommender/schemas.py`, `recommender/components.py`, `recommender/data_loader.py`, and `recommender/audit.py`; structured scoring in `recommender/m1.py`, with the compatibility facade in `recommender/structured.py`; registry and orchestration in `recommender/registry.py` and `recommender/ranker.py`; evaluation and comparison in `recommender/evaluation.py`, `recommender/evaluate.py`, and `recommender/compare.py`.

From the repository root with Python 3.10+:

```powershell
python -m unittest discover -s tests -v
python -m recommender.cli
python -m recommender.evaluate
```

Automatic output paths are unique; use a fresh directory when overriding `--output`. See the [M2 report](../reports/ranking/m2/DECISION.md) for the machine-specific Python path and comparison command.

1. Read the objectives, current status above, and [DEVELOPMENT.md](DEVELOPMENT.md).
2. Confirm the O*NET eligibility rule and upstream/downstream contracts with teammates.
3. Preserve the accepted **M1 RIASEC MVP** and **M2 evaluation baseline** as controls for subsequent milestones.
4. For each material change, copy [EXPERIMENT_TEMPLATE.md](EXPERIMENT_TEMPLATE.md) into `reports/ranking/experiments/`, run the control and treatment, and record the decision.
5. Keep M5 as the required completion checkpoint; evaluate M6/M7 as research extensions; freeze before M9.

**Guiding question for every iteration:** *Does this help deliver a correct occupation-ranking system or test whether conversational clarification improves recommendations? If not, why are we building it?*
