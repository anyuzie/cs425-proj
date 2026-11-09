# Ranking experiment architecture and repository memory

Read this document before changing ranking architecture. It is the authoritative guide to component boundaries, variant versioning, and experiment provenance. The [project README](../cs425-ranking-docs/README.md#current-implementation-status) remains the canonical milestone status summary; keep the current-status section here consistent with it. Milestone reports preserve historical acceptance evidence.

## Current implementation status

Explanation map, Phase A part 1 (9 November 2026, docs/SEMANTIC_LEDGER_PLAN.md §10): explanation-only, no scoring, fusion or default change. `RankingPipeline.rank_with_snapshot` returns the unchanged `rank()` response plus every eligible occupation's fused and component ranks (`rank()` delegates to it; `RankingResponse` and evaluator/CLI exports keep their schema; no snapshot with description reranking or no signal). `System._rank` stores them in each `ranking_trace` row as `rank_snapshot` (integer arrays, null = unranked, in ascending eligible-code order with `codes_sha256`) plus `slot_confidence`; `tracker_record` exports are unchanged. `recommender/build_occupation_map.py` builds `artifacts/ranking/occupation_semantic_v1/occupation_map_v1/` once (t-SNE, cosine, perplexity 30, seed 0, over the fine-tuned index; hexagon from catalog z-scores; manifest with index, checkpoint and catalog hashes; refuses to overwrite; rebuild verified byte-identical). `GET /api/occupation_map` serves the hexagon always and the t-SNE only when its fingerprints match. The debug panel's raw JSON block is replaced by a map with semantic/hexagon tabs, a turn slider, rose/fell/top-10 encoding, component-rank hover/tap details ("moved by" M1, M4 or both; provisional thresholds), the tracker-profile marker and trail on the hexagon, and a slot bar chart. Evidence card, leave-one-turn-out and ledger markers (rest of Phase A, Phase B) remain pending. Tests: `tests/test_occupation_map.py`.

Z-score fusion live default (9 November 2026): `configs/ranking.yaml` now combines M1 `cosine_similarity@baseline_v1` and M4 `occupation_dual_encoder@finetuned_v1` with `zscore_sum@equal_v1` (identical to `configs/ranking/occupation_semantic_fusion_zscore_v1.json`) for the website, terminals and zero-argument evaluator, by explicit owner decision. Evidence: conflict personas MRR .639 vs .114 for RRF (+.525 [+.328, +.722]), agree personas .733 vs .743 (−.010 [−.192, +.136]), task retrieval .507 vs .200; accepted cost: opening-turn development profiles .064 vs .096 (−.032 [−.059, −.011]). Equal-weight RRF k=60 stays selectable (`configs/ranking/occupation_semantic_fusion_v1.json`, `configs/system.rrf.json`); cosine only stays selectable (`configs/system.cosine.json`); workflow selections are unchanged. `RankedOccupation.rrf_score` now holds the z-score sum (field name kept for schema compatibility). See [decision](../reports/ranking/fusion_zscore_default/DECISION.md).

Fusion variants (9 November 2026, #1 in docs/SEMANTIC_LEDGER_PLAN.md §12.3): new fusion selections `{algorithm: rrf_tie_aware, variant: k60_v1}` (tied component scores share their average rank) and `{algorithm: zscore_sum, variant: equal_v1}` (per-query z-scored component scores, equal-weight sum; unranked occupations get the component minimum; constant components contribute 0), registered in `recommender/models/fusion.py` `FUSION_VARIANTS` beside the unchanged `rrf` k=60 control; presets `configs/ranking/occupation_semantic_fusion_{tieavg,zscore}_v1.json`. Nothing tuned. Persona dialogues (`reports/ranking/dialogue_fusion_v2/`, control reproduced exactly): z-score MRR .680 (conflict .639, agree .733) vs control .391 (.114, .743); z-score − control conflict +.525 [+.328, +.722], agree −.010 [−.192, +.136]; tie-aware identical to control (no cosine ties in personas). Single-turn: task retrieval z-score .507 vs control .200 vs semantic .518; 120 development profiles z-score .064 vs control .096 (−.032 [−.059, −.011]). Promoted to the live default the same day, see the next entry. See [decision](../reports/ranking/fusion_variants_v1/DECISION.md).

Fusion dialogue evaluation (9 November 2026, #0 in docs/SEMANTIC_LEDGER_PLAN.md §12.3): `python -m eval.dialogue_fusion` runs 25 team-reviewed personas (`data/eval/fusion_personas_v1.jsonl`, 14 where RIASEC profile and stated topic conflict) through the real System in six arms (RRF fusion / cosine / semantic × probing on/off). Primary metric, fixed in advance: MRR of the first acceptable occupation. Fusion MRR .391 (conflict .114, agree .743), cosine .198, semantic .758 (conflict .881, agree .603); fusion − semantic on conflict −.767 [−.877, −.646], on agree +.141 [−.073, +.366]; probing effects span 0. Equal-weight RRF overrules a confident semantic match in conflict cases, while semantic alone is weaker on agree cases. Defaults unchanged; evaluation only, never tune on the personas. See [SEMANTIC_LEDGER_PLAN.md §12.5](SEMANTIC_LEDGER_PLAN.md#125-results-of-0-fusion-dialogue-evaluation-9-november-2026).

Fusion live default (9 November 2026): `configs/ranking.yaml` now runs M1 `cosine_similarity@baseline_v1` and M4 `occupation_dual_encoder@finetuned_v1` in parallel, combined by equal-weight RRF k=60 (identical to `configs/ranking/occupation_semantic_fusion_v1.json`), for the website, terminals and zero-argument evaluator, by explicit owner decision; it is the new baseline. With no observed RIASEC type the semantic ranking is used alone. On 120 development profiles with real extractor inputs fusion is tied with cosine (MRR .096 vs .117, -.021 [-.059, +.014]) and beats semantic-only (+.056 [+.023, +.092]); on held-out task retrieval .201 vs cosine .032. Single-turn evidence only; negation is not handled by M4. Requires the gitignored local checkpoint/index, otherwise fusion degrades to cosine. Cosine alone remains selectable (`configs/system.cosine.json`); the workflow selection and weight-free mock mode stay on cosine. See [decision](../reports/ranking/occupation_semantic_default/DECISION.md).

Occupation semantic retrieval (9 November 2026): opt-in M4 `occupation_dual_encoder@pretrained_v1` / `@finetuned_v1` (`recommender/occupation_semantic.py`, `training/occupation_semantic.py`) embeds chronological user-only text with MiniLM-L6 and scores all 378 eligible occupations against a validated title/description/task/context index; it runs without extracted preferences or RIASEC and is topic/background association, not preference (negation is not handled). Fine-tuned on source-grouped task/context statements (no dev_profile). Frozen test (1,133 task groups, binary ownership proxy): MRR control .032, pretrained .483, fine-tuned .518 (+.035 [+.020, +.049]), fine-tuned+cosine RRF .201; 38-occupation held-out: .364 to .418. RRF with single-sentence RIASEC hurts on this proxy; conversational relevance is unmeasured and XGBoost is deferred. Terminal/HTTP/restart acceptance pass; browser review pending. Presets `configs/system.occupation-semantic-*-v1.json` and `configs/ranking_terminal/occupation_semantic_*_v1.json`; defaults unchanged. See [report](../reports/ranking/occupation_semantic_v1/README.md).

Confidence-weighted distance (8 November 2026): new M1 algorithm `weighted_distance@kappa3_v1` (`recommender/models/weighted_distance.py`, preset `configs/ranking/m1_weighted_distance.json`) scores -sum over observed t of w_t (3 u_t - z_t)^2 with the tracker slot weight w_t; it is `unavailable` without weights (oracle inputs). Optional `RankingRequest.slot_weights` / `ComponentInput.weights` carry the weights (here from format-3 replay sessions; System and terminal wiring land with the separate tracker/policy change); dot product and cosine ignore them and stay frozen. Tier 2 (`data/eval/extractor_sessions_dev_v3`, v2 readings replayed with weights; vectors identical): MRR .143 vs cosine .117 and dot .112; on the 51 profiles with three or fewer observed types distance minus cosine is MRR +.056 [+.014, +.105] and minus dot +.065 [+.025, +.114]; on four or more types differences are near zero with CIs spanning it. Defaults unchanged. See [report](../reports/ranking/weighted_distance_v1/README.md). Tier 3 (`python -m eval.dialogue_distance`, 120 simulated dialogues per set; dot/cosine dialogues reproduce dialogue_probing_v1 exactly): the primary hypothesis is rejected. Full arm, distance in loop + final versus cosine in loop + final: MRR .158 vs .234 (-.076 [-.122, -.036]); distance still beats the dot product (+.066 [+.019, +.111]). Every final profile has four or more observed types, so the sparse regime where distance won at Tier 2 does not occur at the final turn; probing helps distance less than cosine (+.018 vs +.035 MRR). The in-loop ranker barely changes the dialogues. Cosine stays the default; see [Tier 3 report](../reports/ranking/dialogue_distance_v1/README.md).

Cosine live default (8 November 2026): `configs/ranking.yaml` now selects `cosine_similarity@baseline_v1` (fusion null; identical to `configs/ranking/m1_cosine.json`) for the live app, terminal and zero-argument evaluator. Evidence: simulated multi-turn dialogues with the real System and NLI give final-turn MRR .234 vs .092 (+.142 [+.087, +.206]); probing helps only with cosine (+.035 [+.010, +.066]). `weighted_dot_product@baseline_v1` stays frozen as the programmatic `BASELINE_CONFIG` and `configs/ranking/baseline_v1.json`; the workflow's `configs/ranking/active.json` and initial `candidate.json` now also select cosine with fusion disabled, aligning workflow chat/evaluation with the web app and ordinary terminals. This is an explicit configuration alignment using the existing cosine variant and saved dialogue evidence, not a new automated promotion result. Later workflow promotions can change the workflow selection independently of the app YAML. See [decision](../reports/ranking/cosine_default/DECISION.md) and [dialogue report](../reports/ranking/dialogue_probing_v1/FINDINGS.md).

Familiarity-aware profile policies (8 November 2026): optional ranking config key `profile_policy` (`raw` default; frozen `familiarity_excluded_v1`, `familiarity_half_v1` in `recommender/profile_policy.py`) is applied by `RankingPipeline` to the structured input only, before components; components and text are unchanged, and `raw` output is byte-identical. `RankingRequest.slot_reasons` (optional, validated against observed types) carries the tracker reason; replay sessions may include it per turn; a non-raw policy is rejected for oracle inputs or turns without reasons. Cosine + exclusion: MRR .135 vs .117 (CI spans zero), +.216 MRR on misleading-familiarity profiles, -.086 on genuine ones. See [report](../reports/ranking/familiarity_policy_v1/README.md).

Extractor-derived inputs (8 November 2026): development profile openings are converted to `actual_tracker` sessions by the configured extractor and tracker (`scripts/build_extractor_sessions.py`; one turn, no policy). On these inputs cosine versus dot product is Hit@5 .175 vs .117 (CI spans zero), MRR .117 vs .112, NDCG@10 .134 vs .142, compared with the oracle gap of .683 vs .167. Upstream code stays in `scripts/`; `recommender/` reads only the saved sessions. Defaults unchanged. See [report](../reports/ranking/extractor_inputs_v1/README.md).

Sentence-to-task reranking (8 November 2026): `recommender/semantic/clause_tasks.py` adds four frozen `*_clause_tasks_v1` / `*_clause_tasks_signed_v1` reranking variants via `SemanticMatchingPipeline` (M4 max-task/mean-sentence formula inside dot-product groups of 10). All regress versus the dot product and `context_tasks_v1`; within-group target position is worse than random. Earlier variants are untouched; the terminal can select them with `--reranking`. See [CLAUSE_TASK_RERANKING.md](CLAUSE_TASK_RERANKING.md).

Context/task text reranking experiment (2 November 2026): six separate frozen treatments add all raw task text to description/context prose, using runtime chunk encoding and unchanged weighted-dot-product first-stage groups. All ten metrics are evaluated on the same 120 development/oracle profiles and 378 occupations; six rows are appended to the preserved eight historical rows. No rated Work Context features or classifier scores are inputs. See [experiment report](../reports/ranking/context_tasks_v1/README.md) for actual metrics, chunk aggregation, grouping and limitations. Active defaults and frozen controls remain unchanged.

Context acceptance revision (2 November 2026): experimental context matching now abstains on unresolved references/preferences, uses explicit setting evidence, and allows no attribute match. Other wording uses a provisional uncalibrated cosine cutoff of .5 instead of forced top-three matches. Informal negation is recognized. Terminal layout, classifier, ranking and fusion are unchanged; see docs/CLAUSE_CONTEXT_PIPELINE.md for limits.

Per-clause context occupation associations (2 November 2026): context-routed clauses display a separate top ten across all local occupations using similarity-weighted available ratings for the three closest context attributes. Coverage and supporting evidence are shown; negative preferences label associated occupations as potential conflicts. These are context associations, not calibrated preference compatibility; RIASEC ranking and task routing remain unchanged. Details: docs/CLAUSE_CONTEXT_PIPELINE.md.

Context clause diagnostics (2 November 2026): terminal occupation-clause details and historical occupation-classifier loading removed. Context-routed user clauses now independently produce a lexical preference sign (unknown for unresolved cases) and 57 ordered MPNet attribute similarities. This is display-only semantic projection, not occupation ratings or ranking integration. Task routing remains independent; its matcher is deferred. See docs/CLAUSE_CONTEXT_PIPELINE.md for semantics and limits.

Stanza terminal baseline (2 November 2026): user clauses now default to offline Stanza dependency parsing with conservative explicit-subject coordination rules (`stanza_conservative_v1`). Embedded complements and shared-subject negation stay together; spaCy remains selectable with `--clause-splitter spacy`. Occupation annotation parsing, shared System, classifier weights and ranking calculations are unchanged. Example-based regressions do not establish general conversational accuracy.

Terminal clause scoring (2 November 2026): `app.ranking_terminal` now displays independent BERT Task/Context sigmoid scores and routes for each exact spaCy clause. Model loads once; this is display-only and does not change extraction, splitting, ranking, fusion or reviewed annotations. Known paraphrase performance remains Task F1 .564103, Context F1 .457831, exact match 35%. The pre-extension ranking snapshot below remains the control.

Pre-extension snapshot (2 November 2026): [ranking and classifier baseline](PRE_EXTENSION_BASELINE.md) records all ten ranking metrics for all eight existing variants. Dot-product-only Hit@5 = .166667; equal-weight MiniLM/MPNet groups-of-10 Hit@5 = .216667 (MRR .133342 versus .141508 for dot product). Active workflow remains dot-product-only. Current source-labelled task/context BERT has source-validation F1 1.000 for both outputs, but the 80-case assistant-authored paraphrase diagnostic gives Task F1 .564103, Context F1 .457831 and exact match 35% (28/80). These are separate evaluations; conversational reliability and ranking benefit are not established. No new classifier/matching integration or scoring change is included in this snapshot.


Historical occupation clause classifier pilot (1 November 2026; different checkpoint and label definitions from the current task/context BERT): BERT is trained on 123 human-reviewed clauses from 24 occupations and validated on 32 clauses from six disjoint occupations. Ten-epoch pilot selected by validation macro F1: Activity .625, Job nature .893, exact match .500. No Neither training examples or independent test set. Trained checkpoint installed; real terminal smoke confirms ten displayed occupations and 50 underlying candidates. Occupation prose uses conservative sentence boundaries; user-utterance splitting and ranking scores are unchanged. See docs/OCCUPATION_CLASSIFIER_TRAINING.md for datasets, corpus prediction outputs and limitations.

Utterance preprocessing: optional independent spaCy clause splitting is available through
`settings.utterance_processing`. Results are recorded in the turn trace; original text,
RIASEC scoring and reranking are preserved. Pattern classification is deferred.
See [utterance processing](UTTERANCE_PROCESSING.md).

Recall@50 extension (31 October 2026): evaluator m2-v4 now reports ten permanent
metrics, including macro Recall@5/10/20/50. All eight configurations have verified
Recall@50 extensions derived from saved predictions and fingerprint-matching labels.
Original historical metrics/manifests are preserved. CSV, notebook, history and
new paired comparisons include Recall@50. See [metric extension](RECALL50.md).


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
metrics. See [integration guide](SEMANTIC_INTEGRATION.md) and the consolidated experiment CSV.


Central workflow automatic promotion (26 October 2026): workflow ranking now
uses mutable `configs/ranking/active.json`, initialized to weighted dot product.
`candidate.json` and `promotion.json` select the next experiment and policy.
Workflow evaluation re-evaluates incumbent and challenger under matching conditions
and promotes higher Hit@5 only when none of the other nine permanent metrics regress.
Reports, previous bytes and the comparison are preserved; publication is atomic
and serialized. Workflow chat uses the winner. Direct evaluator calls remain
report-only, and frozen variants plus the separate application YAML are unchanged.
See [promotion usage and recovery](RANKING_MODULES.md#run-the-recommended-configuration).

Application integration audit: real cosine and cosine+BM25 now have HTTP-level
acceptance with explicitly mocked upstream components, covering recommendation
order, component reuse and participant-mode ranking logs. Live System rejects
replay-only `active_only` selection at startup until upstream lifecycle evidence
is available. Defaults and frozen variants are unchanged. See the
[integration audit](RANKING_INTEGRATION_AUDIT.md).

- Modular package layout: implementation now lives in `recommender/models/`,
  `recommender/data/`, `recommender/preparation/`, and `recommender/experiments/`.
  Original imports/CLI entry points remain compatible. Central parameter snapshots
  and additional named variants live under `configs/ranking/`; the new
  `workflow.json` selects the active incumbent for evaluation and terminal use.
  Existing production/default baseline selection is preserved. Terminal I/O lives
  in `app/terminal.py`, delegating to the shared `System.from_config` composition
  root. No ranking fitting is introduced: preparation builds pretrained resources.
  See [module and iteration guide](RANKING_MODULES.md).

- M0: data validation and local contracts implemented; external teammate sign-off remains pending.
- M1: signed weighted dot product preserved as `weighted_dot_product@baseline_v1`; observed-dimension cosine implemented and evaluated separately as `cosine_similarity@baseline_v1`. Cosine development/oracle Hit@5=.883333 versus .166667 for dot product; the dot product stayed the production default pending tracker validation; cosine became the live default on 8 November 2026 after the dialogue evaluation ([decision](../reports/ranking/cosine_default/DECISION.md)). See [cosine decision](../reports/ranking/m1_cosine/DECISION.md).
- M2: development/oracle evaluation, six metric cutoffs, reproducible exports, paired comparison, and append-only history implemented.
- Cosine M1 fusion combinations are evaluated and registered: Hit@5=.266667 with BM25, .233333 with MiniLM, .116667 with both, versus .883333 standalone. Equal-weight RRF regresses in every case; defaults remain unchanged. See [cosine fusion report](../reports/ranking/m1_cosine/FUSION_DECISION.md).
- The modular refactor and automatic configuration workflow preserve the accepted M1 formula, response contract, and M2 metrics. All 61 tests pass, including exact comparison of all 22,680 saved oracle baseline occupation results and a no-argument CLI test.
- M3 is implemented: independent `bm25@baseline_v1`, positive-only lexical ranking, and equal-weight occupation-level RRF at `k=60`. Three matching development comparisons are saved in `reports/ranking/m3/`. Hybrid regresses; M1-only remains the default.
- M4 is implemented and evaluated as `minilm@mean_tasks_v1`: pinned MiniLM, normalized task cache, exhaustive FAISS search, occupation aggregation and provisional evidence. M1 cosine is independently selectable through the shared registry.
- RRF now combines two or more usable occupation rankings. Inactive signals do not contribute; a single usable signal bypasses fusion.
- M4 correctness and experiment gates are met. M5 ranking integration is now implemented and frozen with mocked upstream acceptance; production upstream and latency agreement remain pending. At the initial M5 freeze, full discovery ran 155 tests: 154 passed, with one pre-existing error due to the missing upstream extractor checkpoint. All six M5 integration tests pass. Default selection stays M1-only after measured M3/M4 regressions.
- M6 query selection and the controlled mock-tracker stress ablation are complete: six paired-condition evaluations on 60 development fixtures, with preserved structured inputs and mixed relevance effects. Keep `active_only` optional and the default `raw`; representative annotated dialogue validation remains pending. See [M6 decision](../reports/ranking/m6/DECISION.md).

120-profile follow-up: all six M1+M3, M1+M4 and three-way combinations were rerun for dot-product and cosine M1 on identical development/oracle inputs. Every combination regresses against its matching standalone M1 on all six metrics; standalone cosine Hit@5 is .683333. Both histories contain the six new runs. The stale semantic cache was preserved and rebuilt with the same pinned model and parameters. Defaults remain unchanged. See [120-profile combination report](../reports/ranking/combinations_dev120/README.md).

Stance-aware M4 follow-up: independent `minilm_stance@stance_internal_v1` and positive-only `positive_internal_v1` are implemented with conservative runtime text-only extraction, pinned MiniLM resources, and fixed signed task matching. The frozen M4 and defaults are preserved. All 38 targeted tests pass. Five conditions are evaluated on the restored canonical 120-profile dataset and registered in both histories. Hit@5: frozen M4 .041667, positive-only stance .033333, signed stance .033333, cosine .683333, cosine+stance .308333. Signed stance covers 97/120 profiles; defaults remain unchanged. All 45,360 frozen semantic scores/ranks match the earlier canonical-data run. Polarity quality remains unverified without independent review labels. See [stance experiment status](../reports/ranking/stance_dev120_v3/README.md).

## Architecture and boundaries

### M5 integration control

Extractor follow-up: `pipeline/extractor.py` loads the local trained 45-logit checkpoint once, resolves semantic head indices from `id2label`, and converts predictions into the existing `Reading` contract. This fixes checkpoint/shared-class ordering and positive-strength conversion without changing ranking. `scripts/check_m5_extractor.py` tests real extraction with configured ranking and explicitly mocked remaining upstream components. Actual weights are absent locally, so real inference is pending; see [extractor integration status](../reports/ranking/m5/EXTRACTOR_INTEGRATION.md). The original baseline freeze is preserved.

Follow-up verification: 159 tests run, 158 pass, with one existing missing-checkpoint error; all four new adapter regressions pass. These are interface tests, not trained-model accuracy evidence.

`configs/system.json` selects the `configured` ranker factory. It loads `configs/ranking.yaml` once when `System` is constructed, retaining existing component instances and caches across turns. The legacy `structured` factory remains M1-only. The frozen control selection is `configs/ranking/baseline_v1.json`; optional three-way RRF uses the existing M4 preset without algorithm changes.

`System` sends the current tracker vector, observed types and complete chronological user-only history before policy invocation. Policy receives up to 100 occupations including their RIASEC values. Recommendation cards retain up to five occupations and three provisional task matches each, including structured evidence metadata. Unknown and observed-zero dimensions remain distinct. No confidence merging, score accumulation, stance filtering or contradiction-aware text removal is added.

State retains per-turn input/status/reason/ranking/timing snapshots. Debug traces expose them; server turn logs retain input mode, component identity, resolved ranking and the current snapshot even when participant debug is disabled. `System.tracker_record` exports replayable scoring inputs; independent annotations supply relevance labels. Explicit `actual_tracker`, `mock_tracker` and `unverified_tracker` runtime modes prevent silent attribution to a production tracker; unverified records cannot be exported for evaluation. Known test component names cannot claim actual-tracker mode. Caller-declared provenance still requires external verification.

The evaluator accepts development `actual_tracker` or `mock_tracker` session files with upstream extractor/tracker/revision provenance and validated per-turn runtime snapshots. It reconstructs history from ordered user turns, reranks every turn, exports top-100 intermediate results, and evaluates the final turn once per session at full ranking depth using unchanged M2 metrics. The default remains the fixed oracle adapter. Mode, evaluation unit, raw query policy and provenance/input fingerprints are recorded; the existing comparison checks reject incompatible conditions.

See [M5 acceptance and freeze](../reports/ranking/m5/DECISION.md) and the [session format/workflow](../cs425-ranking-docs/README.md#multi-turn-tracker-evaluation). The frozen ranking control includes exact configurations, input traces, source archive, dependency/data/model/cache fingerprints and local latency. Actual upstream operation and team latency agreement remain pending; fixture metrics are not actual-tracker quality evidence. The M6 query-selection path and controlled synthetic stress experiment are complete; representative dialogue validation and all M7+ evidence experiments remain pending.

| Module | Responsibility |
|---|---|
| `recommender/components.py` | Common input, scored occupation, component result, and scoring protocol; no algorithm dependencies |
| `recommender/bm25_ranker.py` | Independent BM25 baseline, frozen tokenizer/stopwords, eligible in-memory index and positive-only lexical ranks |
| `recommender/m1.py` | Independent weighted-dot-product algorithm and its explicit baseline parameters |
| `recommender/cosine.py` | Independent observed-dimension cosine algorithm, explicit normalization and zero-norm policies |
| `recommender/models/weighted_distance.py` | Independent confidence-weighted z-space distance; reads `ComponentInput.weights` (tracker slot weights) |
| `recommender/registry.py` | Algorithm factories, named parameter variants, deterministic resolution, and duplicate-registration rejection |
| `recommender/configuration.py` | Default file paths and safe JSON/YAML loading with duplicate-key rejection |
| `recommender/ranker.py` | Request validation, enabled component construction/invocation, candidate validation, standalone/fusion dispatch, and response assembly |
| `recommender/fusion.py` | Registered fusion variants (`rrf_k60` control, `rrf_tie_aware@k60_v1`, `zscore_sum@equal_v1`), fusion protocol and usable-signal dispatch |
| `recommender/structured.py` | Backward-compatible M1-only `StructuredRanker` facade |
| `recommender/schemas.py` | Existing runtime/oracle inputs and policy-facing response envelope |
| `recommender/data_loader.py` | Candidate eligibility, validation, and static data audit |
| `recommender/evaluation.py` | Algorithm-independent ranking metrics and aggregation |
| `recommender/evaluate.py` | Fixed development/oracle input adapter, pipeline invocation, export, provenance, and history registration |
| `recommender/compare.py` | Compatible-run checks and paired metric differences/bootstrap intervals |
| `recommender/history.py` | Shared append-only CSV registration and original baseline import |

The dependency direction is contracts â†’ algorithms/registry â†’ pipeline â†’ evaluator. Scoring components never call another scoring component, fusion, or the evaluator. The metric layer consumes occupation codes and evaluator-side labels, regardless of which component produced the codes.

Upstream owns extraction, confidence merging, contradiction resolution, the cumulative RIASEC profile, and user-only chronological history. Recompute rankings from each snapshot; do not accumulate occupation scores across turns or apply confidence merging again. Downstream policy/UI consume `RankingResponse`; System now wires configurable ranking to tracker snapshots and policy output; stance-aware supporting evidence remains future work.

### Component input and output

`ScoringComponent.score(ComponentInput) -> ComponentRanking` is the shared interface.

`ComponentInput` contains optional `interests`, an observed-dimension frozenset, and a tuple of user utterances. A component reads only the fields it needs. M1 ignores text; future lexical/dense components can ignore the structured profile. Target occupations, relevance labels, and hidden evaluation annotations must never enter this interface. The pipeline validates runtime values through `RankingRequest`, preserves oracle z-scale values through `OracleInput`, and supplies independent snapshots to each component.

Each `ComponentScore` holds occupation `code`, numeric `score`, and one-based `rank`. A `ComponentRanking` explicitly declares:

- `usable`: nonempty ranking with unique occupation codes, finite scores, contiguous one-based ranks, sorted by descending score then ascending code.
- `unavailable`: a required input/resource is absent; empty ranking and reason.
- `unusable`: inputs exist but provide no usable signal; empty ranking and reason.

The pipeline validates that every emitted occupation belongs to the same eligible pool. Do not fill an inactive signal with zero-score rankings. Negative M1 scores remain valid, and observed zero stays distinct from unknown even though both contribute zero. A future lexical query with no matches must return `unusable`, rather than an arbitrary code-sorted list. Components return their full intended ranking before the pipeline applies policy `top_k`; do not truncate M1 to 100 before offline MRR evaluation.

No usable components gives `RankingResponse("no_signal", reason=...)`. One usable component passes through without fusion, preserving raw component scores/ranks and `rrf_score=None`. Two or more use the selected fusion variant (`zscore_sum@equal_v1` in `configs/ranking.yaml` since 9 November 2026; equal-weight RRF k=60 before, and still the fusion entry of the programmatic `BASELINE_CONFIG`). `Fusion.fuse(mapping_of_usable_rankings)` returns the shared occupation ranking; fusion operates on occupation-level component rankings, never raw texts. Rank-based variants use ranks only; `zscore_sum@equal_v1` standardises each component's scores per query before combining, so raw scales are never compared directly. New fusion rules are added as new `FUSION_VARIANTS` entries, never by editing an existing one. Component diagnostics remain available in `RankedOccupation.signals` (`structured`, `lexical`, `semantic`). `TaskEvidence` already exists in the policy schema; M4 now attaches provisional task matches; application wiring is implemented at M5; stance-aware supporting evidence remains future work.

### Configuration and selection

The evaluation entry point automatically loads **two separate default files** from the repository root:

- `configs/ranking.yaml`: component algorithm/variant selection and fusion settings.
- `configs/m2.json`: evaluation-specific run prefix, split, input mode, datasets, eligibility, ranking depth, and seed. It contains no scoring implementation selection.

YAML parsing uses pinned `PyYAML==6.0.2` from `requirements.txt` with a safe loader and duplicate-key rejection. Install dependencies once in the project environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The local `.venv` already contains the dependency on the development workstation. Activate it in each new shell before invoking `python`. The editable ranking default is:

```yaml
components:
  m1:
    algorithm: weighted_dot_product
    variant: baseline_v1
  m3: null
  m4: null
fusion:
  algorithm: rrf
  k: 60
```

Edit the ranking YAML to select implementations; edit `configs/m2.json` only for evaluation changes. Then run:

```powershell
python -m recommender.evaluate
```

No config or output arguments are required. Each run creates `reports/ranking/experiments/<UTC timestamp>_<UUID>/`, prints its report path and metrics, and appends one row to the central CSV after all artifacts are saved. Explicit `--config`, `--ranking-config`, `--output`, and `--history` overrides remain available for reproduction and isolated tests. Missing default files are errors; the evaluator never silently falls back to an inline ranking baseline or to `configs/ranking.json`.

All three component keys are explicit; `null` disables construction and invocation. Fusion `null` is allowed for standalone execution. RRF is implemented and is bypassed with one usable signal; multiple usable signals use its occupation ranks. Unknown algorithms/variants, invalid schemas/settings, duplicate keys, and unsafe/malformed YAML fail clearly before profile evaluation. Candidate eligibility and the default development/oracle workflow remain unchanged. Explicit development actual_tracker and mock_tracker replay adapters are implemented; held-out evaluation remains out of scope.

`RankingPipeline(config=Path("configs/ranking.yaml"))` uses YAML selection directly. Its programmatic no-config constructor and `StructuredRanker()` retain the accepted inline M1 baseline for compatibility; the evaluator always passes its loaded selection explicitly. The old `configs/ranking.json` remains a compatibility fixture, not an active default. JSON selection overrides remain supported.

Each new run saves:

| Artifact | Contents |
|---|---|
| `evaluation_config.json` | Validated evaluation settings only |
| `ranking_config.json` | Normalized component selections and fusion only |
| `resolved_ranking.json` | Exact resolved algorithm/variant parameter snapshots |
| `config_sources/evaluation.json` | Exact original evaluation source bytes |
| `config_sources/ranking.yaml` | Exact original ranking source bytes, including comments; suffix follows an overridden source format |
| `config.json` | Compatibility reproduction bundle containing evaluation settings, selection, and resolved parameters |
| `manifest.json` | Source/snapshot hashes, source paths, resolved variants, dataset/input/source fingerprints, code revision and dirty state, dependency versions, evaluator version, and reproduction command |

The manifest separately records `evaluation_config_sha256`, `ranking_config_sha256`, and `resolved_ranking_sha256`. `configuration_sources` records the raw byte hashes and snapshot paths. `config_sha256` and CSV `config_hash` cover the compatibility bundle, tying both configurations and resolved parameters together without changing the CSV schema. Legacy bundled snapshots are accepted explicitly for reproduction, but do not change the two-file default workflow. Saved stale resolved parameters are rejected rather than silently replaced.

To reproduce a new run from its separate snapshots:

```powershell
python -m recommender.evaluate --config <report>/evaluation_config.json --ranking-config <report>/ranking_config.json
```

For strict variant-parameter checking, the saved compatibility `config.json` can also be passed to `--config`; its embedded selection is used and its resolved parameters must match the registry. Use the original code revision and fingerprint-matching datasets for exact implementation provenance. Metrics remain `m2-v2`, with unchanged definitions. Record future model IDs/revisions, tokenizer settings, aggregation, preprocessing, seeds, and static artifact hashes in resolved parameters/manifests when those components exist.

### Registering a component

Use the simple explicit `AlgorithmRegistry`, not dynamic imports or a plugin framework:

1. Implement the scoring protocol in an independent module.
2. Register its factory with `register_algorithm(component, algorithm, factory)`.
3. Register named JSON-safe parameter dictionaries with `register_variant(component, algorithm, variant, parameters)`.
4. Enable a registered selection in `configs/ranking.yaml`. No pipeline dispatch or metric changes are needed for a different registered scoring algorithm/variant.

Factories receive validated eligible occupations and an owned copy of variant parameters. A future text/dense factory should load its own additional artifacts from explicit parameter paths; do not make M1 load task data, text indices, or models. Keep occupation-code joins aligned with the candidate pool and record additional static artifact fingerprints in the run manifest when introducing them. For isolated tests/experiments, pass a separate registry to `RankingPipeline(..., registry=registry)` or `evaluate.run(..., registry=registry)`. Accepted production registrations belong in `default_registry()` so CLI selection works without source edits between already registered variants.

The default registry registers `weighted_dot_product@baseline_v1`, `cosine_similarity@baseline_v1`, `weighted_distance@kappa3_v1`, `bm25@baseline_v1`, and `minilm@mean_tasks_v1`; the default selection (`configs/ranking.yaml`) is M1 cosine only since 8 November 2026; the dot product remains the programmatic baseline. Tests use a temporary alias to exercise switching; that alias is not a production variant or a new algorithm. The M1 parameters fix dimension order `R,I,A,S,E,C`, observed-only scoring, supplied occupation z-scores, and ascending-code tie-breaking. Cosine normalizes both vectors over observed dimensions; zero user norm is unusable and zero occupation norm is omitted. Select `configs/ranking/m1_cosine.json` for cosine. Algorithms reject unsupported parameter settings rather than pretending to support unimplemented formulas.

## Algorithm, variant, and run identity

- **Algorithm** names the mathematical/retrieval approach, e.g. `weighted_dot_product`. Register a different implementation when the actual algorithm changes.
- **Variant** names a reproducible parameter configuration, e.g. `baseline_v1`. Reuse the implementation when only supported settings change; do not create one class per hyperparameter setting.
- **Run ID** names one evaluation execution. It is unique per execution and must not be used as an algorithm version.

Registry registration never overwrites an existing algorithm or variant key. Parameter dictionaries are copied on registration/resolution/build; caller mutation cannot rewrite registered settings. This guards in-process replacement, while Git revisions, saved resolved configurations, tests, and review protect accepted definitions across code changes. Never edit the meaning of an accepted identifier in place. Add a new variant or algorithm, retain the old one, and document the reason and evidence. If a formula change cannot be represented by the same implementation honestly, introduce a separate algorithm.

To roll back selection, restore the earlier algorithm/variant in the selection JSON. To reproduce an exact historical implementation, use its original code revision and fingerprint-matching data/configuration, including recorded source changes for dirty-worktree experiments; a variant name alone does not recreate uncommitted code. Source hashes detect differences but do not store missing source contents. Preserve/commit accepted implementation snapshots alongside their reports.

### Central history mapping

`reports/ranking/evaluation_history.csv` retains its existing schema. New component fields use `algorithm@variant`; unused components and standalone `fusion` stay empty. The original M2 row remains exactly as recorded with `m1=weighted_dot_product`; that legacy identifier maps to the original baseline, not to a new result. Do not rewrite it to add a version suffix. No CSV migration is needed for this refactor.

`config_hash` covers the saved configuration including resolved parameters. Detailed `report_path` artifacts retain everything needed to assess input mode, split, eligibility, and provenance. A successful complete export appends once; duplicate IDs are skipped. New runs get new IDs even for repeated configurations. Tests use temporary registries and do not append test fixtures to the central CSV.

The CSV is not a global leaderboard. `compare.py` checks compatible input mode, split, dataset/input hashes, pool, eligibility, ranking depth, metric definitions, evaluator version, and profile IDs. Keep comparisons controlled beyond these mechanical checks. Original `m2-v1` artifacts retain three metrics; their self/paired comparisons remain supported, but cross-version comparisons are rejected. Reproduce both sides under a matching evaluator rather than altering original artifacts.

## Rules for future development

1. Never modify an accepted baseline silently.
2. Preserve previously accepted configuration variants and their meaning.
3. Keep scoring components independent; load only enabled dependencies.
4. Do not modify M2 metric definitions to accommodate a particular algorithm.
5. Change one material experimental variable at a time.
6. Maintain deterministic ranking, explicit tie-breaking, and reproducible evaluation.
7. Preserve detailed predictions, configuration, provenance, and historical rows.
8. Compare against an appropriate frozen control with matching evaluation conditions.
9. Better standalone performance does not imply better fused performance.
10. Document accepted changes and the evidence supporting them, including regressions and limitations.

Do not change Job Zone 4â€“5 eligibility or supplied z-scores during an algorithm experiment. Runtime `[-1,1]` tracker inputs and oracle z-scale inputs remain distinct. Never tune on held-out profiles or feed target/relevance labels into components. Passing correctness tests does not prove recommendation quality or probing effectiveness.

## Future implementation workflow

1. State the hypothesis and proposed change before inspecting results.
2. Decide whether it is a new algorithm or a supported configuration variant.
3. Register the implementation and explicit versioned parameters, preserving old selections.
4. Run correctness tests, including deterministic ordering, eligibility, signal status, and input/output validation.
5. Evaluate the component independently with the shared M2 metrics.
6. Evaluate it within the fused pipeline where applicable, after implementing and validating RRF at the authorized milestone.
7. Compare against the frozen control with matching inputs, pool, and evaluator.
8. Save complete artifacts, append history, inspect regressions, and record latency and quality separately.
9. Accept, revise, or revert with written evidence; update this memory, README status, and milestone report consistently.

M3 is complete; see [its acceptance and decision report](../reports/ranking/m3/DECISION.md). Select structured-only, lexical-only or hybrid with the saved `configs/ranking/m3_*.json` presets, or enable `m3: {algorithm: bm25, variant: baseline_v1}` in the ranking YAML. M1's legacy facade and default selection stay unchanged. The live application's `pipeline/registry.py` now selects the configurable `RankingPipeline`; the legacy structured factory remains available.

BM25's baseline is frozen in `recommender/bm25_ranker.py`: `k1=1.5`, `b=.75`, positive smoothed IDF, lowercase ASCII/apostrophe tokens, an explicit stopword list retaining negators, no stemming, and linear query term counts across accumulated user text. Only eligible prepared documents are indexed. Missing documents fail when BM25 is enabled. All strictly positive matches are ranked; no zero padding or lexical cutoff. Raw dislike and familiarity wording is not filtered, and cannot be interpreted semantically by BM25. Index metadata and resolved parameters are saved in each experiment manifest.

Keep the negative baseline result: structured Hit@5=.166667, BM25=.016667, hybrid=.133333 on the same 60 development/oracle inputs. Do not claim actual-tracker or probing effectiveness. M4 was subsequently implemented and evaluated; M5 application wiring is now implemented with fixture acceptance. Lexical tuning requires a new preserved variant and a separate hypothesis/control comparison.

## M4 implementation and experiment memory

See [M4 decision and acceptance mapping](../reports/ranking/m4/DECISION.md) for complete evidence. New modules are `dense_config.py` (frozen parameters and lazy factory), `dense_ranker.py` (local encoder, validated cache, exhaustive search and aggregation), and `build_dense.py` (explicit one-time download/build). M1/M3 do not import model dependencies when M4 is disabled.

`minilm@mean_tasks_v1` pins `sentence-transformers/all-MiniLM-L6-v2` at `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. CPU float32, one thread, seed 425, batch 32, max sequence length 256, normalized 384-dimensional vectors. The exact same local snapshot encodes both tasks and query sentences. Preserve this variant and its parameters; changes require a new variant/cache path.

Only eligible tasks are indexed, sorted by `(occupation code, task ID)`. The natural-language splitter uses punctuation/newlines, preserving negation, wording and repetition; no BM25 stopwords or evidence filters apply. Search all cached tasks with `IndexFlatIP`; per sentence, take the maximum task similarity for each occupation, then average those maxima across sentences. Retain signed similarity scores. Missing-task occupations have no dense rank. Empty/punctuation-only text gives an unusable signal without any encoder call. No task corpus embedding or model download occurs in a turn.

`ComponentRanking.evidence` carries task IDs, occupation ownership, text, score and provisional status. `RankingPipeline` attaches up to three matches per occupation to the existing `RankedOccupation.evidence`. Task evidence uses maximum similarity across sentences with task-ID tie-breaking, distinct from the occupation mean-of-maxima formula. Never call these matches supporting evidence solely because they are semantically similar; a dislike can match strongly.

Install optional dependencies from `requirements-ranking.txt`; the exact evaluation/validation environments are saved in the M4 report. On a fresh checkout, run `python -m recommender.build_dense --download-model` once. The current workstation is already prepared. The gitignored `artifacts/ranking/` cache/model snapshot is validated against task source/mapping hashes, pool, model revision/file hashes, parameters, dependency versions and serialized artifact integrity. Build protects existing caches. Publication inherits project permissions and is atomic within the cache parent. Do not manually alter a cache to reuse a different model or dataset under the same identity.

Select the M4 presets in `configs/ranking/m4_*.json`, or set `m4: {algorithm: minilm, variant: mean_tasks_v1}` in the default YAML. Evaluation still consumes the same M2 request/metric interfaces, saves exact resolved snapshots, and appends real runs to the existing CSV. The shared CSV dataset fingerprint identifies evaluation profiles/occupation data; auxiliary task/model/index fingerprints live in detailed manifests. Paired runs that both use dense retrieval reject a changed task-corpus hash. Reviewers still verify fixed model/aggregation conditions for evidence experiments.

Seven saved development conditions cover all standalone/pairwise/three-way combinations. The three earlier controls reproduce exactly; all four dense conditions reproduce exact predictions (including provisional task matches), inputs and metrics after cache reload. Dense-only Hit@5=.05 versus BM25=.016667, but structured+dense=.116667 and three-way=.05 remain worse than structured=.166667. Keep M1-only default. Record negative results; diagnose aggregation/evidence before elaborate ANN or adaptive fusion.

The live app factory now selects configurable ranking. The upstream extractor checkpoint is absent, and tracker/NLI/policy completion and latency agreement remain team integration work. No held-out tuning, actual-tracker evaluation, or end-to-end probing claim is made.
