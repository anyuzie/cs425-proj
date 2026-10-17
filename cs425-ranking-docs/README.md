# CS425 — Occupation Scoring & Ranking

> **Project objective:** Build and evaluate a conversational occupation-ranking component that uses a user's evolving RIASEC interest profile and relevant conversation evidence to rank O*NET occupations after each turn.
>
> **Research question (whole project):** Can probing contradictions and challenging familiarity-based justifications give a better occupation ranking than accepting the user's stated profile as given?

This README is the **source of truth for objectives, scope, implementation status, and milestone priorities**. See [DEVELOPMENT.md](DEVELOPMENT.md) for acceptance criteria and implementation steps, [EVALUATION.md](EVALUATION.md) for controls and metrics, and [EXPERIMENT_TEMPLATE.md](EXPERIMENT_TEMPLATE.md) for recording each experiment. The [ranking architecture memory](../docs/RANKING_EXPERIMENT_ARCHITECTURE.md) is the authoritative guide to module boundaries, configuration variants, and future implementation workflows; read it before architectural changes.

## Current implementation status

This section is the canonical current status; milestone reports preserve their own scope and recorded acceptance results. The architecture and milestone requirements below include planned work.

| Milestone | Current status | Evidence |
|---|---|---|
| M0 — Data + contracts | Implemented and accepted locally; teammate integration sign-off pending | [M0 report](../reports/ranking/m0.md) |
| M1 — Structured ranking | Implemented and accepted; deterministic RIASEC-only runtime ranker and mock CLI | [M1 report](../reports/ranking/m1.md) |
| M2 — Evaluation | Implemented and accepted; frozen structured-only oracle development baseline, metrics, exports, and paired comparison | [M2 decision](../reports/ranking/m2/DECISION.md) |
| M3–M9 | Not implemented; next milestone is M3 (BM25 and two-signal RRF) | [Development roadmap](DEVELOPMENT.md#m3--bm25-and-two-signal-fusion) |

**Current gate: stop before M4.** The M2 acceptance report records 38 passing M0–M2 tests. Its 60-profile oracle development baseline reports Hit@5 = 0.166667, MRR = 0.135636, and NDCG@10 = 0.203471. These are offline synthetic-oracle results, not actual-tracker or end-to-end probing results. BM25 and occupation-level RRF are now implemented at M3. Dense indexing, task evidence generation, and live configurable-hybrid integration remain pending.

The earlier modular refactor originally registered only `weighted_dot_product@baseline_v1`, preserved the `StructuredRanker` compatibility API, and added a shared component interface, registry, configurable pipeline, and fusion boundary subsequently implemented at M3. The refactor originally passed 54 tests; the default-configuration workflow originally passed 61 tests. The current ranking/evaluation suite passes 72 tests; broader team API/model tests require their dependencies and model artifacts. See the [refactor acceptance report](../reports/ranking/refactor/DECISION.md).

## Component configuration

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

The registered variants are `weighted_dot_product@baseline_v1` and `bm25@baseline_v1`. `null` disables a component; RRF `k=60` is implemented. One usable signal passes through, no usable signal returns `no_signal`, and multiple usable signals use equal-weight occupation-level RRF. M1 scoring and all six M2 metrics are unchanged. Evaluation remains development/oracle-only with the established candidate pool.

In Python, use `RankingPipeline(config=Path("configs/ranking.yaml"))`; existing `StructuredRanker()` callers remain valid. See the [architecture memory](../docs/RANKING_EXPERIMENT_ARCHITECTURE.md) for configuration provenance, variant preservation, and reproduction rules.

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

Future rankers can call `recommender.history.append_history(report_path)` after exporting a complete run. Set manifest `components` to stable algorithm/variant IDs, e.g. `{"m1": "weighted_dot_product@baseline_v1", "m3": "bm25@baseline_v1", "m4": "minilm@mean_tasks_v1", "fusion": "rrf_k60"}`. M1/M3/RRF are implemented; the MiniLM entry illustrates future M4 usage. The original M2 row retains its unversioned `weighted_dot_product` identifier unchanged; no CSV schema migration is needed. Keep exact parameters, input mode, split, eligibility, fingerprints, and evaluator version in the detailed manifest/configuration. The evaluator uses the common pipeline and remains development/oracle-only; M1 and BM25 are its implemented scoring components today.

**Comparison guidance:** the CSV is a registry, not a leaderboard. Compare only matching input modes, datasets, candidate pools, and metric definitions. Use `python -m recommender.compare <control-directory> <treatment-directory> --output <new-directory>` for compatible paired runs. Original M2 artifacts retain evaluator `m2-v1` and three metrics; new runs use `m2-v2` and all six. Legacy-to-legacy comparisons still work. Cross-version comparisons are rejected: reproduce the control and treatment with the same evaluator before comparing them. Prediction scores/order reproduce, while run IDs, timing, timestamps, and source metadata vary.

## 1. Three levels of objectives

| Level | Objective | Definition of success | Ownership |
|---|---|---|---|
| **1 — Required engineering** | Implement a deterministic, modular, per-turn occupation ranker using **RIASEC fit, BM25, MiniLM task matching, and RRF (k=60)**. | Correct, runnable, reproducible ranking pipeline with occupation-level output, top 100 for policy, and task evidence for recommendations. | Scoring & Ranking |
| **2 — Component research** | Test whether **evidence-aware retrieval** improves rankings compared with retrieval over accumulated raw user text. | Controlled comparisons of stale/contradictory evidence handling and familiarity-aware evidence selection, including regressions. | Scoring & Ranking, with upstream evidence metadata |
| **3 — End-to-end research** | Test whether **contradiction clarification and familiarity probing** improve final recommendations over accepting the initial user profile. | Evaluated dialogue runs with/without interventions, using the real extractor, state tracker, contradiction check, and policy. | Whole team |

**Do not confuse these claims.** An improvement using manually corrected evidence (Level 2) does not prove the chatbot can reliably elicit that correction (Level 3). Likewise, improving BM25 or fusion alone does not prove probing helps.

## 2. What the ranking component does — and does not do

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

**Required baseline (M1–M5):** RIASEC → BM25 → dense task retrieval → occupation-level aggregation → equal-weight RRF → integration. This baseline must remain runnable and evaluable even after optional extensions are introduced.

**Experimental extensions (M6–M8):** Use lifecycle/stance/reason metadata to choose *which* sentences influence text retrieval, optionally separating positive and negative channels. Keep the original rankers and fusion available as controls.

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
| **M0 — Data + contracts** | Verified datasets, eligibility rule, typed I/O contract | Can upstream and downstream components exchange the needed data without ambiguity? |
| **M1 — MVP** | Runnable RIASEC-only ranker | Does structured fit produce valid, deterministic occupation rankings? |
| **M2 — Evaluator** | Verified Hit@5, MRR, NDCG@10 and saved baseline | Can we trust the comparisons before introducing retrieval complexity? |
| **M3 — BM25** | Lexical ranker + two-signal RRF | Does lexical matching add useful retrieval signal, and in which cases does it hurt? |
| **M4 — MiniLM + FAISS** | Cached task embeddings/index, task-to-occupation aggregation, three-signal RRF | Is semantic retrieval useful beyond structured/lexical ranking at acceptable cost? |
| **M5 — Integration** | Per-turn API, top 100 for policy, task evidence and trace | Can the *required proposal baseline* operate in a realistic conversation? |
| **M6 — Contradiction-aware evidence** | Status-aware selection/filtering, controlled ablation | Do superseded or disputed statements distort retrieval after the profile is corrected? |
| **M7 — Familiarity-aware evidence** | Reason-aware selection/weighting, controlled ablation | Does reducing unconfirmed familiarity improve retrieval without suppressing genuine interests? |
| **M8 — Optional signed channels** | Positive/negative text retrieval variant | Does explicitly handling disliked activities add value beyond M6–M7? |
| **M9 — Freeze + final evaluation** | Frozen configurations, held-out report, end-to-end results if available | Which changes are justified, and what can we actually claim about probing? |

See [DEVELOPMENT.md](DEVELOPMENT.md) for each milestone's implementation requirements, tests, and exit gate.

**Advance / revise / revert:** Advance when acceptance tests pass and the previous change has been evaluated and documented. Revise when results expose a diagnosable problem. Revert or defer a feature when its extra complexity has no demonstrated benefit, or when its effects cannot be evaluated reliably. **Accuracy improvement is not mandatory to complete an experiment.**

### What counts as a complete project?

- **M5:** A complete engineering deliverable for the proposed scoring-and-ranking subsystem, with baselines and metrics.
- **M7:** The main *component-level* evidence-aware research extensions have been implemented and compared with controls (whether or not they win).
- **M9 + team integration:** Evidence sufficient to address the **overall project research question**. Without complete dialogue integration, explicitly mark the end-to-end claim as untested.

Do not spend time on adaptive fusion, ranking-impact question selection, or elaborate explanations unless core milestones are stable and a specific hypothesis warrants the extra effort.

## 6. Evaluation philosophy — always retain a control

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

- The project proposal describes **1,016 occupations with RIASEC scores**; the working implementation plan restricts eligibility to **378 Job Zone 4–5 occupations** with complete RIASEC data. **Confirm and record** the candidate pool before running benchmarks. Preserve supplied z-scores rather than recomputing them on a restricted pool.
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

M0–M2 are implemented. Contracts and validation live in `recommender/schemas.py`, `recommender/components.py`, `recommender/data_loader.py`, and `recommender/audit.py`; structured scoring in `recommender/m1.py`, with the compatibility facade in `recommender/structured.py`; registry and orchestration in `recommender/registry.py` and `recommender/ranker.py`; evaluation and comparison in `recommender/evaluation.py`, `recommender/evaluate.py`, and `recommender/compare.py`.

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
