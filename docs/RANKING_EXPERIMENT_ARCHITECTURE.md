# Ranking experiment architecture and repository memory

Read this document before changing ranking architecture. It is the authoritative guide to component boundaries, variant versioning, and experiment provenance. The [project README](../cs425-ranking-docs/README.md#current-implementation-status) remains the canonical milestone status summary; keep the current-status section here consistent with it. Milestone reports preserve historical acceptance evidence.

## Current implementation status

- M0: data validation and local contracts implemented; external teammate sign-off remains pending.
- M1: signed weighted dot product implemented as `weighted_dot_product@baseline_v1`.
- M2: development/oracle evaluation, six metric cutoffs, reproducible exports, paired comparison, and append-only history implemented.
- The modular refactor and automatic configuration workflow preserve the accepted M1 formula, response contract, and M2 metrics. All 61 tests pass, including exact comparison of all 22,680 saved oracle baseline occupation results and a no-argument CLI test.
- M3 is implemented: independent `bm25@baseline_v1`, positive-only lexical ranking, and equal-weight occupation-level RRF at `k=60`. Three matching development comparisons are saved in `reports/ranking/m3/`. Hybrid regresses; M1-only remains the default.
- M4 semantic retrieval, MiniLM, embeddings, FAISS, and cosine similarity remain unimplemented.
- RRF now combines two or more usable occupation rankings. Inactive signals do not contribute; a single usable signal bypasses fusion.
- M3 correctness and experiment gates are met. Stop before M4; update this section when later milestones are accepted. The current ranking/evaluation suite passes 72 tests. Broader API/model tests require the team dependencies and model artifacts.

## Architecture and boundaries

| Module | Responsibility |
|---|---|
| `recommender/components.py` | Common input, scored occupation, component result, and scoring protocol; no algorithm dependencies |
| `recommender/bm25_ranker.py` | Independent BM25 baseline, frozen tokenizer/stopwords, eligible in-memory index and positive-only lexical ranks |
| `recommender/m1.py` | Independent weighted-dot-product algorithm and its explicit baseline parameters |
| `recommender/registry.py` | Algorithm factories, named parameter variants, deterministic resolution, and duplicate-registration rejection |
| `recommender/configuration.py` | Default file paths and safe JSON/YAML loading with duplicate-key rejection |
| `recommender/ranker.py` | Request validation, enabled component construction/invocation, candidate validation, standalone/fusion dispatch, and response assembly |
| `recommender/fusion.py` | Equal-weight RRF at k=60, fusion protocol and usable-signal dispatch |
| `recommender/structured.py` | Backward-compatible M1-only `StructuredRanker` facade |
| `recommender/schemas.py` | Existing runtime/oracle inputs and policy-facing response envelope |
| `recommender/data_loader.py` | Candidate eligibility, validation, and static data audit |
| `recommender/evaluation.py` | Algorithm-independent ranking metrics and aggregation |
| `recommender/evaluate.py` | Fixed development/oracle input adapter, pipeline invocation, export, provenance, and history registration |
| `recommender/compare.py` | Compatible-run checks and paired metric differences/bootstrap intervals |
| `recommender/history.py` | Shared append-only CSV registration and original baseline import |

The dependency direction is contracts → algorithms/registry → pipeline → evaluator. Scoring components never call another scoring component, fusion, or the evaluator. The metric layer consumes occupation codes and evaluator-side labels, regardless of which component produced the codes.

Upstream owns extraction, confidence merging, contradiction resolution, the cumulative RIASEC profile, and user-only chronological history. Recompute rankings from each snapshot; do not accumulate occupation scores across turns or apply confidence merging again. Downstream policy/UI consume `RankingResponse`; actual integration and evidence generation remain future work.

### Component input and output

`ScoringComponent.score(ComponentInput) -> ComponentRanking` is the shared interface.

`ComponentInput` contains optional `interests`, an observed-dimension frozenset, and a tuple of user utterances. A component reads only the fields it needs. M1 ignores text; future lexical/dense components can ignore the structured profile. Target occupations, relevance labels, and hidden evaluation annotations must never enter this interface. The pipeline validates runtime values through `RankingRequest`, preserves oracle z-scale values through `OracleInput`, and supplies independent snapshots to each component.

Each `ComponentScore` holds occupation `code`, numeric `score`, and one-based `rank`. A `ComponentRanking` explicitly declares:

- `usable`: nonempty ranking with unique occupation codes, finite scores, contiguous one-based ranks, sorted by descending score then ascending code.
- `unavailable`: a required input/resource is absent; empty ranking and reason.
- `unusable`: inputs exist but provide no usable signal; empty ranking and reason.

The pipeline validates that every emitted occupation belongs to the same eligible pool. Do not fill an inactive signal with zero-score rankings. Negative M1 scores remain valid, and observed zero stays distinct from unknown even though both contribute zero. A future lexical query with no matches must return `unusable`, rather than an arbitrary code-sorted list. Components return their full intended ranking before the pipeline applies policy `top_k`; do not truncate M1 to 100 before offline MRR evaluation.

No usable components gives `RankingResponse("no_signal", reason=...)`. One usable component passes through without fusion, preserving raw component scores/ranks and `rrf_score=None`. Two or more use the implemented equal-weight RRF baseline. `Fusion.fuse(mapping_of_usable_rankings)` returns the shared occupation ranking; fusion must operate on occupation ranks, not raw texts or incompatible raw scores. Component diagnostics remain available in `RankedOccupation.signals` (`structured`, `lexical`, `semantic`). `TaskEvidence` already exists in the policy schema; task retrieval/evidence attachment is not part of this refactor.

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

All three component keys are explicit; `null` disables construction and invocation. Fusion `null` is allowed for standalone execution. RRF is implemented and is bypassed with one usable signal; multiple usable signals use its occupation ranks. Unknown algorithms/variants, invalid schemas/settings, duplicate keys, and unsafe/malformed YAML fail clearly before profile evaluation. Candidate eligibility and development/oracle-only evaluation remain unchanged. Editable evaluation run prefixes and seeds are validated; held-out/actual-tracker adapters remain separate future work.

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

The default registry registers `weighted_dot_product@baseline_v1` and `bm25@baseline_v1`; the default selection remains M1-only because the M3 hybrid regressed. Tests use a temporary alias to exercise switching; that alias is not a production variant or a new algorithm. The baseline parameters fix dimension order `R,I,A,S,E,C`, observed-only scoring, supplied occupation z-scores, and ascending-code tie-breaking. The algorithm rejects unsupported parameter settings rather than pretending to support unimplemented formulas.

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

Do not change Job Zone 4–5 eligibility or supplied z-scores during an algorithm experiment. Runtime `[-1,1]` tracker inputs and oracle z-scale inputs remain distinct. Never tune on held-out profiles or feed target/relevance labels into components. Passing correctness tests does not prove recommendation quality or probing effectiveness.

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

M3 is complete; see [its acceptance and decision report](../reports/ranking/m3/DECISION.md). Select structured-only, lexical-only or hybrid with the saved `configs/ranking/m3_*.json` presets, or enable `m3: {algorithm: bm25, variant: baseline_v1}` in the ranking YAML. M1's legacy facade and default selection stay unchanged. The live application's `pipeline/registry.py` still selects `StructuredRanker`; full app wiring remains an integration follow-up.

BM25's baseline is frozen in `recommender/bm25_ranker.py`: `k1=1.5`, `b=.75`, positive smoothed IDF, lowercase ASCII/apostrophe tokens, an explicit stopword list retaining negators, no stemming, and linear query term counts across accumulated user text. Only eligible prepared documents are indexed. Missing documents fail when BM25 is enabled. All strictly positive matches are ranked; no zero padding or lexical cutoff. Raw dislike and familiarity wording is not filtered, and cannot be interpreted semantically by BM25. Index metadata and resolved parameters are saved in each experiment manifest.

Keep the negative baseline result: structured Hit@5=.166667, BM25=.016667, hybrid=.133333 on the same 60 development/oracle inputs. Do not claim actual-tracker or probing effectiveness. Next planned milestone is M4 semantic task retrieval, after a subsequent instruction. Lexical tuning requires a new preserved variant and a separate hypothesis/control comparison.
