# M4 — MiniLM task retrieval and three-signal baseline

**Decision: M4 baseline implemented, tested and evaluated. Retain it for experiments; keep structured-only as the default. Stop before M5.** Dense improves on lexical-only in this sample, but dense fusion does not justify replacing the structured control.

The applicable instruction checklist is [DEVELOPMENT.md, M4](../../../cs425-ranking-docs/DEVELOPMENT.md#m4--minilm--task-level-dense-retrieval). The [pre-evaluation protocol](EXPERIMENT.md) freezes the hypothesis, model, aggregation and comparison conditions. See the [failure review](failure_review.md) for explicit dislikes, misleading/genuine familiarity, and gains.

## Implemented architecture

- `recommender/dense_config.py`: optional-dependency-safe registration of `minilm@mean_tasks_v1`, with all parameters explicit.
- `recommender/build_dense.py`: explicit model download and immutable offline cache build. Existing caches are protected from overwrite; publication uses a complete staging directory with inherited project permissions.
- `recommender/dense_ranker.py`: validated eligible-task mapping, local pinned model loading, normalized embeddings, cached FAISS `IndexFlatIP`, exhaustive retrieval, occupation aggregation and provisional task matches.
- Shared component output now carries validated task evidence; `RankingPipeline` attaches it to the existing response envelope. M1/M3 response scores and rankings remain unchanged.
- The existing registry, evaluator, RRF and history support dense-only and fused configurations. Manifests record model revision, task-index fingerprint, task-source hash, cache metadata and actual package versions. Paired comparison rejects changed auxiliary task corpora when both runs use dense retrieval.

No model dependencies are imported by disabled M4. No model download or task embedding happens in a turn. Cache/model absence or fingerprint/configuration drift fails clearly at startup, before a successful evaluation can be registered. M1/M3 remain runnable without the optional M4 environment.

## Frozen model and retrieval

Model: [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/1110a243fdf4706b3f48f1d95db1a4f5529b4d41), revision **`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`**. CPU float32, one thread, seed 425, batch size 32, maximum 256 wordpieces, normalized 384-dimensional embeddings. The same verified local model snapshot handles tasks and queries; no training or remote custom code is used. Pinned direct dependencies are in `requirements-ranking.txt`; [evaluation environment](environment.txt) and [validation environment](validation_environment.txt) record resolved packages.

Only tasks whose occupation code belongs to the unchanged eligible pool are indexed. Stable row identity is `(occupation code, task ID)`, sorted lexically. User utterances are split into natural-text sentences at `. ! ?` and newlines, preserving case, wording, negation and repetition; no lexical stopwording or evidence filtering is used.

For each query sentence, exact FAISS search visits all tasks. Each occupation receives its highest task cosine/inner-product similarity for that sentence; its dense score is the arithmetic mean of these sentence maxima. Negative values are retained and no relevance threshold is tuned. Missing-task occupations receive no dense rank; empty text gives an unusable dense signal. Score ties use occupation code ascending.

Up to three tasks per occupation are selected by maximum similarity across sentences, with task-ID tie-breaking. All are marked `provisional`, including tasks matching a dislike. This supplies task-match diagnostics without claiming stance-aware recommendation support or completing M5 integration.

## Cache and measured costs

The cache covers **8,243 tasks and all 378 eligible occupations**, with no missing-task occupations. [Cache manifest](cache_manifest.json) and [model-file manifest](model_snapshot_manifest.json) preserve configuration, model/file hashes, task CSV/mapping hashes, pool, versions and artifact fingerprints.

| Item | Observation |
|---|---:|
| Task embedding time | 41.172 s |
| FAISS flat-index add time | .001494 s |
| Build duration including initial model loading, validation and serialization | 81.104 s |
| Embedding/index/owner-map cache artifacts | 26,810,790 bytes (25.57 MiB), plus metadata |
| Local model snapshot files | 91,608,367 bytes (87.36 MiB), excluding download bookkeeping |

The build duration excludes model download. Model/cache artifacts live under gitignored `artifacts/ranking/`; fresh checkouts must install dependencies, download the pinned snapshot, and build the cache. Runtime verifies the corpus, pool, parameters, model files, dependencies, artifact hashes, embedding norms, task-owner mapping and exact agreement between serialized vectors and the FAISS index.

## Controlled development results

All seven runs use identical 60 development/oracle request snapshots and the same 378 occupation pool, unchanged M1/M3 parameters, RRF k=60, and all six existing M2 metric definitions. Structured uses synthetic oracle vectors, whereas retrieval uses the supplied opening texts. These are not actual-tracker or end-to-end dialogue results. No held-out scoring or tuning occurred.

| Configuration | Hit@1 | Hit@5 | Hit@10 | MRR | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| [Structured](structured/metrics.json) | .033333 | .166667 | .433333 | .135636 | .151459 | .203471 |
| [BM25](bm25/metrics.json) | .000000 | .016667 | .033333 | .016099 | .014219 | .020544 |
| [Structured + BM25](hybrid/metrics.json) | .016667 | .133333 | .200000 | .084887 | .098101 | .116610 |
| [Dense](dense/metrics.json) | .000000 | .050000 | .050000 | .025520 | .028859 | .031444 |
| [Structured + dense](structured_dense/metrics.json) | .016667 | .116667 | .200000 | .086552 | .103499 | .121640 |
| [BM25 + dense](bm25_dense/metrics.json) | .000000 | .033333 | .050000 | .023789 | .018654 | .023739 |
| [Three-way RRF](three_way/metrics.json) | .000000 | .050000 | .116667 | .045221 | .051315 | .068509 |

The three controls reproduce the M3 metrics exactly. [Paired comparisons](comparisons.json) use 2,000 profile-bootstrap resamples, seed 425:

- Dense versus BM25: Hit@5 difference +.033333, interval [0, .083333]; NDCG@10 difference +.010899, interval [−.002864, .027740]. This is not strong evidence of a robust semantic improvement.
- Structured+dense versus structured: Hit@5 difference −.05 (5 gains, 8 losses), interval [−.166667, .066667]; NDCG@10 difference −.081830, interval [−.133586, −.032634].
- BM25+dense versus BM25: Hit@5 difference +.016667, interval [0, .05].
- Three-way versus structured+BM25: Hit@5 difference −.083333 (0 gains, 5 losses), interval [−.15, −.016667]; NDCG@10 difference −.048101, interval [−.073716, −.021520].

Standalone gains do not guarantee fusion gains. Retain the M4 baseline, but do not activate it as the default. Diagnose aggregation/query evidence before more elaborate ANN search. No accuracy threshold was invented to redefine correctness acceptance.

| Configuration | Pipeline load | Preloaded median | Preloaded p95 |
|---|---:|---:|---:|
| Structured | 41.15 ms | 1.310 ms | 1.541 ms |
| BM25 | 80.91 ms | 1.278 ms | 1.806 ms |
| Structured+BM25 | 87.09 ms | 2.500 ms | 2.804 ms |
| Dense | 5,051.58 ms | 45.193 ms | 58.777 ms |
| Structured+dense | 399.61 ms | 47.077 ms | 58.559 ms |
| BM25+dense | 339.52 ms | 46.512 ms | 60.181 ms |
| Three-way | 325.43 ms | 47.814 ms | 65.579 ms |

Runs executed sequentially in one process; the first dense load includes heavy-import/model warmup that later loads reuse. Each query timing covers one full pipeline call, including sentence encoding, exact search, aggregation, provisional evidence and response assembly, but excludes metric/export time. These are local small-sample timings, not an agreed service latency budget.

## Instruction-sheet acceptance mapping

| Requirement / acceptance criterion | Evidence | Result |
|---|---|---|
| Pin MiniLM model/revision and use the same encoder for tasks and queries | Frozen `dense_config.py`, model file hashes, shared `MiniLMEncoder`, saved revision/environment | Pass |
| Cache normalized eligible task embeddings and owner mapping | 8,243 validated rows; hashes and cache manifest; per-file/vector/index integrity checks | Pass |
| Exact/exhaustive inner-product search | Normalized float32 embeddings in FAISS `IndexFlatIP`; all tasks searched without truncation | Pass |
| Best task per occupation per sentence, then sentence mean | Hand-built two-sentence fixture gives occupation scores 1.0 and .7 | Pass |
| Paraphrase fixture retrieves a meaningful task | Real pinned model ranks machinery diagnosis/repair above nursing/teaching for a factory-equipment paraphrase | Pass |
| Valid occupation mapping; missing tasks receive no fictitious rank | Corpus validation, partial-pool fixture, duplicate/out-of-pool rejection and real 378-pool validation | Pass |
| Cached/reloaded scores agree within tolerance | Fixture reload equality; all four real dense conditions reproduce exactly, exceeding the 1e-6 tolerance | Pass |
| Empty text omits dense; no corpus re-embedding per turn | Empty/whitespace/punctuation fixture makes no encoder call; task encoding is build-only; reloads encode queries only | Pass |
| Compare dense-only, structured+dense, BM25+dense and three-way against controls | Seven saved runs and six paired comparisons; unchanged inputs/pool/evaluator | Pass |
| Record build time, cache size and preloaded median/p95 | Cache manifest and each run's latency JSON; tables above | Pass |
| Preserve modularity, variants, metrics, history and prior baselines | Lazy factory, no heavy imports when disabled, separate snapshots, unchanged M1/M3 scores and six metrics; seven real history rows appended | Pass |

## Verification and remaining limits

**85 ranking/evaluation tests pass**, including all M0–M3 tests, 13 M4 tests, real model paraphrase and cached-pipeline integration. [Reload checks](reproducibility_check.json) confirm identical inputs, metrics and full predictions excluding unique run IDs for all four dense configurations; task scores, ranks and evidence also match exactly. Repeats used temporary history files. Original history rows and reports are preserved. Every saved evaluation records its source/config hashes, model/index fingerprints, auxiliary task-source hash and resolved component IDs.

The CSV dataset fingerprint continues to identify shared evaluation profiles and occupation data. Dense-specific task/model/index fingerprints are recorded separately in each detailed manifest and cache metadata. Pairing a non-dense control with a dense treatment intentionally introduces the task corpus; when both use dense retrieval, a changed auxiliary task fingerprint is rejected. Reviewers must still check the declared experimental variable and model/aggregation settings.

The broader suite has one remaining upstream integration error: the production extractor checkpoint at `models/extractor` is absent. This does not involve MiniLM or the M4 cache. See `full_suite_final.log`. Production tracker/NLI/policy scaffolds and the live app's structured factory were not modified; full M5 integration and an agreed latency budget remain pending. All task matches are provisional, and end-to-end probing effectiveness remains untested.

## Usage

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-ranking.txt
# Fresh checkout only: explicit pinned model download and one-time immutable cache build.
python -m recommender.build_dense --download-model
python -m recommender.evaluate --ranking-config configs/ranking/m4_three_way.json
python -m unittest discover -s tests -p test_dense.py -v
```

The local environment/model/cache are already prepared; do not rebuild over existing artifacts. The four dense presets are `m4_dense.json`, `m4_structured_dense.json`, `m4_bm25_dense.json`, and `m4_three_way.json`. To select one through the no-argument workflow, edit `configs/ranking.yaml` and set M4 to `{algorithm: minilm, variant: mean_tasks_v1}`, enabling M1/M3 as desired. Keep `configs/m2.json` evaluation-only.

Next planned work is M5 application integration with tracker/policy and a baseline freeze. Any aggregation improvement requires a new named variant, a separate hypothesis and controlled development comparison. Do not overwrite `mean_tasks_v1`, tune on held-out profiles, or attribute these results to conversational probing.
