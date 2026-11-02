# Recommender integration and consolidated results

Integrated into `C:/Users/aanny/Documents/GitHub/year5/cs425/project` on
31 October 2026. Changes use the destination's existing architecture.

## Two separate matching pipelines

**Structured matching:** the existing extractor/tracker converts user utterances
into a RIASEC snapshot; `models/m1.py` scores occupations using weighted dot
product. This implementation and the destination's upstream `pipeline/system.py`
remain byte-for-byte unchanged. Structured matching neither loads an embedding
model nor uses semantic scores to alter the RIASEC vector.

**Semantic matching:** `recommender/semantic/pipeline.py` consumes ordered
candidate records and user-only text. It does not read or compute RIASEC values.
Independent MiniLM and MPNet encoders in `models/description_reranker.py` embed
only occupation descriptions and the query. `RankingPipeline` passes the
structured top-100 shortlist to this module and applies the display limit after
semantic processing. Original structured scores/ranks remain available.

The semantic pipeline supports MiniLM across the top 100, MiniLM groups of 5
or 10, MPNet groups of 10 or 20, and MPNet groups of 20 followed by MiniLM
groups of 10 **formed from MPNet's output**. Both encoders have separate caches;
all previous implementations remain available. The default is still weighted
dot product; semantic variants are selected explicitly through configuration.

## Select a workflow

From the project root, after installing `requirements-description-ranking.txt`
in your chosen environment:

```powershell
python -m recommender.workflow chat --config configs/ranking/workflow.sequential.json
python -m recommender.workflow evaluate --config configs/ranking/workflow.sequential.json --output reports/ranking/sequential_new_run
```

The MPNet-only workflow is `configs/ranking/workflow.mpnet.json`. Other ranking
presets are under `configs/ranking/description*.json`. Existing default workflows
and the local terminal edit are preserved. Chat still requires the existing
upstream trained checkpoints.

Model cache defaults to `artifacts/ranking/models`. Set
`RECOMMENDER_MODEL_CACHE` to reuse another local Hugging Face model cache without
copying model weights. The integration verification used the existing workspace
cache and `.runtime` dependencies; it did not replace the destination's virtual
environment or existing model resources. Both models use pinned revisions.

## Consolidated CSV

[semantic_experiments_consolidated.csv](../reports/ranking/semantic_experiments_consolidated.csv)
contains six comparable configurations:

1. Weighted dot product alone.
2. MiniLM across the top 100.
3. MiniLM groups of 5.
4. MiniLM groups of 10.
5. MPNet groups of 10.
6. MPNet groups of 20 → MiniLM groups of 10.

Columns include all nine permanent metrics, model IDs/revisions, block sizes,
profile/candidate counts, run IDs, evaluator version, data/input fingerprints
and the source report path. Metric values are fractions, not percentages.
Recall counts distinct eligible positive-grade relevance labels and is
macro-averaged across all profiles. Hit@K measures the exact target instead.

Regenerate with:

```powershell
python -m recommender.experiments.consolidate
```

The exporter rejects mismatched datasets, input snapshots or metric definitions.
The CSV summarizes the existing synthetic development/oracle runs (120 profiles,
378 occupations); it makes no held-out or real-tracker performance claim.

## Provenance and preservation

Frozen experiment directories were imported from the working experiment
workspace without rewriting their manifests or claiming they ran in this Git
repository. `semantic_integration_manifest.json` records imported files, original
and final hashes, and protected files that remain unchanged. Original versions
of modified integration files are backed up in the source workspace's
`integration-backup` directory.

A separate real-model run of the destination's integrated code is saved at
`reports/ranking/mpnet20_minilm10_integrated_v1`; its manifest records the actual
destination source hashes. The consolidated CSV retains original experiment
provenance. Verification checks compare the integrated output to the frozen
sequential run, including both intermediate stage traces.

The integrated evaluator reports all nine metrics in per-profile/aggregate JSON,
paired comparisons and history. Appending to an older history backs it up before
adding recall columns, leaving unknown historical recall values blank. Historical
evaluator versions and frozen reports remain intact.

See [sequential behavior and signals](SEQUENTIAL_RERANKING.md),
[MPNet comparison](../reports/ranking/mpnet_blocks10_v1/README.md), and
[rank-block results](../reports/ranking/description_blocks_v1/README.md).


## Reverse-sequence follow-up

`workflow.reverse.json` selects MiniLM groups of 20 then MPNet groups of 10 formed from MiniLM output. Both sequence variants are preserved. The updated seven-experiment results are in [semantic_experiments_consolidated_v2.csv](../reports/ranking/semantic_experiments_consolidated_v2.csv), saved separately because the previous CSV was locked. See [reverse results](../reports/ranking/minilm20_mpnet10_v1/README.md).


## Aggregation follow-up

`workflow.aggregate.json` combines MiniLM and MPNet cosine scores with equal weights, then sorts once in groups of 10. The [aggregation guide](AGGREGATED_RERANKING.md) describes configuration and score traces. The standard consolidated CSV and adjacent notebook now include eight experiments. All 36 targeted tests pass; RIASEC matching remains separate.
