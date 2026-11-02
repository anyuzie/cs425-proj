# Sequential MPNet → MiniLM description reranking

The existing ranking workflow can now select the frozen
`sequential_descriptions@mpnet20_minilm10_v1` configuration.

1. Weighted dot product ranks the eligible pool using RIASEC and selects the top 100.
2. MPNet reranks those candidates within consecutive baseline groups of 20:
   positions 1–20, 21–40, 41–60, 61–80 and 81–100.
3. MiniLM receives **MPNet's ordered output**, partitions it into consecutive
   groups of 10, and reranks within those new groups. For example, MiniLM's
   first group is MPNet's first ten results, which may include occupations
   originally at baseline ranks 11–20.
4. Only after both stages does the pipeline apply the requested display count.
   The baseline tail after rank 100 remains unchanged for full-pool MRR.

Both encoders use only occupation descriptions and the same newline-joined
user-only history, with normalized vectors and a 256-token limit. Each model
uses its existing pinned revision and has a separate reusable model/vector
cache. Tokenizers differ, so equal token limits need not retain equal character
spans. No raw-score blending or RRF is introduced. Blank text preserves the
baseline; absent structured signal produces no recommendations.

## Architecture and observability

`RankingPipeline` delegates the ordered list of stages to the separate
`recommender/semantic/SemanticMatchingPipeline`; scoring components do not invoke
each other. The existing `DescriptionReranker` implements one semantic stage.
`sequential_description_config.py` freezes the stage order and parameters;
`mpnet_description_config.py` adds the standalone MPNet groups-of-20 variant.
Single-model variants and all historical configurations are preserved.

Top-100 prediction signals retain:

- `structured`: original weighted-dot-product score and baseline rank.
- `semantic_mpnet`: MPNet cosine score and unrestricted semantic rank among the 100 candidates.
- `stage_mpnet`: MPNet cosine score and position after its groups-of-20 stage.
- `semantic_minilm`: MiniLM cosine score and unrestricted semantic rank among the 100 candidates.
- `stage_minilm`: MiniLM cosine score and position after its groups-of-10 stage.
- `semantic`: compatibility alias for the final MiniLM semantic signal.

`RankedOccupation.rank` is the final output position. The two `stage_*` signals
allow intermediate candidate groups to be reconstructed without another model
call. Tail occupations retain structured signals only. Resolved configuration
and metadata contain both ordered stages, model revisions, dimensions,
description fingerprints and dependency versions.

## Use

```powershell
python -m recommender.workflow chat --config configs/ranking/workflow.sequential.json
python -m recommender.workflow evaluate --config configs/ranking/workflow.sequential.json --output reports/ranking/sequential_new_run
```

Choose a new evaluation directory. The workflow preserves the existing system
and report-only evaluation path, with all nine permanent metrics and history
registration. Normal active/default selection is unchanged; chat still needs
the upstream extractor resources. Both model snapshots are cached locally.

Dependencies remain `requirements-description-ranking.txt`. On this workstation
use `C:/Users/aanny/AppData/Local/Programs/Python/Python313/python.exe` with
`$env:PYTHONPATH = (Join-Path $PWD '.runtime')`.

## Interpretation

MiniLM can rearrange MPNet's first ten but cannot change their membership.
It can change Hit@1, Hit@5, MRR, NDCG and Recall@5 relative to MPNet's intermediate
output, but not Hit@10 or Recall@10/20. Relative to the dot-product baseline,
MPNet's groups of 20 can change the first ten members, while Recall@20 is
structurally preserved. Fixed rank groups do not guarantee similar raw RIASEC
scores. These constraints should be considered when interpreting tied metrics.

The integration tests deliberately give the encoders opposing orderings to
verify stage order, MPNet-derived MiniLM groups, score traces, cache reuse,
blank/no-signal behavior, configuration validation and post-stage truncation.

See the [development run](../reports/ranking/mpnet20_minilm10_v1/README.md)
for real-model verification and metrics.
