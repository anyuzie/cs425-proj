# MPNet description reranking

MPNet is selectable through the existing configuration-driven ranking pipeline.
Weighted dot product selects the top 100. The shared description reranker then
uses cosine similarity within consecutive blocks of 10, retaining block
membership, original RIASEC scores/ranks and the unchanged full-pool tail.
User-only history is joined with newlines; occupation embeddings use the
`Description` column only. No task retrieval, score blending or RRF is added.

## Model selection and preservation

`recommender/models/mpnet_description_config.py` declares a new frozen variant:
`mpnet_descriptions@mpnet_blocks10_v1`. The model is
`sentence-transformers/all-mpnet-base-v2`, pinned to revision
`e8c3b32edf5434bc2275fc9bab85f82640a19130`, with 768 normalized dimensions.
The pinned identity is recorded by the [model repository API](https://huggingface.co/api/models/sentence-transformers/all-mpnet-base-v2);
its [configuration](https://huggingface.co/sentence-transformers/all-mpnet-base-v2/blob/e8c3b32edf5434bc2275fc9bab85f82640a19130/config.json)
specifies the hidden dimension. The experiment keeps the existing 256-token
limit for both encoders. Tokenizers differ, so the character span retained at
that limit can still differ. This is a model replacement under the current
workflow, not a separate test of MPNet's maximum context capacity.

`DescriptionReranker` now obtains the model identity, revision and dimension
from the selected frozen parameters. All MiniLM parameters, variants and
task-based MiniLM implementations remain present. No existing workflow file
or active production selection is replaced. Each pipeline owns its encoder
and vector cache, so vectors from different models are not mixed.

## Use the existing workflow with MPNet

```powershell
python -m recommender.workflow chat --config configs/ranking/workflow.mpnet.json
python -m recommender.workflow evaluate --config configs/ranking/workflow.mpnet.json
```

The new workflow has the same structure and upstream system as the previous
description experiment. Evaluation through it is report-only; it does not
automatically publish a new active selection. Chat retains existing upstream
resource requirements. MPNet downloads once to `artifacts/ranking/models`.
The existing `requirements-description-ranking.txt` supports both models.

## Reproduce the paired comparison

```powershell
python -m recommender.experiments.mpnet_comparison --output reports/ranking/mpnet_blocks10_new_run
```

Choose a new directory. On this workstation, set
`$env:PYTHONPATH = (Join-Path $PWD '.runtime')` and use the working interpreter
`C:/Users/aanny/AppData/Local/Programs/Python/Python313/python.exe`.

Both fresh runs use the same 120 synthetic development/oracle profiles,
candidate pool, labels and evaluator `m2-v3`. All nine permanent metrics are
saved, alongside full predictions, original structured scores/ranks, semantic
scores/ranks, per-profile metrics, inputs/configuration snapshots, source/data
fingerprints, model metadata and latency. A paired bootstrap comparison uses
2,000 profile resamples, seed 425. Histories record both model variants through
their experiment description and exact configuration hash.

See [results and decision](../reports/ranking/mpnet_blocks10_v1/README.md).
