# Aggregated semantic matching

The separate semantic pipeline supports a frozen 50/50 aggregation of MiniLM
and MPNet cosine scores, preserving every existing sequential and single-model
variant. Both models score the same weighted-dot-product top 100. Their scores
are averaged, then sorted once inside baseline groups of 10. The original
RIASEC scores, baseline ranks and tail after rank 100 are retained.

```python
combined = 0.5 * minilm_score + 0.5 * mpnet_score
```

## Select this workflow

```powershell
python -m recommender.workflow chat --config configs/ranking/workflow.aggregate.json
python -m recommender.workflow evaluate --config configs/ranking/workflow.aggregate.json --output reports/ranking/aggregate_new_run
```

Chat uses the existing upstream system and checkpoint requirements. The default
active workflow remains weighted dot product. No automatic promotion is added.
Dependencies remain requirements-description-ranking.txt; model snapshots are
cached independently with their pinned revisions.

## Code and configuration

- `recommender/models/description_reranker.py`: `score(codes, utterances)` returns
  model cosine scores without changing order or accessing RIASEC. Existing
  `rerank()` delegates to it and retains previous sequential behavior.
- `recommender/semantic/pipeline.py`: `_aggregate()` collects both score maps,
  applies weights and performs the one final sort. The model evaluation order
  does not affect the result; no concurrent threads are required for aggregation.
- `recommender/semantic/aggregation_config.py`: frozen weights, aggregation
  method, candidate limit, model selections and block size.
- `configs/ranking/description_aggregate10.json`: selectable snapshot of those
  parameters, used by `workflow.aggregate.json`.

Weights and grouping are versioned: to experiment with different weights, add a
new named parameter variant and matching configuration rather than changing
past experiment identities. This first variant averages raw cosine scores with
no additional calibration. Equal numeric weights need not imply equal influence
because model score distributions differ. Embeddings themselves are not averaged.

## Prediction signals

`structured` retains the RIASEC score and original rank. `semantic_minilm` and
`semantic_mpnet` retain each cosine score and unrestricted semantic rank among
the shortlist. `semantic_aggregate` is the combined score and unrestricted rank;
`semantic` is its compatibility alias. `stage_aggregate` records final grouped
position. There are no model-specific intermediate reordered stages.

## Validation and results

All 36 targeted tests pass. Saved aggregate scores and group boundaries are
verified for all 120 development profiles. The updated consolidated CSV contains
eight experiments, and the adjacent notebook is executed with eight rows and
nine metric columns. See [full results and failure review](../reports/ranking/aggregate_minilm_mpnet10_v1/README.md).
