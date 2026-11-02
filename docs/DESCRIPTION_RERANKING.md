# Description-only MiniLM reranking

Follow-up: [rank-block experiments](../reports/ranking/description_blocks_v1/README.md)
compare fixed groups of 5 and 10 against fresh dot-product and unrestricted
semantic controls. Recall@5/10/20 are now permanent evaluator `m2-v3` metrics,
included in metrics JSON, per-profile output, paired comparisons, histories and
promotion guards. The supplementary-only description below records the earlier
experiment's historical format.

Implemented 31 October 2026 in this recovered implementation workspace.

## Behavior and boundaries

1. The existing signed weighted dot product ranks the eligible occupation pool using the user's observed RIASEC dimensions. It is the sole candidate selector.
2. Take its first 100 occupations, regardless of the requested display count.
3. Encode each candidate's `Description` column only, and encode chronological user-only utterances joined with newlines as one query.
4. Sort those candidates by cosine similarity descending, then occupation code ascending. There is no score blending or RRF. Original structured scores and ranks remain in `signals.structured`; cosine scores and stage ranks appear in `signals.semantic`.
5. Keep the baseline tail unchanged for full-pool offline metrics; return only requested `top_k` at runtime. Missing structured signal returns no recommendations; blank utterances preserve baseline order.

`recommender/models/description_reranker.py` owns embeddings, description validation and a per-instance lazy vector cache. `RankingPipeline` owns stage ordering and candidate selection. Components remain independent; no scorer calls another scorer. Description text never contains titles, alternate titles, tasks or relevance labels. Only descriptions encountered in a request's top 100 are encoded; vectors are reused across requests. Existing task-based MiniLM and historical variants are preserved for reproducibility.

Model: `sentence-transformers/all-MiniLM-L6-v2`, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, CPU, 384 normalized dimensions, 256 word-piece limit. Both long descriptions and long joined histories are truncated by the model at that limit. This is a declared first experiment, not an evaluation of long-history chunking. Model and description fingerprints, exact stage parameters and dependency versions are saved in evaluation manifests. The model downloads once and is reused from the project cache. See the [official model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

## Performance

The historical evaluator used **Hit@1, Hit@5, Hit@10, MRR, NDCG@5, NDCG@10**, not Recall@20/50. The saved promotion smoke control and the fresh control have identical aggregate scores. The original three metrics were Hit@5, MRR and NDCG@10 before expansion to six.

Paired runs: `reports/ranking/description_top100_verified/{control,treatment}`. Both use the same 120 synthetic development/oracle profiles, 378 eligible occupations, labels and input snapshots. No upstream tracker accuracy is measured.

| Metric | Weighted dot product | Description reranking |
|---|---:|---:|
| Hit@1 | 0.050000 | 0.025000 |
| Hit@5 | 0.166667 | 0.075000 |
| Hit@10 | 0.383333 | 0.141667 |
| MRR | 0.141508 | 0.073640 |
| NDCG@5 | 0.149974 | 0.075118 |
| NDCG@10 | 0.195757 | 0.099685 |
| Recall@5 (new supplementary metric) | 0.089856 | 0.047520 |
| Recall@10 (new supplementary metric) | 0.167393 | 0.100079 |
| Recall@20 (new supplementary metric) | 0.303874 | 0.203944 |
| Recall@50 (new supplementary metric) | 0.589844 | 0.459094 |

Supplementary recall is macro-averaged coverage of **all eligible labeled relevant occupations**, including the exact target and all related tiers with positive grade, with zero for an empty relevance set. It is not exact-target Hit@K. Historical metric definitions and CSV history columns remain unchanged. The candidate gate retains 117/120 exact targets (97.5%); semantic reranking cannot recover the other three into the top 100.

The paired bootstrap 95% interval for change in Hit@5 is [-.175, -.00833], and for MRR [-.11580, -.02141], using 2,000 paired profile resamples, seed 425. Seven profiles gain a top-five target and eighteen lose one. Results do not support activating this variant as the default. The normal application and central workflow retain weighted dot product alone. The new semantic stage is available through its explicit configuration.

Reports include predictions with both scores, per-profile metrics, frozen inputs/configuration, dependency and source fingerprints, paired bootstrap comparison, supplementary recall, failure ranks and latency. The timing includes lazy first-query model loading and incremental description encoding; it is not a clean warmed-cache latency benchmark.

## Review of concrete cases

These are qualitative observations from inspecting synthetic profile text against the target description, not independent adjudications of label correctness. All targets below are eligible.

- D099, Librarians and Media Collections Specialists: rank 68 → 1. The user emphasizes organizing messy information and teaching juniors; the description explicitly covers organizing information and instruction. A plausible semantic gain.
- D037, Farm and Home Management Educators: rank 92 → 15. The user describes childcare; the target description mentions child development. Semantic overlap exists, but the occupation also covers agricultural/home-management instruction. Improved target rank does not establish whole-job fit.
- D034, Conservation Scientists: rank 94 → 18. The utterance concerns curiosity, bridges and tinkering, without natural-resource conservation. This target-rank gain has weak domain support and flags the limitations of synthetic RIASEC-derived targets.
- D045, Orthopedic Surgeons: rank 8 → 92. The user describes experimentation and repairing a fan but no clinical work. The short surgical description lacks these general interest cues. The profile also contains unresolved opposing statements about hands-on work.
- D080, Claims Adjusters: rank 14 → 100. Budget tracking and market analysis have little direct overlap with insurance claims and litigation. Contradictory equipment preferences remain in the raw query.
- D112, Compliance Managers: rank 6 → 93. Selling ideas and neat budgets lack explicit ethical/regulatory-compliance content. Short, domain-specific descriptions can lose a RIASEC-based match.
- D029, Urologists: rank 2 → 90. Biomedical research and helping others do not identify genitourinary medicine. The target label expresses interest-vector fit more strongly than domain intent.
- D100, Cytotechnologists: rank 5 → 100. Outdoor practical work contrasts with a cell pathology description. The semantic demotion is understandable even though the synthetic target metric penalizes it.

Decision: keep weighted dot product as the default; retain the implemented stage for controlled experimentation. Do not tune on held-out data. These labels favor RIASEC profile matching, so independently reviewed semantic-intent labels would help determine whether observed demotions are real recommendation errors.

## Run

From the implementation directory with a working Python environment:

```powershell
python -m pip install -r requirements-description-ranking.txt
python -m recommender.experiments.description_comparison --output reports/ranking/description_new_run
python -m recommender.workflow chat --config configs/ranking/workflow.description.json
```

The comparison output must be a new directory, preserving earlier experiments. The separate workflow explicitly enables the stage and does not automatically promote it. Chat still requires the existing upstream extractor resources.

To add Recall@5/10/20/50 to an existing paired experiment without rerunning inference:

```powershell
python -m recommender.experiments.description_comparison --output reports/ranking/description_top100_verified --recall-only
```

This verifies the original dataset fingerprints and saves aggregate and per-profile recall in each run's `recall_metrics.json` and the paired `supplementary_recall.json`. Original six-metric results and manifests remain preserved.

For this workstation, dependencies were installed locally under `.runtime`; its working interpreter is `C:/Users/aanny/AppData/Local/Programs/Python/Python313/python.exe`. Set `$env:PYTHONPATH = (Join-Path $PWD '.runtime')` when using that interpreter. The downloaded model is under `artifacts/ranking/models`. No training is required.
