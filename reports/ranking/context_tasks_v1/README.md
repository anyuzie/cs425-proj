# Occupation description-context and task text reranking, v1

## Hypothesis and scope

Adding occupation task statements to descriptive job-context prose may improve semantic reranking over description-only controls. The six requested configurations were fixed before observing treatment metrics. No training, hyperparameter search, held-out evaluation or automatic configuration promotion was performed.

**Context means the raw occupation `Description` field in this generic experiment.** All raw `Task` statements are added by occupation code, ordered by task ID. No BERT classifier labels/probabilities or clause routing are used. The O*NET Work Context sidecar is deliberately excluded: occupation-specific levels require ratings; including its mostly shared attribute names without ratings would not express occupation-specific working conditions. This experiment therefore does not test rating-aware Work Context preference compatibility.

## Fixed pipeline and text scoring

1. Use the unchanged weighted RIASEC dot product on the existing oracle interests and supplied occupation z-scores, including the existing observed-dimension and ascending-code tie rules. Rank the full 378 eligible Job Zone 4/5 occupations.
2. Take only the first 100 as semantic candidates. Send only occupation codes for joining, raw `Description` and `Task` text, and newline-joined chronological user-only utterances to the semantic scorer. No occupation numerical scores, precomputed embeddings, relevance labels, target identities, classifier probabilities or Work Context ratings enter semantic scoring.
3. Form `Context:\n{Description}\nTasks:\n{all tasks}`. Tokenize locally with the selected frozen model tokenizer, split into contiguous non-overlapping 224-token chunks, decode each, and verify that re-tokenized chunks including special tokens fit the frozen 256-token encoder limit. Every source token is retained. Encode all chunks into normalized float32 vectors during this evaluation. For occupation c and query q, score(c,q) = mean_j cos(chunk_cj, q). The arithmetic mean vector is deliberately not renormalized. In-memory vectors may be reused across profiles within one run; no disk embedding/index input is read or written. Query handling is frozen: newline join of user history, normal model truncation at 256 tokens.
4. Single-model variants sort by decreasing semantic score, then ascending occupation code, inside the original dot-product groups of 5 or 10. Sequential variants first sort within original groups of 20, then form fresh groups of 10 from that output and sort with the other model. Both stages retain the same 100 candidates. The equal-weight variant scores the identical original shortlist with both models and sorts once within original groups of 10 using .5*MiniLM_cosine_mean + .5*MPNet_cosine_mean. There is no numerical RIASEC/semantic score fusion, RRF, score normalization, context-rating weight or fitted weight.
5. Preserve the tail at ranks 101–378 and every structured score/rank. Reassign final ranks and evaluate at full depth.

Adding text and explicitly avoiding its truncation is a joint treatment; comparisons do not isolate text content from document chunking. Equal weighting is over token chunks, not individual task statements or context/task sections. Generic cosine matching does not reliably model dislikes or requested levels of working conditions.

MiniLM is `sentence-transformers/all-MiniLM-L6-v2` at `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` (384 dimensions). MPNet is `sentence-transformers/all-mpnet-base-v2` at `e8c3b32edf5434bc2275fc9bab85f82640a19130` (768 dimensions). Both use the existing local pinned weights, CPU, one thread, seed 425, deterministic Torch algorithms, evaluation mode and max sequence length 256. Detailed manifests record model-file/package hashes, raw text source hashes and encoded chunk counts.

## Evaluation and leakage controls

Only the canonical `data/dev_profiles.jsonl` (120 synthetic profiles) is loaded. `profiles.jsonl` and `profiles_v4.jsonl` are final-evaluation datasets and were not loaded. Occupation source text is static catalogue data, not relevance supervision. The scorer never receives a target/relevance field. Evaluation applies the same existing `relevance_labels`, `evaluate_ranking` and `aggregate` code after ranking. No relevance labels influence candidate formation, model text, grouping or score fusion. Oracle RIASEC values come from the existing synthetic benchmark; this is not actual-tracker or real-user performance.

Hit@1/5/10 and full-pool MRR use the exact target occupation. NDCG@5/10 uses existing graded relevance (target 4, short 3, long 2, supplemental 1; maximum grade per eligible code; gain 2^grade-1). Recall@5/10/20/50 is macro coverage of all eligible positive-grade labels, with zero for empty relevance sets. Every profile remains in the denominator. Full definitions are in each manifest.

All eight historical prediction files were re-evaluated under m2-v4 after checking raw dataset hashes and profile identities; all ten metrics match the original CSV within 1e-12. These are verified historical predictions, not rerun encoder inferences. A fresh dot-product control also reproduces all ten original metrics. The new rows identify m2-v4 directly; old rows and their original evaluator/Recall@50-extension provenance remain unchanged. Historical source reports are never rewritten.

## Actual metrics

| Experiment | hit_at_1 | hit_at_5 | hit_at_10 | mrr | ndcg_at_5 | ndcg_at_10 | recall_at_5 | recall_at_10 | recall_at_20 | recall_at_50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| weighted_dot_product | 0.050000 | 0.166667 | 0.383333 | 0.141508 | 0.149974 | 0.195757 | 0.089856 | 0.167393 | 0.303874 | 0.589844 |
| minilm_groups5__context_tasks_v1 | 0.025000 | 0.166667 | 0.383333 | 0.120868 | 0.139874 | 0.186190 | 0.089856 | 0.167393 | 0.303874 | 0.589844 |
| minilm_groups10__context_tasks_v1 | 0.033333 | 0.216667 | 0.383333 | 0.129812 | 0.145725 | 0.185563 | 0.082855 | 0.167393 | 0.303874 | 0.589844 |
| mpnet_groups10__context_tasks_v1 | 0.041667 | 0.208333 | 0.383333 | 0.138693 | 0.143323 | 0.186462 | 0.083250 | 0.167393 | 0.303874 | 0.589844 |
| mpnet20_then_minilm10__context_tasks_v1 | 0.025000 | 0.150000 | 0.291667 | 0.110165 | 0.117120 | 0.153191 | 0.077755 | 0.146681 | 0.303874 | 0.576314 |
| minilm20_then_mpnet10__context_tasks_v1 | 0.016667 | 0.150000 | 0.275000 | 0.105176 | 0.112168 | 0.155120 | 0.069680 | 0.149261 | 0.303874 | 0.582610 |
| minilm_mpnet_equal_weights_groups10__context_tasks_v1 | 0.025000 | 0.200000 | 0.383333 | 0.128329 | 0.137418 | 0.184217 | 0.082683 | 0.167393 | 0.303874 | 0.589844 |

Among the six treatments, the highest observed Hit@5 is 0.216667 for `minilm_groups10__context_tasks_v1`. The highest treatment MRR is 0.138693 for `mpnet_groups10__context_tasks_v1`. Dot-product control Hit@5 is 0.166667 and MRR is 0.141508. These are different ranking objectives; the largest Hit@5 alone does not establish an overall benefit. Defaults remain unchanged.

| Treatment | Hit@5 delta vs dot, 95% CI | MRR delta vs dot, 95% CI |
|---|---:|---:|
| minilm_groups5 | +0.000000 [+0.000000, +0.000000] | -0.020640 [-0.048498, +0.004283] |
| minilm_groups10 | +0.050000 [-0.041667, +0.133333] | -0.011696 [-0.052387, +0.027157] |
| mpnet_groups10 | +0.041667 [-0.041667, +0.125000] | -0.002815 [-0.045356, +0.037582] |
| mpnet20_then_minilm10 | -0.016667 [-0.108333, +0.066667] | -0.031343 [-0.076015, +0.009085] |
| minilm20_then_mpnet10 | -0.016667 [-0.108333, +0.066667] | -0.036332 [-0.080839, +0.005824] |
| minilm_mpnet_equal_weights_groups10 | +0.033333 [-0.058333, +0.125000] | -0.013179 [-0.053198, +0.023509] |


## Change from matching description-only configurations

| Treatment minus matching description control | hit_at_1 | hit_at_5 | hit_at_10 | mrr | ndcg_at_5 | ndcg_at_10 | recall_at_5 | recall_at_10 | recall_at_20 | recall_at_50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| minilm_groups5 | +0.008333 | +0.000000 | +0.000000 | +0.004552 | +0.002101 | +0.001944 | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| minilm_groups10 | +0.000000 | +0.008333 | +0.000000 | -0.004188 | +0.002584 | -0.001153 | +0.002402 | +0.000000 | +0.000000 | +0.000000 |
| mpnet_groups10 | +0.000000 | +0.033333 | +0.000000 | +0.005579 | +0.003420 | -0.000970 | +0.001303 | +0.000000 | +0.000000 | +0.000000 |
| mpnet20_then_minilm10 | +0.008333 | +0.025000 | +0.041667 | +0.008394 | -0.005691 | -0.006828 | -0.001902 | -0.010929 | +0.000000 | -0.007866 |
| minilm20_then_mpnet10 | -0.025000 | -0.025000 | -0.050000 | -0.022013 | -0.026944 | -0.020062 | -0.011603 | -0.007399 | +0.000000 | -0.007875 |
| minilm_mpnet_equal_weights_groups10 | -0.008333 | -0.016667 | +0.000000 | -0.005013 | -0.007650 | -0.002142 | +0.000885 | +0.000000 | +0.000000 | +0.000000 |


The group constraints explain some exact metric ties: groups of 5 preserve top-5 membership; groups of 10 preserve top-10 membership; sequential groups of 20 preserve top-20 membership but can change top-10 and top-50 membership. Ties at these cutoffs are structural, not proof that the encoder ignored the added text.

Each treatment folder contains all 45,360 final predictions, inputs, per-profile metrics, exact configurations, latency, manifests, paired differences and paired bootstrap comparisons against dot product (2,000 resamples, seed 425), plus differences against its verified historical description control. Bootstrap intervals are descriptive across development profiles; choosing the largest observed value among six treatments is not independent validation.

## Artifacts and reproduction

The consolidated CSV preserves the original eight rows byte-for-byte and appends six names ending `__context_tasks_v1`. The notebook displays all fourteen rows and ten metrics, treatment-minus-description differences and treatment-minus-dot-product differences. `consolidate.py` now preserves separately appended experiments when refreshing the historical comparisons. Existing ranking defaults and frozen description variants remain unchanged.

Reproduce individual treatments from repository root with the saved evaluation and ranking configuration snapshots, choosing a fresh output directory. Set `CONTEXT_TASK_MINILM_MODEL` and `CONTEXT_TASK_MPNET_MODEL` to the pinned local model folders if not using the offline Hugging Face cache. These paths choose model resources only; their content hashes are recorded. Optional dependencies already exist in the project environment. `python -m recommender.experiments.context_tasks --historical-root <original-repository>` runs the whole protocol on a fresh experiment output and original eight-row CSV; it refuses overwriting completed results. The implementation source archive accompanies this report.

Validation: 27 targeted tests pass, including all existing description/sequential/reverse/aggregation/evaluation checks plus raw text, full task-token coverage, all six candidate/group contracts, unchanged RIASEC signals, untouched full-pool tails, the unnormalized chunk mean and rejection of altered frozen text settings.
