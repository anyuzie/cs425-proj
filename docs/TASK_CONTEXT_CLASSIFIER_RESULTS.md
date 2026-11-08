# Source-labelled BERT results

Input: clause text only. Two independent sigmoid scores, with Task and Context routing thresholds both fixed at 0.5.

Training: 14,065 examples. Validation: 3,435 examples, including 22 texts from 11 unseen context families and 3,413 tasks from held-out occupations.

| Model | Task F1 | Context F1 | Macro F1 | Exact match |
| --- | ---: | ---: | ---: | ---: |
| Majority baseline | 0.997 | 0.000 | 0.498 | 99.36% |
| BERT, weighted BCE | 1.000 | 1.000 | 1.000 | 100.00% |

## Per-output validation

| Output | Precision | Recall | Positive examples | TP | FP | FN | TN |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| task | 1.000 | 1.000 | 3413 | 3413 | 0 | 0 | 22 |
| context | 1.000 | 1.000 | 22 | 22 | 0 | 0 | 3413 |

## Held-out context variants

- context: 11/11 pass Context threshold; mean context score 1.000.
- description: 11/11 pass Context threshold; mean context score 1.000.

## Scope of the result

These results evaluate source-labelled task/context recognition. No source identifiers, occupation names, ratings, or scores enter BERT.
Class weights use training labels only. Validation supplies neither gradient updates nor class weights; thresholds were not tuned on validation.
The model sees only [1,0] and [0,1] targets. Two sigmoids support simultaneous routing or rejection mathematically, but these behaviours have not been validated on mixed/irrelevant conversational clauses.
Context wording differs from task wording and may create source-style shortcuts. Context definitions phrased as questions are especially distinct. Scores are not calibrated probabilities.
Source membership is not perfect clause-level ground truth. Only 57 context families exist, and context recall here is measured on 22 held-out texts.
Current utterance splitting, preference extraction and occupation scoring/ranking are unchanged.

## Artifacts

`data/` holds frozen train/validation JSONL, exclusions and hashes. The checkpoint holds training_manifest.json, validation_report.json, validation_predictions.jsonl, validation_errors.jsonl and context_variant_diagnostics.json.
The public scoring API returns task_score, context_score, route_task and route_context. The scorer CLI accepts --task-threshold and --context-threshold independently.

## User-style phrase smoke check

These illustrative phrases are a diagnostic, not an independent labelled evaluation set.

| Phrase | Task score | Context score |
| --- | ---: | ---: |
| Repair machinery. | 0.9972 | 0.0001 |
| I enjoy repairing machinery. | 0.0195 | 0.9977 |
| I prefer working outdoors. | 0.0039 | 0.9999 |
| I enjoy repairing machinery outdoors. | 0.0100 | 0.9992 |
| I am not sure yet. | 0.0042 | 0.9998 |

The model misroutes “I enjoy repairing machinery” to Context and assigns high Context to “I am not sure yet”. The perfect source-validation score therefore does not transfer to conversational clause routing. Source-style shortcuts are a plausible explanation, not a proven causal diagnosis.
Readiness: NOT READY for reliable conversational routing. The checkpoint is available for further experiments; the current user-utterance/ranking pipeline has not been switched to it. Additional conversational variation and simultaneous/rejection evaluation are needed before integration.

## Paraphrase diagnostic and pre-extension comparison baseline

The later frozen-checkpoint, 80-case assistant-authored paraphrase diagnostic scores Task F1 **0.564103**, Context F1 **0.457831**, macro F1 **0.510967**, and exact match **35% (28/80)** at unchanged 0.5 thresholds. Mixed and irrelevant exact match are both 0/12; routing changes in 17/20 semantic families. These diagnostic labels are not independently reviewed, and related variants are correlated. They demonstrate limitations that perfect source validation misses; they do not establish real-user accuracy.

See the [pre-extension baseline](PRE_EXTENSION_BASELINE.md) for precision/recall, confusion counts, checkpoint fingerprint, all ten existing ranking metrics, and requirements for a matched later integration comparison. The [paraphrase report](../reports/task_context/paraphrase_v1/REPORT.md) and [per-clause scores](../reports/task_context/paraphrase_v1/scores.csv) preserve the evidence. No retraining or ranking changes were made for this documentation snapshot.
