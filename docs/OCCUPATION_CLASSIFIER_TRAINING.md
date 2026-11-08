# Occupation classifier pilot

Your completed occupation CSV is the only supervised label source. All 155 active
clauses were reviewed: 28 Activity, 88 Job nature, 39 Both, no Neither. Your decisions
were used as saved. No conversational annotations or machine-generated labels were
added to supervised training.

## Files

- `training_data/occupation_train.jsonl`: 123 reviewed clauses, 24 occupations.
- `training_data/occupation_validation.jsonl`: 32 reviewed clauses, six occupations.
- `training_data/split_manifest.json`: grouping, label distributions and fingerprints.
- `training_data/occupation_corpus_unlabelled.jsonl`: 20,812 source-grounded candidates
  from descriptions/tasks covering 1,016 occupations.
- `classified_data/occupation_classified.jsonl`: all corpus predictions.
- `classified_data/occupation_activity.jsonl`: predicted Activity, including Both.
- `classified_data/occupation_job_nature.jsonl`: predicted Job nature, including Both.
- `classified_data/occupation_neither.jsonl`: predicted Neither.
- `classified_data/occupations_grouped.jsonl`: all predictions grouped by occupation.

The JSONL source corpus uses conservative spaCy sentence boundaries to retain shared
objects and coordinated actions rather than creating bare-verb fragments. These
20,812 candidates have not each been manually repaired or reviewed. Source text and
offsets are retained, and no generic Workers prefix is introduced. The occupation
terminal uses the same conservative segmentation; user-utterance splitting is unchanged.

Occupation codes do not overlap between train and validation. Occupations sharing
an identical normalized clause are grouped together to avoid exact-text leakage.
Split seed is 42. Validation distribution was balanced using label counts only.
No independent test set was created; validation results are not test performance.

## Model and validation

Hugging Face `google-bert/bert-base-uncased`, two independent outputs in order
Activity/Job nature, BCE-with-logits training, sigmoid inference, thresholds 0.5.
Learning rate 2e-5, AdamW, batch size 16, maximum length 256, seed 42, CPU.

| Run | Activity F1 | Job nature F1 | Exact match |
| --- | ---: | ---: | ---: |
| 3 epochs | 0.300 | 0.897 | 0.500 |
| 10 epochs (selected) | 0.625 | 0.893 | 0.500 |

The ten-epoch checkpoint was selected by validation macro F1. This is a small-data
pilot and has no Neither training examples. Its per-clause probabilities are not
calibrated confidence estimates. Neither detection and broader generalization need
additional reviewed examples and an independent evaluation set.

Corpus predictions are explicitly `annotation_status=model_predicted` and
`training_eligible=false`. They are not ground truth or additional training data.
Every prediction includes both sigmoid probabilities and the checkpoint fingerprint.
Mixed clauses appear in both category-specific files; do not add their counts to
estimate the number of unique clauses.

## Use and reproduction

The selected checkpoint is installed in the project's `models/occupation_classifier`.
Reviewed split files are also copied to project `data/`. The existing command
`.venv/Scripts/python.exe -m app.ranking_terminal` now loads the trained classifier
and displays its clauses for the top ten occupations. Ranking algorithms, weights,
candidate cutoffs and user-utterance annotations are unchanged.

Dataset generation: `build_training_data.py`. Supervised training:
`python -m training.occupation_classifier --train <train.jsonl> --validation
<validation.jsonl> --output <empty-checkpoint-directory> --epochs 10`.
Corpus inference: `classify_corpus.py --checkpoint trained_bert_10epochs`.
`verify_training_results.py` checks split isolation, source spans and prediction coverage.
Training manifests and `model_selection.json` preserve the measured comparison.
