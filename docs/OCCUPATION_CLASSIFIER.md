# Occupation clause classification

The ranking terminal displays only its first ten results. Ranking still requests
50 results and scoring, reranking and evaluation cutoffs are unchanged. For each
displayed occupation, descriptions and task text are split using the existing
spaCy clause processor and classified by a local fine-tuned BERT/RoBERTa model.
Both labels are independent; a mixed clause appears in both lists. Neither is
not an uncertainty estimate. Empty lists mean no label was detected in available
text, not that the occupation lacks that characteristic.

No retired stance ranker, preference extraction, or semantic matching is added.
User clause preference extraction belongs to point 2 of the implementation.

## Train the checkpoint

Install the existing requirements-description-ranking.txt and requirements-utterance.txt.
Use reviewed JSONL rows with `text`, `labels.activity`, `labels.job_nature`:

```json
{"text":"Repairing machinery outdoors","occupation_code":"example","labels":{"activity":1,"job_nature":1}}
```

Include activity-only, job-nature-only, mixed and neither examples. Labels mean
content presence, independent of preference. Keep occupations (or conversation
groups) in separate train/validation/test sets; provide occupation_code/group_id
so training can reject overlapping groups. Review occupation examples rather
than relying solely on conversational annotations or source-field pseudo-labels.
Never use the four illustrative clauses as a production training dataset.

```powershell
.venv/Scripts/python.exe -m training.occupation_classifier --train data/occupation_train.jsonl --validation data/occupation_validation.jsonl --output models/occupation_classifier
.venv/Scripts/python.exe -m app.ranking_terminal
```

To use RoBERTa, supply `--base-model FacebookAI/roberta-base`. Training uses
AutoModelForSequenceClassification, num_labels=2, multi_label_classification,
float labels and Hugging Face's BCE-with-logits loss. Inference uses sigmoid
and independent thresholds of 0.5. Training saves validation per-label F1 and
exact match, data fingerprints, seed and hyperparameters. Thresholds are defaults,
not calibrated confidence estimates. Assess on a separate reviewed test set
before relying on model quality. DAPT is not part of this initial baseline.

The terminal requires a trained checkpoint and fails clearly if absent. It never
initializes a random classification head for runtime use. Classifications retain
source fields and offsets and are cached within the process. Inputs beyond 256
tokens are truncated. Only Description and Task are used; additional qualitative
fields need explicit source adapters when they become available. Missing nature
information is not inferred or invented.

This delivers the training and runtime integration, not trained model accuracy:
a reviewed occupation training corpus and a trained checkpoint are required to
complete an end-to-end real-model run.
