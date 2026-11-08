# Source-labelled BERT task/context scores

This model has two independent sigmoid outputs: `task_score` and `context_score`.
It returns `route_task` and `route_context` using independently configurable thresholds,
initially 0.5. There is no four-way classification, preference extraction or ranking
logic. Both routes may be active, or neither may be active. Scores need not sum to one.

Training input is text only. Labels come from source membership: task statements
are [1,0]; context names/descriptions are [0,1]. Occupation metadata is used only
for grouping and provenance. Numeric occupation context ratings are never model
inputs or extra text examples. No previous manually labelled pilot is reused.

Frozen data: 14,065 training examples (13,973 tasks and 92 context texts); 3,435
validation examples (3,413 tasks and 22 context texts). Eleven context families
and 185 task occupations are held out. Names and definitions of one context stay
together. Normalized duplicates do not cross splits. Exclusions are recorded.

Training uses pretrained google-bert/bert-base-uncased with two outputs,
multi_label_classification and BCEWithLogitsLoss. Training-only positive weights
are approximately 0.00658 for Task and 151.88 for Context. Every retained training
example is visited once per epoch. Length buckets reduce padding, with shuffled
buckets/batches. Default settings: 3 epochs, batch 32, max length 128, AdamW 2e-5,
seed 42. Long inputs are truncated. Validation does not contribute gradients,
class weights or threshold selection.

## Commands from the project folder

```powershell
.venv/Scripts/python.exe -m training.task_context --train data/task_context_source_v1/train.jsonl --validation data/task_context_source_v1/validation.jsonl --output models/task_context_bert_new
.venv/Scripts/python.exe scripts/score_task_context.py --text "I enjoy repairing machinery" --text "I prefer working outdoors"
.venv/Scripts/python.exe scripts/score_task_context.py --text "Repair machinery outdoors" --task-threshold 0.6 --context-threshold 0.7
```

Python interface: `recommender.utterance.task_context.TaskContextClassifier`, method
`predict(list_of_clause_texts)`. A caller must supply clauses; this module does not
split or change utterances. The terminal now passes exact source clauses to this classifier
for display only; the spaCy processor and ranking calculations remain unchanged.

## Evaluation limits

This is source-labelled supervision. Source membership does not guarantee every
task statement is context-free or every context name is activity-free. Source
wording (e.g. questions in context descriptions) can make validation easier than
real user clauses. There are only 57 context families and no explicit simultaneous
positive/negative examples. Independent mathematical outputs do not establish
reliable simultaneous routing or rejection. No conversational generalisation,
calibrated probability or preference-recognition claim is made.

Reports include per-output precision, recall, F1, confusion counts, distributions,
routing counts, a majority baseline and per-example predictions/errors. Overall
accuracy is misleading because tasks greatly outnumber context examples.

## Source attribution

Training text is derived from the O*NET 31.0 Database by the U.S. Department of
Labor, Employment and Training Administration, used under CC BY 4.0:
https://www.onetcenter.org/database.html and https://creativecommons.org/licenses/by/4.0/.
The project selected, deduplicated, split and assigned source-based labels to the
text. These transformations and the trained model are not endorsed by USDOL/ETA.

## Terminal user-clause classifier integration (2 November 2026)

`python -m app.ranking_terminal` now loads `models/task_context_bert` once and passes the exact spaCy clause texts to its batched predictor. Each clause displays `task_score`, `context_score`, `route_task` and `route_context`; independent checkpoint thresholds default to 0.5. Override the checkpoint with `--task-context-classifier PATH`. The old rule-evidence terminal display is replaced with these model scores; the spaCy splitter, source spans and existing annotation exports remain unchanged.

This is experimental display-only integration: original utterances still feed unchanged RIASEC extraction/ranking, and classifier scores do not enter matching or fusion. Session metadata records the classifier path and thresholds. Reviewed annotations are not replaced with model predictions. See [pre-extension baseline](PRE_EXTENSION_BASELINE.md): source-validation F1 is 1.000, but paraphrase Task F1 is .564103, Context F1 .457831 and exact match 35%; integration does not improve those metrics.

The terminal now defaults to the conservative Stanza splitter; `--clause-splitter spacy` preserves comparison mode. See [splitter rules and setup](UTTERANCE_PROCESSING.md#stanza-conservative-baseline-2-november-2026). Classifier weights and thresholds are unchanged.
