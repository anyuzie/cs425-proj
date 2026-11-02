# Utterance processing

Install with `.venv\Scripts\python -m pip install -r requirements-utterance.txt`.
Enable in system JSON with `settings.utterance_processing = true`.
The model loads once per System. Missing dependencies fail explicitly when enabled.
Existing configurations remain compatible and do not load spaCy unless enabled.

`UtteranceProcessingPipeline.process(request)` accepts the combined RankingRequest
(or ComponentInput), reads only user_sentences, and returns ProcessedUtterance records.
`split(text)` handles a single utterance. Each clause includes an exact source slice,
character offsets, sentence index, dependency relation, parent predicate position,
and an inherited subject for coordinated actions. Original input is never rewritten.

Separate sentences and coordinated verbal predicates split. Subject-bearing advcl
and ccomp clauses split. Object lists, infinitival complements, relative clauses and
fragments remain intact. Discontinuous parents conservatively retain the sentence.
Negation and modifiers stay attached according to the parse; ambiguous scope is not
resolved. Empty input produces no clauses. Parser predictions can be imperfect.

System records clauses under turn_trace.utterance_processing; rankings continue
receiving the original full utterance history and unchanged RIASEC scores.
No activity/job-nature classification, weighting or reranking changes are included.

Known small-model limitation: "I repair machines and install equipment" is parsed
as noun coordination and retained intact; adding "electrical" before "equipment"
produces verbal coordination and splits correctly. No keyword override is applied.
The supplied configs/system.json enables this stage; custom configurations opt in.

## Independent-turn terminal

From the project root run:

```powershell
.\.venv\Scripts\python.exe -m app.ranking_terminal
```

After the occupation list and component matches, each entry prints
`clauses: ["first clause", "second clause"]`. The parser loads once at startup.
This terminal always displays clauses, independently of the conversation setting.
Existing ranking configuration overrides remain supported.

## Initial clause classification
Each clause now exposes activity_detected, job_nature_detected and a derived outcome
(activity, job nature, mixed, uncertain), plus exact evidence offsets in its utterance.
The terminal displays these after the clause list. Activity rules require a curated
work-action lemma and a grammatical target; job-nature rules use explicit work
characteristic phrases. Broad role-purpose inference is not included in this version.
These are content labels, not positive preferences: negation/dislike remains in the
original clause and must be interpreted separately before any future scoring.
Unknown vocabulary remains uncertain. Rule coverage is provisional, not calibrated
confidence. No labels or evidence are passed to semantic scoring or reranking.

## Terminal data collection
Every nonempty entry other than the exit command is saved automatically in
`data/collected/utterances/<session UUID>.jsonl`. Override using `--collection-dir`.
The terminal prints the destination on startup. Existing files are never replaced.
Session metadata records resolved ranking parameters, extractor path, parser and
spaCy versions, and processing source fingerprints. A submitted event saves the raw
input and trimmed processing text before inference. A linked completed event saves
clauses, evidence, derived flags/outcomes, extractor readings, RIASEC values and
observed dimensions, and the complete ranking response including scores/evidence.
Failures produce linked failed events; an interrupted process may leave a submitted
entry without a completion. These remain useful examples for later review.
Predictions are explicitly unlabelled and must not be treated as human annotations.
Keep later annotations linked by entry ID in a separate dataset. No automatic
training, semantic scoring changes, or occupation-corpus labelling is introduced.

## Terminal user-clause classifier integration (2 November 2026)

`python -m app.ranking_terminal` now loads `models/task_context_bert` once and passes the exact spaCy clause texts to its batched predictor. Each clause displays `task_score`, `context_score`, `route_task` and `route_context`; independent checkpoint thresholds default to 0.5. Override the checkpoint with `--task-context-classifier PATH`. The old rule-evidence terminal display is replaced with these model scores; the spaCy splitter, source spans and existing annotation exports remain unchanged.

This is experimental display-only integration: original utterances still feed unchanged RIASEC extraction/ranking, and classifier scores do not enter matching or fusion. Session metadata records the classifier path and thresholds. Reviewed annotations are not replaced with model predictions. See [pre-extension baseline](PRE_EXTENSION_BASELINE.md): source-validation F1 is 1.000, but paraphrase Task F1 is .564103, Context F1 .457831 and exact match 35%; integration does not improve those metrics.

## Stanza conservative baseline (2 November 2026)

The independent ranking terminal now defaults to Stanza for user-clause splitting:

```powershell
.venv/Scripts/python.exe -m pip install -r requirements-utterance.txt
.venv/Scripts/python.exe scripts/prepare_stanza.py
.venv/Scripts/python.exe -m app.ranking_terminal
# Compare with the preserved spaCy splitter:
.venv/Scripts/python.exe -m app.ranking_terminal --clause-splitter spacy
```

Stanza 1.15.0 English tokenization, multiword expansion, POS, lemma and dependency parsing use explicit combined/nocharlm packages under `models/stanza`. Runtime is offline (`download_method=None`), loads once, and uses CPU. This baseline uses dependency parsing, not constituency parsing. Setup downloads models explicitly; runtime does not silently fall back to another splitter.

Rules (`stanza_conservative_v1`): retain sentence units; split only root-level coordinated/paratactic assertions with explicit subjects. Keep embedded complements, relative and adverbial clauses together. Shared-subject coordination stays together, preserving preference and negation scope. Trim boundary punctuation and conjunctions, retain exact source slices/offsets, and fall back to the sentence if proposed spans overlap. Never rewrite pronouns or infer missing participants. Ambiguous/missing-punctuation text may remain too long; conservatism intentionally favours complete units over fragments. This is an initial rule baseline, not a validated conversation-understanding model.

Verified example:

- `i find it funny when i see someone fall down`
- `sometimes i feel like that makes me quite evil`
- `i like to care for people`
- `i occasionally also volunteer at the old folks home`

These four units are retained for both comma-separated and sentence-separated versions of the example. `I like caring for people but I dislike working nights` splits into two units; `I do not enjoy repairing machinery and working outdoors` stays intact. Real-model regressions verify these boundaries, complements, contractions, exact offsets and BERT display input ordering.

Only the terminal's user-clause parser changes. Occupation annotations continue using their existing spaCy processor; shared System configuration, existing utterance annotation files/exports, RIASEC extraction and ranking calculations are unchanged. The terminal records parser backend/version and implementation hashes in new collection-session metadata. BERT continues receiving original clause wording and retains its known errors; no retraining or ranking-performance improvement is claimed. See [the pre-extension baseline](PRE_EXTENSION_BASELINE.md) for frozen comparison results. Future splitter comparison should use reviewed conversational boundaries and measure fragment rate, preference-scope preservation, boundary accuracy and latency.

Stanza setup and dependency interfaces follow the [official download documentation](https://stanfordnlp.github.io/stanza/download_models.html) and [dependency parser documentation](https://stanfordnlp.github.io/stanza/depparse.html).
