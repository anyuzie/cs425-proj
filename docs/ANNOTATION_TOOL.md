# Clause labelling — isolated tool

From the project root:

```powershell
.\.venv\Scripts\python.exe -m annotation_tool
```

Open http://127.0.0.1:8010. Restart this separate service after upgrading the tool.
It does not import or change the ranking terminal, scoring, extractor or spaCy models.

## Left to right
1. **Utterances:** Refresh reads new terminal submissions and puts newest unreviewed
   inputs first. Select any entry. Reviewed entries appear at the bottom.
2. **Select and label:** Select a phrase in Original utterance. It appears automatically
   in the editable clause box. Correct wording if needed. Click Activity, Job nature,
   or Uncertain. Activity and Job nature can both be active. Uncertain clears both.
   Re-selecting the same original span updates its preview rather than duplicating it.
3. **Labelled clauses:** Check the live JSONL preview. Click a clause to revise it,
   or Remove it. Select further passages to add more clauses. No separate evidence,
   ordering, non-overlap, or whole-utterance coverage is required.
4. **Save to training data:** Saves an append-only annotation revision, updates the
   training JSONL, and moves the entry to the bottom marked Reviewed. There is no
   second export step. Re-saving replaces that entry's rows in the current training
   snapshot while retaining annotation history. Save never trains a model.

Reviewer settings are optional: local-reviewer is the default; contributor/split
identity defaults to collection session. Set a consistent anonymous group across
sessions from the same person before preparing train/test splits.

## Files and training contract
- Raw collection logs: data/collected/utterances (never modified).
- Annotation history: data/annotations/utterances (append-only revisions).
- Current training snapshot: data/training/utterances/current.jsonl, rebuilt on Save.
- Manifest: data/training/utterances/current.manifest.json, including dataset hash.

Rows contain edited clause text, original source_text and selection offsets, full
original context, two independent labels, outcome, reviewer, group and source IDs,
and annotation revision. Offsets always address Unicode code points in original
context, not edited text. Edited snippets are not gold boundary or evidence spans.
Activity/job_nature are independent content labels, not preference polarity.
Uncertain rows have null labels and training_eligible=false: retain them for review
or unsupervised contextual adaptation, but exclude them from supervised binary-label
training. They must never be silently converted to two negative labels.

Full utterances are not necessarily exhaustively annotated. Do not treat unselected
text as negative examples or infer that absent labels cover the rest of an utterance.
Overlapping snippets are allowed, so deduplicate/group examples before splitting.
Keep all related contributors and original contexts in the same train/test split.
Original model predictions are never copied into human training labels.

Existing span-review annotations remain readable and exportable; opening and saving
one through the simple UI creates a new clause-label revision. Old drafts stay drafts
until saved. The legacy CLI export still writes timestamped reviewed snapshots:

```powershell
.\.venv\Scripts\python.exe -m annotation_tool export
```

Use --collection-dir, --annotation-dir, --output and --port to override locations.
Run one annotation-service instance per output directory. Concurrent stale revisions
and changed source fingerprints are rejected. Each training file is atomically
replaced; a save failure clearly reports whether an annotation revision was already
saved so you can reload and retry. Local outputs remain ignored by Git.
