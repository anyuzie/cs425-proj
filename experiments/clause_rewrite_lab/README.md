# Standalone clause rewrite and task-match lab

Double-click **Start Clause Rewrite Lab.cmd**, enter one clause, and press Enter.
Type `/quit` to exit. Models load once; the first start also encodes the task texts.
The full DiffuSeq sampling path takes roughly a minute for three candidates on the
tested GPU. The screen displays:

1. Original clause and the selected real DiffuSeq output.
2. Reconstruction from a separate FLAN-T5 model, and fidelity scores/reasons.
3. The exact text ColBERT received and its strongest occupation-task matches.
4. If the rewrite failed the checks, an explicitly separate preview of matches
   for that rejected rewrite, alongside the original-clause fallback matches.

**Accepted DiffuSeq output is the task-matching query. Reconstruction is evaluation
only and never enters the matcher.** All three candidate rewrites and all four
bidirectional NLI scores are retained in the session JSONL for inspection.

## Running one clause

From this folder:

```powershell
./launch.ps1 --clause "I enjoy debugging software errors." --top-k 5
```

Add `--json` for a machine-readable result. `--diffuseq-model qqp` selects the
authors' question-paraphrasing checkpoint instead of the default simplification
checkpoint. `--steps 50` selects experimental accelerated sampling; it produced
poor text in the local smoke checks and is not the default.

`runtime.local.json` identifies the separate CUDA environment and resource folder.
The installed launcher uses that environment even when the original project's
Python installation is CPU-only. The tested environment is Python 3.13,
PyTorch 2.7.1+cu128, Transformers 4.51.3, on an RTX 4060 with 8 GB memory.
The exact installed dependency list is in `reports/dependencies.lock.txt`.

## Isolation

This tool imports only its own package and the separately downloaded upstream
DiffuSeq source. It does not import or connect to the existing ranker, extractor,
classifier, tracker, System, terminal, or semantic experiment pipeline. The old
source, configurations, environment, CSV and notebook are preserved. It does not
rank occupations by RIASEC, apply dislike penalties, train a model, or append new
ranking metrics. This narrower scope follows the final request for a separate
interactive rewrite-and-task inspector.

The existing occupation/task CSVs are read only. Matching covers 8,243 task
statements belonging to the existing 378 eligible occupations (Job Zones 4/5 with
complete supplied RIASEC z-scores). Those numerical columns are used only to
identify the existing eligible pool; they are never model inputs or matching
scores. All task token representations are generated from raw task text in memory
at startup; no task embedding/index file is loaded or written. Session logs and
new model resources stay inside the separate lab workspace.

## Models and checks

- **Forward generator:** the authors' DiffuSeq text-simplification checkpoint,
  SHA-256 `db190fbfc43a23142cd7c8c1637ad8747178106c3e11e9c69c659dee5288c876`.
  Source: the official DiffuSeq repository's linked remaining-task releases.
  Its extensionless release contains weights only. The shared BERT encoder,
  128-dimensional latent state, 2,000-step sqrt schedule, and vocabulary use the
  official QQP configuration; strict complete state loading verifies structural
  compatibility. This does not establish that the missing TS training settings
  are independently verified. The QQP checkpoint and its original configuration
  are retained separately with SHA-256
  `f49b686a28dbb035de00aa987c28278058a04f3bee89f4ae13bcba3f4a5d7bb1`.
- **Reconstruction:** frozen `google/flan-t5-base`, deterministic four-beam
  generation, at most 96 output tokens. It sees only the forward rewrite and a
  fixed reconstruction instruction, never the original or an occupation task.
- **Fidelity:** pinned MPNet at revision
  `e8c3b32edf5434bc2275fc9bab85f82640a19130`, plus the existing frozen local
  three-label DeBERTa NLI checkpoint. Direct and round-trip cosine must each be
  at least .80; entailment in both directions must be at least .50 for original
  versus rewrite and original versus reconstruction. Conservative lexical checks
  also preserve preference sign, strength, frequency, uncertainty, conditions,
  comparison cues, and numeric quantities. These thresholds and rules are
  provisional heuristics, not calibrated truth or a complete semantic guarantee.
- **Matching:** actual `colbert-ir/colbertv2.0` BERT and learned projection weights.
  Contextual normalized token vectors use the checkpoint's Q/D marker tokens,
  document punctuation/padding masks, and masked-document-max/query-sum MaxSim.
  Query windows have 29 content tokens plus markers, overlap 8; task windows have
  177 content tokens plus markers, overlap 32. Scores average across query windows,
  divide by 32 and take the best document window per task. They are association
  scores, not probabilities or preference compatibility. Negative preferences
  remain visible and can identify potential conflicts; this standalone tool does
  not convert them into a recommendation score.

Three sampling seeds (425, 426, 427) generate candidates. Passing candidates are
ordered by mean direct/cycle cosine and then seed. The best passing rewrite is
used; otherwise the original is used. A copy is explicitly flagged. Long clauses
that exceed DiffuSeq's source budget keep the original rather than being silently
truncated. ColBERT retains long matching text through overlapping token windows.

DiffuSeq is **not trained for career-preference standardization**. The QQP model
can turn statements into questions; the simplification model can copy clauses or
make inappropriate edits. Full sampling and fidelity checks reduce some observed
failures but are not evidence of reliable standardization or better recommendations.
No synthetic output is presented as an actual model result.

## Resources and reproducibility

The resource manifest records upstream source and model hashes. Runtime is offline
and all model weights are frozen. Downloads happen only through explicit setup:

```powershell
python scripts/bootstrap.py
./.venv/Scripts/python.exe scripts/prepare_models.py
```

For a different checkout, configure `runtime.local.json` with `python`,
`resources_root`, `project_root`, and the pinned local MPNet snapshot as `encoder`.
The deployment uses the prepared external lab environment instead of relocating
or modifying either virtual environment.

Run the independent contract checks with:

```powershell
./.venv/Scripts/python.exe -m unittest discover -s tests -v
```

See `reports/` for executed real-model smoke results and the original-file hash
audit. Unit tests additionally prove that reconstruction never reaches matching,
accepted forward output does, fallback is labeled, high similarity cannot bypass
explicit polarity checks, and masked MaxSim/window coverage are correct.

Upstream references and licenses:
[DiffuSeq](https://github.com/Shark-NLP/DiffuSeq) (MIT),
[ColBERT](https://github.com/stanford-futuredata/ColBERT) (MIT),
[FLAN-T5](https://huggingface.co/google/flan-t5-base) (Apache 2.0).
