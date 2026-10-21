# Run and test the integrated chatbot

Integration verified on 21 October 2026. Real extractor, real NLI, production
tracker/policy, configured ranking and optional fine-tuned T5 are now connected.
No mocks are used by either production system configuration.

The current application keeps extraction, contradiction rules/NLI, and ranking
behavior from `main` commit `7e065d6`. Only tracker/action policy, T5 wording, and
real-model loading/integration are extended. Extractor context and evidence
selection are unchanged; its readings pass directly to the tracker. V3b remains
the default question model with template fallback. Historical local evaluation
reports describe earlier versions and do not establish current behavior.

## Start on this computer

Open PowerShell and use the existing Python environment in the parent project:

```powershell
cd C:\Code\CS425_Project\cs425-proj
..\.venv\Scripts\python.exe run.py
```

Open <http://127.0.0.1:8000>. Wait for "Application startup complete" before
loading the page. The models load once at startup; CPU initialization can take
some time. Stop with Ctrl+C in that terminal. If port 8000 is already occupied by
the instance started during integration, use it or stop it before starting another.

The app defaults to V3b (`configs/system.json`), offline model loading, and four
OMP/MKL threads. No environment commands are required. Explicit environment
settings still override these defaults. With an already activated Python
environment, use `python run.py`.

On another computer, first install the dependencies and copy the required model,
index and data assets described below.

For template-only wording, stop the process, change the config and restart:

```powershell
$env:CM_CONFIG = 'configs/system.question-bank.json'
..\.venv\Scripts\python.exe -m uvicorn app.server:app --host 127.0.0.1 --port 8000
```

The researcher panel displays tracked preferences, action reasons and rankings.
Create a fresh session to reset the conversation. The app ends after 5â€“10 user
turns. Sessions are in memory; transcripts remain in `logs/<participant_code>.jsonl`.
Use a test code rather than personal details. The integration checks created
`integration-conflict`, `integration-probe` and `integration-complete` transcripts.

For participant mode, set `$env:CM_STUDY_MODE = '1'` before starting. This hides
the researcher panel. Remove that variable to return to researcher mode:
`Remove-Item Env:CM_STUDY_MODE -ErrorAction SilentlyContinue`.

## Local assets installed

- `models/extractor/`: trained DeBERTa checkpoint from the teammate's
  `NLP_extractor_model.zip`, specifically its `riasec-extractor/` directory.
  Includes safe tensor weights, configuration and tokenizer. Bundled datasets,
  notebooks and redundant pickle-based `.pt` weights were not installed.
- `models/nli/`: pretrained `cross-encoder/nli-deberta-v3-base`, revision
  `6c749ce3425cd33b46d187e45b92bbf96ee12ec7`.
- `models/t5-questions/`: the previously trained local T5 checkpoint, copied from
  the parent project's `artifacts/t5-questions/`.

Models are ignored by Git. Another computer needs these folders as well as the
source code. A clean environment can be created **inside the group repository**:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-models.txt
# Then use .\.venv\Scripts\python.exe instead of ..\.venv\Scripts\python.exe.
```

Keep the T5 training environment separately pinned when retraining; the existing
checkpoint was smoke-tested successfully with the group's runtime dependencies.
Training scripts and provenance remain in the parent project's `training/`.

## Try these conversations

Contradiction:

1. "I love repairing machines and working with tools."
2. "I hate repairing machines and working with tools."
3. Answer the challenge explicitly, e.g. "I would prefer not to repair machines."

Familiarity (start a fresh session):

1. "My dad works in sales, so selling is what I know."
2. "I actually dislike selling and persuading people."

Complete profile (start a fresh session): describe enjoyment of research, helping
people, creative work, persuasion, detailed procedures and practical work across
successive turns. Answer the follow-up questions in full sentences. A real HTTP
check with these inputs finished at turn 8 and returned five occupation cards.

T5 generates only the question wording; Python inserts exact quotations and
retains the policy's action and target interest. A malformed, overlong or repeated
T5 question falls back to the bank. OPENING and RECOMMEND remain deterministic.

## Current rules

- Cumulative confidence-weighted merging, no recency decay. The existing registry
  name `decay` is retained for compatibility. If all merge weights are zero, retain
  the latest reading.
- The unchanged teammate contradiction rule requires opposite signs with both
  confidences > 0.4. NLI confirmation requires P(contradiction) > 0.6, as on main.
  Keep the accepted profile while a confirmed claim is disputed.
- PROBE requires familiarity and not previously probed; no reason-confidence gate.
- All-neutral answers retry the same underlying question, including CHALLENGE or
  PROBE. No automatic "settled" type after two neutral replies.
- A targeted non-neutral answer to CHALLENGE/PROBE replaces that type and clears
  its conflict. New conflicts can be detected later. CHALLENGE still asks the user
  to choose; mixed/contextual resolution remains deferred.
- CLARIFY uses the largest variance among unfilled interests over the top 100
  occupations. If all are filled, refine the least-confident one.
- Normal stopping: at least 5 turns, all six interests filled, each value changes
  by at most 0.10 on both of the last two updates, no unresolved conflict or pending
  probe/retry. Hard stop at 10 turns. The `no_probing` control skips interventions.

Both application defaults use `configs/ranking.yaml`, exactly as on main:
structured-only weighted-dot-product scoring. Existing BM25/MiniLM/RRF and cosine
variants remain available through their original ranking configurations; they
are not silently enabled by this branch.

MiniLM resources were built locally for 8,243 tasks across 378 occupations using
the pinned `all-MiniLM-L6-v2` revision. These ignored assets live under
`artifacts/ranking/`. On another computer, prepare them before selecting a dense-ranking variant:

```powershell
Remove-Item Env:HF_HUB_OFFLINE -ErrorAction SilentlyContinue
..\.venv\Scripts\python.exe -m recommender.preparation.build_dense --download-model
$env:HF_HUB_OFFLINE = '1'
```

## Verification and known limits

Extraction and contradiction-rule source and their tests are restored from main.
Ranking tests and ranking documentation are also restored, including their
original strict assertions. No expectation changes or floating-point tolerance
patches are used to make those tests pass. Current verification is recorded in
`reports/dialogue/scope_revert_20261021/REPORT.md` and its independent
main-reference comparison.

Extractor mistakes can still affect state. A clarification request only receives
a repair response when the original extractor supplies a neutral reading;
application code no longer discards or reclassifies its output. Challenge
resolution uses supplied readings without an additional NLI relevance filter.
T5 wording remains imperfect and can fall back to templates.

Run tracker/action integration tests:

```powershell
..\.venv\Scripts\python.exe -m unittest discover -s tests -p test_dialogue_integration.py -v
```

Run the full suite with `-m unittest discover -s tests -v`. Original ranking tests
contain historical dataset/fingerprint assumptions and strict saved-score checks;
see the scope-reversion verification before interpreting failures as new changes.

## Ranking source recovery

The checkout's wrapper modules imported `recommender.models`, but that source
directory was missing because `.gitignore` excluded any directory named `models`.
Recovered implementations from the versioned
`reports/ranking/stance_dev120_v3/source_snapshot/`, adjusted their relative imports,
and reconnected the existing central parameter files and variant catalog. All
central baseline parameter values match the recovered constants. BM25 k1/b
substitution is preserved for the existing modular variant test. The ignore rule
now excludes only root `/models/`, so ranking Python source can be committed.
`RANKING_SOURCE_RECOVERY.json` records initial source/restoration hashes. Frozen ranking algorithms, baseline selections, historical reports and evaluation
data are preserved. Application defaults retain main's `configs/ranking.yaml` selection.

## Development dialogue evaluation

```powershell
..\.venv\Scripts\python.exe -m eval.dialogue --limit 3
```

This runs real extraction, tracking, NLI and ranking with deterministic question-bank
wording for both full and no-probing conditions. It uses development profiles only;
labels stay in the evaluator and simulated user. Results, source/checkpoint hashes
and configuration are saved in a new `reports/dialogue/` directory. Increase the
limit for broader development evaluation. Do not interpret a small smoke run as a
human study or held-out quality measurement.

See [CORE_FUNCTIONALITY.md](CORE_FUNCTIONALITY.md) for the checked feature list.

## Wider scenario testing

See [SCENARIO_TESTING.md](SCENARIO_TESTING.md) for 19 real-model cases, expected
behavior, five observed model failures, API boundary checks and replay commands.
These historical results predate the scope reversion; current checks use the original main component behavior.

## T5 wording evaluation

The [current T5 report](../reports/t5/current_20261021/REPORT.md) contains 112 controlled
requests and ten complete fictional-user conversations, with raw generation versus
template fallback provenance. Run `python -m eval.t5_wording` to repeat the audit.

## Retained T5 experiments

V1/V2 checkpoints and local experiment presets remain available for rollback.
The current defaults select V3b, with template fallback. Historical evaluation
reports describe the code and ranking configuration used at the time of each run.
