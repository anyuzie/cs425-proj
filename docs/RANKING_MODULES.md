# Ranking development map

The recommended workflow selects standalone `cosine_similarity@baseline_v1`, the
best recorded development/oracle configuration. This is a configuration choice,
not a new quality claim. Legacy production defaults and dot-product controls are
preserved. Cosine has not been established as better with actual tracker inputs.

## Layout and dependency direction

```text
configs/                         All editable experiment/application settings
  ranking/workflow.json          Select model, preparation, evaluation, system
  ranking/parameters/*.json       Exact accepted algorithm parameter snapshots
  ranking/variants.json           Additional named parameter variants
  ranking/preparation.json        Resource preparation selection and input data
  ranking/*.json                 Component/fusion selections, including cosine
  m2.json                        Evaluation conditions and datasets
  system.cosine.json             Application composition with cosine ranking
recommender/
  schemas.py, components.py       Public contracts; no model dependencies
  configuration.py               Central safe configuration loader
  data/
    data_loader.py               CSV/profile loading, validation and eligibility
    tasks.py                     Task mapping and sentence preprocessing
    lexical.py                   BM25-specific tokenization
    evidence.py                  Evidence validation and query selection
  models/
    m1.py, cosine.py              Independent structured scorers
    bm25_ranker.py                Lexical scorer
    dense_ranker.py               Encoder/cache loading and semantic scorer
    dense_config.py               Dense factory and parameter selection
    task_preferences.py           Explicit activity preference extraction
    stance_ranker.py              Experimental signed task scorer
    stance_config.py              Stance factory and parameter selection
    fusion.py                     Occupation-level RRF
    registry.py                   Algorithm factories and named variants
    ranker.py                     Component orchestration and response assembly
    structured.py                 Legacy dot-product facade
  preparation/build_dense.py     Offline pretrained model/task-index preparation
  experiments/
    evaluation.py                Metrics only
    evaluate.py                  Experiment runner, snapshots and provenance
    runtime_inputs.py            Tracker replay adapter
    compare.py                   Compatible paired comparisons
    history.py                   Append-only experiment history
    audit.py                     Dataset audit entry point
  workflow.py                    CLI dispatch using central workflow selection
pipeline/system.py               Conversation orchestration shared by all UIs
app/terminal.py                   Conversation terminal presentation and I/O
app/ranking_terminal.py           Independent-turn extractor/ranker diagnostic
app/server.py                     HTTP presentation
```

Data and models never import evaluation, application or conversation code.
Scorers implement `ScoringComponent.score(ComponentInput) -> ComponentRanking`.
The registry injects eligible occupations and an owned parameter dictionary.
`RankingPipeline` coordinates components and fusion; it does not train models or
read terminal input. `System` coordinates extraction, tracking, ranking and
dialogue policy. Terminal and HTTP adapters call `System`.

Flat `recommender/*.py` modules remain compatibility entry points forwarding to
the same implementation module, including existing patch targets and CLI calls.
Use the package paths above for new implementation work. Public contracts remain
at their original paths.

## Run the recommended configuration

From the repository root, using an environment with project dependencies:

```powershell
python -m recommender.workflow evaluate
python -m recommender.workflow chat
python -m recommender.workflow prepare --download-model
```

All three read `configs/ranking/workflow.json`; `--config` selects another workflow.
Its paths are relative to the repository root. Evaluation additionally accepts
`--output` and `--history` for isolated reports. Chat receives the workflow's same
ranking selection, overriding the system file's ranking path in memory.

Cosine requires no resource preparation. `prepare` explicitly builds the optional
MiniLM task index selected in `preparation.json`; it is never invoked by chat or
evaluation and refuses to overwrite an existing cache. There is currently **no
ranking model fitting**: dot product/cosine are formulas, BM25 indexes documents,
and MiniLM is pretrained. Extractor training is an upstream subsystem, outside
this package. Add a separate fitting layer if a future ranker requires learning;
do not mix fitting into evaluation or runtime scoring.

Direct entry points remain available:

```powershell
python -m recommender.experiments.evaluate --ranking-config configs/ranking/m1_cosine.json
python -m recommender.preparation.build_dense --config configs/ranking/preparation.json
python -m app.terminal --system-config configs/system.cosine.json
python -m app.ranking_terminal --ranking-config configs/ranking/m1_cosine.json
```

Conversation chat and the independent-turn diagnostic require the trained
extractor checkpoint; conversation chat also requires the configured NLI resource.
These are not replaced with mocks automatically.

## Iterate without changing accepted baselines

1. Select a control/treatment in `configs/ranking/*.json` and point the workflow
   at it. Keep evaluation conditions in `configs/m2.json` separate from selection.
2. Accepted files under `parameters/` preserve the existing parameter values;
   do not edit their meaning in place. Add a new variant for changed settings.
3. For an already supported parameter change, add an entry to `variants.json`
   containing `component`, `algorithm`, `variant`, and a complete `parameters`
   dictionary, then select that new name. Duplicate IDs are rejected.
4. BM25 supports configurable positive finite `k1` and `b` in `[0,1]`; its other
   preprocessing choices remain frozen. Other algorithms reject settings they
   do not implement. A different model/formula/preprocessing method needs an
   explicit implementation and registry entry rather than an ignored setting.
5. To swap a component implementation, implement the scoring protocol, register
   its factory and parameters in `models/registry.py`, then change selection.
   Pipeline, metrics, terminal and conversation code need no algorithm dispatch edits.

For example, a new BM25 catalog entry copies the complete baseline parameters
from `parameters/bm25_ranker_baseline_parameters.json`, changes `k1`, and uses a
new variant name such as `k1_2_v1`. Never reuse `baseline_v1` for changed behavior.

New evaluation runs save selection, resolved parameters, original selection and
evaluation bytes, predictions, metrics and hashes. Source fingerprints recursively
cover the new packages. Frozen historical reports and history rows are unchanged.
Tests use temporary report/history paths; they are not experiment history entries.

## Verification

```powershell
python -m unittest discover -s tests -v
```

`tests/test_modular_ranking.py` checks canonical/legacy identity, dependency
boundaries, cosine workflow selection, parameter-variant substitution and terminal
delegation. Existing scoring/evaluation/integration tests remain the regression
controls. Optional semantic tests require the optional model dependencies.

Refactor verification (20 October 2026): all complete ranking responses match the
pre-refactor implementation on 120 development profiles for dot product, cosine,
BM25 and cosine+BM25. The cosine workflow reproduces Hit@5 `0.6833333333333333`.
Full discovery ran 171 tests: 152 passed, 14 skipped, two failed and three errored.
The five unsuccessful checks also fail before the refactor: two tests assume
60 development profiles, two legacy-history tests encounter the changed dataset
fingerprint, and one pipeline test requires the absent extractor checkpoint.
All five new modularity tests pass. Real MiniLM/cache integration remains
unverified in this environment; no model weights were downloaded or rebuilt.
