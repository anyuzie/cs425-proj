# Ranking application integration audit

Reviewed 20 October 2026 against the current working tree, including the modular
refactor. This is an interface audit, not actual-tracker relevance evidence.

## Runtime path

```text
static/index.html + app.js
  -> POST /api/session, POST /api/chat (app/server.py)
  -> System.load -> System.from_config -> pipeline.registry.build
  -> configured ranker -> RankingPipeline -> algorithm/variant registry

Each user turn: extractor -> tracker -> contradiction rule/NLI
  -> ranking(current slots, observed keys, chronological user-only history)
  -> policy(state.ranking: top 100 with occupation RIASEC values)
  -> filler + recommendation cards (top five, up to three task matches)
  -> API response -> frontend
```

The ranker and enabled scoring components are constructed once per System and
reused across turns. Rankings are recomputed from each current tracker snapshot;
occupation scores are not accumulated. Unknown dimensions remain absent from
observed keys, while observed zero values remain present. Policy, researcher
panel and cards reuse the same turn ranking. Labels and oracle hidden slots do
not cross the live ranking interface.

Scoring and data packages do not depend on application or evaluator code.
Preparation is offline; chat does not fit models or rebuild semantic indexes.
The semantic component owns its pinned resource/cache validation. Legacy imports
forward to canonical implementation modules.

## Configuration and limits

- The web server loads `configs/system.json` unless `CM_CONFIG` overrides it.
  That file selects `configs/ranking.yaml`: dot product, with text components off.
- `configs/system.cosine.json` selects standalone cosine for the same web API.
  In PowerShell, set `$env:CM_CONFIG = 'configs/system.cosine.json'`, then run
  `.venv/Scripts/python.exe -m uvicorn app.server:app` from the repository root.
  This requires completed upstream components and their model resources.
- `recommender.workflow` reads `configs/ranking/workflow.json` for evaluate/chat;
  the web server does not implicitly read that workflow file. To use a different
  model on the web, select a system file whose ranking path names that model.
- `active_only` is supported by evaluator tracker replay with supplied lifecycle
  snapshots. Live State has no lifecycle-evidence contract. System now rejects
  this selection at construction instead of failing on the first user turn.
  Do not infer lifecycle status from raw text or silently fall back to raw.
- Tracker, NLI and manual policy remain explicit production stubs. The extractor
  adapter is implemented, but its trained checkpoint is absent locally.
- MiniLM/FAISS integration requires prepared optional resources. Cosine and BM25
  application acceptance does not establish real dense-resource integration.
- Recommendation `why` is a generic profile-based explanation, not per-result
  attribution. Retrieved task matches remain provisional.

## Acceptance coverage

`tests/test_ranking_app_integration.py` exercises real configured cosine and
cosine+BM25 through FastAPI with explicitly named mock upstream components. It
checks static-page serving, exact HTTP trace scores/order against direct runtime
ranking, cumulative user history, state ranking, final recommendation order,
component reuse, and retained ranking provenance in logs with debug disabled.
It also checks early rejection of replay-only evidence filtering.

Existing M5 tests cover changing tracker values, conflict/probe flows, empty and
short rankings, replay exports and experiment provenance. The legacy HTTP suite
uses a keyword fake ranker; the new HTTP tests close that coverage gap without
registering test components in production.

Verification in the project `.venv`: both new application tests pass; all five
modularity tests pass; the M5 suite reports four passes and one optional-cache
skip. Full discovery runs 173 tests: 154 pass, 14 skip, two fail and three error.
The failures assert the old 60-profile count against the current 120-profile
dataset. Two errors reject legacy-history dataset fingerprint mismatches, and
one requires the absent extractor checkpoint. These are the same five categories
documented before this audit in `RANKING_MODULES.md`. The full suite is not green;
the new application checks pass. No historical data or reports were rewritten
to suppress those failures.

Run with the project environment:

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -p test_ranking_app_integration.py -v
.venv/Scripts/python.exe -m unittest discover -s tests -p test_integration.py -v
```

Actual-tracker quality, trained-extractor inference, real semantic resources and
team latency acceptance remain separate gates. No defaults, scoring formulas,
frozen variants or experiment history were changed by this audit.
