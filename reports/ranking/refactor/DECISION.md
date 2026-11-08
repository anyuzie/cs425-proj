# Modular ranking refactor acceptance

**Decision: refactor accepted; stop before M3.** M1 and M2 remain implemented. No new scoring formula, lexical/semantic retrieval, fusion algorithm, or eligibility policy was introduced.

## Changes

- Extracted M1's unchanged signed dot product into `m1.py` and registered `weighted_dot_product@baseline_v1` with explicit parameters.
- Added common component input/output contracts and usable/unavailable/unusable signal statuses.
- Added a registry that rejects algorithm/variant overwrites and returns owned parameter snapshots.
- Added `configs/ranking.json` and a configuration-driven `RankingPipeline`. Existing `StructuredRanker` callers retain their API and output behaviour.
- Added the fusion protocol/dispatch boundary; standalone rankings bypass fusion and inactive signals never contribute. Actual RRF remains planned.
- Routed M2 through the shared pipeline, preserving its original command and metrics. New runs save resolved parameters and versioned component IDs; saved configurations can be evaluated again and stale resolved parameters are rejected.
- Added the [architecture memory](../../../docs/RANKING_EXPERIMENT_ARCHITECTURE.md) and root `AGENTS.md` discovery reference.

## Validation

`python -m unittest discover -s tests -v`: **54 tests passed** (44 existing tests plus 10 architecture tests).

The real-data regression test compares every occupation result against the original saved M2 baseline: **60 profiles × 378 occupations = 22,680 results**, with exact equality of codes, titles, ranks, scores, vectors, signal diagnostics, and empty fusion/evidence fields. Runtime tests also compare the original formula independently, including observed dimensions, negative scores, and deterministic tie-breaking.

Other checks cover registry determinism/overwrite prevention, old variant selection, unknown selections before data loading, direct component execution, shared output validation, disabled/unusable signals, explicit unimplemented-fusion failure, configuration round-tripping, tamper rejection, and unchanged history prefixes. Existing M0–M2 tests, metric/history tests, and legacy paired comparisons remain valid.

Tests use temporary output directories/history files; this refactor does not create fabricated experiments or append test fixtures to the central CSV. The original history bytes and historical baseline artifacts are preserved. No dataset content was changed and no held-out rankings were run.

## Usage and remaining limits

```powershell
python -m recommender.evaluate --config configs/m2.json --ranking-config configs/ranking.json --output reports/ranking/m2/new_selected_run
```

Omit `--ranking-config` for the accepted M1 default; choose a fresh output directory. Only M1 is registered in production. More than one usable component requires implementing the specified RRF baseline at an authorized later milestone. Runtime tracker/policy integration, actual-tracker evaluation, and evidence generation remain pending.

The next recommended step is M3: implement independent BM25 and the specified RRF baseline, then compare lexical-only and structured+lexical under matching development/oracle conditions. No M3 work is authorized by this refactor.
