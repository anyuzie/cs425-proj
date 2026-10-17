# M5 integrated ranking baseline

## Decision and scope

Freeze `baseline_v1` as M1-only, with raw chronological user-only text retained for independently configurable BM25/MiniLM experiments. M1, M3, M4, RRF and M2 metric implementations are unchanged. Production selection remains `configs/ranking.yaml`; the frozen control is `configs/ranking/baseline_v1.json`. The existing three-way preset remains an experimental comparison, not the default.

Ranking integration and scripted interface acceptance are implemented. Full team end-to-end acceptance remains pending: `DecayTracker`, `NliConfirmer` and `ManualPolicy` still raise `NotImplementedError`, the extractor checkpoint is unavailable, and teammates have not agreed a latency budget. No actual-tracker quality or probing-effectiveness claim is made.

## Acceptance criteria mapping

| Instruction sheet criterion | Implementation and evidence | Status |
|---|---|---|
| Load caches/indexes at startup | `pipeline.registry` constructs one configured `RankingPipeline` per `System`; existing component constructors load resources once; repeated-turn tests retain component identity | Met |
| Tracker input; top 100 for policy with occupation RIASEC | Each `System.step` reranks current tracker values/observed dimensions and all user utterances before calling policy; no confidence merging or score accumulation | Met with mock upstream; production acceptance pending |
| Top five, up to three task statements | Recommendation cards keep text plus task IDs, occupation codes, scores and status; M1-only legitimately has no task evidence | Met |
| Similarity alone must not imply supporting evidence | Existing M4 matches retain `provisional`; UI labels provisional matches, including raw dislike matches | Met |
| Multi-turn changes and conflict resolution | Script one: positive machines, negative machines, CHALLENGE answer, analytical interest; script two: unknown input, familiarity PROBE, negative clarification, analytical interest. Original texts are retained | Met with explicitly mocked tracker/policy |
| Different states change ranks | Signed tracker changes produce different top occupations; current vectors are reranked independently | Met |
| Valid task/occupation relationships | Real three-way cache test joins each task ID, occupation code and text back to `data/tasks.csv` | Met |
| No duplicate/ineligible occupations | Checked across every scripted three-way result; existing component/fusion validation retained | Met |
| Policy handles fewer than 100 and no ranking | Integration tests supply two results and `no_signal`; recommendation safely ends with two cards or abstention | Met with fixture policy |
| Median/p95 preloaded latency | 100 top-100 calls after warmup per configuration; results below | Met locally |
| Agree explicit budget with teammates | No team agreement exists in this checkout; freeze records this as pending | Pending external agreement |
| Freeze configuration, raw query policy and integration trace | `frozen/baseline_contract.json`, separate resolved configs and manifests, trace, model/cache/data/input hashes and source archive | Met for ranking integration control |

## Reproducibility and input conditions

The evaluator still loads `configs/m2.json` and `configs/ranking.yaml` by default and therefore remains oracle/M1-only unless explicitly changed. New `actual_tracker` and `mock_tracker` modes consume chronological per-turn snapshots, validate runtime values in `[-1,1]`, and require upstream extractor/tracker/revision provenance. Known test components cannot claim actual-tracker status. Provenance is a caller declaration, not cryptographic proof that a tracker produced the data.

Every turn is reranked and recorded. Metrics use only the final turn of each session, with the unchanged full-pool M2 definitions; a session contributes once to aggregation and paired bootstrap. Each report records input mode, input/provenance hashes, evaluation unit, raw query policy, exact configs, component variants and runtime dependencies. Comparison already rejects mismatching input modes or input fingerprints. Successful exports still append to the centralized history; the two new rows are explicitly described as `mock_tracker` evaluations.

The two scripted sessions and their illustrative target labels are interface fixtures, not representative relevance judgements. Their metrics must not be compared with the 60-profile oracle results. No actual-tracker evaluation was run. Frozen snapshots cover the production ranking implementation and mocked conversation integration, not missing teammate implementations.

Reproduce into a new directory:

```powershell
python scripts/freeze_ranking_baseline.py --output reports/ranking/m5/reproduction
```

Use the saved source archive or matching source hashes and the data/model/cache fingerprints in each run manifest. `source_snapshot.zip` preserves the dirty implementation used for this freeze, independently of its parent Git commit. Generated run IDs and measured latencies will differ; ranking predictions should match under matching resources. The freeze refuses an existing output directory.

## Verification and latency

Full discovery: **155 tests, 154 pass, one pre-existing error**. The error is `test_unknown_component_lists_valid_names` attempting to load the missing production extractor checkpoint; it is not a ranking failure. All six new M5 tests pass, including actual-tracker adapter validation and cached three-way integration. Existing exact M1 oracle regression coverage and metric tests pass. Scoring/fusion/metric source files have no changes.

| Preloaded top-100 ranking | Median | p95 | Samples |
|---|---:|---:|---:|
| Frozen M1-only | 0.73 ms | 0.90 ms | 100 |
| Optional three-way RRF | 24.69 ms | 28.87 ms | 100 |

These are local repeated calls on one fixed four-turn history, after warmup. They exclude extraction, tracker updates, policy, serialization and network/UI time; they are not end-to-end latency claims. Recorded startup measurements run in an already warm Python process and do not represent cold launch. Teammates must agree both the ranking-call and end-to-end budgets before full M5 acceptance.
