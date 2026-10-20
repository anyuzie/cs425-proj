# M5 validation with installed extractor weights

The checkpoint is now installed locally in `models/extractor/`. The current adapter loads its Hugging Face `model.safetensors`, tokenizer and calibrated 45-logit configuration successfully. This supersedes the missing-checkpoint limitation in the earlier extractor integration report; the original reports and frozen baseline are preserved.

## Results

| Check | Result |
|---|---|
| Full test discovery | 159 tests pass; no failures or errors |
| Real extractor + M1-only M5 ranking | Passed across three chronological user turns |
| Real extractor + M1/BM25/MiniLM RRF | Passed with the prepared M4 cache |
| Ranking eligibility and uniqueness | Checked by both smoke runs |
| Additional implementation changes during validation | None |

The real checkpoint uses negative/neutral/positive direction logit order. The current adapter maps these labels into the neutral/positive/negative shared contract. Its existing label-mapping and strength regression tests pass with the rest of the suite.

Example from the M1-only smoke trace: “I enjoy building and repairing machines.” produces R=0.55; the subsequent dislike statement updates R=-0.55. Both states produce eligible rankings. These examples establish data flow, not general model accuracy. Other predicted types in the traces still require semantic quality review.

## Evidence and limits

- [Full test output](with_weights_test_results.txt)
- [Real extractor / M1-only trace and checkpoint hashes](extractor_weights_smoke.json)
- [Real extractor / three-way trace and checkpoint hashes](extractor_weights_three_way_smoke.json)

Both smoke conditions are explicitly `mock_tracker`: the extractor and ranker are real, while the tracker, NLI confirmer and policy are test fixtures. The production tracker/NLI/policy remain unfinished, so this does not establish full production application readiness or actual-tracker ranking quality. No relevance metrics, evaluation-history rows or replacement baseline freeze were generated.

The weights load with the current updates; no integration break was observed in these checks. The only pre-existing working-tree code change is the restoration of the extractor's introductory instructions from the earlier request. Validation adds these reports without modifying upstream implementations or scoring algorithms.
