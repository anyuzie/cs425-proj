# Known issues and pending improvements

Snapshot: 9 November 2026, branch `feat/ranking-scoring`, commit `1948162`.

Live default (`configs/ranking.yaml`, since 9 November 2026): RIASEC cosine, signed lexical (`bm25_signed@ignore_negative_v3`) and signed semantic retrieval (`occupation_dual_encoder_signed@finetuned_v2`), fused with `zscore_sum@equal_v1`, then the post-fusion dislike stage `dislike_demotion@title_v3`. See the [default decision](../reports/ranking/d2_title_v3_default/DECISION.md). Earlier entries below describe the state when they were written.

## 1. Current errors

### 1.1 Test suite

`python -m unittest discover -s tests -p "test_*.py"` gives 443 tests and **5 errors**. All 5 existed before the 9 November changes; none come from the semantic pipeline.

| Test | Error | Likely cause | Suggested fix |
|---|---|---|---|
| `test_dense.RealMiniLMTests.test_real_cached_pipeline_provisional_evidence_and_manifest` | `ValueError: stale M4 cache: configuration, corpus, model, pool or dependency version changed` | The legacy MiniLM/FAISS M4 cache was built under older dependency versions. Reproduces with the original registry (`reports/ranking/occupation_semantic_v1/preexisting_m4_cache_check.json`). | Rebuild the legacy M4 cache with the current environment, or pin/record the dependency versions it expects. |
| `test_integration.IntegrationTests.test_three_way_cached_pipeline_tasks_and_repeated_turns` | Same stale M4 cache error | Same as above (uses `configs/ranking/m4_three_way.json`). | Same as above. |
| `test_history.HistoryTests.test_failed_and_incomplete_runs_leave_history_unchanged` | `ValueError: dataset fingerprint mismatch: data/dev_profiles.jsonl` | A history fixture records an older hash of `data/dev_profiles.jsonl`. | Regenerate the fixture's recorded fingerprint, or confirm whether `dev_profiles.jsonl` was changed intentionally. |
| `test_history.HistoryTests.test_legacy_import_preserves_artifacts_and_skips_duplicates` | Same fingerprint mismatch | Same as above. | Same as above. |
| `test_occupation_classifier.OccupationClassificationTests.test_terminal_only_cuts_display` | `TypeError: show_result() got an unexpected keyword argument 'occupation_annotations'` | The test calls `app.ranking_terminal.show_result` with a keyword the function no longer accepts. | Align the test with the current `show_result` signature (or restore the parameter if that display mode is still wanted). |

Run the suite with `discover -s tests` from the repository root. Some tests import `tests/fakes.py` by bare module name and fail with `ModuleNotFoundError: fakes` if run as `python -m unittest tests.<module>`.

### 1.2 Runtime and operational issues

- **Model files are local only.** The checkpoint and index in `artifacts/ranking/occupation_semantic_v1/` are gitignored. Without them M4 is `unavailable`. Fusion then falls back to cosine only, and nothing tells the user; only the researcher diagnostics show the M4 status and reason. Groupmates need the folder copied (e.g. via `/share/`) or rebuilt with the commands in the experiment report.
- **The index validates exact file bytes.** It checks SHA-256 hashes of `recommender/occupation_semantic.py`, `training/occupation_semantic.py`, `data/occupations.csv`, `data/tasks.csv`, `data/work_context/onet_31_0/source_manifest.json` and the local `occupation_work_context.jsonl`. `.gitattributes` now pins line endings for the tracked ones. **Any edit to the two `occupation_semantic.py` files requires a new versioned index**, otherwise loading fails with `Semantic implementation changed`.
- **Preparation and indexing refuse to overwrite.** Each stage of `training.occupation_semantic` errors if its output exists. Rerunning needs a new config/version.

## 2. Known behavioural limitations

These are not crashes, but they affect recommendation quality.

- **Negation is not handled by semantic matching.** "I do not want to teach mathematics" ranks mathematics teachers first in the semantic half. The preference extractor already records it as an avoid task, but task records are not scored.
- **Contradictions** ("I enjoy laboratory work but I do not want laboratory work") return laboratory occupations.
- **Vague or nonsense text** gets a full, confident-looking semantic ranking when RIASEC has no signal.
- **One-dimensional RIASEC ties.** With one observed type, cosine gives hundreds of occupations the same score (e.g. all 1.0), ordered by code. Through RRF these ties can pull in weakly related occupations (e.g. Education Administrators at #4 for an information systems student).
- **Equal RRF weighting is untuned.** On single task sentences, fusion (.201 MRR) is far below semantic-only (.518); on development profiles it is tied with cosine (.096 vs .117). On conflict personas it overrules a confident semantic match (MRR .114). Since 9 November 2026 the default is `zscore_sum@equal_v1` (conflict .639; opening-turn development profiles −.032 vs RRF); `rrf_tie_aware@k60_v1` exists but is not the default; see [SEMANTIC_LEDGER_PLAN.md §12.6](SEMANTIC_LEDGER_PLAN.md#126-results-of-1-fusion-variants-9-november-2026).
- **Reply wording mismatch.** The final chatbot message says "based on the preferences you described" even when the ranking is driven by semantic association. The recommendation cards use the correct explanation.
- **Context retrieval is effectively unevaluated.** The test split contains 3 context groups out of 1,133; context statements collapse into large shared groups.
- **Fine-tuning was bounded.** Only 768 of the 3,470 training groups were used (`max_train_groups`), for 3 epochs on CPU.
- **Preference extractor recall gaps** (rules `explicit_context57_rules_v1`, currently stashed):

  | Input | Result |
  |---|---|
  | "I don't want to teach maths" / "I hate teaching" / "I avoid public speaking" | Caught |
  | "Teaching is not for me" / "Anything but sales" | Missed |
  | "I'd rather not work with patients" / "I would never want to be a nurse" | Missed (`no_explicit_preference`) |
  | "I like science, just not lab work" | Abstains (`complex_scope`) |
  | "I enjoy coding but I don't want to manage people" | Only the positive half |

## 3. Evaluation gaps

- **No conversational relevance labels.** Task-retrieval labels are O*NET task ownership; development-profile targets come from hidden RIASEC profiles. Each benchmark favours its own approach.
- **Single-turn only.** The fusion default was not evaluated with the multi-turn simulated dialogues (`eval/dialogue*.py`) that justified the cosine default.
- **Browser and user review pending.** HTTP, terminal and restart checks pass; no observed browser session or user qualitative review has been recorded.
- **Held-out profile sets unused.** `profiles.jsonl` / `profiles_v4.jsonl` have not been used for any final claim.

## 4. Pending improvements (suggested order)

Items 1 and 2 are expanded into a single design in [SEMANTIC_LEDGER_PLAN.md](SEMANTIC_LEDGER_PLAN.md) (semantic evidence ledger).

1. **Negation-aware semantic matching** (new variant, e.g. `occupation_dual_encoder@finetuned_signed_v1`; keep current variants):
   1. Remove the evidence spans of active "avoid" records from the semantic query.
   2. Score each active task record's target against occupations with the same encoder: bounded bonus for "prefer", bounded penalty for "avoid". Soft demotion only, with diagnostics.
   3. Match against individual task statements (task-level index) rather than the whole-document vector, so "avoid teaching maths" demotes maths teachers more than mathematicians.
   4. Evaluate on a labelled synthetic negation set built from held-out task statements: avoid-query owners should drop out of the top 10; positive queries must not regress.
2. **Multi-turn accumulation in semantic matching** (new variant, e.g. `occupation_dual_encoder@finetuned_turns_v1`; keep `finetuned_v1`). Current behaviour: every turn, `OccupationSemantic.score` (`recommender/occupation_semantic.py:98`) joins all user utterances in chronological order into one string, encodes it once and re-ranks from scratch. Earlier turns are not thrown away, but nothing carries over from the previous turn's scores either. Issues:
   - **Truncation drops new turns.** `max_seq_length = 256` keeps the *first* 256 wordpieces (about 180–200 words), so once a conversation is long enough, later turns no longer reach the semantic query. `query_diagnostics.query.truncated` flags this in `ranking_trace`.
   - **Corrections are averaged in, not applied.** All the text goes into one embedding, so "actually I don't want outdoor work" is mixed with an earlier "I like being outdoors" rather than cancelling it. Earlier topics never fade. Only the RIASEC tracker (M1) can represent a change of mind.
   - **Short answers lose their meaning.** Assistant text is excluded, so answers like "yes, a lot" or "not really" carry almost no semantic signal without the question they answer.
   - **No smoothing between turns.** Neither M4 nor the RRF fusion uses the previous turn's ranking, so one new sentence can reshuffle the top 5.

   Options:
   1. Encode each turn separately and combine the vectors with a running or recency-weighted mean. This removes the 256-token cap and makes each turn add to the score instead of competing for space. The legacy dense M4 already averages per-sentence scores (`recommender/models/dense_ranker.py:160-171`).
   2. Smooth scores across turns: `score_t = α·new_t + (1-α)·score_{t-1}`, with α recorded in the variant config and diagnostics.
   3. Feed only active, stance-filtered evidence (`rank_with_evidence`, `evidence_filter: active_only`) so retracted statements leave the query (ties in with item 1).
   4. Pair short answers with the question they answer when building the query.

   Evaluate with the multi-turn dialogue evaluation (item 4), checking rank stability across turns and target rank at the final turn. Note: editing `recommender/occupation_semantic.py` in place invalidates the index (section 1.2), so add this as a separate versioned implementation/index.
3. **Extractor recall:** add rule patterns for the misses above, then compare the planned Qwen extractor (checkpoint 1b) against the rules on the same cases.
4. **Multi-turn evaluation:** rerun the dialogue evaluation for fusion vs cosine vs semantic-only with probing on and off. **Done 9 November 2026**, results in [SEMANTIC_LEDGER_PLAN.md §12.5](SEMANTIC_LEDGER_PLAN.md#125-results-of-0-fusion-dialogue-evaluation-9-november-2026): `python -m eval.dialogue_fusion` on `data/eval/fusion_personas_v1.jsonl` (25 personas, 14 conflict; team-reviewed). Primary metric MRR of the first acceptable occupation, see [SEMANTIC_LEDGER_PLAN.md §12.4](SEMANTIC_LEDGER_PLAN.md#124-decision-primary-metric-for-the-0-fusion-evaluation-9-november-2026). A full run needs `models/nli-probing-v2` and an extractor checkpoint matching `pipeline/extractor.py` (the local `models/extractor/config.json` lacks `custom_name`).
5. **Fusion weighting:** **Done 9 November 2026** as untuned variants `rrf_tie_aware@k60_v1` and `zscore_sum@equal_v1` ([decision report](../reports/ranking/fusion_variants_v1/DECISION.md)). z-score promoted to the default the same day ([decision](../reports/ranking/fusion_zscore_default/DECISION.md)); RRF k=60 via `configs/system.rrf.json`. Any tuned weighting must use data other than the 25 personas.
6. **Retraining:** use all training groups and design a split that makes context retrieval measurable; version as `occupation_semantic_v2`.
7. **Fix the 5 pre-existing test errors** (section 1.1).
8. **Reply wording:** make the final message reflect the ranking basis (semantic association vs stated interests).
9. **Robustness:** surface a visible startup warning when the semantic model files are missing instead of silently degrading to cosine.
10. **Qualitative review:** browser session plus user feedback on recommendation quality, recorded per turn.

## 5. Repository state notes

- `git stash list`:
  - `stash@{0}`: preference experiments (checkpoint 1, context57, preference fusion/XGBoost, feedback API/UI) and uncommitted reports from the cosine, clause-task, weighted-distance and dialogue experiments. Restore with `git stash pop`; expect conflicts in `recommender/models/ranker.py`, `pipeline/system.py`, `app/ranking_terminal.py` and `static/app.js` if those change first.
  - `stash@{1}`: full backup taken before the commit split; drop once the commit is confirmed.
- `artifacts/preferences/` and `reports/preferences/` (about 3.4 GB of ignored pilot files) remain on disk and are hidden locally via `.git/info/exclude`.
- Mock mode (`configs/system.mock.json`), the M5 test fixture and the experiment workflow (`configs/ranking/active.json`) intentionally stay on cosine.
