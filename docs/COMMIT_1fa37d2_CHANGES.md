# Changes in commit `1fa37d2`: new live ranking default

Commit `1fa37d2` on `feat/ranker-v2` (9 November 2026), "Promote signed lexical + signed semantic + dislike stage as the live default". 37 files, +2,865 / −28 lines.

**In one sentence:** the ranker now ignores what the user says they *don't* want when it searches for matching jobs, adds keyword matching, and then pushes jobs the user disliked down the final list.

Design history and all experiments: [LEXICAL_MATCHING_PLAN.md](LEXICAL_MATCHING_PLAN.md). Decision and accepted costs: [DECISION.md](../reports/ranking/d2_title_v3_default/DECISION.md).

## 1. Why

In the previous default, a dislike was read as interest. For "I study psychology … I like helping others, but teaching, not as much", the semantic model encoded "teaching" like any other topic, so the top 10 filled with teaching jobs (10 of 10 after six turns in the owner's session `anon-ebffdc4a`). Nothing in the pipeline could act on a dislike after fusion either: lowering a job in one of three components could not outweigh the other two.

## 2. Pipeline: before and after

| Stage | Before | After |
|---|---|---|
| M1 interest profile | `cosine_similarity@baseline_v1` | unchanged |
| M3 lexical | off | **`bm25_signed@ignore_negative_v3`** |
| M4 semantic | `occupation_dual_encoder@finetuned_v1` | **`occupation_dual_encoder_signed@finetuned_v2`** |
| Fusion | `zscore_sum@equal_v1` | unchanged |
| After fusion | nothing | **`dislike_demotion@title_v3`** (new `post_fusion` stage) |

```
user turns ─► clause routing v3 ─┬─► M3 signed BM25 (positive + neutral clauses)   ─┐
                                 ├─► M4 signed MiniLM (dislike clauses deleted)    ─┼─► z-score fusion ─► dislike stage ─► final ranking
RIASEC tracker ──────────────────┴─► M1 cosine                                     ─┘                     (negative clauses)
```

## 3. New components

### Clause routing v3 (`recommender/models/clause_routing.py`)
Shared by M3, M4 and the dislike stage, so all three treat a sentence the same way. Each user clause (split by the existing rule extractor `explicit_scoped_v1`) gets a route:

| Route | Rule | Used for |
|---|---|---|
| positive | explicit like ("I love …") | query text |
| neutral | no stance and no negation word ("I study psychology") | query text |
| negative | explicit dislike, or a negation word: `not`, `n't`, apostrophe-less `dont`/`cant`/`isnt`…, `never`, `hate`, first-person `avoid`, `except`… | never query text; feeds the dislike stage |
| skipped | non-preference ("I don't mind", "not sure", "doesn't matter") | ignored |

Routing-v3 rules: "I love biology, just not lab work" is split into a liked part and a disliked part (no split when the tail is only the cue, e.g. "teaching, not as much"); "anything but X" marks X as the dislike; "avoid" counts only in first person, so "help people avoid debt" stays neutral.

### Signed lexical matching (`recommender/models/bm25_signed_ranker.py`)
Reuses the frozen `bm25@baseline_v1` index and scoring unchanged; only the query differs: positive and neutral clauses only. If no such clause exists, lexical matching does not run (`unusable`) and fusion uses the other components. The job documents include titles, alternate titles and task text, so lexical matching rewards exact job and task words.

### Signed semantic retrieval (`recommender/models/occupation_semantic_signed.py`)
Same fine-tuned MiniLM checkpoint and occupation index as before; it wraps `OccupationSemantic` without editing it (that file's bytes are part of the index identity). Each turn has its negative and skipped clauses **deleted in place**; all other text stays as typed, so a conversation with no dislikes gives a byte-identical query and ranking. "i like helping others. but teaching, not as much" is encoded as "i like helping others. but".

### Dislike stage (`recommender/models/dislike_demotion.py`)
Runs on the final ranking after fusion, whatever the components scored.
1. **Target:** each negative clause minus cue and filler words ("teaching, not as much" → "teaching"; "not a people person" → no target, no effect).
2. **Match (`title`):** a target word and a job-title word share a stem (teaching ↔ Teachers, nursing ↔ Nurses, judge ↔ Judges). Suffixes are stripped leaving at least 4 letters, so "lab" does not match "labor" and "cars" does not match "care".
3. **Strength from wording** (owner review):
   - *strong* ("hate", "never", "can't stand", "not for me", "don't want", "anything but", "just not", "except", …): matched jobs move **below every other job**;
   - *mild* ("not as much", "not really", "rather not", "not a fan", "don't care much", …, and anything unclear): the job's fused score drops by **1.5 standard deviations** of that query's fused scores, then the list is re-sorted. A job with a very large lead can stay near the top.

   A job matched by both keeps the strong treatment. Nothing is ever removed from the list.

## 4. Pipeline and config changes

- `RankingPipeline` (`recommender/models/ranker.py`) accepts an optional `post_fusion: {algorithm: dislike_demotion, variant: …}` key, applied after fusion. It is rejected together with description `reranking`. The `rank_snapshot` used by the explanation map reflects the final order.
- New diagnostics key `dislike_demotion` in `query_diagnostics`: targets, their strength, the mild penalty and the demoted jobs. M4 query diagnostics now list the removed clauses.
- `recommender/models/registry.py` registers the new algorithms and variants. Earlier variants of each (signed BM25 ignore/filter v1–v2, signed semantic v1, dislike title/semantic v1–v3) are registered too, so their parameter files are included in the commit.
- `configs/ranking.yaml` selects the new default; identical to `configs/ranking/occupation_semantic_signed_v2_fusion_zscore_bm25_signed_ignore_v3_dislike_title_v3.json`.
- `configs/system.zscore.json` runs the **previous** default without editing anything; `configs/system.signed-v3-dislike-title-v3.json` runs the new one explicitly.

## 5. Evidence

| Benchmark | Metric | Before | After |
|---|---|---|---|
| Personas (25), probing on | MRR | .680 | **.770** (+.090 [−.037, +.226]) |
| | Hit@5 / NDCG@10 | .80 / .539 | **.96 / .620** |
| Personas, probing off | MRR | .653 | **.777** (+.123 [−.003, +.258]) |
| Negation probes (24, owner-reviewed) | dislike-matched jobs in top 10 | 2.79 | **0.08** |
| | acceptable-job MRR / Hit@5 | .317 / .50 | **.676 / .91** |
| Development profiles (120) | MRR | .064 | .057 (−.007, n.s.) |
| | Recall@50 | .391 | .355 (**−.036 [−.067, −.008]**) |
| Ranking latency | median per turn | 19.4 ms | 20.8 ms |

### Full comparison: what each part contributes

Four configurations, each changing one part relative to its neighbour. "Live default" is the previous default; "Signed v2 + D2, with lexical" is the new default. All columns use cosine (M1) and z-score fusion; D2 is `dislike_demotion@title_v3`. Source: [lexical_ablation_d2/README.md](../reports/ranking/lexical_ablation_d2/README.md) (not in this commit).

| Benchmark | Metric | Live default | Plain semantic + D2 | Signed v2 + D2, no lexical | Signed v2 + D2, with lexical |
|---|---|---|---|---|---|
| Personas, probing on | MRR | .680 | .680 | .728 | **.770** |
| | Hit@5 / Hit@10 | .80 / .92 | .80 / .92 | .80 / .96 | **.96 / .96** |
| | NDCG@10 | .539 | .551 | .591 | **.620** |
| | MRR conflict / agree | .639 / .733 | .639 / .733 | .613 / **.874** | **.780** / .758 |
| Personas, probing off | MRR / Hit@5 / NDCG@10 | .653 / .80 / .538 | .653 / .80 / .548 | .749 / .84 / .586 | **.777 / 1.00 / .643** |
| Probes | MRR / Hit@5 | .317 / .50 | .405 / .64 | .631 / .86 | **.676 / .91** |
| | avoided in top 10 | 2.79 | 1.04 | 0.17 | **0.08** |
| Development profiles | MRR / Hit@5 | .064 / .075 | .065 / .075 | **.073 / .092** | .057 / .042 |
| | NDCG@10 / Recall@50 | .094 / .391 | .094 / .393 | **.102 / .402** | .077 / .355 |

How to read the columns, moving left to right:
- **Live default → Plain semantic + D2** (adds only the dislike stage): personas and development profiles unchanged; probes improve (avoided 2.79 → 1.04). D2 acts only when a dislike names a job word, and few persona dislikes do.
- **→ Signed v2 + D2, no lexical** (dislike clauses removed from the semantic query): the largest single gain. Agree personas .733 → .874; probes .405 → .631, avoided 1.04 → 0.17; development profiles slightly up. Conflict personas dip slightly (.639 → .613).
- **→ Signed v2 + D2, with lexical** (adds signed BM25): conflict personas .613 → .780 and Hit@5 .80 → .96, probes up again, but agree personas down (.874 → .758) and development profiles down (Recall@50 .402 → .355, significant). This is the trade accepted in the promotion.

Owner's session `anon-ebffdc4a`: teaching jobs in the top 10 went from 10 to 1; Psychology Teachers moved from #1 to #2 (mild wording). With strong wording ("teaching is not for me") it drops to the bottom.

## 6. Accepted costs and known limits

- **Development-profile regression** (Recall@50 −.036) comes from the lexical component; those labels follow the RIASEC profile and favour cosine.
- **Lexical's persona gain is probably an upper bound:** conflict personas name their field outright, which exact title matching rewards.
- **Mild dislikes can keep a large lead**, especially early in a conversation.
- **Title matching only acts on dislikes that name a job word.** "I'd rather not work with patients" is handled by the signed queries, not by the dislike stage.
- "No" answers to the system's own questions carry no topic words, so they cannot be acted on yet (needs question pairing, ledger plan §6).
- Without the local MiniLM files, ranking falls back to cosine + signed lexical + the dislike stage.

## 7. Tests

- New: `tests/test_bm25_signed.py` (routing, early stop, ignore vs filter, apostrophe-less cues), `tests/test_occupation_semantic_signed.py` (in-place deletion, byte-identical query without dislikes, early stop), `tests/test_dislike_demotion.py` (routing v3, stemming, strong vs mild, pipeline integration).
- Updated to pin the new default: `tests/test_configuration.py` (also keeps the previous default, RRF and cosine selectable), `tests/test_ranking_app_integration.py` (web app ranking equals the preset), `tests/test_history.py`.
- Full suite: 507 tests; the 23 failures are all pre-existing (extractor checkpoint, history fingerprints, stale legacy M4 cache, NLI preset, terminal signature). On a checkout without the gitignored model files, two `test_configuration` tests also fail; they fail at the previous commit too.

## 8. How to use

```powershell
python -m uvicorn app.server:app --host 127.0.0.1 --port 8000      # new default (configs/ranking.yaml)
$env:CM_CONFIG = 'configs/system.zscore.json'                        # previous default, for comparison
```
In a debug session, `query_diagnostics.dislike_demotion` shows each dislike's target, strength and demoted jobs; `occupation_semantic.query.removed_clauses` shows what the semantic query ignored.

**Roll back:** replace `configs/ranking.yaml` with the content of `configs/ranking/occupation_semantic_fusion_zscore_v1.json`, or delete the `post_fusion` key to keep everything except the dislike stage.

## 9. Not in this commit

The other experiment presets and system configs, the evaluation harness changes (`eval/dialogue_fusion.py` arms, `eval/negation_probes.py`, `eval/calibrate_dislike.py`), the negation probe set with its builder, review file and reviewer tool (`devtools/review_probes.py`), and the experiment reports. The decision record and the plan link to some of those reports (`lexical_ablation_d2/`, `dislike_d2/calibration/`, `d1_baseline_check/`), so the numbers above cannot yet be rerun from git alone.
