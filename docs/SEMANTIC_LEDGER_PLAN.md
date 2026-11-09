# Plan: negation-aware, multi-turn semantic matching (semantic evidence ledger)

Status: **ledger is a proposal, not implemented.** Explanations Phase A (§10) is partly implemented (9 November 2026): `rank_snapshot`, component-rank hover, the RIASEC hexagon and the t-SNE map with the turn slider; the evidence card and leave-one-turn-out are still pending, see §10.8. #0 (fusion evaluation) is done, see §12.5. #1 (fusion variants) is implemented and evaluated, and `zscore_sum@equal_v1` is the live default since 9 November 2026, see §12.6. Written 9 November 2026 against branch `feat/ranking-scoring`, commit `1948162`; §12.6 and the fusion notes in §3, §5.3, §8.2 and §10 updated 9 November 2026 on `feat/ranker-v2` (base `9fd317b`).
This expands items 1 and 2 of [section 4 of the known-issues list](KNOWN_ISSUES_AND_PENDING.md#4-pending-improvements-suggested-order).
Read [RANKING_EXPERIMENT_ARCHITECTURE.md](RANKING_EXPERIMENT_ARCHITECTURE.md) before implementing. The ledger work does not change `occupation_dual_encoder@finetuned_v1`, the fusion default or any accepted variant. (The fusion default itself changed under #1, §12.6.)

## 1. Goals

1. **Negation:** "I don't want to teach maths" should push maths teachers down, not up.
2. **Accumulation:** every turn adds clauses to a lasting semantic profile. Nothing is re-ranked from a single joined string that gets truncated at 256 wordpieces.
3. **Stable weights:** a clause keeps its weight in later turns. A weight changes only when a later statement contradicts the clause (or, within a bounded limit, repeats it). There is no silent decay and no overwriting.
4. **Compact storage:** the profile stores encodings and references to the source text, not copies of it. Each turn encodes only its new clauses.
5. **Short answers:** "yes, a lot" or "not really" gets its meaning from the question it answers.
6. **Credible explanations:** each turn shows which jobs became more or less relevant, on a fixed map of occupations. Each response shows what the system captured from it and which matches it caused. Every explanation is computed from the actual scores, not written afterwards (§10).

## 2. Current behaviour (what this replaces)

`OccupationSemantic.score` (`recommender/occupation_semantic.py:96-107`) joins all user utterances with newlines, encodes the result once and takes its dot product with one vector per occupation. As a result:

- the first 256 wordpieces win, so later turns can drop out of the query entirely;
- negated statements are encoded like positive ones, because bi-encoders mostly ignore "not";
- a correction gets averaged together with what it corrects;
- assistant questions are excluded, so short answers carry almost no signal;
- the whole history is re-encoded on every turn (cost grows with the history).

**Prior art to reuse, and a warning.** `recommender/models/stance_ranker.py` (`minilm_stance@stance_internal_v1`) already scores signed task preferences: rule-extracted activity text, task-level maximum similarity, and `pos - negative_weight * neg`. On the 120 development profiles it scored Hit@5 .033, against .042 for frozen M4 ([report](../reports/ranking/stance_dev120_v3/README.md)). That set contains little explicit negation, so the result says little about negation itself. It does show that signed scoring alone does not improve general ranking. This plan differs in four ways: it uses the fine-tuned encoder, it accumulates across turns, it applies a bounded penalty that only kicks in above a threshold (not a mean subtraction), and it is evaluated on a set built specifically for negation. Its acceptance gate is "no regression on positive inputs", not "beats M4 on development profiles".

## 3. Where accumulation lives (architecture fit)

The architecture rules say: *"Upstream owns extraction, confidence merging, contradiction resolution, the cumulative RIASEC profile ... Recompute rankings from each snapshot; do not accumulate occupation scores across turns."*

This plan follows that rule. The ranking component accumulates nothing:

```
user turn ──► clause split (existing UtteranceProcessingPipeline)
          ──► polarity / target / weight (existing rule extractors)
          ──► question pairing for short answers
          ──► SemanticLedger.update(state, records)      ◄── new, upstream, owned by System (like DecayTracker)
                 • encode only the new targets
                 • detect/confirm contradictions (reuse NLI confirmer)
                 • revise statuses/weights; never delete
          ──► ComponentInput(semantic_evidence=ledger.snapshot())
          ──► occupation_semantic_ledger component       ◄── new, stateless; recomputes from the snapshot
          ──► fusion with cosine (unchanged by the ledger; the selected fusion variant, `zscore_sum@equal_v1` by default, §12.6)
```

- **What accumulates is evidence, never occupation scores.** The ranking is recomputed from the ledger snapshot each turn, the same way cosine is recomputed from the tracker vector.
- **Score smoothing between turns (`α·new + (1-α)·old`) is rejected.** It would keep the influence of superseded statements after a contradiction, which is exactly what goal 3 forbids, and it would break the "recompute from snapshot" rule. The ledger already provides stability: one new clause is one more weighted term, not a new query.
- **Contract change:** add an optional `semantic_evidence: tuple[SemanticEvidence, ...] | None = None` to `ComponentInput` (`recommender/components.py`), passed in through `RankingPipeline.rank`. This works the same way as `weights` (added for `weighted_distance`). Every existing component ignores the new field, so their outputs stay identical.
- `System.__init__` currently rejects any ranker whose `evidence_filter` is not raw. The ledger path does not use `evidence_filter`; it is a separate input, so that guard stays as it is.

## 4. The ledger

### 4.1 Record schema (one per clause or answered question)

| Field | Type | Notes |
|---|---|---|
| `id` | str | `t{turn}c{clause}`; stable, unique, never reused |
| `turn`, `clause_index` | int | chronological position |
| `source` | `(start, end)` | character offsets into `state.user_texts[turn-1]`; **no copy of the text** |
| `source_sha1` | str | hash of the source slice, to detect accidental drift |
| `kind` | `statement` / `answer` | `answer` = short reply paired with a question (§6) |
| `question_ref` | str or None | question-bank ID or `task:<task_id>`; never the question text |
| `polarity` | `+1` / `-1` / `0` (unknown) | from the extractors (§5.1) |
| `w0` | float in (0, 1] | initial weight = extractor confidence × strength (fixed at creation) |
| `w` | float in [0, 1] | current weight; changed only by §4.3 |
| `status` | `active` / `disputed` / `superseded` | the M6 lifecycle vocabulary (`recommender/data/evidence.py`); superseded records cannot be reactivated |
| `vector` | float16[384] | L2-normalised encoding of the **target** (not the whole clause) |
| `history` | list | `(turn, event, w_before, w_after, by_id)` for every revision, kept for audit |

The ledger header records `encoder_checkpoint_sha256` (the same fingerprint as `occupation_semantic.fingerprint`), `ledger_rules_version`, the extractor revisions, and the dimension.

### 4.2 Storage and compute

- **Size:** 384 dimensions × 2 bytes (float16) is about 0.8 KB per record. A 20-turn conversation with about 3 clauses per turn is roughly 50 KB. Int8 quantisation is not needed.
- **Compute:** each turn encodes only its new targets (usually 1–4 short strings), so cost no longer grows with history length. Scoring is one `(n_records × 384) @ (384 × n_occupations)` product, under 1 ms at this scale.
- **Why per-record vectors and not one running vector:** the positive part of the score (§5.2) is linear. In principle, keeping `Σ w·v` and `Σ w` would give the same ranking in O(1) space, and the ledger does keep that sum as a cache. But per-record vectors are still needed for three things: (a) undoing a superseded record's contribution, (b) contradiction detection, which compares new targets with each earlier one, and (c) the nonlinear negative penalty.
- **Why not drop the text completely:** a vector is only valid for the checkpoint that produced it. If the fingerprint changes, the ledger has to re-encode from source text. The text already lives in `state.user_texts` and the turn logs, so the ledger keeps offsets into it rather than its own copy. If the fingerprint differs and the source is unavailable (e.g. a ledger replayed without its logs), the component returns `unavailable` with a reason. It never uses vectors from a different model.
- **Logs:** turn logs record record metadata plus a vector hash. Vectors can be regenerated from text and checkpoint, which keeps logs small.

### 4.3 Weight rules ("keep significance, revise only on contradiction")

| Event | Detection | Effect |
|---|---|---|
| New clause, no related record | — | append with `w = w0` |
| **Reinforcement**: same polarity, target cosine ≥ `τ_same` | vector comparison against active records | no duplicate weight; existing `w ← min(1, w + w_new·(1 − w))` (noisy-OR, bounded), and the new record is stored with `w = 0`, `status=active`, linked as `reinforces` |
| **Contradiction candidate**: opposite polarity, target cosine ≥ `τ_conflict` | vector comparison | confirm with the existing NLI confirmer (`System.confirmer`, `confirm_threshold` 0.6), the same one used for RIASEC contradictions |
| Confirmed contradiction | NLI > threshold | older record → `superseded`, `w ← 0`; newer stays `active` (matches the tracker's "newer statement kept" outcome). Optional phase 2: mark both `disputed` and let the policy issue a CHALLENGE, as the RIASEC flags do |
| Unconfirmed opposite | NLI ≤ threshold | both stay active (scope differs, e.g. "I like science" / "not lab work"); mirrors `merge_unconfirmed` |
| Explicit retraction ("ignore what I said about X", existing `needs_repair` cues) | repair detector + target match | matched record → `superseded` |
| Neither | — | **no change.** No turn-based decay (`decay` is a parameter, default 1.0, as in `DecayTracker`) |

`τ_same` and `τ_conflict` start as provisional, uncalibrated defaults (e.g. 0.80 and 0.70). They are recorded as `calibrated=false` until they have been set on a labelled pair set (§8.3), in line with the `min_similarity` precedent in [CLAUSE_CONTEXT_PIPELINE.md](CLAUSE_CONTEXT_PIPELINE.md).

## 5. Negation-aware scoring

### 5.1 Polarity and target per clause

Use the parsers already in the repo. No new model in v1:

1. Split clauses with `UtteranceProcessingPipeline` (already enabled in `configs/system.json`; clauses are already in `turn_trace.utterance_processing`). If it is disabled, the whole utterance is one clause.
2. Get polarity and target from `TaskPreferenceExtractor` (`explicit_scoped_v1`) and the context preference rules (`recommender/utterance/context_matching.py`). Both already separate the target ("teach maths") from the cue ("don't want to"). If they disagree, the polarity is unknown.
3. **Safety rule:** a clause with unknown polarity that *contains a negation cue* (`not`, `n't`, `never`, `avoid`, ...) gets `w = 0` in the positive part. It is never treated as positive. Unknown polarity with no negation cue keeps today's behaviour (a positive topic association) at a reduced weight `β_unknown` (e.g. 0.5). Without this rule, the extractor's recall gaps ("Teaching is not for me", "I'd rather not work with patients") would still push teachers and nurses up.
4. Extractor recall (known-issues item 3) improves this layer without any change to the component. Add a Qwen-based polarity extractor later as a separate `ledger_rules_version`.

### 5.2 Score

For occupation *o*, with document vector `d_o` (the existing fine-tuned index) and task vectors `T_o` (new task-level index, §7):

```
pos(o)  = Σ_{i∈P} w_i · (v_i · d_o) / Σ_{i∈P} w_i                         # P = active, polarity +1 (and β-weighted unknowns)
neg(o)  = max_{j∈N} w_j · relu( max_{t∈T_o} (v_j · t) − τ_neg ) / (1 − τ_neg)   # N = active, polarity −1
score(o) = pos(o) − λ · neg(o)
```

- **Negatives don't enter `pos`**, so avoided activities no longer pull their occupations up. This alone fixes the ranking for "I don't want to teach maths".
- **Penalty at task level, with a threshold:** an occupation is demoted only if one of its *tasks* closely matches the avoided activity. So "avoid teaching maths" demotes maths teachers more than mathematicians. Below `τ_neg` there is no effect, which limits collateral damage. `λ` bounds the penalty, so this is a soft demotion, not a filter.
- **Only negatives:** `P` is empty, so the component returns `unusable` ("only avoid evidence"). Fusion then falls back to cosine. Ranking "least like teaching" would mean nothing.
- **Linearity check:** the `pos` part equals the dot product of `d_o` with the normalised weighted mean vector, so the ledger can cache that mean, as noted in §4.2.
- **Diagnostics** (in `query_diagnostics`): for the top occupations, `pos`, `neg`, the record ID behind the penalty and its matched task. Also record counts by status, the weight total, and how many records were dropped by the safety rule.
- **Task evidence:** matches with negative polarity are shown in traces as conflicts, never as provisional supporting evidence. The current M4 already warns that "a dislike can match strongly".

### 5.3 Known limitation that stays: fusion

M4 now demotes maths teachers, but cosine (M1) still ranks them high when the user's RIASEC profile is Social. Under the default `zscore_sum@equal_v1` (§12.6) a strongly demoted M4 score counts by its size rather than only its rank, which reduces this compared with equal-weight RRF, but fusion still cannot remove an occupation that cosine ranks highly. Fixing that needs a post-fusion stage (the `reranking` slot in the selection config) that applies confirmed avoid records to the fused list. That is a separate experiment, deliberately left out of scope here so that only one variable changes at a time. It is **not built** yet.

## 6. Short answers: keep the question

### 6.1 When to pair

Use the policy the extractor already uses (`pipeline/answers.py`), so the two layers agree:

- the previous action was a question (`state.last_action.kind in QUESTIONS`), **and**
- the reply has at most `PAIR_MAX_WORDS` (9) words and matches `_ANSWERISH`, or the reply has no extractable target of its own.

Longer replies speak for themselves and are not paired. This matches the extractor's rule that "longer statements about something else never borrow the question".

### 6.2 What gets encoded

Do **not** encode "question + answer" as one string. "Do you enjoy teaching? Not really" encodes close to "enjoy teaching", which brings back the negation problem. Instead, the record is split into parts:

- **target** = the topic of the question, not its wording:
  - task questions: `last.payload['task']` already identifies the O*NET task, so the target is its activity text (`task_history` records this);
  - RIASEC slot/probe questions: the question-bank item's topic phrase;
- **polarity and strength** = from the answer alone: `task_stance(text)` for task CLARIFY questions, or the sign and strength of the extractor reading for slot questions;
- `kind = answer`, `question_ref = task:<id>` or the bank item ID.

Question targets come from a finite bank and a fixed task list, so **their vectors can be precomputed offline** as part of the versioned index. At runtime, pairing needs no encoder call and no question text is stored. T5-reworded questions don't matter, because the target is taken from the task or bank item, not from the generated wording.

### 6.3 Should this be lexical? No, not in this plan.

- BM25 (M3) is off in the default (`m3: null`) and had a negative baseline (Hit@5 .017 against .167 for structured).
- BM25 can't represent polarity. Adding the question's words to its query would treat "Do you like teaching? No" as a positive mention of teaching, which is the same failure we are trying to fix.
- Pairing is an **evidence-building step**, not a matching method. It belongs in the ledger (upstream), so any consumer can use the record. The semantic component uses it through `target`, `polarity` and `w`.
- The only place exact matching helps is resolving which task a question was about. That is already an ID lookup (`payload['task']`) and needs no lexical scoring.
- If lexical matching is ever turned back on, a separate variant (e.g. `bm25@question_paired_v1`) could add only **positive** paired targets to its query. Treat that as its own experiment against the BM25 control.

## 7. Versioned artifacts and code layout

| Item | Location | Notes |
|---|---|---|
| Ledger (upstream state) | `pipeline/semantic_ledger.py` (new) | Owned by `System`, like the tracker; holds `State.semantic_ledger` |
| Polarity and pairing rules | `pipeline/semantic_ledger.py`, `ledger_rules_version = "ledger_rules_v1"` | Recorded in system config and turn logs |
| Encoder access | reuse `recommender.occupation_semantic.load_bundle` (lru-cached; one model in memory) | **Import only, never edit** `occupation_semantic.py`: its bytes are part of the index identity |
| Scoring component | `recommender/occupation_semantic_ledger.py` (new) | New algorithm `occupation_semantic_ledger`, since the formula changes |
| Task-level index | `artifacts/ranking/occupation_semantic_v1/task_index_finetuned_v1/` | Eligible tasks × 384 from the fine-tuned checkpoint, sorted `(code, task_id)`; same manifest/hash validation as the document index |
| Question-target vectors | same index folder, `question_targets.npy` + IDs | Precomputed from the task list and question bank |
| Contract field | `ComponentInput.semantic_evidence` | Optional; default `None` keeps all existing outputs identical |

Variants (changing one thing at a time; all registered in `default_registry()` and none replacing `finetuned_v1`):

| Variant | Accumulation | Polarity | Task-level penalty | Question pairing | Contradiction revision |
|---|---|---|---|---|---|
| `occupation_semantic_ledger@clauses_v1` | ✓ | all treated +1 (except the §5.1 negation safety rule) | – | – | – |
| `@signed_v1` | ✓ | ✓ | – (negatives only removed from `pos`) | – | – |
| `@signed_tasks_v1` | ✓ | ✓ | ✓ | – | – |
| `@signed_tasks_qa_v1` | ✓ | ✓ | ✓ | ✓ | – |
| `@signed_tasks_qa_rev_v1` | ✓ | ✓ | ✓ | ✓ | ✓ |

Question pairing and contradiction revision are ledger behaviour. The component variant only declares which record kinds and statuses it reads (e.g. `clauses_v1` reads every record as active). That lets one ledger serve every variant during evaluation. Record `λ`, `τ_neg`, `τ_same`, `τ_conflict`, `β_unknown` and `decay` in the resolved parameters.

## 8. Evaluation

Every comparison uses the frozen control `occupation_dual_encoder@finetuned_v1` (alone, and fused with cosine) under matching inputs and pool. Results and regressions are registered in history.

### 8.1 Negation set (synthetic, labelled)
Build it from **held-out** task statements: "I don't want to {task}", "{task} is not for me", "anything but {task}", and mixed statements ("I like {A} but not {B}").
Metrics: rank of the owning occupation for the avoided task (target: out of the top 10), rank of A's owner in mixed statements, and the rate of collateral demotion for unrelated owners. Run it with both the rule extractor and gold polarity, to separate extractor errors from scoring errors.

### 8.2 Positive regression gate
Single task sentences: MRR must not fall significantly below `finetuned_v1` semantic-only (.518). Development profiles: no significant regression against the current fusion. Report the fused result under the default `zscore_sum@equal_v1` and under the `rrf_k60` control (§12.6), because the two combine M4 differently: a ledger change that helps under one may not help under the other. If `clauses_v1` alone regresses, the accumulation formula (mean of per-clause similarities vs one joined encoding) is the cause, and that has to be solved before adding polarity.

### 8.3 Multi-turn (`eval/dialogue*.py`)
- Rank of the target at the final turn, with probing on and off.
- **Truncation case:** dialogues where the decisive topic appears after 256 wordpieces. `finetuned_v1` should miss it and the ledger should not.
- **Correction recovery:** "I like being outdoors" … "actually I don't want outdoor work". After the correction, outdoor occupations should drop and stay down.
- **Stability:** overlap of the top 5 between consecutive turns, and Kendall τ, to check that one new sentence no longer reshuffles the list.
- **Contradiction thresholds:** calibrate `τ_same` and `τ_conflict` on a small set of hand-labelled clause pairs (same / opposite / unrelated) drawn from dialogue data that was **not** used for the metrics above.

### 8.4 Short answers
Dialogues with yes/no task questions. Compare `signed_tasks_v1` with `signed_tasks_qa_v1` on the target rank and on how many task answers produce a usable record.

### 8.5 Correctness tests
Ledger: appends only, superseded records stay superseded, offsets and hashes are stable, `w0` never changes, the weight changes only through the events in §4.3, and a fingerprint mismatch gives `unavailable`.
Component: deterministic order and tie-breaks, only-negatives gives `unusable`, an empty ledger gives `unusable`, and existing components give identical outputs when `semantic_evidence` is set.

## 9. Implementation order

1. Contract field plus the ledger skeleton with `clauses_v1` (accumulation only). Evaluate on §8.2 and the truncation case. **Checkpoint:** keep going only if there is no positive regression.
2. Polarity plus the safety rule (`signed_v1`). Run the negation set.
3. Task-level index and threshold penalty (`signed_tasks_v1`).
4. Question pairing with precomputed question targets (`signed_tasks_qa_v1`).
5. Contradiction revision with the NLI confirmer (`signed_tasks_qa_rev_v1`), then calibrate thresholds.
6. Run the multi-turn evaluation for every variant. Write a decision report. Only then consider changing the default, and the post-fusion avoid stage (§5.3).

The explanation work (§10) runs alongside this: its Phase A needs no ledger and can start before step 1; Phase B follows step 2; Phase C follows step 4.

After each accepted step, update `RANKING_EXPERIMENT_ARCHITECTURE.md` (current status, and a note that semantic evidence is accumulated upstream as a ledger while scores are still recomputed per snapshot) together with `cs425-ranking-docs/README.md`, as `AGENTS.md` requires.

## 10. Explainability: per-turn job map and per-response evidence

Motivation: today the recommendations don't feel credible, because nothing shows *why* a job appears. Before building a "what triggered this" panel for users, first build a researcher-facing map that shows how each turn moves the rankings. The panel will later reuse the map's data.

**Rule for every view below:** an explanation must come from the score the ranker actually computed (a score decomposition or a recomputation). It is never generated text written after the fact. Explanations are read-only and never feed back into the ranking.

### 10.1 What already exists

| Data | Where | What it gives the plot or the evidence view |
|---|---|---|
| Fused top 100 per turn, with `signals` (component ranks and scores) and `rrf_score` (the fused score of whichever fusion variant is selected; the name is kept for schema compatibility) | `state.ranking_trace[*].occupations` | Rank changes between turns, and which component moved an occupation |
| Clauses with exact offsets | `turn_trace.utterance_processing` | The clause behind each piece of evidence |
| RIASEC readings (slot, sign, strength, confidence, source quote) | `turn_trace.readings`, slot `history` | What the cosine half captured from each response |
| Question, reply and task stance for task questions | `state.task_history` | Short answers paired with their questions |
| NLI contradiction checks | `turn_trace.contradiction_checks` | Contradiction evidence |
| Provisional task matches | `RankedOccupation.evidence` / `task_evidence` | Matched tasks (association only) |
| Debug panel | `static/app.js` `renderTrace` | Currently shows raw JSON; the plot replaces the "Occupation semantic experiment" block |

What's missing: (a) ranks beyond the top 100, which are needed to colour all 378 eligible occupations, (b) fixed 2D map coordinates, and (c) contributions per clause, which require the ledger.

### 10.2 Feasibility of the map: high

- **Size:** 378 eligible occupations, each with a 384-dimension fine-tuned document vector (`finetuned_index/vectors.npy`, shape `(378, 384)`). A vanilla-JS inline SVG with 378 circles in the existing `static/app.js` handles this easily, so no chart library is needed.
- **Projection, measured:** a 2D PCA keeps only **15.3%** of the variance (PC1 .085, PC2 .068), which is too little to read as a map. Use **t-SNE** with the cosine metric and a fixed seed. scikit-learn 1.9.1 is already in `.venv`. UMAP is not installed and would be a new dependency for little gain at n = 378.
- **Coordinates are computed once offline**, so the map never moves between turns or sessions. Store them as a versioned artifact (`artifacts/ranking/occupation_semantic_v1/occupation_map_v1/`: `coords.json` with both the t-SNE and the hexagon coordinates, plus a manifest with the checkpoint fingerprint, index hash, catalog hash, method, perplexity and seed). Serve them from a read-only endpoint (`/api/occupation_map`). If the t-SNE part is missing or stale, only the semantic tab is hidden. The hexagon tab depends only on the catalog, so it still works without the semantic model.
- **Honest caveat shown on the plot:** t-SNE keeps neighbourhoods, not global distances. Jobs in the same cluster are similar in text, but the distance between two clusters means nothing.
- **Two tabs, one panel.** The map panel has two views of the same 378 occupations, sharing the per-turn colours (§10.3), the turn slider and the hover details. Switching tabs keeps the selected turn and occupation.

| Tab | Positions from | Explains | User marker |
|---|---|---|---|
| **Semantic map** (t-SNE) | Fine-tuned occupation vectors | The semantic half (M4): which jobs are similar in text | Ledger records, from Phase B (§10.3) |
| **RIASEC hexagon** | Occupation RIASEC z-scores | The cosine half (M1): how jobs spread across interest types | The tracker's growing profile, every turn |

#### RIASEC hexagon

- **Layout:** Holland's hexagon, with the six types at the corners in the order R, I, A, S, E, C. Neighbouring corners are similar types and opposite corners are least alike. Each occupation is placed at `Σ_t max(0, z_t) · c_t / Σ_t max(0, z_t)`, where `c_t` is the position of corner *t*. Only positive z-scores pull, so a job sits towards the types it is high in. A job with no positive z-scores goes in the centre. The coordinates come straight from `data/occupations.csv`, so this tab needs no model artifact. It is still computed once and saved in `occupation_map_v1/` with the catalog hash, so both tabs use the same code order.
- **User profile marker:** each turn, place the tracker profile on the same hexagon with the same formula, using `state.slots` values over the **observed** types only. That matches what cosine scores. A faded trail joins the marker's positions from earlier turns, so you can see the profile build up over the conversation, and a confirmed contradiction shows as a visible jump back. The data already exists in `state.profile_history`; no new state is needed.
- **Side bar chart:** the six slot values, each with its confidence. Unobserved types are shown as "unknown", never as 0. The chart shows the exact profile that the 2D marker only approximates.
- **Caveats shown on the tab:**
  - The hexagon squashes six dimensions into two. Opposite types cancel out, so a strong R + S profile and a flat profile both land near the centre.
  - Two jobs at the same point can rank differently, because cosine uses all observed dimensions, not the 2D position.
  - Nearness to the marker is only a rough guide to the cosine rank; the colours show the actual rank changes.
  - The marker shows only the cosine half. With fusion on, a job far from the marker can still rank highly because of the semantic match; the hover details show which half moved it.

### 10.3 Per-turn colours

**Data:** each ranking-trace row gets a compact `rank_snapshot`. This is three integer arrays (fused, M1 and M4 rank) covering all eligible occupations, in the map artifact's code order. That is about 378 × 3 small integers, around 5 KB of JSON per turn. It is stored separately from the top-100 policy list so that the list keeps its meaning.

**Encoding** (computed in the browser from consecutive snapshots):

| Category | Rule (provisional thresholds) | Mark |
|---|---|---|
| Current top 10 | fused rank ≤ 10 | filled, dark outline, labelled |
| Rose | `log(prev_rank) − log(rank) ≥ 0.5` | blue, intensity ∝ change |
| Fell | `log(rank) − log(prev_rank) ≥ 0.5` | orange, intensity ∝ change |
| Entered / left top 10 | crossed rank 10 | ring / hollow ring |
| Little change | otherwise | grey, faded |

- A log scale is used because moving from 5th to 1st matters more than moving from 300th to 296th.
- The blue/orange palette is colour-blind safe, and ring shapes carry the top-10 information so the plot does not rely on colour alone.
- **Hover/tap** shows the title, the fused rank now and before, the M1 and M4 ranks, and which half moved it: "interest profile" if the M1 rank changed most, "semantic match" if M4 did, or "both".
- **A turn slider** replays the conversation. Turn 1 has no earlier turn to compare with, so it shows the top 10 only.
- **With the ledger (Phase B):** each record from the current turn appears as a marker (a star for positive, a cross for negative). Its position is the similarity-weighted average of its 5 nearest occupations' coordinates. t-SNE can't place new points directly, so the marker is labelled as approximate.

### 10.4 Per-response evidence: "what the model captured"

There is one card per user response. It is stored in the trace and turn log, and the UI shows it next to that message:

1. **Clauses:** each clause highlighted in the original reply (using offsets), with its route (task / context / skipped and why).
2. **What was captured for each clause:**
   - polarity and the rule that fired;
   - the target phrase, its weight and its status;
   - the paired question, if any (its `question_ref`, shown with the bank wording);
   - or "not interpreted" when the polarity is unknown.
3. **RIASEC readings:** slot, direction, strength, confidence and source quote, shown as the change to the profile this turn (from `profile_history`).
4. **Ledger events:** reinforced or superseded records, with their IDs and the NLI score that decided each one.
5. **Matches:** the top 3 O*NET tasks for each record, from the task-level index, with similarity scores. A negative-polarity match is labelled "conflicts with what you said", never "matches".
6. **Effect:** the occupations this response moved up and down the most, with the size of each move.

**How the attribution is computed (faithful by construction):**
- **M4 positive part:** this part is linear (§5.2), so each record's share of an occupation's score is exact: `w_i·(v_i·d_o)/Σw`.
- **M4 negative part:** the record that triggers the `max` is the cause.
- **Fused rank:** fusion is not linear (RRF and z-score alike), so use **leave-one-record-out**. Recompute the ledger score without record *i*, rerun the selected fusion variant with the unchanged M1 ranking, and report how far each occupation's rank changes. Each record costs one `(n_records × 384) @ (384 × 378)` product plus one fusion over 378 occupations. That should take milliseconds (to be measured). Compute it for the current turn's records by default, and for older records on request.
- **M1 (cosine):** in v1, report the profile change for that turn and the M1 rank changes. Leave-one-turn-out for M1 would mean replaying the tracker; that is deferred.

**Wording rules:** use "associated with" and "matched task" for semantic matches, never "you prefer". Use "captured as a dislike of …" for avoid records. The card uses the same basis as the recommendation cards (`ranking_basis`), which also fixes the final-message wording mismatch listed in the known issues.

**Before the ledger exists (Phase A fallback, debug only):** leave-one-turn-out for the current `finetuned_v1` M4. Re-encode the joined history without turn *t*, which costs one extra encode per earlier turn, so cost grows with the number of turns. Enough to check that the idea works, not to deploy.

### 10.5 Code layout

| Item | Location |
|---|---|
| Map builder (offline, refuses to overwrite) | `recommender/build_occupation_map.py` (new) |
| Explanation builder (read-only, no scoring imports beyond the ledger component's decomposition API) | `pipeline/explain.py` (new); `System.step` stores `turn_trace["explanation"]` |
| Rank snapshot | added to the `ranking_trace` row in `System._rank` |
| API | `/api/occupation_map`; the explanation travels inside `trace` (debug sessions) |
| UI | `static/app.js`: `renderMap(trace, map, tab)` (tabs `semantic` / `hexagon`, shared colours and slider) and `renderEvidence(trace)` replace the JSON block; styles in `static/styles.css` |

**Who sees it:** at first, debug/researcher sessions only. `CM_STUDY_MODE=1` already hides the profile from participants, and the explanation is hidden the same way. Showing a simplified version to participants (Phase C) is a team decision, because explanations can raise trust without making recommendations any better.

### 10.6 Checks

- **Faithfulness tests:** removing the top-attributed record changes the M4 score by exactly the attributed amount (positive part). Leave-one-out rank changes match a full rerun.
- **Plot correctness:** colours recomputed from logged `rank_snapshot`s match the UI categories. Map coordinates are deterministic for a fixed seed and checkpoint, and a stale fingerprint hides the panel.
- **Qualitative review** (pending item 10): record per turn whether the evidence card matched what the reviewer meant. This is also the cheapest way to find extractor polarity errors.

### 10.7 Phases

| Phase | Needs | Delivers |
|---|---|---|
| **A** | Nothing new upstream | Map artifact, `rank_snapshot`, coloured per-turn map with both tabs (semantic t-SNE and RIASEC hexagon with the user-profile marker, trail and slot bar chart) and a slider, hover with component ranks, an evidence card built from existing traces (clauses, readings, provisional tasks), and debug leave-one-turn-out |
| **B** | Ledger with `signed_v1` or later | Record-level captured evidence, exact M4 contributions, fused leave-one-record-out, clause markers on the map |
| **C** | B, with `signed_tasks_qa_v1` or later | A participant-facing "what triggered this" section on the recommendation cards, simplified from B, plus the qualitative review |

### 10.8 Implementation notes (Phase A part 1, 9 November 2026)

- **Snapshot:** `RankingPipeline.rank_with_snapshot` (the pipeline already computed full component rankings; `rank()` now delegates to it and returns the identical response). `System._rank` writes `rank_snapshot` = `{code_order, codes_sha256, fused, structured, semantic[, lexical]}`, arrays of ranks with `null` for unranked, plus `slot_confidence` for the bar chart. No snapshot under description reranking (it reorders only the top) or with no signal.
- **Map artifact:** `python -m recommender.build_occupation_map` → `artifacts/ranking/occupation_semantic_v1/occupation_map_v1/` (`coords.json`, `manifest.json`). t-SNE: cosine, perplexity 30, PCA init, seed 0, 1000 iterations, coordinates scaled into [−0.5, 0.5] keeping aspect ratio. A rebuild was byte-identical. The server recomputes the hexagon from the catalog and serves the t-SNE only if the index files and checkpoint hash still match.
- **"Moved by"** (provisional): with d = |Δ log rank| per component since the previous turn, "interest profile (M1)" if d_M1 ≥ 1.5·d_M4, "semantic match (M4)" if d_M4 ≥ 1.5·d_M1, "both" otherwise, and "neither moved much" if both are < 0.1.
- **Hexagon marker:** same formula as the occupations, over the observed types of each `ranking_trace` row (the profile the ranker actually received). If nothing is positive it sits at the centre.
- **Not done yet:** evidence card (§10.4), debug leave-one-turn-out, ledger markers (Phase B), and the §10.6 check that recomputes colours from logged snapshots.

## 11. Open decisions for the team

- **Resolving contradictions:** should the newer statement win automatically (v1 proposal), or should both be marked `disputed` and trigger a CHALLENGE question as RIASEC conflicts do?
- **Reinforcement:** should repeating a preference raise its weight (noisy-OR, capped at 1), or should repeats be ignored altogether?
- **Unknown polarity:** is `β_unknown = 0.5` acceptable, or should unknown clauses be left out like negations?
- **Penalty strength `λ`:** it has to be set on development data only, and that data must not be reused for the final claim.
- **Explanation audience:** should it stay researcher-only, or should participants see the Phase C version? This affects the user study, since explanations may change how much participants trust the recommendations.
- **Map (decided 9 November 2026):** both tabs, the semantic t-SNE map and the RIASEC hexagon, are in Phase A.
- **Fusion default (decided 9 November 2026):** `zscore_sum@equal_v1` promoted over `rrf` k=60 by owner decision; see §12.6.

## 12. Review and priority (added 9 November 2026)

This section checks the plan against a real failure, then sets the order of work. The case is session `logs/anon-c509a1d7.jsonl`: a musicology student who said they enjoy research in music and rejected helping, persuading, data checking, equipment and leading. The final top 5 was Mathematicians, Astronomers, Computer and Information Research Scientists, Historians, Materials Scientists.

### 12.1 What actually went wrong in that session

The semantic half was **right**. The failure came from the RIASEC half and from fusion.

| Turn 10 | Fused rank | M1 (cosine) rank | M4 (semantic) rank | M4 score |
|---|---|---|---|---|
| Mathematicians | 1 | **1** | 26 | .143 |
| Astronomers | 2 | 11 | 19 | .161 |
| Music Therapists | 7 | 65 | **1** | .499 |
| Musicians and Singers | 10 | 118 | **2** | .433 |
| Music Directors and Composers | 16 | 144 | **4** | .381 |
| Art, Drama, and Music Teachers, Postsecondary | 15 | 132 | **5** | .323 |

1. **Equal-weight RRF (k=60) rewards occupations that are "fairly good in both" over occupations that are "excellent in one".** Mathematicians gets 1/61 + 1/86 = .0280; Music Therapists gets 1/125 + 1/61 = .0244. Whenever the profile and the topic disagree, the winner is the overlap, which here is generic Investigative science.
2. **RRF discards how different the two score distributions are.** M1 cosine is flat: rank 1 is .714 and rank 30 is .561 (−21%). M4 is peaked: rank 1 is .499 and rank 30 is .137 (−73%). Rank 26 in M4 means "unrelated topic", but rank 26 in M1 means "almost as good as the best". RRF treats both the same.
3. **The cosine profile itself was skewed (upstream).**
   - Music research was read as I only. The only A evidence came from two generic probe answers.
   - Turn 9, "yea, i think thats alright" (A, strength .20, confidence .998), lowered A from .55 to .35. `DecayTracker._merge` weights by confidence, so a confident lukewarm answer outweighs an earlier, less confident strong one (.78).
   - Turn 4, "sure, though i am not sure if i would like client facing jobs", answered the asked question with "sure". That part was dropped, and the hedge became a full-strength E −.55 reading with reason `neither` and confidence .45.
   - Turn 1's familiarity reading was merged into I before the probe answered it. Turn 2 confirmed it, so this did not matter here, but it would in general.
   - Mathematicians fits the resulting sign pattern (z: I +2.16, A +1.08, E −1.06, S −.57, R −.70) better than any music occupation, whose S/E z-scores are positive. Given its input, M1 is not buggy.
4. **M4 did drift, but only slightly.** At turn 2 the M4 top 5 was all music (top score .56). By turn 10 it was .50, with "Agents and Business Managers of Artists" at #3 and Lawyers at #12. This is noise from the joined, negated short answers, and the ledger targets exactly this. It was not what put Mathematicians first.
5. **One-dimensional ties showed up too.** At turns 1–2 only I was observed, so every I-positive occupation scored 1.0 and was ordered by SOC code. That is why Education Administrators and Investment Fund Managers were in the fused top 6. Already listed in the known issues.

Re-weighting fusion on this one session would change the answer completely (e.g. halving M1's RRF weight puts Music Therapists and Historians on top). That shows the output depends heavily on fusion weighting. It is **not** evidence for any particular weight, because tuning on this session would be overfitting.

### 12.2 Do I agree with this plan?

Yes, as a design for M4. Recompute-from-snapshot, append-only records, polarity-aware scoring with a bounded task-level penalty, offsets instead of copies, and the faithfulness rules in §10 are all sound. Three changes:

1. **It does not fix the failure above.** M4 already ranked the right occupations first, and the fused output is capped by fusion, which §5.3 defers to the end. A better M4 only helps if fusion lets M4 win when it is confident. Fusion should come before the later ledger steps (§12.3). *Done as #1 (§12.6): `zscore_sum@equal_v1` lets a confident M4 match win; live default since 9 November 2026.*
2. **Question pairing (§6) can contaminate the topic channel with the system's own questions.** Most turns answer generic probe questions whose targets come from unrelated O*NET tasks. In this session, "yes!" to a task from *Architectural and Engineering Managers* would become a positive record pulling M4 towards managers. With `pos(o)` as a weighted **mean**, each such record also dilutes the one statement that carried the topic ("research … in the domain of music"). Proposed change: give `kind=answer` records their own weight `β_answer` (start low, e.g. 0.3, and calibrate), and make paired **negative** answers feed the penalty at full weight. Then pairing mainly adds avoid evidence. Add a §8 metric: rank of the topic owner after N generic probe answers.
3. **Add a fusion arm to the §8 evaluations.** Every ledger variant should be reported fused as well as alone. The decision that matters is the fused output the user sees.

### 12.3 Recommended order

Ordered by expected effect on the fused ranking per unit of work:

| # | Fix | Why this position | Cost |
|---|---|---|---|
| 0 | **Done 9 Nov, §12.5.** **Multi-turn evaluation for fusion** (known issues §4 item 4), plus ~25 scripted personas across fields with acceptable-occupation lists, where profile and topic sometimes disagree. Primary metric: MRR of the first acceptable occupation (see §12.4); Hit@5 and NDCG@10 secondary; report overall, per field and conflict vs agree. | Every fix below changes the fused output. The current benchmarks each favour one component, and none tests "profile says X, topic says Y". Without this, any change is a guess. Keep the personas out of all tuning. | 1–2 days |
| 1 | **Done 9 Nov, §12.6; `zscore_sum@equal_v1` promoted to default.** **Fusion** (new versioned variants; keep `rrf` k=60 as control): (a) tie-aware ranks, where tied cosine scores share the average rank, or M1 is excluded until ≥2 types are observed; (b) score-aware fusion, i.e. per-query normalisation of each component's scores (e.g. z-score over the eligible pool) and a weighted sum, so a peaked M4 match can beat a flat cosine lead; ~~(c) optionally, M1 weight scaled by profile evidence (observed types × confidence)~~ dropped 9 Nov, see §12.5 decisions. Main fix is (b); (a) covers sparse early turns. Choose using #0, not this session. | This is the step that decides the failure in §12.1, and it is general: when two signals disagree, magnitudes and confidence should decide rather than rank position alone. It touches only the fusion component, with no retraining and no index rebuild. | 1–2 days |
| 2 | **Tracker input quality**: (a) do not let a weak same-sign reading lower a stronger one (e.g. merge strength as max or noisy-OR for same-sign readings, or weight by strength × confidence); (b) down-weight `reason=neither` and hedged readings, and keep the answer to the asked question when a hedge follows it; (c) keep familiarity readings out of the profile until confirmed. | Better cosine input for every user. Small, testable changes to `DecayTracker` and answer pairing, evaluated with #0. | 1 day |
| 3 | **Ledger steps 1–2** (§9: `clauses_v1`, `signed_v1`), with the §12.2 change to answer records. | Fixes M4 drift, truncation and negation. These matter more once fusion lets M4 win. | as planned |
| 4 | **Explanations Phase A** (§10), the `rank_snapshot` and component-rank hover at minimum. | Can run in parallel from day one. The hover would have shown this failure (M1 #1 vs M4 #26) immediately, without reading logs. | in parallel |
| 5 | Ledger steps 3–5 (task penalty, question pairing, contradiction revision), then the post-fusion avoid stage (§5.3). | Valuable but narrower; each is a refinement on top of #1–#3. | as planned |

**First fix: #1 (fusion), preceded by the small evaluation in #0.** It is the only change that would have altered this result. It applies to every user whose topic and RIASEC profile point in different directions, and it is the cheapest to build and reverse. The ledger stays the main M4 work, but its benefit reaches users only once fusion stops overruling a confident semantic match.

### 12.4 Decision: primary metric for the #0 fusion evaluation (9 November 2026)

Decided with the team before any run, so the metric cannot be picked after seeing results.

- **Primary: MRR of the first acceptable occupation** (full eligible pool), with paired bootstrap CIs between arms, reported overall and split by conflict vs agree personas.
- **Secondary:** Hit@5 (the app shows the top 5, so this is "did the user see a right answer") and NDCG@10 (graded relevance).
- **Per-turn diagnostic:** rank of the best acceptable occupation in each component (M1, M4, fused).

Why not Hit@10, as first proposed in §12.3:
1. It would have passed the §12.1 session: Music Therapists was fused rank 7 and Musicians and Singers rank 10, yet the user saw Mathematicians and Astronomers. The cutoff is looser than what the user sees.
2. With about 25 personas a binary metric moves in 4-point steps, so CIs are wide and real differences between fusion variants are hidden. MRR is continuous and counts a move from rank 7 to rank 2, which is the effect the fusion fixes target.
3. MRR answers the question directly: does a confident match reach the top? Its weakness (rank 30 vs 80 barely differs) concerns ranks nobody sees.

NDCG@10 stays secondary because its grades come from the drafted acceptable lists, so it inherits their subjectivity.

### 12.5 Results of #0: fusion dialogue evaluation (9 November 2026)

This section replaces a separate findings file. The raw run (`manifest.json`, `results.json` and the generated `REPORT.md` with per-persona tables) is in `reports/ranking/dialogue_fusion_v1/run_20261108T181100Z/`.

**Setup**
- **Personas:** `data/eval/fusion_personas_v1.jsonl`, 25 personas, hand-drafted and team-reviewed on 9 November. Built by `data/scripts/build_fusion_personas_v1.py`. A persona is a conflict case when its best acceptable occupation ranks worse than 20 under oracle cosine (14 conflict, 11 agree; every intended label matched). Persona text was written without looking at any M4 ranking.
- **Harness:** `python -m eval.dialogue_fusion`, real System. Six arms, {RRF k=60 fusion, cosine only, semantic only} × {probing on, off}; each arm uses its ranker both inside the dialogue and for the final ranking. Question-bank wording, at most 10 turns. Metrics as fixed in §12.4, with a paired persona bootstrap (10,000 resamples, seed 425). Tests are in `tests/test_dialogue_fusion.py`.
- **Checkpoints:** the extractor is `models/extractor`, a custom `CustomConcatWithLayer` (safetensors `804050d42c47…`); the previous plain checkpoint is kept as `models/extractor_plain_20261017`. NLI is `models/nli-probing-v2` (`d2a54b245854…`). The working tree had uncommitted changes; source hashes are in the manifest.

**Results** (probing on; probing off is within noise for every ranker)

| Ranker | All MRR | Conflict MRR | Agree MRR | Conflict Hit@5 | Agree Hit@5 |
|---|---|---|---|---|---|
| fusion (RRF k=60, live default) | .391 | .114 | .743 | .21 | .91 |
| cosine only | .198 | .007 | .440 | .00 | .55 |
| semantic only | .758 | .881 | .603 | 1.00 | .73 |

1. **The §12.1 failure is general.** On all 14 conflict personas, M4 puts an acceptable occupation at rank 1–3, and fusion puts it at rank 3–37 (median about 17). Fusion − semantic on conflict personas: −.767 [−.877, −.646].
2. **Fusion is still the best choice for agree personas.** Fusion − semantic on agree personas: +.141 [−.073, +.366]. Semantic alone fails accounting (rank 33), geology (19) and sales (22), where fusion gets 5, 1 and 7. Fusion beats cosine only in every group (all personas +.193 [+.087, +.316]).
3. **Cosine through the real tracker is far below its oracle.** Agree personas are rank 1–2 under oracle cosine, but cosine only reaches MRR .440 in dialogue (nursing reaches rank 74 without probing). This supports #2.
4. **Probing has no measurable effect** on any ranker here; every CI spans 0 (cosine +.057 [−.004, +.156]).

**Decisions**
- **Keep the RRF k=60 fusion default.** Do not switch to semantic only: it wins overall only because conflict personas make up 14 of 25, and it is weaker on agree personas. Do not revert to cosine only either. The fault is in the combination rule, which #1 addresses.
- **Target for #1:** reach semantic's conflict results (MRR about .88) while keeping fusion's agree results (about .74), with `rrf` k=60 kept as the control.
- **Drop option (c) (M1 weight scaled by profile evidence) from #1.** (1) Every persona in every arm has all six types observed by the final turn (5–6 by turn 3 for most), so the observed-types factor is constant where the evaluation measures. (2) Conflict cases have full, confident profiles that point elsewhere (§12.1: confidence .998 on a lukewarm reading), so (c) would keep M1's weight high exactly where it hurts. The failure is about which direction the signals point, not how much evidence there is. (3) Extractor confidences are currently uncalibrated (temperatures 1.0). (4) Sparse early turns are already covered by (a), with no tuned parameter. Could be revisited after #2 with a calibrated extractor, but only together with a per-turn metric, since the final-turn metric cannot show a benefit.
- **The personas are now the decision benchmark.** Design #1 variants on other data or from first principles, then confirm them here. Do not iterate fusion weights against these 25 personas.

**Caveats**
- Conflict openings name the field outright ("I want to be a surgeon"), so .881 is probably an upper bound for semantic on real conflict cases.
- The extractor's metadata looks unfinished (`best_epoch` 0, `best_val_score` −1, all calibration temperatures 1.0), so tracker confidences are uncalibrated. Rerun with a calibrated export when one exists.
- With only 11 agree personas, their CIs are wide.

### 12.6 Results of #1: fusion variants (9 November 2026)

Full report: [reports/ranking/fusion_variants_v1/DECISION.md](../reports/ranking/fusion_variants_v1/DECISION.md).

**Built** (fusion component only; `rrf` k=60 kept unchanged as the control; nothing tuned):
- `rrf_tie_aware@k60_v1` (#1a): RRF where exactly tied component scores share their average rank.
- `zscore_sum@equal_v1` (#1b): per-query z-score of each component's raw scores, summed with equal weights. An occupation a component did not rank gets that component's minimum z; a component with all-equal scores contributes 0.
- Configs `configs/ranking/occupation_semantic_fusion_{tieavg,zscore}_v1.json`. The pipeline and diagnostics now report the fusion identifier actually used.

**Persona dialogues** (`reports/ranking/dialogue_fusion_v2/run_20261108T184532Z/`; the control reproduces §12.5 exactly). Probing on:

| Ranker | All MRR | Conflict MRR | Agree MRR | Conflict Hit@5 | Agree Hit@5 |
|---|---|---|---|---|---|
| fusion `rrf_k60` (control) | .391 | .114 | .743 | .21 | .91 |
| `rrf_tie_aware@k60_v1` | .391 | .114 | .743 | .21 | .91 |
| `zscore_sum@equal_v1` | .680 | .639 | .733 | .71 | .91 |
| semantic only | .758 | .881 | .603 | 1.00 | .73 |

- z-score − control: conflict **+.525 [+.328, +.722]**, agree −.010 [−.192, +.136] (probing off: +.556, −.066).
- z-score − semantic: conflict −.242 [−.426, −.084], agree **+.130 [+.030, +.249]**.
- On conflict personas, z-score puts an acceptable occupation at rank 1 in 8 of 14 (control median about 17). Accounting, which semantic alone fails, stays at 5.
- Tie-aware is identical to the control here: personas have 2 or more types observed from their first ranked turn, so cosine ties do not occur.

**Single-turn checks:**
- **Held-out task retrieval** (MRR, test split): control .200, tie-aware .402, z-score .507, semantic .518. z-score − semantic −.010 [−.020, −.002].
- **120 development profiles, opening turn** (labels follow RIASEC): control .096, tie-aware .097, z-score .064. z-score − control **−.032 [−.059, −.011]**.

**Decision (owner, 9 November 2026): `zscore_sum@equal_v1` is the live default** in `configs/ranking.yaml` ([decision](../reports/ranking/fusion_zscore_default/DECISION.md)). It recovers most of the conflict gap at no measurable agree cost. The accepted costs are the opening-turn development regression (−.032) and the §12.5 caveat that persona openings name the field outright, so the conflict gain is probably an upper bound. Tie-aware never does worse but does not fix the conflict failure; it stays selectable and is not the default. Roll back by restoring `fusion: {algorithm: rrf, k: 60}` (or run `configs/system.rrf.json`).

**Reproduce and try it:**
- Personas: `python -m eval.dialogue_fusion` (all five rankers × probing on/off, output `reports/ranking/dialogue_fusion_v2/`); `--rankers fusion fusion_zscore --probing probing` runs a subset.
- Task retrieval: `python -m eval.task_retrieval_fusion` (output `reports/ranking/fusion_variants_v1/task_retrieval/`).
- Development profiles: `python -m recommender.evaluate --config reports/ranking/occupation_semantic_v1/dev_extractor_inputs/fusion/evaluation_config.json --ranking-config configs/ranking/occupation_semantic_fusion_zscore_v1.json --output <new-directory>`, then `python -m recommender.compare`.
- Website: the default `configs/system.json` now runs z-score fusion. For the previous RRF behaviour, start the server with `$env:CM_CONFIG = 'configs/system.rrf.json'`. The debug panel's component ranks and the `composition` diagnostic (`zscore_sum@equal_v1` or `rrf_k60`) show which fusion ran.

**Still open after #1:**
- A browser check of the live site with the new default, and a qualitative look at real sessions (e.g. `logs/anon-c509a1d7.jsonl`) under both fusions.
- Tie-aware RRF cannot be tested by the personas (no cosine ties). Only single-sentence task retrieval shows its effect.
- The remaining conflict gap to semantic only (.639 vs .881: F07, F08, F11, F13, F14) is where cosine still strongly favours another field. That most likely points to #2 (tracker input quality) rather than to fusion; not yet checked per persona.
- Next in §12.3 order: #2, then #3 (ledger steps 1–2) with both fusion arms reported (§8.2).

## 13. Change summary: increments on `feat/ranker-v2` (9 November 2026)

Summary of the changes made on top of commit `9fd317b` (all staged, nothing unstaged at the time of writing). Each increment lists what was built, how it works, and the problem it addresses. Details and numbers are in §10.8 and §12.6.

### 13.1 Fusion variant registry
- **What:** `recommender/models/fusion.py` now has `FUSION_VARIANTS`, `build_fusion` and `fusion_identifier`. `RankingPipeline` builds its fusion once from the config and reports the identifier of the fusion actually used (`fusion_identifier`, diagnostics `composition`).
- **How:** each accepted selection (e.g. `{algorithm: zscore_sum, variant: equal_v1}`) maps to a frozen identifier and factory. Unknown selections fail with an error that lists the registered variants. New rules are added as new entries, never by editing an existing one.
- **Problem solved:** `validate_fusion` only accepted `rrf` k=60, and `"rrf_k60"` was hard-coded in the pipeline labels. No other fusion rule could be added or evaluated without breaking variant preservation or provenance.

### 13.2 Tie-aware RRF (`rrf_tie_aware@k60_v1`, #1a)
- **What:** `TieAwareReciprocalRankFusion`, preset `configs/ranking/occupation_semantic_fusion_tieavg_v1.json`.
- **How:** the same equal-weight RRF as the control, but occupations with exactly equal component scores share the average of the ranks they span (`average_ranks`).
- **Problem solved:** with one observed RIASEC type, cosine gives large groups of tied scores, and plain RRF broke those ties by SOC code, an arbitrary order. Result: task retrieval MRR .200 → .402, never worse elsewhere. It does not fix the conflict failure, and the personas cannot test it (no cosine ties). Selectable, not the default.

### 13.3 Z-score sum fusion (`zscore_sum@equal_v1`, #1b), now the live default
- **What:** `ZScoreSumFusion`, preset `configs/ranking/occupation_semantic_fusion_zscore_v1.json`. `configs/ranking.yaml` now selects it (owner decision, `reports/ranking/fusion_zscore_default/DECISION.md`).
- **How:** per query, each usable component's raw scores are z-scored over the occupations it ranked (population SD) and summed with equal weights. An occupation a component did not rank gets that component's minimum z; a component with all-equal scores contributes 0. Nothing is tuned. `RankedOccupation.rrf_score` holds the fused score so the response schema is unchanged.
- **Problem solved:** the §12.1 failure. RRF uses rank position only, so a flat cosine (M1) lead overruled a confident semantic (M4) match whenever profile and topic disagreed (conflict persona MRR .114). Z-score lets a peaked M4 score count by its size: conflict MRR .639 (+.525 [+.328, +.722]), agree unchanged (.733 vs .743), task retrieval .507. Accepted cost: opening-turn development profiles .064 vs .096 (−.032).
- **Rollback:** `configs/system.rrf.json` runs the previous RRF default without editing anything (`$env:CM_CONFIG = 'configs/system.rrf.json'`); or restore `fusion: {algorithm: rrf, k: 60}` in `configs/ranking.yaml`.

### 13.4 Evaluation harness for fusion variants
- **What:** `eval/dialogue_fusion.py` gains the tie-aware and z-score arms, their contrasts against the control and semantic only, and `--rankers` / `--probing` subset flags (output moves to `reports/ranking/dialogue_fusion_v2/`). New `eval/task_retrieval_fusion.py` runs held-out task retrieval for every fusion arm. Development-profile reruns via `recommender.evaluate` / `recommender.compare`.
- **How:** each benchmark reruns the RRF control first and checks it reproduces the saved result (personas and development profiles exactly; task retrieval .200 vs .201, explained by the changed local extractor checkpoint). Paired bootstrap CIs for every contrast.
- **Problem solved:** a variant can only be chosen fairly if it is measured on the multi-turn decision benchmark (personas) and on the single-turn checks, against a reproduced control. Results and provenance are in `reports/ranking/dialogue_fusion_v2/run_20261108T184532Z/` and `reports/ranking/fusion_variants_v1/` (`DECISION.md`, `task_retrieval/`, `dev_extractor_inputs/`).

### 13.5 Explanation map, Phase A part 1 (§10)
- **What:**
  - `RankingPipeline.rank_with_snapshot` (`rank()` delegates to it and returns the identical response).
  - `System._rank` stores `rank_snapshot` and `slot_confidence` in each `ranking_trace` row.
  - New `recommender/build_occupation_map.py` builds the map artifact (t-SNE over the fine-tuned index, plus RIASEC hexagon coordinates), served by the new `GET /api/occupation_map`.
  - The debug panel in `static/app.js` / `static/styles.css` replaces the raw JSON block with the map.
- **How:** the snapshot holds every eligible occupation's fused and per-component ranks as integer arrays in a fixed code order with a `codes_sha256` check. It is explanation-only: no scoring change, exports keep their schema, and there is no snapshot under description reranking or with no signal. The map has semantic and hexagon tabs, a turn slider, rose/fell/top-10 colouring, component-rank hover ("moved by" M1, M4 or both), the tracker-profile marker and trail on the hexagon, and a slot confidence bar chart. The map artifact is deterministic (rebuild byte-identical) and the server only serves the t-SNE when its index and checkpoint fingerprints match.
- **Problem solved:** recommendations did not feel credible because nothing showed why an occupation moved, and diagnosing failures like §12.1 (M1 #1 vs M4 #26) meant reading logs. The hover now shows which component pushed an occupation up or down on each turn.
- **Still pending:** evidence card (§10.4), leave-one-turn-out, ledger markers (Phase B), the §10.6 colour check from logged snapshots.

### 13.6 Tests and documentation
- **Tests:**
  - `tests/test_fusion.py`: the new variants, ties, missing occupations, constant components, the registry.
  - New `tests/test_occupation_map.py`.
  - `tests/test_configuration.py`: pins the YAML to z-score and keeps RRF and `system.rrf.json` selectable.
  - `tests/test_ranking_app_integration.py`: the web app's ranking equals the z-score preset's.
  - Small updates in `test_dialogue_fusion`, `test_history` and `test_integration`.
  - The 23 pre-existing failures (offline HuggingFace, fingerprints, caches) are unchanged from `9fd317b`.
- **Docs:** status entries in `docs/RANKING_EXPERIMENT_ARCHITECTURE.md` and `cs425-ranking-docs/README.md` (kept consistent per `AGENTS.md`), known-issues item 5 marked done, and §3, §5.3, §8.2, §10.8, §11 and §12.6 of this plan updated.

### 13.7 Not done in these increments
- Browser check of the live site with z-score fusion, and a qualitative replay of real sessions under both fusions.
- Remaining conflict gap to semantic only (F07, F08, F11, F13, F14), probably #2 (tracker input quality).
- Next: #2, then #3 (ledger steps 1–2) reported under both `zscore_sum@equal_v1` and `rrf_k60`.
