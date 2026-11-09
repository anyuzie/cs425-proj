# Plan: lexical matching (L0–L3)

Status (9 November 2026, branch `feat/ranker-v2`): **promoted.** `configs/ranking.yaml` now runs signed lexical (`bm25_signed@ignore_negative_v3`) + signed semantic (`occupation_dual_encoder_signed@finetuned_v2`) + z-score fusion + the dislike stage `dislike_demotion@title_v3` ([decision](../reports/ranking/d2_title_v3_default/DECISION.md)). Accepted cost: development-profile Recall@50 −.036. Previous default: `configs/system.zscore.json`. L1c on hold; L0b, L2, L3, D3 are proposals.
Read [RANKING_EXPERIMENT_ARCHITECTURE.md](RANKING_EXPERIMENT_ARCHITECTURE.md) first. Nothing here changes `bm25@baseline_v1`, the live default (`zscore_sum@equal_v1` over M1 cosine + M4 `finetuned_v1`) or any accepted variant. Context: [SEMANTIC_LEDGER_PLAN.md §6.3](SEMANTIC_LEDGER_PLAN.md#63-should-this-be-lexical-no-not-in-this-plan).

## Why consider lexical matching at all

What BM25 could add that the fine-tuned bi-encoder (M4) does not:

- **Exact rare terms:** tools, acronyms and niche fields ("GIS", "CAD", "actuarial", "phlebotomy") that MiniLM may blur or dilute.
- **Exact titles:** `bm25_doc` already contains the title and all O*NET alternate titles, so "I want to be a sonographer" is an exact hit.
- **No truncation:** BM25 reads the whole history; M4 keeps only the first 256 wordpieces.

## Concerns

1. **Negation gets worse.** BM25 has no polarity. "I don't want to work with patients" ranks nurses highly. The kept negator tokens match nothing useful.
2. **Z-score fusion amplifies spikes.** BM25 scores are heavy-tailed, so one rare matched word gives a large z. BM25 ranks only positive matches; every non-matching occupation gets the floor z.
3. **Dilution of M4.** Equal three-way weights cut M4's share from ½ to ⅓, right after z-score fusion let a confident M4 match win the conflict cases.
4. **Earlier evidence is negative but biased.** Hit@5 .017 (BM25) vs .167 (structured) on development profiles, whose labels follow RIASEC and so disadvantage any text component. Never run on the personas.
5. **Task retrieval is leaked for BM25.** `bm25_doc` contains every task statement verbatim (checked: 2,000/2,000), so the held-out task-retrieval benchmark is an exact-string lookup for BM25. Do not use it as L0 evidence.
6. **Benchmarks may not show the benefit.** Persona openings name the field outright (M4 already finds it), and no benchmark has rare-term or past-truncation dialogues. A "no gain" result on personas alone is not conclusive.

## Steps (one variable at a time)

| Step | Change | Hypothesis | Status |
|---|---|---|---|
| **L0** | Add the frozen `bm25@baseline_v1` as M3 to the live default: `configs/ranking/occupation_semantic_fusion_zscore_bm25_v1.json`. Config only, no code. Plus BM25 alone (`configs/ranking/m3_bm25.json`) as a diagnostic arm. | Lexical control: measures what unfiltered lexical matching does to the fused ranking. Expected: little or no gain, some harm from negation and dilution. | persona run done; fails gate (below) |
| L1c (on hold) | Clause handling for the signed query: score each clause separately, then combine, and decide which clauses count (see [L1c](#l1c-clause-handling-on-hold)). Applied to both `ignore` and `filter`. | Stop filler clauses (short probe answers) from drowning the clause that names the topic. | on hold |
| L0b | Small hand-drafted probe set: rare-term dialogues, dialogues whose decisive topic comes after 256 wordpieces, and a few negation cases as a regression gate. Team-reviewed like the personas, never tuned on. | Gives lexical a fair chance to show its specific benefit. | not started |
| L1 | New algorithm `bm25_signed` (see [L1 design](#l1-design-signed-lexical-matching)), two variants: `ignore_negative_v1` and `filter_negative_v1`. | Removes the negation harm while keeping the rare-term and no-truncation benefits. | built; persona run done (below); gate still fails on agree |
| L2 | After ledger step 2: `bm25@ledger_v1` reads positive targets (and positive paired answers) from `semantic_evidence`. This is §6.3's `bm25@question_paired_v1` idea. | Same polarity source as M4; short answers contribute. | blocked on ledger |
| L3 | Only if L1 helps but adds noise: a new fusion variant that caps BM25's z or gives it less weight. Tuned on data other than the personas. | Bounds lexical spikes and dilution. | proposal |

## Evaluation

- **Personas** (`python -m eval.dialogue_fusion`): decision benchmark, never tuned on. Arms `fusion_zscore` (live default, control), `fusion_zscore_bm25` (L0), `lexical` (BM25 alone, diagnostic), plus `semantic`. Primary metric MRR of the first acceptable occupation; contrasts `fusion_zscore_bm25 − fusion_zscore` and `− semantic`, overall and conflict vs agree.
  Run: `python -m eval.dialogue_fusion --rankers fusion_zscore semantic lexical fusion_zscore_bm25 --probing probing`
- **Development profiles** (`recommender.evaluate` + `recommender.compare`, as in §12.6): opening-turn regression check against the default.
- **RRF control** (§8.2): report L1 onward under `rrf_k60` too.
- **Not** task retrieval (concern 5).

**Acceptance gate for L0 → L1:** no significant regression on agree personas, and some sign of benefit (on L0b if personas show nothing). Otherwise record the negative result and stop.

## L0 results (9 November 2026)

Run: `reports/ranking/lexical_l0/run_20261109_personas/` (probing on; `REPORT.md` has per-persona ranks). The live-default arm reproduces §12.6 exactly (.680 / .639 / .733), as does semantic only (.758).

| Ranker | All MRR | Conflict MRR | Agree MRR | All Hit@5 | Conflict Hit@5 | Agree Hit@5 |
|---|---|---|---|---|---|---|
| `fusion_zscore` (live default, control) | .680 | .639 | .733 | .80 | .71 | .91 |
| `fusion_zscore_bm25` (L0) | .658 | .744 | .548 | .92 | 1.00 | .82 |
| `lexical` (BM25 alone) | .518 | .669 | .326 | .60 | .79 | .36 |
| `semantic` only | .758 | .881 | .603 | .88 | 1.00 | .73 |

L0 − control (MRR): all −.023 [−.163, +.121], conflict +.105 [−.078, +.293], **agree −.185 [−.386, −.013]**.

- **Gate fails:** agree personas regress significantly. The largest drops are statistics (1 → 7), accounting (5 → 10), graphic design (1 → 3) and software (1 → 2). BM25 alone ranks these poorly (173, 37, 28, 23), and with equal weight it pulls them down.
- **Conflict gain is real in shape but not significant:** 4 of the 5 remaining conflict gaps improve (food 7 → 1, aviation 9 → 2, finance 13 → 4, sport 12 → 3), and every persona has an acceptable occupation in the top 10. This is most likely exact title matching on openings that name the field ("I want to be a surgeon"), i.e. the §12.5 caveat: probably an upper bound for real users.
- **Negation (concern 1) is not what hurt here:** the agree losses look like lexical noise and dilution, not negated mentions. So L1 alone would probably not fix the agree regression; a bounded weight (L3) targets it, but needs tuning data other than the personas.

**Decision pending (owner).** Options: (a) record L0 as negative and stop; (b) build L0b to test lexical on rare-term / long dialogues before investing in L1–L3. Do not tune a BM25 weight on these personas.

## L1 design: signed lexical matching

Built 9 November 2026 (owner decision to proceed despite the L0 gate, to test whether polarity changes the picture). Module `recommender/models/bm25_signed_ranker.py`, algorithm `bm25_signed` under M3; parameters in `configs/ranking/parameters/bm25_signed_{ignore,filter}_negative_v1_parameters.json`; fused presets `configs/ranking/occupation_semantic_fusion_zscore_bm25_signed_{ignore,filter}_v1.json`; tests `tests/test_bm25_signed.py`. It reuses the frozen `bm25@baseline_v1` index and scoring unchanged: only the query text differs, so with no negative clauses both variants equal `bm25@baseline_v1` exactly.

**Clause routing** (extractor `explicit_scoped_v1`, which splits clauses on punctuation and "but", "because", ...):

| Route | Rule | Used as |
|---|---|---|
| positive | explicit positive preference | query |
| neutral | no explicit stance, no negation cue (background and topic statements, e.g. "I've been writing music since I was twelve") | query |
| negative | explicit negative preference, or no explicit stance but a negation cue (`not`, `n't`, `never`, `avoid`, `hate`, `rather not`, `anything but`, ...) | never in the query; filter variant only |
| skipped | non-preference wording ("I don't mind", "not sure", "doesn't matter"), or uncertain wording with a negation cue | ignored |

Neutral clauses are included (owner decision): the extractor often keeps only an activity phrase and labels the topic-bearing clause unknown (F01's only positive phrase is "working out why a harmony works"; "music" is in unknown clauses). Positive-only would leave BM25 a thin query.

**Early stop:** no positive or neutral clause → `unusable`, so lexical matching does not run and fusion uses M1 and M4 alone.

**Variants** (differ only in negative handling):
- `ignore_negative_v1`: negative clauses have no effect; only the positive/neutral query adds scores.
- `filter_negative_v1`: an occupation is removed from the lexical ranking when its BM25 score against the negative clauses is higher than against the query (owner decision; no tuned threshold). Matching "any negative term" would remove almost everything, since dislikes contain common words such as "work". Removed means "no lexical match", the same as an occupation that matched nothing (floor z in `zscore_sum`), not a penalty below those.

Routing on the 25 persona dialogues (coverage only): 210 neutral, 63 negative, 62 skipped, 39 positive clauses.

## L1 results (9 November 2026)

Run: `reports/ranking/lexical_l1/run_20261109_personas/` (probing on). Controls reproduce: live default .680, L0 .658, semantic .758.

| Ranker | All MRR | Conflict MRR | Agree MRR | All Hit@5 |
|---|---|---|---|---|
| `fusion_zscore` (live default) | .680 | .639 | .733 | .80 |
| L0 `bm25@baseline_v1` | .658 | .744 | .548 | .92 |
| L1 `bm25_signed@ignore_negative_v1` | .683 | .780 | .560 | .92 |
| L1 `bm25_signed@filter_negative_v1` | .685 | .780 | .564 | .96 |

- **Polarity helps lexical a little and consistently:** ignore − L0 +.025 [−.001, +.071], conflict +.036 [+.000, +.107]. Climate policy 2 → 1 and sales 5 → 3.
- **Filter ≈ ignore:** +.002 [+.000, +.005]; differs only on statistics (6 → 5) and accounting (13 → 11).
- **The agree regression remains:** ignore − default −.173 [−.378, +.000], filter −.169 [−.368, +.000] (borderline). Statistics 1 → 5/6, graphic design 1 → 3, accounting 5 → 11/13. This confirms the L0 reading: the agree loss is lexical noise and dilution, not negation.
- **Conflict gain over the default:** +.141 [−.019, +.311] for both, not significant; still probably title matching on openings that name the field.

Gate (no agree regression) still fails. Next options: owner qualitative comparison of ignore vs filter in the browser (`configs/system.bm25-signed-{ignore,filter}-v1.json`), L0b, or L3 (bounded lexical weight, tuned off the personas).

## L1c: clause handling (on hold)

**On hold (owner, 9 November 2026):** the problem may not be clause splitting. Findings kept for reference.

Added 9 November 2026. Owner question: do longer inputs drown out the clauses that matter, and would clause splitting fix it? Decision: settle clause handling first, then choose between `ignore_negative_v1` and `filter_negative_v1` with it in place.

### What exists today
- **Splitting already happens.** `bm25_signed` routes each clause from the `explicit_scoped_v1` splitter (punctuation, "but", "because", "since", ", I ..."). On the persona dialogues: median 8 words per user turn, **6 words per clause** (max 29), 100 words per dialogue (max 147).
- **But the kept clauses are joined back into one BM25 query.** The System's Stanza clause splitter (`stanza_conservative_v1`) also runs, but only into `turn_trace`; no ranking component sees it.

### Findings
1. **Splitting alone cannot change BM25.** BM25 with linear query term counts adds up over query terms, so summing per-clause scores gives exactly the joined score (checked on F24: max difference 4e-15). Any effect has to come from *how clause scores are combined* or *which clauses count*.
2. **Drowning is real, but it comes from filler clauses, not long sentences.** F24 (statistics), lexical alone, acceptable occupation at rank 141 with the joined query. The topic clause "Data is my favourite thing" alone puts an acceptable occupation at rank 2, but its top BM25 score is only 1.7. Short probe answers score far higher and point elsewhere: "Either is fine" 7.7 (Cytotechnologists), "I'd lean towards my own tasks, slightly lah" 8.0, "I'd lean towards doing the work behind it" 8.8 (Manufacturing Engineers). These are answers to the system's questions, so their words describe the question's topic or nothing at all, not the user's field.
3. **Equal-vote combining alone would not fix this.** Normalising each clause to its own maximum gives "Either is fine" the same vote as the topic clause, so one topic clause still loses against about ten filler clauses.
4. **Truncation is not a factor in the personas** (no dialogue over 180 words, so M4 never hits 256 wordpieces). Clause-level semantic matching is ledger step 1 (`occupation_semantic_ledger@clauses_v1`), a separate experiment.

F24 is a single diagnosed failure, used to understand the mechanism, not to choose parameters. The variants below are fixed from these principles before any run.

### Proposed variants (one variable each; applied on top of both ignore and filter)

| Variant | Which clauses count | How clause scores combine | Hypothesis |
|---|---|---|---|
| `…_sum_v1` (current L1) | positive + neutral | sum (= joined query) | control |
| `…_answers_excluded_v1` | positive + neutral, **excluding short replies to a system question** (≤ 9 words, the `PAIR_MAX_WORDS` rule in `pipeline/answers.py`); explicit positive preferences are always kept | sum | Probe-answer filler stops drowning the topic. Main candidate. |
| `…_clause_mean_v1` | positive + neutral | each clause's scores divided by that clause's maximum, then the mean over clauses (each clause one vote) | Long or generic clauses stop dominating. Expected to help less (finding 3). |

- The short-reply rule needs to know which turns answered a question. `ComponentInput.user_sentences` has one entry per user turn, and the first turn is the opening, so v1 can use turn length alone ("turn after the first, ≤ 9 words"). Passing the actual previous action would need a contract field; leave that to the ledger (`kind=answer` records, ledger §6).
- Short answers then carry no lexical signal, which is acceptable: the ledger plan gives answers their meaning through the question target (§6.2) rather than through their own words.
- Early stop still applies: if every clause is excluded, lexical does not run.

### Evaluation
Same as L1: personas once (never tuned on), arms `fusion_zscore` (control), the current ignore/filter arms, and the new variants on top of each; contrasts against the L1 arm they extend. Also report lexical alone (diagnostic): the clause rules are meant to fix lexical's own ranking first. Watch the agree personas (statistics, graphic design, software, accounting): the L1 gate still requires no agree regression against the live default.

## L1 v2: apostrophe-less negation cues

Added 9 November 2026. `negation_cue_pattern` in v1 matches contractions only with an apostrophe (`n't`), so casual typing ("i dont want to work with patients", "cant draw", "isnt for me") is routed **neutral** and its words go into the query. The same gap exists in the non-preference pattern: "i dont mind" would be read as a dislike.

- **New variants** `bm25_signed@ignore_negative_v2` and `@filter_negative_v2` (v1 unchanged, since it has recorded results). Same rules, plus:
  - negation cues: `dont, doesnt, didnt, wont, wouldnt, cant, couldnt, shouldnt, isnt, arent, wasnt, werent, havent, hasnt, hadnt, aint, nah, nope`, and `cant stand`;
  - non-preference (skipped): `dont mind`, `wouldnt mind`, `dont know`, `idk`, `doesnt matter`, `dont matter`.
- Whole-word matches only, so words like "cantonese" are unaffected (tested).
- Presets `configs/ranking/occupation_semantic_fusion_zscore_bm25_signed_{ignore,filter}_v2.json`; browser configs `configs/system.bm25-signed-{ignore,filter}-v2.json`.
- **Personas affected:** 3 turns use apostrophe-less forms. Two clauses change route: F05 "cant draw at all tho" and F22 "dont care much for science stuff" (F22 graphic design is one of the regressed agree personas). F14's clause also contains "not", so it was already negative. Expect small persona differences; the change matters more for real users who type casually.
- **Persona results** (`reports/ranking/lexical_l1_v2/run_20261109_personas/`, probing on; controls reproduce): **MRR and Hit@5 identical to v1** for both versions (v2 − v1 = +.000 [+.000, +.000]); every persona's best acceptable rank is unchanged. Only NDCG@10 moves slightly (conflict .701 → .714 ignore, .700 → .713 filter), i.e. lower-ranked acceptable occupations shift a little. F22 stays at rank 3, so "dont care much for science stuff" was not what pushed graphic design down.
- **Decision:** keep v2 as the version to compare in the browser (it fixes a real gap for casual typing at no measured cost). The agree regression is unaffected; ignore vs filter is still open (.683 vs .685).

## Decision: ignore over filter (9 November 2026)

**Keep `bm25_signed@ignore_negative_v2`; do not develop the lexical filter further.** Evidence:

- **Personas:** filter − ignore +.002 MRR [+.000, +.005], from two personas (statistics 6 → 5, accounting 13 → 11). Too small to choose on.
- **Browser, 3 paired sessions** (owner, 8002 filter / 8004 ignore; logs `anon-d189ba45`/`335c8921` musicology, `anon-d5f7604a`/`f8f93bd0` and `anon-03cd2dee`/`c160cd58` psychology): final top 5 identical in the musicology pair, and identical apart from one swap at rank 5 in both psychology pairs. Pairing is not exact: the follow-up questions came in a different order in one pair.
- **The filter did not remove the target and removed relevant jobs.** Replaying `anon-c160cd58` (filter v2): for "i like helping others. but teaching, not as much", Psychology Teachers, Postsecondary stays fused #2 (lexical 26th, semantic **1st**). It is not removed because "psychology / helping / students" matches it more than "teaching, not as much" does. Meanwhile vague negative clauses ("i dont really care as much about an objects appearance…", "not as much persuading others") removed Marriage and Family Therapists and Mental Health and Substance Abuse Social Workers from the lexical list.
- **Structural reason** (owner analysis): the filter acts inside one of three components. A job removed from the lexical list only gets the lexical floor z and keeps its semantic and cosine z, so z-score fusion still ranks it highly. A dislike enforced in one component cannot override the other two.

The filter's purpose moves to a post-fusion stage (D2). `filter_negative_v1/v2` stay registered and selectable.

## Dislike handling (D1–D3)

Goal: "I like helping others, but teaching, not as much" should push teaching jobs down in the **final** list, whatever the components say. One variable per step; each evaluated against the step before.

| Step | Change | Why this position | Status |
|---|---|---|---|
| **D1** | **Drop negative clauses from the semantic query.** New M4 algorithm `occupation_dual_encoder_signed@finetuned_v1`: same checkpoint and index, but encodes only the clauses `bm25_signed@ignore_negative_v2` routes positive or neutral (shared routing; early stop if none). Wraps `OccupationSemantic`; **never edits** `recommender/occupation_semantic.py` (its bytes are part of the index identity). | Cheapest; removes the pull at the source and also cleans positive matching. Replay of `anon-c160cd58`: semantic top 5 goes from 5 teacher jobs to 1 (Clinical and Counseling Psychologists, Psychiatrists, Psychiatric Nurses move up); other teacher jobs fall to 8 and 13. **But Psychology Teachers stays semantic #1** ("I study psychology" matches its text best), so D1 alone cannot enforce the dislike. | built; probes + personas done (below) |
| **D2** | **Post-fusion dislike stage.** New pipeline stage after fusion (beside the existing `reranking` slot), e.g. `post_fusion: {algorithm: dislike_demotion, variant: …}`. For each clause routed negative: extract the target (strip cue words: "teaching, not as much" → "teaching"), score every job against it, and demote any job that matches strongly, regardless of component scores. Matchers, from cheapest: (a) title / occupation-group match (e.g. "teach" → SOC major group 25 Education, Training, and Library, and titles containing "Teacher"); (b) semantic similarity of the target to the job's fine-tuned vector or its best task (task-level index, ledger §7). Demotion: move matched jobs below the top 10, or remove them; record the choice and threshold as variant parameters. | The only step that can override the components. Needs a threshold, which must be set on labelled negation data, **not** the personas. | proposal |
| **D3** | **Per-clause lexical test** (only if D2 leaves gaps): penalise a job if it strongly matches **any single** negative clause, even when the field word also matches, instead of all-negative vs all-positive. | Likely redundant once D2 exists (lexical is one third of the fused score). Kept as an option, not planned work. | optional |

**Prerequisites and evaluation:**
- **Negation probe set** (ledger §8.1), built before D2: hand-drafted statements such as "I like helping others but not teaching", "I love biology, just not lab work", "anything but sales", each with *avoided* occupations (must leave the top 10) and *acceptable* ones (must stay). Team-reviewed; split into a calibration half (for D2's threshold) and a test half. Metrics: rate of avoided occupations in the top 10, collateral demotion of acceptable ones, MRR of acceptable.
- **Personas** stay the regression gate for every step (never tuned on). Personas contain dislikes too ("I hate having to sell myself", "avoid sales").
- **Negative-route quality matters more once dislikes act on the final list.** Clauses like "i dont really care as much about an objects appearance" are negative by cue but name no job; D2's threshold and target extraction must keep them from demoting unrelated jobs. Track collateral demotion explicitly.
- Overlap with the ledger: D1 is a joined-query interim version of ledger `signed_v1`; D2 is the post-fusion avoid stage of ledger §5.3. Building them here keeps the ledger as the later, per-clause version.

## D1 built (9 November 2026)

- **Component:** `occupation_dual_encoder_signed@finetuned_v1` (`recommender/models/occupation_semantic_signed.py`, parameters `configs/ranking/parameters/occupation_dual_encoder_signed_finetuned_v1_parameters.json`). Same checkpoint and index as `finetuned_v1`; wraps `OccupationSemantic` without editing it. Each user turn has its negative and skipped clauses **deleted in place** (clauses located in order, so "no" is never cut out of "know"); everything else stays as typed. A history with no such clause gives a byte-identical query and ranking (tested). Early stop if no positive or neutral clause. Query diagnostics list the removed clauses.
- **Shared routing:** `recommender/models/clause_routing.py` (`ClauseRouter`), now used by both `bm25_signed` and D1, so lexical and semantic routes are identical. The BM25 refactor is behaviour-preserving (all signed-BM25 tests unchanged).
- **Presets:** `configs/ranking/occupation_semantic_signed_{finetuned,fusion_zscore,fusion_zscore_bm25_signed_ignore_v2}_v1*.json`; browser configs `configs/system.semantic-signed-fusion-v1.json` (D1 alone on the live default) and `configs/system.semantic-signed-bm25-signed-ignore-v2.json` (D1 + signed lexical).
- Tests: `tests/test_occupation_semantic_signed.py` (7, using the local checkpoint).

## Negation probe set (draft, 9 November 2026)

`data/eval/negation_probes_v1.jsonl`, built by `data/scripts/build_negation_probes_v1.py`. **Draft, pending team review.** 24 probes, 12 calibration / 12 test, written from occupation titles only. Kinds: explicit (4), casual apostrophe-less (4), extractor_miss ("is not for me", "anything but", "rather not"; 5), narrow ("engineering, just not cars"; 5), collateral (negative clause names no job; 2), only_negative (2), control (2). Avoided families are title rules (e.g. teaching = "Teacher", "Instructors", "Teaching Assistants", "Educators", "Adapted Physical Education Specialists"; patient-facing restricted to SOC 29). Evaluator: `python -m eval.negation_probes`. Only the calibration split may inform D2's threshold.

### D1 on the probes (`reports/ranking/dislike_d1/negation_probes_run_20261109/`)

| Arm | Avoided in top 10 (mean) | Any avoided in top 10 | Acceptable MRR | Acceptable in top 10 |
|---|---|---|---|---|
| live default (`fusion_zscore`) | 2.79 | .79 | .317 | .40 |
| **D1 fused** (`fusion_zscore_d1`) | **0.79** | **.29** | **.485** | .55 |
| semantic only | 3.33 | .83 | .350 | .44 |
| semantic signed only | 0.92 | .42 | .546 | .59 |

- Control and collateral probes unchanged (nothing demoted that should not be). N01 (owner's case): avoided teaching jobs in the top 10 8 → 2, best acceptable 15 → 2.
- **Routing bug 1, "X, just not Y":** the splitter does not break at the comma, so "I love biology, just not lab work" is one clause, routed negative and deleted *with its positive half*. N02, N10, N20 lose their topic (semantic signed early-stops; best acceptable 117 / 338 fused). Narrow-kind MRR .333 → .322.
- **Routing bug 2, "anything but X":** "but" is a split word, so "Anything" and "healthcare" become separate neutral clauses and the cue `anything but` never matches; "healthcare" stays in the query. N06 and N18 unchanged by D1.
- **Expected, needs D2:** N13 "i love kids but teaching isnt for me" gets *more* children's-teaching jobs (3 → 4): the dislike clause is gone, but "i love kids" alone points at kindergarten teachers.

Both bugs are in the shared splitter/routing and were found on this probe set, including test probes (N18, N20); a fix must be a general rule (split "just not" / "but not" scope inside a clause; keep "anything but X" together) decided from first principles, recorded as a new routing version, and re-checked on calibration only before the test split is read again.

### D1 on the personas (`reports/ranking/dislike_d1/run_20261109_personas/`, probing on; controls reproduce)

| Ranker | All MRR | Conflict MRR | Agree MRR | All Hit@5 |
|---|---|---|---|---|
| live default (`fusion_zscore`) | .680 | .639 | .733 | .80 |
| signed lexical only added (`bm25_ignore_v2`) | .683 | .780 | .560 | .92 |
| **D1 fused** | .731 | .619 | **.874** | .80 |
| **D1 + signed lexical** | **.770** | .780 | .758 | .96 |
| semantic only | .758 | .881 | .603 | .88 |
| **semantic signed only** | **.855** | .881 | .821 | .96 |

- **D1 passes the regression gate and helps:** D1 fused − live default +.050 [−.030, +.142]; agree **+.141 [+.000, +.310]** (geology 4 → 1, sales 9 → 2); conflict −.021 [−.111, +.048] (surgery 5 → 8).
- **D1 removes the lexical agree regression:** D1 + signed lexical − signed lexical alone +.087 [+.017, +.174], agree +.198 [+.054, +.364]. Against the live default: all +.090 [−.037, +.226], conflict +.141 [−.019, +.311], agree +.025 [−.190, +.244]. This is the first lexical arm with no agree loss.
- **Semantic signed alone is the strongest arm:** − semantic +.096 [+.018, +.194] (agree +.219); − live default +.174 [+.063, +.300]. The personas' dislikes ("I hate having to sell myself", "avoid sales") were pulling the semantic query towards the jobs they reject. As in §12.5, "semantic only" is not a candidate default on this evidence alone (persona openings name the field outright).
- Reading: most of the earlier "lexical noise" on agree personas was dislike text reaching the *semantic* query; with D1 in place, signed lexical adds conflict gains without the agree cost.

**Next:** fix the two routing bugs (new routing version, checked on the calibration probes only), then D2. Candidate default after D2 and review: D1 + signed lexical, decided by owner.

## Candidate baseline check (9 November 2026)

D1 + signed lexical (`configs/ranking/occupation_semantic_signed_fusion_zscore_bm25_signed_ignore_v2.json`) was verified as a candidate live default: [VERIFICATION.md](../reports/ranking/d1_baseline_check/VERIFICATION.md). It passes the persona (probing on and off), probe, latency, test-suite and live-server checks. Costs: development-profile Recall@50 −.036 [−.067, −.008] (MRR −.007, not significant; from the lexical part), and the "X, just not Y" routing bug. **Owner decision: not promoted yet**; the live default is unchanged and the candidate stays selectable for browser review (`configs/system.semantic-signed-bm25-signed-ignore-v2.json`).

## Routing v3 and D2 built (9 November 2026)

### Routing v3 (`recommender/models/clause_routing.py`; new versions, v1/v2 unchanged)
Used by `bm25_signed@ignore_negative_v3`, `occupation_dual_encoder_signed@finetuned_v2` and D2 (parameters record the same patterns).
- **Scope split:** "X, just not Y", "X, except / apart from / other than / aside from Y", "X just not Y" become two clauses, each routed on its own; "except …" pieces count as dislikes. No split when the tail carries only the cue ("teaching, not as much" stays one negative clause) or for "no doubt / no problem / no worries".
- **"anything / everything but X":** the clause after "anything" is the dislike.
- **"avoid" only in first person** ("I'd rather avoid", "I try to avoid"); "help people avoid debt" stays neutral.
- Pieces stay substrings of the turn, in order, so D1 deletes them in place.

### D2: post-fusion dislike stage (`recommender/models/dislike_demotion.py`)
- New optional ranking-config key `post_fusion: {algorithm: dislike_demotion, variant: …}` (`RankingPipeline`; rejected together with description reranking). Applied to the final ranking after fusion; the map snapshot's fused ranks include it; diagnostics `dislike_demotion` list targets and demoted jobs.
- **Target:** negative clause (extractor activity text when explicit) through the frozen BM25 tokenizer, minus cue/filler words ("teaching, not as much" → "teaching"; "not a people person" → none).
- **Matchers (variants):** `title_v1` (target and title words share a stem; suffix strip leaving ≥4 letters); `semantic_v1` (cosine of the encoded target to the fine-tuned job vector ≥ .35); `title_semantic_v1` (either, threshold .70).
- **Placement:** matched jobs move below every non-matched job, relative order kept; nothing removed; no target → ranking unchanged (tested).
- **Calibration** (`python -m eval.calibrate_dislike`, `reports/ranking/dislike_d2/calibration/`): calibration probes only; rule fixed beforehand (highest threshold within 0.1 of the best avoided-in-top-10, MRR ≥ base). Chosen: semantic .35; title_or_semantic .70 (title alone is already within tolerance, so it behaves as title).
- Tests: `tests/test_dislike_demotion.py` (14). A stemming bug found by the tests ("labor" → "lab") was fixed before the reported runs; all D2 runs were redone after the fix.

### Results
**Probes** (`reports/ranking/dislike_d2/negation_probes/`; test split read once, after calibration):

| Arm | Test: avoided in top 10 | Test: any avoided | Test: MRR | All: MRR | Narrow MRR | Collateral/control |
|---|---|---|---|---|---|---|
| live default | 1.67 | .75 | .283 | .317 | .333 | unchanged |
| D1 + signed lexical (v2 routing) | 0.50 | .25 | .594 | .614 | .189 | unchanged |
| routing v3 | 0.42 | .17 | .685 | .668 | .294 | unchanged |
| **+ D2 title** | **0.00** | **.00** | **.700** | .678 | .328 | unchanged |
| + D2 semantic | 0.00 | .00 | .683 | .683 | .333 | unchanged |

N01 (owner's case) and N13 ("i love kids but teaching isnt for me") go to 0 avoided; N20 ("I like art, just not music") recovers from rank 338 to 2–3 (routing v3). Semantic over-demotes near neighbours: N17 ("cant stand cybersecurity") best acceptable 29 → 65; in the owner's session it also demotes School Psychologists.

**Personas** (`reports/ranking/dislike_d2/run_20261109_personas/`, probing on): routing v3 and every D2 variant have **identical MRR/Hit@5** to D1 + signed lexical v2 (.770; − live default +.090 [−.037, +.226]); no persona's best acceptable rank changes. Title D2 changes nothing at all; semantic D2 lowers NDCG@10 (conflict .661 → .617) by demoting some lower-ranked acceptable jobs.

**Owner's session `anon-ebffdc4a` replayed:** live default 10 teaching jobs in the top 10 with Psychology Teachers #1; routing v3 1 (still #1); **D2 title 0** (top: School Psychologists, Neuropsychologists, I-O Psychologists, Clinical and Counseling Psychologists); D2 semantic 0 (School Psychologists also demoted). Latency 17–29 ms per ranking.

**Reading:** D2 title is the safer choice: it enforces named dislikes with no threshold, no measured cost on personas, and no collateral on the probes. Semantic catches dislikes that name no title word but demotes neighbours. Not promoted; browser configs `configs/system.signed-v3.json`, `configs/system.signed-v3-dislike-title.json`, `configs/system.signed-v3-dislike-semantic.json`.

## Probe review and two-tier D2 (9 November 2026)

### Probe set reviewed
Owner review with `python -m devtools.review_probes` (answers: `data/eval/negation_probes_v1_review.jsonl`): 23 of 24 probes approved as drafted. **N07** corrected: device/technical roles (Orthotists and Prosthetists, Anesthesiologist Assistants, Low Vision Therapists) are not "working with patients" (55 → 52 avoided; `avoid_except` in the builder). The set is now marked owner-reviewed. Re-running the calibration on the reviewed set (`reports/ranking/dislike_d2/calibration_reviewed/`) gives the same thresholds (.35, .70).

### Dislike strength (owner note on N01)
"The avoided jobs are correct, but that should mainly reduce the scoring, not necessarily completely remove. Remove only if the user's phrasing is very strongly negative." Implemented as new variants; v1 (binary) unchanged:
- **Strength from wording** (`strength_tiers` in the parameters): mild patterns first (not as much, not really, not a fan, rather not, don't really, don't care much, not particularly, ...), then strong (hate, never, can't stand, not for me, don't want / like, anything but, just not, except, absolutely not, avoid, ...); anything else counts as **mild**.
- **Strong:** matched jobs move below every other job (as v1). **Mild:** the fused score drops by a penalty in standard deviations of that query's fused scores, then the list is re-sorted. A job matched by both gets the strong treatment.
- `title_v2` / `semantic_v2` / `title_semantic_v2`: mild penalty **1.0 SD** (first principles; not calibratable, since the probes do not say how far a mild dislike should move a job). In the owner's session Psychology Teachers stayed #1: it leads by 1.06 SD.
- `title_v3` / `semantic_v3` / `title_semantic_v3`: mild penalty **1.5 SD** (owner decision). Recorded as `mild_penalty_calibrated: false`.

### Results (`reports/ranking/dislike_d2_tiers/`)
- **Probes (reviewed set):** v2 and v3 equal v1 everywhere except N01, the one mild-wording probe, where one teaching job may now stay in the top 10 (explicit-kind avoided 0.25 instead of 0). Test split unchanged: 0 avoided in top 10; MRR .700 (title), .683 (semantic). Collateral and control probes unchanged.
- **Personas:** every D2 version (v1, v3, title or semantic) has MRR .770 and Hit@5 .96, identical to routing v3 without D2 (− live default +.090 [−.037, +.226]); v3 − v1 +.000. NDCG@10: title unchanged (.620); semantic v1 .595, v3 .610.
- **Owner's session `anon-ebffdc4a`** ("teaching, not as much" is mild): title v1 Psychology Teachers #332 (removed); title v2 #1; **title v3 #2** behind School Psychologists, the only teaching job in the top 10; semantic v3 #1, because semantic matching also penalises School Psychologists, so the order at the top does not change. With strong wording ("teaching is not for me") every version moves it to #332.

**Reading:** `dislike_demotion@title_v3` follows the owner's rule (mild lowers, strong removes) with no measured cost. Semantic matching is weaker here: it penalises near neighbours too. Not promoted. Browser: `configs/system.signed-v3-dislike-title-v3.json` (recommended), `configs/system.signed-v3-dislike-semantic-v3.json`.

### Lexical ablation under D2 title v3 (9 November 2026)

See [reports/ranking/lexical_ablation_d2/README.md](../reports/ranking/lexical_ablation_d2/README.md). Lexical helps conflict personas and probes, and costs agree personas and the development profiles; without lexical, D2 title v3 beats the live default on every benchmark with no significant loss.
