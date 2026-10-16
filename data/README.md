# CS425 career matcher: data package (v3, 8 Oct 2026)

All text in this package was drafted by an LLM and checked by blind LLM readers. A human spot-check by the team is still needed before training.

## Files

| File | Rows | Used by | Notes |
|---|---|---|---|
| occupations.csv | 1,016 | Scoring, evaluation | Raw RIASEC scores (1–7) and z-scores computed over the 923 rated occupations. Also `job_zone`, alternate titles and `bm25_doc`. **Recommend only job_zone 4–5 (378 occupations).** |
| tasks.csv | 18,838 | Dense index, quoted evidence | Task statements |
| related_occupations.csv | 18,460 | Metrics | O*NET Related Occupations: 5 Primary-Short, 5 Primary-Long and 10 Supplemental per occupation |
| activities.jsonl | 188 | Source only | O*NET Interests Illustrative Activities, each mapped to a type |
| synthetic_seeds.jsonl | 90 | Source only | The first synthetic seeds |
| extractor.jsonl | 2,102 | Extractor training | See below |
| riasec_lexicon.json | 6×60 terms | Coverage checks, keyword baseline, labelling help | O*NET-derived terms per type (weighted log-odds over task statements), plus strong/mild/negator/familiarity cue lists |
| labelling/ | – | Team | GUIDE.md, real_sentences_template.csv and import_real.py for collecting and merging real student sentences |
| question_bank.json | – | Policy + simulator | Opening, 18 CLARIFY questions, CHALLENGE / PROBE / ABSTAIN / RECOMMEND templates, and ANSWERS with 5 levels per question |
| labels.json | – | All training code | Label indices, class weights (inverse frequency), strength values, input format, NLI binary mapping |
| profiles.jsonl | 100 | **Final evaluation only** | Fresh v3 test set. Never tune on it. |
| dev_profiles.jsonl | 60 | Tuning | Same format as the test set (20 original + 40 added 9 Oct from fresh targets) |
| nli_pairs.jsonl | 139 | Contradiction-check validation | dev 75 (28 contradiction, 33 elaboration incl. 20 hard, 14 unrelated); test 64 (31 contradiction, 27 elaboration incl. 20 hard, 6 unrelated). Field `source` is hand / hard / profiles / dev_profiles. |
| nli_pairs_blind.jsonl | 60 | Human sort check | The hand-made pairs with no labels |
| eval/simulator.py, eval/metrics.py, eval/lexicon_baseline.py | – | Evaluation harness | Reference simulator and metrics, plus a keyword baseline (test macro-F1 0.48) |
| scripts/ | – | Rebuilding | Every script that produced the files above |

## extractor.jsonl
- **Splits:** train 1,488 / val 304 / test 310. Rows are assigned by `seed_id`, and no seed appears in two splits.
- **Fields:**
  - `text`
  - `context`: the bot's question, for short replies
  - `slots`: per type, the **direction**: positive, neutral or negative. Neutral means no stance, including "not sure".
  - `strength`: per non-neutral type, **how much**, from 0 to 1: strong 1.0, mild 0.55, don't mind 0.2 ("don't mind" is a weak positive)
  - `strength_label`: the same as words (strong / mild / dontmind)
  - `reason`: enjoyment, familiarity or neither
  - `kind`, `split`
  - `weight`: 0.5 for bare O*NET activity phrases, which appear only in train
- **Training:** the input is a tokenizer sentence pair `tokenizer(context, text)` when `context` is set, else `tokenizer(text)` (max length 128). Use two heads per type. The **direction head** (768→6×3, softmax, cross-entropy on `slots`, class weights [0.41, 2.95, 4.99] (index order neutral, positive, negative — see labels.json)) gives the class and its **confidence** (softmax probability of the chosen class). The **strength head** (768→6, sigmoid, MSE on `strength`, non-neutral types only) gives **how much**. Multiply each row's loss by `weight`. Tracker value = sign × strength, and merge weight = confidence. Check on test that the mean predicted strength is about 1.0 / 0.55 / 0.2 per label.
- **Conventions:**
  - "don't mind / either is fine" counts as a weak positive (dontmind).
  - "not sure / depends" counts as no stance.
  - Familiarity rows carry the stated lean as their slot label, and `reason: familiarity`.

- **8 Oct additions:**
  - 30 strong Social/Artistic dislikes
  - 60 long multi-clause sentences (25–40 words)
  - 48 sentences covering O*NET terms the data missed
  - a second, rule-guided strength pass (95% agreement with the first pass; the rule-guided pass is used)

## Profiles (v3)
- **Targets:** 120 new occupations, none of them used in any earlier check. They come from Job Zones 4–5 and are stratified by each job's raw dominant type (R 18, I 18, A 9, S 18, E 18, C 19 in the test set).
- **People are not job averages:** `hidden_slots` = the target's z-row + N(0, 0.3). `target_profile` holds the job's own z-row.
- **Style:** a third of Tier 1 profiles are short, messy first turns (`style: messy`). A quarter mention a background that matches a genuine interest (`background: true`).
- **Tier 2 (40):** the writers never saw the job title. Each profile uses one of five plans:
  - `contradiction_like`: the genuine side is a like
  - `contradiction_dislike`: the genuine side is a dislike
  - `familiarity_misleading`: exposure to a low-scoring type
  - `familiarity_genuine`: background that the student really does enjoy, so probing should keep it
  - `both`
- **Simulator:**
  - CLARIFY is answered by z band, with five levels.
  - CHALLENGE returns `resolutions[slot]`.
  - **Only PROBE returns `latent_reveal`.**
- **Metrics (eval/metrics.py):**
  - hit@5 (headline)
  - hit@3
  - related-hit@5 (target or a Primary-Short relation)
  - twin-hit@5 (target or a near-identical RIASEC job)
  - MRR
  - NDCG@10, with grades target 4 / Primary-Short 3 / Primary-Long 2 / Supplemental 1, and IDCG computed over codes in the pool

## Checks run (blind LLM readers)
| Check | Result |
|---|---|
| Training labels | 96–100% agreement, 0 polarity flips |
| Profile texts vs hidden scores | r = 0.89 (full), 0.78 (messy), 0.93 (Tier 2); 0 of 720 slots outside their band |
| Resolutions and reveals | 747 of 748 consistent with the hidden scores and planted design |
| Target vocabulary leak (TF-IDF on opening text only) | hit@5 is 0.10 / 0.00 / 0.00 for full / messy / Tier 2 |
| Information ceilings | Perfect knowledge of the person: hit@5 0.82, related-hit@5 0.90 |

## Known limits
- Everything is LLM-drafted.
- Profiles are simulated people built from job averages plus noise.
- Contradiction plants: 16 have a liked genuine side and 8 a disliked one.
- Background words appear in 20/24 familiarity plants and in 19/76 other profiles (up from 7/76 after neutral background sentences were added to 18 Tier 1 texts).
- Only 9 Artistic-dominant degree-level jobs were available.
- NLI test: 31 contradictions and 33 non-contradictions (20 of them hard elaborations). The unrelated class is small (6), so report the binary contradiction-vs-not result.

## 9 Oct: second deep review (confidence/strength scheme)
- Blind readers relabelled 1,843 rows for presence and strength:
  - presence F1 0.965
  - strength: 94.7% same level, 100% within one level, 0 flips
- 243 disputed rows went to a third reader, and the majority vote was applied. Two-type activities and O*NET's own Career Type labels were kept as designed.
- Convention:
  - **indifference** ("neutral", "no strong preference", "just okay", "don't mind") = **+0.2**
  - **uncertainty** ("not sure", "depends", "idk") = neutral (no stance)
- question_bank.json: the 36 strong answers were rewritten to be emphatic, and all 90 answers now read at their intended level (90/90).
- Profiles: 663/663 type slots within band, and resolutions 720/720 consistent.

## 9 Oct: final extractor scheme (team decision)
Direction (positive / neutral / negative) with a **confidence**, plus a separate **strength** from 0 to 1 for non-neutral types. This replaces the earlier `soft` and `present`/`liking` fields. The labels are the same as those checked in the second deep review.

## 9 Oct: training-readiness review
- **labels.json** (new) holds the label indices, inverse-frequency class weights (direction [0.41, 2.95, 4.99], reason [0.4, 3.44, 4.55]), the strength values, the input format (tokenizer sentence pairs, max length 128), the NLI binary mapping and the job-zone filter.
- **dev_profiles.jsonl** goes from 20 to 60. The 40 new profiles use fresh targets that were never used before, are built the same way, pass the blind band check (0 of 960 slots outside band across test and dev), and have resolutions written blind.
- **nli_pairs.jsonl:**
  - +40 hard elaborations: opposite-sounding statements that are compatible, all confirmed by a blind sort (70/70)
  - +15 dev contradictions taken from the dev profiles (dev now has 28 contradictions in all)
- **Learning curve** (stand-in model, 25→100% of train): direction F1 0.54→0.67, strength Spearman 0.39→0.55, reason F1 0.66→0.91. Reason has levelled off; direction and strength would still gain from more data, especially real student sentences.
- **Reporting:** some val/test cells are small (I-negative val 14; don't mind 18–20 per split). Report strength calibration pooled over types, and give bootstrap confidence intervals.
- eval/simulator.py: the CLARIFY variant now wraps around (variant % 3).

## 9 Oct: second alignment review (build manual revision 33)
- The build manual was checked line by line against this package and updated to match it (file names, columns, counts, the 378-job pool, training code, metrics, simulator).
- eval/simulator.py: ABSTAIN with a slot now returns the CLARIFY answer for that slot and variant. To retry a CHALLENGE, call `answer("CHALLENGE", slot)` again.
- eval/metrics.py: in `ndcg_at`, codes outside the pool earn nothing, so NDCG stays at or below 1 even if a ranking includes them.
- Type keys: letters (R…C) in slots, strength, hidden_slots, resolutions and z columns; full names in question_bank.json, contradiction.slot and nli_pairs.slot.
