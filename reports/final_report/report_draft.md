# Career Matcher: A Task-Oriented Dialogue System that Separates Interest from Familiarity

CS425 Final Group Report · Chen Hua-En, Koon Wei Heng, Yap Xinyan, Leong An Yu, Chua Qihan

[TO CHECK: team roles; the deck's slide and speaker notes assign them differently. Add a one-line role table here if the course requires one.]

---

## 1. Introduction

### 1.1 Motivation

Students who are choosing a career often say two things. The first is "I cannot find a job related to my degree. What other jobs suit me?" The second is "I studied marketing, but I am not sure I really like it." In the first case the student needs options beyond their degree. In the second, what the student knows is not the same as what they enjoy.

Existing career tools take stated preferences at face value. The O*NET Interest Profiler scores 60 like/dislike ratings as given. GovTech's Career Kaki tailors advice to a profile, skills and goals. NUS AI-Xplore uses what users share about their interests. None of their public descriptions mention checking where an interest comes from. A student who "likes sales" because a parent runs a sales company would be steered towards sales jobs.

Two findings suggest that asking follow-up questions should help. Brown, Ryan Krane et al. (2003) found that career interventions with individualised interpretation and feedback have much larger effects (mean effect size 0.99 with three critical ingredients, against 0.22 with none). Aliannejadi et al. (2019) showed that asking a single good clarifying question improved precision@1 by over 170% in an oracle experiment on open-domain information seeking.

### 1.2 Goal and hypothesis

We built **Career Matcher**, a task-oriented dialogue system that recommends careers a student enjoys, not only careers they are familiar with. In at most ten turns it:

- reads each RIASEC interest the student expresses, together with the *reason* behind it (enjoyment or familiarity);
- **probes** interests that rest on familiarity;
- **challenges** statements that contradict earlier ones;
- ranks all 378 degree-level O*NET occupations and recommends the top five, each with quoted O*NET task statements as evidence.

**Research question.** Does probing and identifying contradictions improve the accuracy of job recommendations, compared with accepting the stated profile as given?

**Hypothesis.** On the same simulated students, the full system will read hidden familiarity and contradiction traps correctly more often than a no-probing control, and will rank the student's true job higher.

### 1.3 Background

**RIASEC and O*NET.** Holland's theory describes vocational interests with six types: Realistic (hands-on work with tools and machines), Investigative (ideas and research), Artistic (creating and self-expression), Social (helping and teaching), Enterprising (leading and persuading) and Conventional (data, records and procedures). O*NET, the US Department of Labor's occupational database (version 31.0, 1,016 occupations, CC BY 4.0), gives each occupation a score from 1 to 7 on each type. Career Matcher builds the same six scores for the student and matches the two profiles. RIASEC was chosen because it gives the student and the job the same six-number representation.

**Task-oriented dialogue.** The system follows the traditional modular architecture taught in Week 11: natural-language understanding, dialogue state tracking, a dialogue policy and natural-language generation, with the ranker in the role of the knowledge base. There is one domain and one intent (career advice), so no domain or intent classifier is needed. Slot filling is framed as classification rather than BIO tagging, because a stance ("I'd rather not deal with customers") is not a span of words.

### 1.4 Differences from existing work

1. **Reason-aware slots.** Each slot stores a signed value, a strength and a *reason* (enjoyment, familiarity or neither). Existing profilers store only like/dislike.
2. **Probing and challenging as dialogue acts.** The policy has explicit PROBE and CHALLENGE acts, driven by the extractor's reason head and an NLI model fine-tuned on "tensions worth clarifying" rather than strict logical contradiction.
3. **Hybrid ranking.** The ranker fuses the RIASEC profile with what the student actually said: a fine-tuned semantic encoder and a negation-aware BM25 lexical matcher. Explicit dislikes are demoted after fusion.
4. **Controlled evaluation of probing.** Every simulated conversation is run twice, with and without probing, on occupations the semantic encoder never saw during fine-tuning.

---

## 2. System Overview

### 2.1 Architecture

Each user turn passes through seven steps (Figure 1; `pipeline/system.py`, `System.step`):

| Step | Component | Type | What it does |
|---|---|---|---|
| 1 | Extractor (NLU) | Fine-tuned RoBERTa-base | Reads each sentence into signed RIASEC slot readings, each with strength, confidence and reason |
| 2 | State tracker | Rule-based | Merges readings into a six-slot frame; holds opposite-sign readings as conflict candidates |
| 3 | Contradiction rule | Rule-based | Flags the same slot with opposite signs when both readings have confidence > 0.4 |
| 4 | NLI confirmer | Fine-tuned DeBERTa-v3 cross-encoder | Confirms a flag only if P(contradiction) between the two source sentences > 0.90 |
| 5 | Ranker | RIASEC cosine + fine-tuned MiniLM + signed BM25, z-score fusion, dislike demotion | Ranks all 378 eligible occupations every turn |
| 6 | Policy | Rule-based | Chooses CLARIFY, PROBE, CHALLENGE, ABSTAIN or RECOMMEND |
| 7 | NLG | Fine-tuned T5-small, with question-bank fallback | Writes the question; code inserts the user's exact quotes |

[FIGURE 1: pipeline diagram, from deck p. 10]

Steps 2–6 form the dialogue manager, and steps 2–4 are dialogue state tracking. The output each turn is a reply. On RECOMMEND the output is five job cards, each with a "why" line, the O*NET description and up to three task statements. Every turn is logged as one JSON line with the profile, the readings, the NLI scores, the policy's reason, the top five jobs and the reply source.

The system is served by a FastAPI backend (`app/server.py`) with a plain HTML/JavaScript chat page. A researcher panel shows the RIASEC bars, the action log and the current top five.

### 2.2 Shared data: the O*NET catalogue

All components share one occupation catalogue, built from O*NET 31.0 by `data/scripts/build_onet.py`:

- **Join.** Occupation Data, Interests, Task Statements, Job Titles and Job Zones are joined on the O*NET-SOC code, giving 1,016 rows in `data/occupations.csv`.
- **Normalisation.** Raw interest scores are lopsided: the mean over all jobs is 4.57 for Realistic and 4.53 for Conventional, but only 1.94 for Artistic. Each type is therefore z-scored across the 923 jobs that have interest data (population SD). After this, 0 means "a typical job", which matches the student profile, where 0 means neutral and negative means dislike. Example, School Psychologists: raw R/I/A/S/E/C = 1.00/5.00/3.31/6.77/2.82/4.01 becomes z = −1.85/+0.97/+1.14/+2.19/0.00/−0.52.
- **Eligibility.** Only Job Zones 4–5 (jobs needing a degree) with interest data are kept: 378 occupations. All recommendations and test targets come from this pool.
- **Text.** 8,243 task statements (of 18,838) for the 378 jobs, Work Context (57 conditions) for 372 of them, and a `bm25_doc` per job (title, alternate titles, description and tasks).
- **Related occupations.** O*NET Related Occupations (up to 20 per job, in three tiers) give partial credit in evaluation.

### 2.3 Rule-based components

**State tracker** (`pipeline/tracker.py`). A new reading with value v = sign × strength and confidence c is merged into the slot as a confidence-weighted average: value = (v_old·w_old + v·c) / (w_old + c). The old weight can be decayed per turn (`decay`). The live setting is 1.0, meaning no decay. Values stay in [−1, 1]. A reading whose sign is opposite to the current slot is not merged. It is held as a conflict candidate until the contradiction check decides. An answer to a PROBE or CHALLENGE replaces the slot. A familiarity answer such as "it's just what I know" resets the slot to 0.

**Dialogue policy** (`pipeline/policy.py`). The policy applies these rules in priority order:

| Act | Fires when |
|---|---|
| ABSTAIN | The answer to a question carried no readable stance. Re-ask once in simpler words (or with an example), then settle the slot as "no clear preference". |
| RECOMMEND | Turn 10, or from turn 5 if all six types are covered, every value moved ≤ 0.10 over the last three answers, and nothing is left to challenge or probe. |
| CHALLENGE | The NLI model confirmed a contradiction. Quote both statements and ask which is closer. |
| PROBE | A slot's reason is familiarity with reason confidence > 0.6, and it has not been probed yet. |
| CLARIFY | Default. Ask about the uncovered type whose z-scores vary most across the current top 100 jobs. From turn 3, use an O*NET task question where one fits. |

A contradiction that is still unresolved at turn 10 is kept at half confidence. In the no-probing control used for evaluation, PROBE and CHALLENGE are disabled and opposite statements are merged immediately.

### 2.4 How to run the system

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-models.txt -r requirements-ranking.txt
# Model folders are not in git; copy the shared folders into place:
#   models/extractor/   models/nli-probing-v2/checkpoint/   models/t5-questions-v3b/
#   models/occupation_semantics_v1/   (semantic encoder checkpoint + index)
.venv/bin/python scripts/check_nli_checkpoint.py --smoke   # verifies the NLI checkpoint hash
.venv/bin/python run.py                                    # web app on http://127.0.0.1:8000
.venv/bin/python chat_cli.py                               # terminal chat
.venv/bin/python -m unittest discover -s tests             # test suite
```

`CM_CONFIG` selects an alternative configuration, for example `configs/system.cosine.json` for the RIASEC-only ranker. `CM_STUDY_MODE=1` hides the researcher panel for participants. `python -m devtools.run_mock serve` runs the whole dialogue without any model files.

The four modelled components (extractor, contradiction NLI, ranker, NLG) are described below with the same structure: (1) data and pre-processing, (2) model and training, (3) key hyper-parameters, (4) how to run, (5) evaluation.

---

## 3. Extractor (NLU)

### 3.1 Data and pre-processing

`data/extractor.jsonl` contains **2,102 labelled sentences**. The seeds are O*NET's 188 Interests Illustrative Activities (for example "Build kitchen cabinets") and 90 synthetic seeds. These were expanded into rephrasings, negations, conversational frames, short question–answer pairs and combinations (seed 913, rephrase 376, frame 249, QA 216, negation 188, combo 160). An LLM drafted the text to cover dislikes, hedges, familiarity and short replies. Blind LLM readers then checked the labels: stance F1 0.965, and strength 94.7% exact and 100% within one level. Disputed rows were settled by a third reader.

Each row is labelled per RIASEC type:
- **Direction:** positive, neutral or negative.
- **Strength**, for non-neutral types: strong 1.0, mild 0.55, or "don't mind" 0.2.
- **Reason:** enjoyment, familiarity or neither.

Short replies carry the bot's question as `context` (216 rows). For example, "I'd love to build kitchen cabinets from scratch with my own hands" is labelled R positive, strong, enjoyment, and "I never took D&T in secondary school so tools aren't really my thing" is R negative, mild, familiarity.

The split is by seed, so rephrasings of one sentence never cross splits: **train 1,488 / validation 304 / test 310**. A separate stress set of 400 hard phrasings (`data/extractor_stress.jsonl`; indirect interest, ambivalence, informal text and others) was never used for training.

At run time a message is split into sentences, and fragments of six words or fewer are joined to the previous sentence. Each sentence is read separately, and for each type the most confident non-neutral reading is kept. A reply is first read on its own. The bot's question is attached as context only for short CLARIFY replies that name nothing.

### 3.2 Model and training

The extractor is a pre-trained encoder with a custom multi-task head (`pipeline/extractor.py`, `CustomConcatWithLayer`):

- **Pooling.** Mean-pool each of the last two hidden layers, then concatenate them (1,536 dimensions).
- **Head.** Dropout 0.1 → Linear 1,536→512 → ReLU → Dropout 0.2 → Linear 512→45.
- **Output split.** The 45 logits form three heads:
  - 18 direction logits (6 types × negative/neutral/positive);
  - 3 reason logits;
  - 24 strength logits (6 types × inactive/don't mind/mild/strong).
- **Loss.** Direction loss, plus weighted reason and strength losses, with label smoothing.

We selected the encoder, the head size and the pooling on the validation data:

| Experiment | Options (overall accuracy) | Chosen |
|---|---|---|
| Encoder (same head) | roberta-base 91.4%; deberta-v3-base and bert-base-uncased 89.3% / 85.8% [TO CHECK: the deck's model labels for these two rows are swapped]; distilbert-base-uncased 81.0% | **roberta-base** |
| Head hidden size | 768: 86.4%; **512: 88.4%**; 256: 84.0%; 128: 83.1% | **512** |
| Pooling over last n layers | mean: 90.5 / 90.5 / 91.4 / 91.7% (n = 1–4); concatenation: 90.5 / **92.3** / 90.8 / 92.0% | **concatenate last 2** |
| Loss weights | reason 0.2, strength 0.8 performed best overall | 0.2 / 0.8 |

The reason head was close to 100% for every encoder. The differences came mostly from strength.

### 3.3 Key hyper-parameters

Beyond the architecture choices above: learning rate 3e-5 for the encoder and 1e-3 for the head, 50 epochs, warm-up ratio 0.1, label smoothing 0.15, and a maximum length of 128 tokens. Direction and reason losses use inverse-frequency class weights ([0.41, 2.95, 4.99] and [0.40, 3.44, 4.55], from `data/labels.json`). The data-build seed is 425.

### 3.4 How to run

- **Training.** Training was done in a separate notebook workflow [TO CHECK: add the notebook path or commit it under `training/extractor/`]. The data contract (labels, class weights, input format) is in `data/labels.json`.
- **Inference.** The app loads the checkpoint from `models/extractor/` at start-up (`pipeline/registry.py`).
- **Checks:**
  ```bash
  .venv/bin/python scripts/check_m5_extractor.py --checkpoint models/extractor   # smoke test with the real ranker
  .venv/bin/python training/extractor/context_leak_check.py                    # does the question leak into readings?
  .venv/bin/python data/eval/lexicon_baseline.py test                           # keyword baseline
  ```

### 3.5 Evaluation

On the test split, a keyword lexicon baseline reaches direction macro-F1 0.48 and a TF-IDF stand-in 0.67. The final RoBERTa extractor reaches **96.7%** RIASEC-slot accuracy, **100%** reason accuracy, **93.8%** strength accuracy and **92.9%** overall.

[TO CHECK: the deck does not say which split the final-model accuracies are measured on, and they are accuracies rather than macro-F1, so they are not directly comparable with the baselines. Re-run the checkpoint on the test split with the same metric.]

Qualitative behaviour (real outputs):

| Input | Reading | What it shows |
|---|---|---|
| "Kept the screws from the old chair in a jar, just in case." | R +0.55 | Picks up implicit interest from context |
| "Just another boring weekend fixing the bike engine again." | E −0.2 (don't mind) | Strength follows the wording |
| Same sentence prefixed with "Ugh" | E −1.0 (strong) | |
| "I wanna be the very best Pokemon gotta catch them all" | E +1.0 | **Failure:** confident labels on off-topic text |
| "This is ridiculous, I hate candy and clouds" | A and C −1.0 | **Failure** |

The off-topic failure is one reason the system has ABSTAIN and an NLI check before it challenges anything. The installed checkpoint's confidence scores are also not temperature-calibrated (all temperatures 1.0), and these confidences feed the tracker's merge weights.

---

## 4. Contradiction Detection (Rule + NLI)

### 4.1 Data and pre-processing

`data/nli_natural_v1.jsonl` has **688 sentence pairs**: train 407, validation 138, test 143 (79 contradiction, 64 non-contradiction). Each pair (a, b) is labelled contradiction, elaboration or unrelated, and the labels are collapsed to contradiction vs not. The data consists of:

- **175 earlier evaluation pairs:**
  - 60 hand-written pairs;
  - 40 "hard" pairs: opposite-sounding statements that are actually compatible, such as "Leading a team is fine for me" / "I'd rather be a strong contributor than the team lead";
  - 75 pairs drawn from simulated student profiles.
- **13 further hand-written pairs.**
- **500 generated pairs** in 50 groups, covering six patterns:
  - interest vs a different career (250);
  - specialisation dislike, want vs related dislike, same-domain relative preference, intensity or conditions, and interest vs ability (50 each).

The labelling policy is deliberately broader than standard NLI. A cross-domain tension worth clarifying counts as a contradiction. For example, "I have a soft spot for designing posters" against "For a living, I'm more drawn to physiotherapy".

Leakage guards (`scripts/finetune_nli.py`): no pair, or its reverse (casefolded, whitespace-normalised), may appear in two splits, and no generation group may cross splits.

At run time the **premise is the student's earlier sentence and the hypothesis is the current sentence**, both taken verbatim from the extractor's source text. Short replies to a question are first restated by a template. For example, "Not really" to "Would you enjoy repairing machines?" becomes "I do not like repairing machines."

### 4.2 Model and training

- **Stage 1, rule.** The cheap rule flags a candidate when the same slot has readings of opposite sign with both confidences > 0.4. A flip right after a PROBE counts as the answer to the probe, not as a contradiction.
- **Stage 2, NLI.** Only flagged candidates go to the model. It is `cross-encoder/nli-deberta-v3-base`, a 3-way NLI cross-encoder. The pair is encoded together and the system uses the softmax P(contradiction).
- **Fine-tuning.**
  - Objective: binary negative log-likelihood of P(contradiction) against P(entailment) + P(neutral), keeping the original 3-way head.
  - Optimiser: AdamW (weight decay 0.01) with gradient clipping at 1.0.
  - Checkpoint selection: the epoch with the lowest validation loss.

We compared four adaptation methods on the 143 held-out test pairs:

| Method | Accuracy | Log loss |
|---|---|---|
| Stock model (vanilla) | 77.6% | 1.161 |
| Temperature scaling | 77.6% | 0.470 |
| Frozen encoder (head only) | 76.9% | 0.745 |
| **Full fine-tune** | **91.6%** | **0.389** |

Full fine-tuning cut false alarms from 18 to 1. Decisions improved only when the encoder itself was updated.

### 4.3 Key hyper-parameters

The learning rate was chosen by 5-fold grouped cross-validation on the 545 train + validation pairs, without touching the test set. Across 1.25e-5, 1.50e-5, 1.70e-5 and 1.75e-5, CV log loss fell steadily (0.215 → 0.188 → 0.173 → 0.158), and 1.75e-5 was best, at 96.5% CV accuracy at the 0.90 threshold. Raising the confirm threshold from 0.60 to 0.90 removed 2 false alarms with no extra misses.

Other settings: 2 epochs (epoch 1 selected by validation loss), batch size 8, maximum length 512, seed 425. The rule cut-off (> 0.4) and optimiser settings are given in Section 4.2.

### 4.4 How to run

```bash
# fine-tune (output directory must not exist)
.venv/bin/python scripts/finetune_nli.py train --data data/nli_natural_v1.jsonl --output <dir> \
    --epochs 2 --batch-size 8 --learning-rate 1.75e-5 --seed 425 --allow-unreviewed
# stock vs tuned on the test split (thresholds are fixed on dev before test inference)
.venv/bin/python scripts/finetune_nli.py compare --data data/nli_natural_v1.jsonl --tuned <dir>/checkpoint --output <dir2>
# deploy: copy the checkpoint to models/nli-probing-v2/checkpoint/, then
.venv/bin/python scripts/check_nli_checkpoint.py --smoke
```

### 4.5 Evaluation

- **Test accuracy.** Full fine-tuning reaches 91.6% on the 143 test pairs, against 77.6% for the stock model.
- **Typical fix.** For the poster/physiotherapy pair, the stock model gives P(neutral) = 99.65%. The fine-tuned model gives P(contradiction) = 99.87%.
- **Compatible pairs pass.** "I like coding" / "I hate front-end" scores 0.87, below the 0.90 threshold, so no CHALLENGE is raised.

**Limitations.**
- **Two stances in one message.** The rule compares readings across turns of the same RIASEC slot. When one message holds two stances, the contradiction can be missed. For example, a pilot user said "I would love being part of the team, but I do not enjoy making decisions", then later said "yes" to leading a team and making decisions, and no CHALLENGE fired.
- **Contradiction traps.** In the end-to-end test, only 42% of contradiction traps end with the correct sign (Section 7). The end-to-end chain (extractor → rule → NLI → student's resolution) is weaker than the NLI model alone.

---

## 5. Ranker

### 5.1 Data and pre-processing

- **Catalogue.** The 378 eligible occupations with z-scored RIASEC profiles (Section 2.2).
- **Encoder fine-tuning data** (`training/occupation_semantic.py`, `configs/occupation_semantic_v1.json`):
  - Sources: O*NET task statements and Work Context. Each job's most common answer to a work condition becomes a short sentence.
  - Size: 29,074 source units, reduced to 7,500 unique statements.
  - Grouping: near-duplicate groups are formed with character 3–5-gram TF-IDF cosine ≥ 0.85, and groups never cross splits.
  - Split: 70/15/15, giving train 3,470 / validation 1,598 / test 1,133 groups.
  - Held-out occupations: **38 occupations are held out entirely**, and none of their 903 groups is seen in fine-tuning. These 38 occupations are later the targets of the end-to-end test.
  - Pairs: each training pair is (task or context statement → the owning occupation's document). The document is the title, description, tasks and context, capped at 24/64/112/48 wordpieces, with the query's own group removed.
- **Query pre-processing** (clause routing, `recommender/models/clause_routing.py`):
  - The user's turns are split into clauses on punctuation and on "but / however / although / because".
  - Each clause is routed by stance:
    - Positive and neutral clauses form the query.
    - Negative clauses (explicit dislikes, negation cues including "dont/cant", "avoid", "X, just not Y", "anything but X") are removed from the query and sent to the dislike stage.
    - "Don't mind / not sure" clauses are ignored.

### 5.2 Model and architecture

Three components score every eligible occupation:

1. **M1 RIASEC cosine.** Cosine between the tracked profile and the occupation's z-scores, over the observed types only. No training.
2. **M3 signed BM25.** BM25 over `bm25_doc`, using only the positive and neutral clauses. Tokens are lower-cased with stop-words removed and no stemming.
3. **M4 signed semantic encoder.** A dual encoder (shared weights) fine-tuned from `sentence-transformers/all-MiniLM-L6-v2`. The query is all user turns with negative clauses deleted, truncated at 256 wordpieces. It is scored by cosine against a pre-computed index of the 378 occupations.
   - Loss: multi-positive contrastive softmax (InfoNCE), log Σ_all − log Σ_positives of exp(q·d/τ), with in-batch and 4 random negatives.

**Fusion.** Each usable component's scores are z-scored per query and summed with equal weights (`zscore_sum`). An occupation a component did not score gets that component's minimum z. If only one component is usable, fusion is skipped and that component's ranking is used.

**Why z-score fusion.** Rank fusion (RRF, k = 60) keeps only list positions. With RRF, a flat RIASEC lead overruled a confident semantic match whenever the profile and the student's text disagreed. Z-scores let a peaked component count in proportion to its confidence.

**Dislike demotion.** After fusion, jobs whose title words match a negative clause are demoted:
- strong wording ("hate", "never", "not for me") moves them below every other job;
- mild wording lowers their fused score by 1.5 SD.

### 5.3 Key hyper-parameters

- **BM25:** k1 = 1.5, b = 0.75.
- **Encoder fine-tuning:** learning rate 1e-5, weight decay 0.01, batch size 8 with 4 random negatives, τ = 0.07, up to 3 epochs with early-stop patience 1 (checkpoint chosen by validation MRR), 768 training groups, seed 425.
- **Fusion and demotion:** equal weights and the 1.5 SD mild penalty (Section 5.2) are both untuned.

### 5.4 How to run

```bash
# semantic encoder: each stage refuses to overwrite earlier outputs
.venv/bin/python -m training.occupation_semantic prepare  --config configs/occupation_semantic_v1.json
.venv/bin/python -m training.occupation_semantic train    --config configs/occupation_semantic_v1.json
.venv/bin/python -m training.occupation_semantic index    --config configs/occupation_semantic_v1.json
.venv/bin/python -m training.occupation_semantic evaluate --config configs/occupation_semantic_v1.json
# BM25 needs no build step: the index is built from bm25_doc at load time.
# persona benchmark (ranker x probing)
.venv/bin/python -m eval.dialogue_fusion --probing both
.venv/bin/python -m eval.task_retrieval_fusion
```

The live ranker is selected in `configs/ranking.yaml`. Every earlier variant remains selectable through its own preset in `configs/ranking/`.

[TO CHECK: the negation-probe and dislike-calibration scripts are on branch `feat/ranker-v2` and not yet on `main`. Merge them, or cite the branch.]

### 5.5 Evaluation

**Benchmarks:**
- **25 team-reviewed personas.** 14 are "conflict" personas, whose text points to a different field than their RIASEC profile; 11 are "agree" personas. The metric is the MRR of the first acceptable job, fixed in advance.
- **120 development profiles.** Opening turn, read by the real extractor.
- **Held-out O*NET task retrieval.** 1,133 test groups.
- **24 negation probes.** 12 calibration and 12 test.

Confidence intervals come from paired bootstraps (seed 425).

**Semantic encoder (task-retrieval MRR):**

| Model | Test (1,133) | 38 unseen occupations (903) |
|---|---|---|
| RIASEC cosine | 0.032 | 0.055 |
| Pre-trained MiniLM | 0.483 | 0.364 |
| **Fine-tuned MiniLM** | **0.518** | **0.418** |

The fine-tuned encoder's gain over the pre-trained one is +0.035 [+0.020, +0.049] on the test groups and +0.054 [+0.040, +0.067] on the unseen occupations.

**Fusion, RRF vs z-score (MRR):**

| Benchmark | RRF k = 60 | z-score | Δ [95% CI] |
|---|---|---|---|
| Conflict personas | 0.114 | 0.639 | +0.525 [+0.328, +0.722] |
| Agree personas | 0.743 | 0.733 | −0.010 [−0.192, +0.136] |
| Task retrieval | 0.200 | 0.507 | +0.307 [+0.284, +0.331] |
| Development profiles | 0.096 | 0.064 | −0.032 [−0.059, −0.011] |

**Step-by-step development (persona MRR, probing on):**

| Variant | All | Conflict | Agree | Hit@5 |
|---|---|---|---|---|
| RRF (cosine + semantic) | 0.391 | 0.114 | 0.743 | – |
| z-score fusion | 0.680 | 0.639 | 0.733 | 0.80 |
| + plain BM25 | 0.658 | 0.744 | 0.548 | 0.92 |
| + signed BM25 (negatives ignored) | 0.683 | 0.780 | 0.560 | 0.92 |
| + signed semantic query | 0.770 | 0.780 | 0.758 | 0.96 |
| **+ clause routing v3 + dislike demotion (live)** | **0.770** | **0.780** | **0.758** | **0.96** |

**Live ranker vs the previous z-score default:**

| Benchmark | Before | After | Δ [95% CI] |
|---|---|---|---|
| Persona MRR, probing on | 0.680 | 0.770 | +0.090 [−0.037, +0.226] |
| Persona MRR, probing off | 0.653 | 0.777 | +0.123 [−0.003, +0.258] |
| Disliked jobs in top 10 (negation probes) | 2.79 | 0.08 | |
| Acceptable-job MRR (negation probes) | 0.317 | 0.676 | |
| Development-profile Recall@50 | 0.391 | 0.355 | −0.036 [−0.067, −0.008] |

The development-profile drop is an accepted cost. Those labels are derived from RIASEC, so they reward RIASEC-only ranking. This is why we moved the main benchmark to personas: on RIASEC-derived labels cosine scores 0.117 and semantic-only 0.040, while on O*NET task retrieval the order is reversed (0.032 vs 0.518). An XGBoost learned fusion reached NDCG@5 0.97, but only on synthetic labels that came from the same RIASEC scores, so the result was circular. We kept the untuned z-score sum.

---

## 6. Natural-Language Generation (T5)

### 6.1 Data and pre-processing

The policy decides the act and its focus. Python packs them into a structured prompt, for example:

`career question v2: {action: PROBE, type: Enterprising, focus: "working in sales", context: "My dad works in sales, so that's what I know.", retry: false, previous_question: "", reference: "Is working in sales something you enjoy, or mainly something you already know?", style: direct}`

T5 writes only the question. Code prepends the user's exact quote ("You said '…'."), so quotes can never be paraphrased wrongly. OPENING, RECOMMEND and ABSTAIN repair turns use fixed templates from `data/question_bank.json`.

The training pairs (prompt → question) are synthetic and template-expanded, over unseen background groups. Two training sets were built: a standard set (864 train / 216 validation), and a standard + varied set (1,512 / 378) that adds 648 examples with more diverse wording. The starting checkpoint was fine-tuned in four stages:
1. contextual question generation (882 examples);
2. action refinement (432);
3. task and retry refinement (648);
4. balanced replay (864).

### 6.2 Model and training

The model is **T5-small** (about 60M parameters; 6+6 layers, d_model 512), trained with teacher forcing and cross-entropy, AdamW, and warm-up with linear decay. Every experiment started from the same staged checkpoint.

We compared full fine-tuning with LoRA (294,912 trainable parameters, against 60.5M). The test set was 432 questions (144 each of CLARIFY, PROBE and CHALLENGE), each generated once:

| Training approach | Valid questions / 432 | Focus errors | Retries accepted / 23 | Task questions / 46 |
|---|---|---|---|---|
| Starting checkpoint | 411 (95.1%) | 0 | 19 | 42 |
| **Full FT, 864 standard (selected)** | **415 (96.1%)** | **0** | **21** | **44** |
| Full FT, 864 + 648 varied | 418 (96.8%) | 0 | 19 | 42 |
| Full FT, 216 + 648 varied | 419 (97.0%) | 1 | 18 | 41 |
| LoRA, 864 + 648 varied | 418 (96.8%) | 1 | 19 | 42 |

A *focus error* is a question that changes the required activity, for example asking "Would you enjoy managing a team?" when the focus was "working in sales". We selected the model with the most valid task questions and retries and no focus errors, not the one with the highest token accuracy (all variants were within 69.9–70.3%).

**Run-time safeguards.** Any of the following falls back to the question-bank template:
- a prompt longer than 256 tokens;
- an output that is not a single 5–60-word question;
- an output that repeats the previous question on a retry;
- an output that leaks prompt field names;
- an output that omits the task activity when the question is about a task;
- a PROBE without familiarity wording, or a CHALLENGE without "which / closer / prefer".

Each logged turn records whether T5 or the template produced the reply.

### 6.3 Key hyper-parameters

Learning rate 1e-4, batch size 8, 2 epochs (epoch 1 kept: validation loss 1.810 vs 1.944), maximum input / target length 256 / 64 tokens, seed 42. Decoding uses beam search with 2 beams, no sampling, no_repeat_ngram_size 3 and at most 64 new tokens.

### 6.4 How to run

- **Run-time.** The app loads `models/t5-questions-v3b` when the config has `"filler": "t5-v3"` (the default `configs/system.json`). Template-only mode: `CM_CONFIG=configs/system.question-bank.json`.
- **Wording audit:**
  ```bash
  .venv/bin/python -m eval.t5_wording --version 2 --checkpoint models/t5-questions-v2
  ```
- **Training scripts.** [TO CHECK: they live in the teammate's training project; add the path, or commit them under `training/`.]

### 6.5 Evaluation

The selected model produces a valid question for 96.1% of the 432 test prompts, with no focus errors, and accepts 21 of 23 retries. In the logged sessions, T5 wrote 105 of 120 CLARIFY questions (87.5%) and all PROBE and CHALLENGE questions. The rest fell back to templates.

Example generations:

| Act | Output |
|---|---|
| CLARIFY | "Imagine a job involving talking with people throughout the day; would that appeal to you?" |
| Task CLARIFY | "Would a working day involving teaching staff or customers about electronics equipment standards feel satisfying to you?" |
| Retry | "Would you enjoy analysing financial data?" → "Would spending your working day examining financial patterns feel satisfying?" |

**Limitations.**
- **Stiff wording.** Some outputs are awkward, for example "work involving doing practical work".
- **Quote punctuation.** A quote that ends in a full stop is followed by `".`, so the stop appears twice.
- **Repeated questions.** Two pilot sessions were asked the same question twice. That is a policy issue, not a generation issue.

---

## 7. End-to-End Evaluation

### 7.1 Setup

We tested the whole system on simulated students whose true job is known.

- **Students.** 38 occupations (exactly the 38 held out from encoder fine-tuning) × 3 students × 2 levels of detail = **228 conversations**.
- **Levels:**
  - L1: interests plus background (degree, internship);
  - L2: as L1, plus a career goal.
- **Planted traps.** Student s3 of each job carries one planted trap:
  - **familiarity:** a liked activity that is only familiar, e.g. "I just grew up around my dad's workshop";
  - **familiar but genuine:** familiar *and* enjoyed;
  - **contradiction:** two opposite statements, one of them genuine.
- **Simulated students** are rule-based. They answer CLARIFY questions according to a hidden profile (the target's z-scores plus Gaussian noise, SD 0.3). They answer PROBE and CHALLENGE questions with a scripted resolution. Labels never reach the system.
- **Arms.** Each conversation runs twice, as the full system and with probing off (no PROBE or CHALLENGE). Every component runs live, with up to 10 turns, and all 378 jobs are ranked.
- **Metrics:**
  - true job in the top 5 / top 10;
  - a related O*NET job in the top 10;
  - median rank of the true job;
  - whether the trap slot ends with the correct sign.

  Random choice would place the true job in the top 10 2.6% of the time.
- **Confidence intervals.** 95% CIs come from a bootstrap that resamples the 38 jobs (2,000 resamples, seed 425), so the six conversations of one job stay together.

### 7.2 Overall accuracy

| Metric (full system, n = 228) | Result |
|---|---|
| True job in top 5 | 65% (95% CI 56–73) |
| True job in top 10 | **76%** (CI 69–84) |
| Related job in top 10 | **93%** (CI 87–97) |
| Median rank of true job | **3rd** (CI 2nd–4th) |

By level, the full system places the true job in the top 5 for 51% of L1 and 79% of L2 students.

**Worked example (J28, field + goal; true job Industrial Ecologist).**
- **Opening:** "I can spend hours researching and building models of how a system behaves. I also like helping people, but honestly that's mostly because my family runs programmes at our community club. I'm doing environmental studies at NUS and want to work in sustainability."
- **Full system:** it probes the helping interest. The student replies "Truthfully, I don't really enjoy the people side… I'd rather be researching." Social flips negative, and Industrial Ecologist ranks **#2 of 378**.
- **Probing off:** the trap stays in, teaching jobs enter the top five, and the true job ranks #7.

### 7.3 With vs without probing

| Metric | Group | With probing | Without | Δ [95% CI] |
|---|---|---|---|---|
| Trap read correctly | Trap students (n = 76) | **82%** | 45% | **+37 pts [+21, +53]** |
| True job in top 5 | Trap students | **45%** | 36% | **+9 pts [+1, +17]** |
| True job in top 10 | Trap students | 58% | 57% | within noise |
| Related job in top 10 | Trap students | 84% | 79% | within noise |
| True job in top 10 | All students (n = 228) | 76% | 76% | none |

Trap read correctly, by trap type:

| Trap type | With probing | Without |
|---|---|---|
| Familiarity (26) | 100% | 0% |
| Familiar but genuine (26) | 100% | 100% |
| Contradiction (24) | 42% | 33% |

**Interpretation.** The hypothesis is partly supported.

- **Where probing helps:**
  - It corrects the profile: every familiarity trap is fixed, and genuine interests are left untouched (100% in both arms).
  - For trap students, the true job reaches the top five more often (+9 points, CI excluding zero).
- **Where it does not:**
  - The top-10 and related-job gains are within noise.
  - Students without a trap get the same result either way, so overall accuracy barely moves.
  - Contradiction handling is the weakest part: only 42% of contradiction traps end with the right sign. Detection depends on the extractor reading both statements into the same slot with confidence above 0.4.

### 7.4 Qualitative analysis: example inputs and responses

The table below shows 18 real inputs from logged sessions (six pilot participants and two team test sessions) and the system's response.

| # | User input | Act | System response (abridged) |
|---|---|---|---|
| 1 | "i studied nursing and i enjoy flexible and easy jobs" | PROBE | You said "…". Is helping people something you enjoy, rather than mainly something you already know? |
| 2 | "i already know" | CLARIFY | Would you enjoy following clear procedures as part of your job? |
| 3 | "I love analysing data. My dad runs a sales company so that's what I know." | PROBE | You said "My dad runs a sales company so that's what I know." Is persuading people something you enjoy, rather than simply something you already know? |
| 4 | "It's just what I know." | CLARIFY | (Enterprising reset from +0.55 to 0) Would you enjoy following clear procedures as part of your job? |
| 5 | "hmm, not sure" | ABSTAIN | By following clear procedures, I mean a task such as checking forms against a written checklist. Would you enjoy that activity, dislike it, or feel unsure? |
| 6 | "Actually, I don't enjoy analysing data at all." | CHALLENGE | Earlier you said "I love analysing data." and then "Actually, I don't enjoy analysing data at all." Which statement better describes whether you want work involving doing research? |
| 7 | "I do love analysing data, I was just tired today." | CLARIFY | Would you enjoy installing or repairing network cables as part of your job? |
| 8 | "I like helping people" → later "No, I dont like talking to people" | CHALLENGE | Earlier you said "I like helping people" and then "No, I dont like talking to people". Which statement better describes whether you want work involving helping people? |
| 9 | "Architecture and design" | CLARIFY | Would you enjoy helping or teaching people as part of your job? *(weakness: ignores the opening)* |
| 10 | "hmm sounds like something ai could do for me, but i do like organization" | CLARIFY | Would you enjoy analysing research data using statistical software as part of your job? |
| 11 | "i studied Aircraft systems engineering, i like collaborative work and having team discussion" | PROBE | …Is doing practical work something you enjoy, rather than simply something you already know? |
| 12 | "i hate research" | CLARIFY | Would you enjoy designing graphics and animation as part of your job? |
| 13 | "not really but doable" | CLARIFY | Would a working day involving teaching staff or customers about electronics equipment standards feel satisfying to you? |
| 14 | "it feels alright" | ABSTAIN | That's fair, it's a tricky one. Let me ask differently. How would you feel about teaching staff or customers about electronics equipment standards…? |
| 15 | "i would love being part of the team, but i do not enjoy making decisions" | CLARIFY | Would a working day involving negotiating deals or selling feel satisfying to you? *(two stances in one message; later contradiction missed)* |
| 16 | "i like doing creative things" (+9 short replies) | RECOMMEND | Art Directors; Art, Drama and Music Teachers; Commercial and Industrial Designers; Special Effects Artists and Animators; Video Game Designers |
| 17 | "I studied Politics, Law and Economics … non-profit … event planning, team building and volunteer management" | RECOMMEND | Sociologists; Political Scientists; Social Work Teachers; Fundraisers; Political Science Teachers |
| 18 | "i like flexible jobs" (+9 mostly "no") | RECOMMEND | Compensation, Benefits and Job Analysis Specialists; Industrial-Organizational Psychologists; Compensation and Benefits Managers; HR Specialists; Equal Opportunity Officers |

**What works:**
- Familiarity probes fire on the intended cases (1, 3, 11), and "just what I know" removes the borrowed interest (4).
- ABSTAIN rephrases instead of guessing (5, 14).
- CHALLENGE quotes both statements verbatim (6, 8).
- When students name a field, the recommendations match it (16, 17).

**What does not:**
- PROBE quotes the whole message rather than the relevant clause (1, 11).
- The first CLARIFY ignores the opening message (9).
- One message with two stances escapes the contradiction check (15).
- With no field and mostly negative answers, the recommendations are generic (18).

### 7.5 Pilot with real users

[TO CHECK: the repository logs six participants on 9 Nov 2026; the deck reports 12 or 14 participants with a feedback form. Use the final count and form results here.]

Six participants used the live system (median 2.5 minutes per conversation). All six finished and reached recommendations. In 5 of 6 sessions the top five matched the field the person named (our own reading; participants did not rate the recommendations).

The main observation is that real users reply in one or two words: 73% of replies, averaging 3.1 words. This leaves the extractor and the contradiction check little text to work with. No CHALLENGE fired in the pilot.

---

## 8. Limitations and Future Work

- **Contradictions.** Only 42% of contradiction traps are resolved correctly. Detection needs both statements read into the same slot. Future work: compare clauses within one message, and route NLI over clauses rather than sentences.
- **Extractor calibration.** The extractor is confident on off-topic text, and its confidences are not calibrated. Future work: temperature-scale it on the validation split, and add an "off-topic" output.
- **Synthetic data.** The extractor, NLI and T5 data and the simulated students were LLM-drafted and checked by blind LLM readers, not by humans. The pilot is small and informal.
- **Ranker.** The fusion weights and the dislike penalty are untuned. The encoder reads only 256 tokens, and adding lexical matching costs 0.036 Recall@50 on the RIASEC-derived development profiles.

---

## 9. Conclusion

Career Matcher shows that a modular task-oriented dialogue system can separate interests a student enjoys from interests they are only familiar with:

- **Profile correction.** On simulated students with hidden traps, probing corrects the profile far more often (82% vs 45%).
- **Ranking.** For trap students, probing raises the true job into the top five more often (45% vs 36%).
- **Overall accuracy.** On simulated students who describe their background, the full system places the true job in the top ten 76% of the time and a related job 93% of the time, out of 378 occupations.

The benefit of probing is concentrated in the students who need it. Contradiction handling is the clearest next step.

---

## References

Aliannejadi, M., Zamani, H., Crestani, F., & Croft, W. B. (2019). Asking clarifying questions in open-domain information-seeking conversations. *Proceedings of SIGIR 2019*, 475–484.

Bowman, S. R., Angeli, G., Potts, C., & Manning, C. D. (2015). A large annotated corpus for learning natural language inference. *Proceedings of EMNLP 2015*.

Brown, S. D., Ryan Krane, N. E., et al. (2003). Critical ingredients of career choice interventions: More analyses and new hypotheses. *Journal of Vocational Behavior, 62*(3), 411–428.

Cormack, G. V., Clarke, C. L. A., & Büttcher, S. (2009). Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. *Proceedings of SIGIR 2009*.

He, P., Gao, J., & Chen, W. (2021). DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training with gradient-disentangled embedding sharing. *arXiv:2111.09543*.

Holland, J. L. (1997). *Making Vocational Choices: A Theory of Vocational Personalities and Work Environments* (3rd ed.). Psychological Assessment Resources.

Hu, E. J., et al. (2022). LoRA: Low-rank adaptation of large language models. *ICLR 2022*.

Liu, Y., et al. (2019). RoBERTa: A robustly optimized BERT pretraining approach. *arXiv:1907.11692*.

National Center for O*NET Development. *O*NET 31.0 Database*. U.S. Department of Labor, Employment and Training Administration. CC BY 4.0. https://www.onetcenter.org/database.html

Raffel, C., et al. (2020). Exploring the limits of transfer learning with a unified text-to-text transformer. *Journal of Machine Learning Research, 21*(140), 1–67.

Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *Proceedings of EMNLP-IJCNLP 2019*.

Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval, 3*(4), 333–389.

Wang, W., et al. (2020). MiniLM: Deep self-attention distillation for task-agnostic compression of pre-trained transformers. *NeurIPS 2020*.

Williams, A., Nangia, N., & Bowman, S. R. (2018). A broad-coverage challenge corpus for sentence understanding through inference. *Proceedings of NAACL-HLT 2018*.

Wolf, T., et al. (2020). Transformers: State-of-the-art natural language processing. *Proceedings of EMNLP 2020: System Demonstrations*.

