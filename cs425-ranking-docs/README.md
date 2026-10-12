# CS425 — Occupation Scoring & Ranking

> **Project objective:** Build and evaluate a conversational occupation-ranking component that uses a user's evolving RIASEC interest profile and relevant conversation evidence to rank O*NET occupations after each turn.
>
> **Research question (whole project):** Can probing contradictions and challenging familiarity-based justifications give a better occupation ranking than accepting the user's stated profile as given?

This README is the **source of truth for objectives, scope, architecture, and milestone priorities**. See [DEVELOPMENT.md](DEVELOPMENT.md) for acceptance criteria and implementation steps, [EVALUATION.md](EVALUATION.md) for controls and metrics, and [EXPERIMENT_TEMPLATE.md](EXPERIMENT_TEMPLATE.md) for recording each experiment.

## 1. Three levels of objectives

| Level | Objective | Definition of success | Ownership |
|---|---|---|---|
| **1 — Required engineering** | Implement a deterministic, modular, per-turn occupation ranker using **RIASEC fit, BM25, MiniLM task matching, and RRF (k=60)**. | Correct, runnable, reproducible ranking pipeline with occupation-level output, top 100 for policy, and task evidence for recommendations. | Scoring & Ranking |
| **2 — Component research** | Test whether **evidence-aware retrieval** improves rankings compared with retrieval over accumulated raw user text. | Controlled comparisons of stale/contradictory evidence handling and familiarity-aware evidence selection, including regressions. | Scoring & Ranking, with upstream evidence metadata |
| **3 — End-to-end research** | Test whether **contradiction clarification and familiarity probing** improve final recommendations over accepting the initial user profile. | Evaluated dialogue runs with/without interventions, using the real extractor, state tracker, contradiction check, and policy. | Whole team |

**Do not confuse these claims.** An improvement using manually corrected evidence (Level 2) does not prove the chatbot can reliably elicit that correction (Level 3). Likewise, improving BM25 or fusion alone does not prove probing helps.

## 2. What the ranking component does — and does not do

**Inputs (dynamic):** the tracked RIASEC vector (six signed values in `[-1,+1]`), which dimensions have been observed, and user utterances. Later milestones may also consume sentence-level evidence metadata such as stance, reason, confidence, lifecycle status, and source turn.

**Inputs (static):** eligible O*NET occupation identifiers, occupation RIASEC z-scores, text fields used for BM25, task statements, and precomputed MiniLM task embeddings/index.

**Outputs:** a ranked list of eligible O*NET occupation codes, component rankings/scores, fused ranking, and linked supporting task statements. The dialogue policy consumes the top 100; displayed recommendations can use the top five and their supporting tasks.

**Boundary:** this component **does not** train the RIASEC extractor, decide whether two statements contradict, maintain the canonical user profile, or choose the next dialogue question. It consumes upstream outputs. The ranking code may select/filter eligible evidence, but the underlying classification and evidence-resolution authority must be agreed with upstream teammates.

**Important:** RRF itself consumes *occupation rankings*, not raw O*NET texts. O*NET data is used by the three individual rankers and later to attach metadata/evidence.

## 3. Pipeline at a glance

```text
                        USER CONVERSATION
                                |
                  Extractor + State Tracker
                     Contradiction Check
                                |
           updated RIASEC profile + user utterances
            (+ evidence labels/statuses when ready)
                                |
             [Evidence selection: OPTIONAL M6+]
                                |
               +----------------+----------------+
               |                |                |
         Structured fit      BM25          MiniLM + FAISS
       user RIASEC vs     user text vs     user sentences vs
       occupation scores occupation text  O*NET task embeddings
               |                |                |
         occupation       occupation       matched tasks ->
         ranking          ranking          occupation ranking
               |                |                |
               +----------------+----------------+
                                |
                   RRF fusion (k = 60)
                                |
            ranked O*NET occupations + task evidence
                                |
             Dialogue Policy (top 100) / UI (top 5)
                                |
                    next question or results
```

**Required baseline (M1–M5):** RIASEC → BM25 → dense task retrieval → occupation-level aggregation → equal-weight RRF → integration. This baseline must remain runnable and evaluable even after optional extensions are introduced.

**Experimental extensions (M6–M8):** Use lifecycle/stance/reason metadata to choose *which* sentences influence text retrieval, optionally separating positive and negative channels. Keep the original rankers and fusion available as controls.

## 4. Non-negotiable research guardrails

1. **Every feature must answer one of three questions:** Does it fulfill a required component capability? Does it improve a measured ranking outcome? Does it directly test the stated research question? If none apply, defer it.
2. **Build before optimizing.** Start with RIASEC-only MVP, then complete the proposal's three-signal baseline before testing evidence-aware variants. Do not make novel extensions prerequisites for core functionality.
3. **One material change per experiment.** Specify the hypothesis, control, treatment, and expected effect *before* looking at results.
4. **Keep comparisons fair.** Hold the candidate pool, test cases, tracker values, embedding model, and evaluation rules fixed when isolating evidence-filter changes. Document any intentional exceptions.
5. **Separate correctness, relevance, and latency.** Passing tests is mandatory; a feature need not improve accuracy to be a valid experiment. Record negative results and regressions.
6. **Do not overclaim.** Oracle/manual evidence results, actual-tracker results, and full dialogue results are distinct evaluation conditions.
7. **Reproduce every run.** Record code revision, configuration, data fingerprint/split, model revision, metrics, per-profile predictions, and timing.
8. **Protect held-out evaluation.** Tune on development cases; freeze configurations before final held-out results.
9. **Keep controls executable.** Do not delete baseline retrieval or overwrite previous experiment reports when introducing extensions.

## 5. Milestones and decision gates

| Milestone | What must exist | Research purpose / go-no-go question |
|---|---|---|
| **M0 — Data + contracts** | Verified datasets, eligibility rule, typed I/O contract | Can upstream and downstream components exchange the needed data without ambiguity? |
| **M1 — MVP** | Runnable RIASEC-only ranker | Does structured fit produce valid, deterministic occupation rankings? |
| **M2 — Evaluator** | Verified Hit@5, MRR, NDCG@10 and saved baseline | Can we trust the comparisons before introducing retrieval complexity? |
| **M3 — BM25** | Lexical ranker + two-signal RRF | Does lexical matching add useful retrieval signal, and in which cases does it hurt? |
| **M4 — MiniLM + FAISS** | Cached task embeddings/index, task-to-occupation aggregation, three-signal RRF | Is semantic retrieval useful beyond structured/lexical ranking at acceptable cost? |
| **M5 — Integration** | Per-turn API, top 100 for policy, task evidence and trace | Can the *required proposal baseline* operate in a realistic conversation? |
| **M6 — Contradiction-aware evidence** | Status-aware selection/filtering, controlled ablation | Do superseded or disputed statements distort retrieval after the profile is corrected? |
| **M7 — Familiarity-aware evidence** | Reason-aware selection/weighting, controlled ablation | Does reducing unconfirmed familiarity improve retrieval without suppressing genuine interests? |
| **M8 — Optional signed channels** | Positive/negative text retrieval variant | Does explicitly handling disliked activities add value beyond M6–M7? |
| **M9 — Freeze + final evaluation** | Frozen configurations, held-out report, end-to-end results if available | Which changes are justified, and what can we actually claim about probing? |

See [DEVELOPMENT.md](DEVELOPMENT.md) for each milestone's implementation requirements, tests, and exit gate.

**Advance / revise / revert:** Advance when acceptance tests pass and the previous change has been evaluated and documented. Revise when results expose a diagnosable problem. Revert or defer a feature when its extra complexity has no demonstrated benefit, or when its effects cannot be evaluated reliably. **Accuracy improvement is not mandatory to complete an experiment.**

### What counts as a complete project?

- **M5:** A complete engineering deliverable for the proposed scoring-and-ranking subsystem, with baselines and metrics.
- **M7:** The main *component-level* evidence-aware research extensions have been implemented and compared with controls (whether or not they win).
- **M9 + team integration:** Evidence sufficient to address the **overall project research question**. Without complete dialogue integration, explicitly mark the end-to-end claim as untested.

Do not spend time on adaptive fusion, ranking-impact question selection, or elaborate explanations unless core milestones are stable and a specific hypothesis warrants the extra effort.

## 6. Evaluation philosophy — always retain a control

Every iteration must preserve a **control** and a **treatment**:

| Field | Record |
|---|---|
| Hypothesis | What behavior is wrong or missing, and why would the change help? |
| Control | Exact runnable configuration of the previous/baseline method |
| Treatment | Exact configuration with one intended change |
| Constants | Same profiles, targets, candidate pool, RIASEC inputs, metric definitions, model/index versions where appropriate |
| Measurements | Hit@5, MRR, NDCG@10, affected-case counts, latency, and per-profile rank shifts |
| Decision | Keep / revise / revert, with evidence and known tradeoffs |

For example, M6 tests accumulated raw user text against status-filtered evidence **using the same final RIASEC vector**. Otherwise a gain might come from changing the tracker state rather than filtering the text.

Use [EVALUATION.md](EVALUATION.md) for precise metric definitions, ablation controls and dataset separation, and [EXPERIMENT_TEMPLATE.md](EXPERIMENT_TEMPLATE.md) for a reusable experiment record.

## 7. Dataset and interface assumptions to resolve at M0

The implementation plan anticipates `data/occupations.csv`, `data/tasks.csv`, `data/dev_profiles.jsonl`, `data/profiles.jsonl`, `data/related_occupations.csv`, and `data/labels.json`. Verify what actually exists; this documentation does not create those datasets.

- The project proposal describes **1,016 occupations with RIASEC scores**; the working implementation plan restricts eligibility to **378 Job Zone 4–5 occupations** with complete RIASEC data. **Confirm and record** the candidate pool before running benchmarks. Preserve supplied z-scores rather than recomputing them on a restricted pool.
- The plan anticipates **60 development profiles** and **100 held-out profiles**. Verify that each split exists and that conversation histories, target labels, and evidence states are available for evidence-aware experiments. A final RIASEC vector alone cannot test contradiction/familiarity filtering.
- Runtime tracker scores in `[-1,+1]` are **not interchangeable** with synthetic oracle `hidden_slots` expressed on a z-scale. Use a separate oracle adapter and report results separately.
- Unknown dimensions must remain distinguishable from observed neutral values (e.g., `observed_types`).
- Choose and document task-to-occupation score aggregation before interpreting dense retrieval results.

## 8. Suggested modular code layout

```text
recommender/
  data_loader.py           # O*NET validation and joins
  schemas.py               # input/output contracts
  structured.py            # signed RIASEC scoring
  bm25_ranker.py           # lexical occupation ranking
  dense_ranker.py          # MiniLM / FAISS task matching
  fusion.py                # occupation-level RRF
  evidence.py              # OPTIONAL M6+ evidence selection
  ranker.py                # orchestration / public API
  evaluate.py              # metric and benchmark entry point
  cli.py                   # local reproducible demos
configs/                   # versioned configurations
artifacts/ranking/          # index/cache metadata, gitignored as appropriate
reports/ranking/
  experiments/             # experiment manifests and per-profile results
  final/
tests/
  fixtures/
  test_structured.py
  test_metrics.py
  test_bm25.py
  test_dense.py
  test_fusion.py
  test_evidence.py
```

These names are suggestions, **not confirmation that the modules already exist**. Implement only the modules needed for the active milestone.

## 9. Quick start

M0 implementation is available in `recommender/schemas.py`, `recommender/data_loader.py`, and `recommender/audit.py`. See the [M0 audit and contract report](../reports/ranking/m0.md) for verified counts, interface examples, pending teammate coordination, and reproduction commands. Run `python -m unittest discover -s tests -v` and `python -m recommender.audit` from the repository root. No ranking algorithm is implemented at M0.

1. Read the objectives and scope above, then begin **M0** in [DEVELOPMENT.md](DEVELOPMENT.md).
2. Confirm the O*NET eligibility rule and upstream/downstream contracts with teammates.
3. Finish **M1 RIASEC MVP** and **M2 evaluation baseline** before BM25 or dense retrieval.
4. For each material change, copy [EXPERIMENT_TEMPLATE.md](EXPERIMENT_TEMPLATE.md) into `reports/ranking/experiments/`, run the control and treatment, and record the decision.
5. Keep M5 as the required completion checkpoint; evaluate M6/M7 as research extensions; freeze before M9.

**Guiding question for every iteration:** *Does this help deliver a correct occupation-ranking system or test whether conversational clarification improves recommendations? If not, why are we building it?*
