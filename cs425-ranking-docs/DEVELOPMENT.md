# Development Roadmap — Incremental Ranking

Return to [README](README.md) · See [Evaluation](EVALUATION.md)

**Current status:** M0?M2 are implemented and accepted locally; the current gate is before M3. See the [canonical status table](README.md#current-implementation-status) and [M2 decision](../reports/ranking/m2/DECISION.md). Checked items below reflect recorded acceptance results, not a new test run. Upstream/downstream teammate sign-off remains pending.

## Rules for every milestone

Each milestone follows the same loop:

```text
Specify acceptance criteria → Implement smallest version → Unit/integration tests
       → Run frozen comparison → Inspect failures → Keep / Revise / Revert
       → Save report → Proceed (or stop)
```

**Definition of Done (DoD):** code runs from a documented command; automated correctness tests pass; same input produces the same results; run config and data fingerprint are saved; observed quality/latency and notable failures are written down; the decision is explicit. A quality regression **does not** mean a correctness test failed.

**Avoid premature complexity.** Prefer a working RIASEC-only system and a trustworthy evaluator over a partially integrated triple-retrieval system.

## M0 — Data and interfaces (no ranking algorithm yet)

**Purpose:** Make sure everyone is solving the same problem.

**Requirements**
- Read dataset schemas; validate occupation code uniqueness, tasks-to-occupation mapping, RIASEC dimension order `R,I,A,S,E,C`, missing values, and eligibility.
- Confirm Job Zone 4–5 restriction (planned: 378 eligible) versus proposal-wide 1,016 occupations. Record why.
- Confirm split membership (planned: 60 development, 100 held-out), target eligibility, and whether graded relevance labels exist.
- Define runtime contract; distinguish unknown RIASEC from a known zero.
- Clarify what text query represents: latest turn, all user turns, or active evidence; keep baseline as accumulated **user** utterances excluding bot questions.
- Agree with policy developer on top-100 output and empty/no-signal behavior (external sign-off remains pending; the local contract is implemented).

**Minimum interface**

```python
rank(
    interests: dict[str, float],       # tracker outputs in [-1, 1]
    observed_types: set[str],          # known dimension can still be 0
    user_sentences: list[str],         # user statements only
    top_k: int = 100,
) -> RankingResponse
```

The implemented envelope exposes `status`, `reason`, and `occupations: list[RankedOccupation]`; external policy agreement remains pending. Each occupation output needs `code`, `title`, final `rank`, signal scores/ranks where available, and task evidence when implemented. Validate finite interest scores and allowed keys. At M1, inputs without observed nonzero interests return a clearly signaled no-ranking result; text-only ranking remains planned. Policy agreement on this envelope is pending.

**Important:** Development `hidden_slots` are z-scale synthetic inputs according to the current implementation plan; do not clamp or silently treat them as `[-1,1]` tracker scores. Provide a separate `oracle` adapter and label its reports.

**Acceptance / gate**
- [x] Data audit saved (counts, missingness, eligibility, split summary).
- [x] Contract documented with one mock request/result.
- [x] Data validation scripts fail clearly on broken fixtures.
- [x] All open decisions recorded; unresolved blocking decisions resolved before M1.

**Deliverable:** `reports/ranking/m0.md`.

## M1 — MVP: structured RIASEC ranker

**Purpose:** Get a real, end-to-end ranking over the eligible occupation pool without text retrieval.

**Requirements**
1. Load occupation RIASEC z-scores without recalculating them on the eligible subset.
2. Compute `score(o) = sum(user[t] * occupation_z[o,t] for t in observed_types)`.
3. Sort high to low; break ties by occupation code; return up to `top_k`.
4. Build CLI with a fixed mock tracker profile; print top five.

**Tests**
- [x] Small fixture matches hand-computed scores.
- [x] Negative values and unobserved dimensions behave as expected.
- [x] Known zero differs from unknown dimension.
- [x] Invalid/nonfinite/out-of-range runtime input is rejected.
- [x] Repeated runs are identical and all outputs belong to eligible pool.

**Gate:** runnable CLI + tests pass. No accuracy threshold yet.

**Deliverable:** code, CLI command, sample output, `reports/ranking/m1.md`.

## M2 — Evaluation foundation (before adding signals)

**Purpose:** Know whether any future ranking change actually helps.

**Requirements**
1. Implement Hit@5, MRR, and graded NDCG@10; use the proposed relevance grades `target=4`, `Primary-Short=3`, `Primary-Long=2`, `Supplemental=1` where valid.
2. Deduplicate related codes by maximum grade; exclude ineligible codes from both labels and ideal DCG.
3. Implement per-profile prediction export, aggregate report, and paired comparison script.
4. Store input mode: `mock_tracker`, `oracle`, or `actual_tracker` (never pool their results). The M2 runner implements only the frozen development/oracle condition; other evaluation modes require future adapters.
5. Run structured-only baseline on development data; do not use target labels as ranking inputs.

**Tests**
- [x] Hand-computed target at rank 1, rank 6, and absent cases.
- [x] Perfect NDCG=1; irrelevant NDCG=0; empty ideal DCG handled.
- [x] Duplicate recommendations cause evaluator error.
- [x] Missing/ineligible targets explicitly reported.
- [x] Frozen baseline can be reproduced using saved config and data fingerprint.

**Gate:** metrics are trustworthy. Fix evaluator before comparing models.

**Deliverable:** `reports/ranking/m2/` with run manifest, predictions, metrics.

## M3 — BM25 and two-signal fusion

**Purpose:** Test whether lexical occupation information adds value to structured fit.

**Requirements**
- Index eligible occupations' prepared `bm25_doc` text with a fixed documented tokenizer.
- Query accumulated user statements only (baseline; no evidence filtering yet).
- Do not pass BM25 results with zero relevance into RRF merely due to tie ordering.
- Implement one-based reciprocal rank fusion `sum(1/(60+rank))` for available signals; absent documents contribute zero.
- Compare structured-only, BM25-only, and structured+BM25 using identical development cases.

**Tests**
- [ ] Exact-term fixture retrieves intended occupation.
- [ ] Empty/stopword/no-positive query omits BM25 contribution.
- [ ] Hand-calculated RRF scores and stable tie breaks.
- [ ] Every returned occupation is unique and eligible.
- [ ] Failure review includes explicit dislike and familiarity statements.

**Decision gate:** Is BM25 offering useful complementary signal? Retain as baseline even if it hurts some cases, but document failures and whether to revise query construction later.

**Deliverable:** `reports/ranking/m3/` experiment and decision.

## M4 — MiniLM + task-level dense retrieval

**Purpose:** Measure semantic value beyond exact word matching.

**Requirements**
- Pin a specific sentence-transformers MiniLM model and revision; use the *same* model for stored tasks and user queries.
- Cache normalized embeddings of eligible O*NET tasks, mapping each task to its occupation code.
- Use exact/exhaustive inner-product search first (a FAISS flat index is sufficient); optimize only if required by measured latency.
- Initial aggregation: for each user sentence, take best task match per occupation, then average across sentences. Treat this as an experimental choice, not a universally optimal method.
- Compare dense-only, structured+dense, BM25+dense, and three-way RRF to earlier controls.
- Record index build time, cache size, and preloaded query median/p95 latency.

**Tests**
- [ ] Paraphrase fixture retrieves meaningful task.
- [ ] Every result maps to a valid occupation; missing-task occupations don't get a fictitious dense rank.
- [ ] Cached/reloaded scores match within tolerance.
- [ ] Empty user text omits dense signal; no task corpus re-embedding during a turn.
- [ ] Aggregation is correct on a hand-built task fixture.

**Decision gate:** Does semantic retrieval add sufficient ranking quality relative to complexity/latency? If not, diagnose aggregation before adopting more elaborate ANN search.

**Deliverable:** `reports/ranking/m4/` comparisons and decision.

## M5 — Integrated, proposal-compliant baseline

**Purpose:** Deliver a working system that the policy can call each turn.

**Requirements**
- Load caches/indexes at application startup, not once per turn.
- Integrate with tracker input and policy output, returning top 100 with occupation RIASEC values and top five with up to three task statements.
- Do not call a task *supporting* evidence if it merely matches a stated dislike; omit or flag as provisional until stance metadata exists.
- Test a scripted multi-turn conversation with changing interest scores and conflict resolution.
- Agree on an explicit latency budget with teammates.

**Tests**
- [ ] Different user states change rank as expected.
- [ ] Results have valid occupation/task relationships.
- [ ] No duplicate/ineligible occupations.
- [ ] Policy handles fewer than 100 results and no-ranking case.
- [ ] Median/p95 latency measured with preloaded indexes.

**Decision gate:** **Baseline freeze point.** Save configuration as `baseline_v1`, including the current raw-text query policy. This is the primary control for evidence-aware experiments.

**Deliverable:** `reports/ranking/m5/`, integration trace, frozen baseline manifest.

## M6 — Experimental: contradiction-aware evidence lifecycle

**Purpose:** Remove superseded/rejected statements from BM25 and dense query construction while leaving the structured vector unchanged in the controlled experiment.

**Dependencies:** tracker/policy must provide stable evidence IDs and statuses, or fixtures must simulate them.

**Requirements**
- Evidence statuses: `active`, `disputed`, `superseded`; resolved contradictions are cases, but each statement still has its own usable status.
- Preserve original utterances for traceability. Exclude superseded and (by initial policy) disputed text from retrieval.
- Provide switch `evidence_filter = raw | active_only` so both variants use the same retrieval stack.
- Avoid editing the structured vector between paired control and treatment runs.

**Tests**
- [ ] Rejected teaching statement cannot influence text retrieval when superseded.
- [ ] Active statements remain usable.
- [ ] Empty active evidence is safe.
- [ ] Source evidence IDs survive to result diagnostics.

**Decision gate:** compare paired target-rank changes, Hit@5, NDCG@10, and stale-evidence sensitivity. **Keep** if useful and no unacceptable regression; **revise** if rule drops meaningful evidence; **revert** if ineffective/unsafe. Log uncertainty and affected-case counts.

## M7 — Experimental: familiarity-aware weighting

**Purpose:** Reduce the influence of exposure-only text that is not confirmed enjoyment.

**Requirements**
- Keep reason label (`enjoyment/familiarity/neither`) separate from stance, confidence, and lifecycle status.
- Implement configurable weighting for unresolved familiarity: candidate values `0, 0.25, 0.5, 1.0`; choose on *development* data only.
- For text rankers, score/aggregate sentences individually if necessary; don't claim stock `rank_bm25` itself applies sentence weights.
- After PROBE response, use updated confirmed evidence rather than implicitly assuming every familiar activity is disliked.
- Compare `baseline_v1`, M6, familiarity-only, and M6+M7 on the same cases.

**Tests**
- [ ] Familiarity-only text has appropriately changed influence.
- [ ] Confirmed genuine enjoyment is not automatically suppressed.
- [ ] Neutral/unusable statements don't create phantom ranks.

**Decision gate:** evaluate gains on misleading familiarity cases **and** losses on genuine enjoyment cases. Do not keep a policy that suppresses true interest without compensating evidence.

## M8 — Optional: explicit positive/negative channels

**Purpose:** Stop lexical/semantic relevance from promoting occupations solely because the user names a disliked activity.

**Requirements**
- Separate sentence-level positive and negative text scores; test a soft penalty on occupation relevance before ranking.
- Tune penalty only on development data. Never treat NLI contradiction probability as a direct dislike score.
- Compare full M6+M7 with and without signed retrieval using fixed structured vectors.

**Tests / gate**
- [ ] A disliked-activity fixture is not presented as positive evidence.
- [ ] Explicit dislikes reduce inappropriate promotion without harming unrelated occupations.
- [ ] Improvements and regressions documented; feature may be reverted.

## M9 — Final freeze and evaluation

**Purpose:** Make defensible project claims.

- Freeze candidate pool, data fingerprints, model revisions, tokenization, retrieval aggregation, evidence rules, RRF settings, software versions, and code revision.
- Re-run all core control/treatment configurations on the untouched held-out set.
- Report exact-target metrics, graded metrics, paired changes, affected-case counts, confidence intervals, and latency.
- When upstream simulator and policy are ready, evaluate probing on vs off *end to end*. Label any oracle-only evaluation clearly.
- Do not tune based on the held-out result. New hypotheses go into a clearly labeled follow-up study.

**Gate:** reproducible final artifact with bounded, transparent claims (including negative results).

## Stretch options — only after M9 if time remains

- **Adaptive weighted RRF:** can test stage-dependent signal reliability but adds tuning complexity and risks obscuring core evidence effects.
- **Question selection by ranking impact:** requires coordination with the policy owner and a credible simulator of possible responses; not required to validate retrieval hypotheses.
- **Task mismatch explanations:** useful to users, but require separate human evaluation of faithfulness/usefulness.

Don't let any of these block the proposal-compliant baseline or the evidence-aware experiments.
