# Ranking diagnostic findings

Analyzed 60 development profiles, 378 eligible occupations, and seven compatible saved M4 configurations. No held-out data was opened for diagnosis.

## Five consequential findings

1. Oracle privilege and numerical ambiguity: median user-target cosine=0.960; M1 Hit@5=16.7%; 29 profiles list RIASEC twins. Alignment does not identify a unique target.
2. Lexical coverage: 12 targets have zero matched M3 query types; BM25 retrieves 12 targets at no positive score. Its Hit@5 is 1.7%.
3. Semantic task versus aggregation: median best target-task similarity=0.353, aggregate=0.257; dense Hit@5=5.0%. This does not isolate representation failure.
4. Fusion regression: three-way RRF loses 9 M1 top-five successes and gains 2; NDCG@10 changes by -0.1350.
5. Benchmark identifiability remains unverified: 18 blind-stage provisional specificity annotations, 0 verified human reviews; no verified-specificity performance claim is made.

## Top three recommended experiments

1. **Evidence selection with fixed RIASEC**: Compare raw user history with independently reviewed active-evidence selection, keeping tracker vector/model/aggregation/fusion fixed. (affected n=47; planning score=0.703).
2. **Blind specificity and benchmark audit**: Complete text-only human review before target reveal, then assess compatibility and family-level ambiguity. (affected n=18; planning score=0.612).
3. **Fixed fusion-weight ablation**: Compare equal-weight RRF with one preregistered fixed structured-weight alternative on the same saved component rankings. (affected n=9; planning score=0.558).

## Section findings and limits

### Integrity
Finding: 60 profiles and 378 eligible occupations; 0 profiles excluded. All seven saved metric sets reproduce with the existing evaluator.
Interpretation: Only matching development/oracle requests and recognized baseline variants are compared. Missing lexical retrieval remains an unavailable rank, not a zero rank.
Limitation: Dataset/profile validity is not a verification of synthetic labels or human relevance. Auxiliary task/model fingerprints are checked separately from shared evaluation data.
Next experiment: Keep these integrity checks when introducing future configurations.

### Overview
Finding: Three-way RRF changes Hit@5 by -11.7 percentage points compared with M1.
Interpretation: A new retrieval signal is not automatically complementary to an oracle profile. Inspect paired cases rather than interpreting aggregate loss as a single cause.
Limitation: This is a small synthetic development benchmark; oracle and text channels have different information access.
Next experiment: Audit evidence and benchmark identifiability before optimizing components.

### A1
Finding: I is the most frequent dominant positive dimension (20/60 profiles); 1 absolute-dominance ties. Scores range from -2.63 to 2.98.
Interpretation: The sample is not a uniform six-type population; polarity and correlations characterize the benchmark's numerical signal.
Limitation: Observed imbalance does not establish that it causes retrieval failures; dimensions are not independent occupations.
Next experiment: Use fixed stratified subsets and report dominant-type sample sizes in future experiments.

### A2
Finding: 60/60 target_profile vectors agree with O*NET within two-decimal rounding. Median cosine is 0.960; 30/37 profiles with cosine >= .95 still miss M1 top five.
Interpretation: Close direction alignment is insufficient for a dot-product winner. Dot products depend on vector norms as well as angle, and the full pool supplies competing vectors.
Limitation: A cosine plot is an analytical diagnostic, not evidence for adopting a new scoring formula. Synthetic targets are not a unique career ground truth.
Next experiment: If justified, preregister a separate structured-scoring comparison with frozen inputs; do not change M1 here.

### A3
Finding: 29/60 profiles list twins; median score-band occupancy is 5.0, excluding no profiles.
Interpretation: Provided twins, cosine neighbours and close score bands describe different forms of ambiguity. Subgroup metrics show whether retrieval adds labeled relevance in each group.
Limitation: The .99 cosine and +/-0.1 score thresholds are exploratory descriptors, not accepted occupation-equivalence definitions.
Next experiment: Test text disambiguation on reviewed ambiguous cases with the same numerical profile and candidate pool.

### B1
Finding: Median length is 66 words and 4 sentences; style counts are {'full': 48, 'messy': 12}.
Interpretation: Length and style associations describe this benchmark. Subgroup error bars are profile-bootstrap intervals; longer text can contain both preferences and distractions.
Limitation: Length is confounded with style/tier/content. Missing BM25 ranks are displayed at 379 only in plots and excluded explicitly from rank correlations, never from hit metrics.
Next experiment: Use matched sentence additions/removals on fixed profiles to test whether extra text helps or contaminates retrieval.

### B2
Finding: 12/60 targets have zero lexical overlap and 12 are absent from positive BM25 retrieval. Median query coverage is 4.5%.
Interpretation: Positive term overlap is not necessarily positive preference evidence. Inspect generic terms and the original text alongside each competing occupation.
Limitation: Token matches do not prove that background or negation caused a particular ranking. Stemming, document coverage and human-relevant vocabulary are not independently tested here.
Next experiment: Compare one query/document or tokenization change at a time; separately test stance/evidence filtering with fixed tracked interests.

### B3
Finding: Median best target-task similarity is 0.353, versus median aggregated score 0.257. Live sentence diagnostics cover 60/60 profiles.
Interpretation: Best single-task similarity and occupation mean-of-maxima answer different questions; task reuse/spread helps identify candidate aggregation bottlenecks.
Limitation: A high exact-target rank cannot diagnose an inadequate embedding model. Task coverage, broad evidence and competing plausible tasks may explain part of the result; all evidence remains provisional.
Next experiment: Freeze model/text and compare one alternative task aggregation in a separate registered experiment after human review.

### C1
Finding: There are 126 method-profile cases with a missed exact top-five target but at least one labeled relevant top-five occupation.
Interpretation: Some exact misses preserve benchmark-related relevance, so Hit@5 alone is incomplete. The graded metrics still quantify how much relatedness survives.
Limitation: Related occupations are O*NET/benchmark labels, not independently verified relevance for this synthetic person. Unlabeled plausible jobs receive grade zero.
Next experiment: Blindly review a sample of top recommendations before interpreting grade-zero results as universally unsuitable.

### C2
Finding: Among 34 profiles with M1 rank>10, dense-only retrieves 2 exact targets in the top five. BM25 has 12 missing target ranks overall.
Interpretation: Complementary cases exist or fail on their own evidence; subgroup figures distinguish different ambiguity descriptors rather than treating all misses alike.
Limitation: Groups and thresholds are exploratory and small. Text sufficiency is not established by either numerical ambiguity or a retrieval miss.
Next experiment: Create reviewed occupation-specific contrasts within ambiguous RIASEC families before claiming robust disambiguation.

### C3
Finding: Blind-stage proposals cover 18 stratified profiles: {'broad': 8, 'partially_specific': 4, 'specific': 3, 'misleading': 3}; verified human reviews: 0.
Interpretation: The table enables a reproducible text-first audit and later target compatibility review without passing labels into the automated judge.
Limitation: The cue heuristic is incomplete and may misclassify broad/task-specific language. No human claim about target inferability can be made until blind review is completed.
Next experiment: Complete the blind human audit, then reveal targets to test benchmark compatibility and occupation-family ambiguity.

### D1
Finding: 3 displayed attribute groups contain fewer than 10 profiles; all group sizes and metric intervals are exported.
Interpretation: Metadata helps locate cases to inspect, but the same profile contributes to multiple overlapping groups.
Limitation: Many exploratory comparisons, correlated attributes, and only 60 profiles prevent strong subgroup or causal conclusions.
Next experiment: Predefine balanced strata for later development comparisons; retain paired profiles and fixed inputs.

### D2
Finding: RRF loses 9 M1 top-five successes and gains 2; 2 standalone text successes are lost.
Interpretation: The contribution breakdown directly explains the fusion arithmetic. It does not explain why an upstream component preferred a competitor.
Limitation: Rank-only fusion discards score confidence; assigning a specific semantic/negation cause requires further controlled evidence.
Next experiment: If warranted, test one fixed fusion-weight change with exactly the same component rankings; keep equal-weight RRF as the control.

### D3
Finding: 252 overlapping provisional indicators cover 60 profiles; no definitive semantic-representation failures are assigned.
Interpretation: The taxonomy records reproducible observations and candidate explanations instead of forcing one cause per profile.
Limitation: Heuristic cue/metadata co-occurrence does not establish contamination or model failure. Category counts are not prevalence estimates of true causal failures.
Next experiment: Blindly review examples, then use one-variable ablations to separate vocabulary, evidence, aggregation and fusion mechanisms.

### D4
Finding: The top three rubric-ranked proposals are: Evidence selection with fixed RIASEC, Blind specificity and benchmark audit, Fixed fusion-weight ablation.
Interpretation: Priority combines observed affected cases with reviewable engineering/research judgments, rather than asserting that an untested change will improve accuracy.
Limitation: The weighting is a transparent planning heuristic. Overlapping affected groups and observational severity do not estimate causal treatment gains.
Next experiment: Preregister control/treatment and constants before implementing any selected experiment; retain all accepted baseline variants.

## Integrity and limitations
{"eligible_occupations": 378, "excluded_profiles": 0, "ineligible_targets": 0, "missing_targets": 0, "missing_text": 0, "nonfinite_or_missing_riasec": 0, "profiles": 60, "related_codes_outside_pool": 271, "unique_ids": 60, "unknown_related_codes": 0}
Actual pinned model/cache reused; task embeddings were not rebuilt.
All annotations/taxonomies are provisional except explicitly supplied verified blind human reviews. Subgroup sizes are small, intervals exploratory, labels synthetic/O*NET-based, and correlations noncausal. Source configurations and metrics were not changed. No diagnostic calculations were registered as model experiments.