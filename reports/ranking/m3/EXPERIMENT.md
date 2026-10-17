# M3 pre-evaluation protocol

Hypothesis: lexical occupation matching adds useful information to structured RIASEC fit, but raw dislike/familiarity statements can promote unwanted occupations. Improvements are not assumed.

Control: unchanged `weighted_dot_product@baseline_v1`. Treatments: `bm25@baseline_v1` alone, then structured+BM25 with equal-weight `rrf_k60`. Comparing hybrid against structured adds one lexical signal with the specified fusion baseline; it does not isolate all effects of fusion versus retrieval separately.

Constants: same 60 development profile IDs, unchanged oracle vectors and opening user texts, same 378 eligible occupations and data fingerprints, same six M2 metrics. No held-out scoring or tuning. All runs receive identical snapshots, even when a component ignores a field. Oracle structured inputs are privileged synthetic information; lexical inputs are the supplied texts. These are component comparisons, not actual-tracker or probing results.

Frozen BM25 baseline: `k1=1.5`, `b=0.75`, positive smoothed IDF `log(1+(N-df+0.5)/(df+0.5))`; eligible `bm25_doc` only. Lowercase ASCII alphanumeric/apostrophe tokens, explicit versioned stopwords, no stemming. Query term frequency is linear; all accumulated user utterances are included. No stance, contradiction, familiarity, or evidence filtering. Only positive matching documents enter rankings; absent documents contribute zero to RRF. All positive matches are returned without a retrieval cutoff. Negative M1 scores remain valid structured ranks.

Measurements: Hit@1/5/10, full-depth MRR (unretrieved target=0), NDCG@5/10, per-profile differences, paired bootstrap intervals, regressions/improvements, preloaded median/p95 latency and index/load time. Inspect explicit dislike and familiarity cases, including genuine familiarity. Repeat predictions/metrics for reproducibility. Preserve the original M2 baseline and history rows.

Decision: retain this baseline even if quality regresses; document failures and whether default activation is justified. Stop before M4 and do not claim that probing helps.
