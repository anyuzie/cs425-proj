# Cosine fusion development experiment

Hypothesis recorded before execution: adding lexical and/or dense retrieval may contribute occupation-specific information beyond cosine RIASEC alignment, but equal-weight fusion may dilute the strong structured signal.

Evaluate the three remaining combinations containing cosine M1: M1+BM25, M1+MiniLM, M1+BM25+MiniLM. Use unchanged baseline variants, RRF k=60, raw text, the same 60 development/oracle inputs and 378 candidates. Compare each with saved cosine-only and matching dot-product fusion runs. Save all six metrics, paired bootstrap intervals and rank shifts. Append successful evaluations to central history. Preserve defaults and prior reports; no held-out access or parameter tuning.
