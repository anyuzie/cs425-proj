# Independent clause context projection

2 November 2026. The ranking terminal displays the top ten occupation names and ranking scores without occupation-clause annotations. It no longer loads the historical occupation classifier or occupation annotation corpus. Existing ranking, classifier weights and occupation context ratings remain unchanged.

## Routing and representation

After Stanza splitting and BERT scoring, route_context independently triggers ContextMatchingPipeline at the classifier's existing threshold (0.5). The independent routing interface also accepts an optional task pipeline; no task matcher is implemented or invoked in this stage. Both can run on a clause; neither requires the other. Nonzero sigmoid scores alone do not trigger every pipeline.

Context names plus descriptions are encoded once per pipeline instance using pinned sentence-transformers/all-mpnet-base-v2 revision e8c3b32edf5434bc2275fc9bab85f82640a19130. Each clause's preference target is encoded by that same model. The displayed 57-value vector contains raw cosine similarities against those 57 attributes in context_dimensions.csv order. It is a semantic relevance projection, NOT measured occupation context levels, a calibrated probability, or the encoder's native 768-dimensional embedding. Negative cosine similarity is not negative user preference. Full-precision results remain available through process(); terminal values are rounded to six decimals. Closest three attributes are displayed for inspection, not accepted matches above a validated threshold.

No occupation scoring, rating comparison, dot product with occupation vectors, desired-level interpretation or score fusion is implemented. In particular, preference for quiet work versus dislike of noise requires later interpretation of the desired attribute level before ratings can be used safely. The present vector alone cannot supply that interpretation.

## Preference baseline

Explicit like/love/enjoy/prefer/want cues produce +1; dislike/hate/avoid/cannot stand and don't like/enjoy/want cues produce -1. The target after the preference cue is embedded, keeping sign separate. Neutral descriptions default to +1 by the agreed policy. Uncertain, complex negation, unresolved negation or detected mixed scope yields unknown (null sign), never fabricated negative/positive certainty. Unknown clauses still receive a diagnostic context vector. These are untrained lexical rules with limited coverage, not an LM or the retired stance ranker. Third-party statements, sarcasm and complex preferences are not reliably handled. All results remain experimental and do not affect ranking.

## Provenance and verification

Collection session metadata records encoder revision, dimension IDs and source hash, vector definition and preference rule version. Existing annotation events/exports are preserved; terminal context diagnostics are not training labels. The encoder loads lazily on the first context-routed clause, separately from description reranking. Initial use may download the pinned model; subsequent use can use its cache. Tests cover independent routes, polarity and uncertainty, stable dimension order, 57 outputs, shared target vectors for opposite preferences, cached attribute encoding and unchanged terminal ranking values. Existing pre-extension ranking/classifier metrics remain in PRE_EXTENSION_BASELINE.md; no new performance gain is claimed.

## Shared low-score exclusion

Before either task or context processor runs, a clause is skipped when both task_score < 0.3 and context_score < 0.3. Exactly 0.3 does not meet this exclusion. Clauses and scores remain visible in the terminal with a skip reason; exclusion does not delete annotations or source text. Passing this floor does not activate either route: existing independent classifier thresholds (currently 0.5) still apply. The terminal session records the 0.3 floor.

## Per-clause occupation associations (2 November 2026)

Each context-routed clause now displays up to ten context-associated occupations, independently of the RIASEC ranking and task pipeline. `context_association_top3_v1` uses the clause's three closest attributes with positive cosine similarities. For each occupation, association = sum(similarity * normalized rating) / sum(similarity), using only available selected ratings. Missing/suppressed ratings never become zero; genuine measured zero remains usable. No available selected ratings means no association result. Reported coverage is available similarity weight / total selected similarity weight. Results sort by association descending, coverage descending, then occupation code. Partial coverage is visible and may overstate associations based on one available attribute; this is experimental evidence, not a recommendation quality claim.

All 1,016 local occupations are considered, including occupations outside the RIASEC eligibility pool. Occupations without usable selected context data are omitted. The ten-result cap is terminal presentation; the matcher exposes an independent limit argument. Names/codes, association scores, coverage, and supporting context ratings/similarities are displayed per clause. Source vector/mask/occupation hashes and matching variant are recorded in session metadata. Source files and saved baseline reports are unchanged.

Preference sign does not reverse the association list: negative clauses label high-exposure occupations as potential preference conflicts; unknown preference labels results as associations only. Positive clauses also show associations, not guaranteed compatibility. Similarity cannot establish whether the user wants a high or low level of a condition (quiet vs noise is a key example). No desired-level inference, validated semantic cutoff, fusion or modification to existing ranking is introduced. A future compatibility scorer must explicitly resolve those semantics and evaluate against the preserved baseline.

## Conservative acceptance revision (2 November 2026)

`explicit_or_threshold_v2` replaces forced top-three selection in the experimental context path. Overall ranking/fusion and the terminal interface are unchanged. Raw 57-value cosine diagnostics remain available even for rejected matches; empty top_contexts means no occupation associations. The process result additionally exposes matching_status and matching_usable.

Unresolved demonstratives/pronouns (it/that/this/those/these/them/there) in the extracted target cause abstention; a narrow expletive-it construction with an explicit infinitive is allowed. This is a conservative heuristic, not coreference resolution. Unknown/mixed preference scope also abstains. Informal dont and don't really want/like/enjoy are normalized for preference detection; original clause/source evidence is preserved.

Explicit office/air-conditioned/climate-controlled phrases select only Indoors, Environmentally Controlled. Generic indoor and outdoor phrases map to their respective indoor/outdoor attribute families without adding communication or schedule attributes. Noise/loud phrases map to noise exposure. Each explicit accepted attribute records its actual source phrase and offsets, selection method, and original semantic similarity; mapping does not manufacture a confidence of 1. Quiet/low-noise/remote/work-from-home conditions currently abstain because the occupation association list has no desired-level scorer. Unsupported conditions are retained in original text for future handling; no high-exposure recommendation is invented.

Other phrasing may select up to three semantic attributes only at cosine >= 0.5. This is a provisional, uncalibrated conservative engineering default, not a learned or reviewed relevance cutoff. min_similarity is configurable on ContextMatchingPipeline; metadata records the setting and cutoff_calibrated=false. The BERT 0.5 route threshold and shared 0.3 exclusion are separate and unchanged. Review positive/non-match conversational examples before choosing a validated attribute threshold. False matches above the cutoff remain possible.

Regression cases cover baking with low similarity producing no matches, unresolved references, informal negation, exactly-at-cutoff semantics, office-only evidence, expletive it, unknown preference, and unsupported low-level/remote conditions. No new classifier accuracy or recommendation improvement is claimed.
