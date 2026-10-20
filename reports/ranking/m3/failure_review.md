# M3 development failure review

All examples below are development cases. These observations describe the saved baseline outputs, not a causal evidence-filtering experiment. We did not remove sentences or tune parameters after inspecting results.

| Profile | Evidence category | Target rank: structured → BM25 → hybrid | Observation |
|---|---|---|---|
| D002 | Explicit dislikes and irrelevant family background | 4 → 210 → 12 | The user dislikes creative writing and repetitive paperwork. BM25 ranks Writers and Authors second and English/Literature Teachers third. Lexical overlap with disliked work can still reward those occupations; hybrid loses the target's top-five placement. |
| D046 | Explicit dislikes and planted contradiction | 7 → 118 → 21 | The text combines genuine creative interests with opposed record-keeping statements and disliked physical work. Lexical top two are writing occupations; the target is a postsecondary architecture teacher. Retrieval shifts the target down, without any mechanism to resolve the contradictory evidence. |
| D047 | Misleading familiarity, plus creative/research dislikes | 2 → 216 → 23 | Workshop/machine exposure appears alongside genuine leadership/planning interests and explicit creative dislikes. BM25's top two are writing occupations. The hybrid loses a correct top-five result. Multiple text effects are mixed here; this does not isolate familiarity as the cause. |
| D048 | Genuine familiarity (Enterprising) | 1 → 150 → 18 | The user's sales background is a genuine-interest case. Lexical top results include HR Managers and Labor Relations Specialists, while the target is Meeting/Event Planners. Hybrid loses the rank-one target. Simply suppressing all familiarity would not be justified by this case. |
| D058 | Genuine familiarity (Social) | 1 → 78 → 5 | Volunteering/counselling exposure coexists with genuine helping interests. Hybrid preserves Hit@5 but reduces reciprocal rank from 1 to 0.2. Familiarity handling must preserve genuine preferences. |
| D005 | Genuine background and a positive example | 6 → 24 → 2 | Accounts/spreadsheet interests are supported by background experience. Hybrid brings the target into the top five despite neither standalone ranker placing it there. Complementarity exists on individual cases despite aggregate regression. |

The baseline has no polarity-aware query model: retaining negation tokens does not make BM25 understand their scope. It also has no synonym matching or stemming, and broad informal wording may overlap weakly with occupation/task language. Prepared documents include titles, alternate titles, description and task wording; generic terms and document-length normalization can affect rankings. These are plausible explanations to test later, not demonstrated causes of every miss.

Hybrid Hit@5 improves on 5 profiles and regresses on 7; full-depth target rank improves on 11, regresses on 48, and is unchanged on 1. BM25 fails to retrieve the target at any positive score on 12 profiles; these remain in metric denominators with MRR zero. Zero-score occupations were not padded into lexical rankings.

Keep the unfiltered baseline executable. Future lexical variants should separately test tokenization/document/query choices on development data. Evidence filtering belongs to the later evidence-aware milestones and must hold tracker values fixed. No claim about actual contradiction detection, familiarity probing, or end-to-end dialogue effectiveness follows from these runs.
