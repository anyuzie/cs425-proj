# M4 development failure and gain review

These examples use only the development split. They describe saved results; no sentences were removed and no model, threshold, aggregation, or fusion parameters were tuned after looking at them. Proposed explanations are hypotheses rather than causal ablations.

| Profile | Evidence category | Target rank: structured / dense / structured+dense / three-way | Observed behaviour |
|---|---|---|---|
| D002 | Explicit dislikes and irrelevant family exposure | 4 / 283 / 20 / 52 | Dense ranks Technical Writers first, with a matched task about maintaining records, despite the user's explicit paperwork dislike. Raw semantic similarity is not signed preference evidence. A top-five structured hit is lost. |
| D047 | Misleading familiarity plus explicit creative/research dislikes | 2 / 208 / 6 / 42 | Dense ranks Graphic Designers first and Landscape Architects second. The text explicitly dislikes creative work but also mentions workshop exposure. Structured+dense narrowly loses Hit@5; three-way falls much further. These mixed effects do not isolate familiarity as the cause. |
| D048 | Genuine familiarity (Enterprising) | 1 / 168 / 14 / 30 | Genuine sales exposure, helping/creative interests, and technical/research dislikes appear together. Dense's top result is Biological Technicians, with a generic logging task. The genuine target falls out of the top five after fusion. Blanket familiarity filtering is not justified. |
| D058 | Genuine familiarity (Social) | 1 / 314 / 5 / 20 | Dense ranks Writers and Authors first with an advertising/sales task, while the user explicitly dislikes persuading/selling and creative work. Structured+dense retains Hit@5 but loses reciprocal rank; three-way loses Hit@5. |
| D005 | Genuine background; regression from M3 gain | 6 / 208 / 30 / 10 | Structured+BM25 had rank 2 for the target. Dense prefers generic record-keeping matches, including Biological Technicians and Sales Engineers. Adding dense to the M3 hybrid removes that top-five gain. |
| D046 | Creative interests, explicit dislikes, and a contradiction | 7 / 24 / 1 / 13 | Structured+dense gains a rank-one target although neither standalone ranker retrieves it in the top five. Three-way does not retain the gain. Complementarity is configuration-dependent. |
| D041 | Genuine organisation/problem-solving background; gain | 25 / 2 / 2 / 8 | Dense retrieves the target Computer Programmers second, with a task about correcting errors and rechecking results. A paraphrase-like task match helps structured+dense recover a target that structured-only misses. Three-way again dilutes that gain. |
| D004 | Creative-project interests; dense gain | 21 / 3 / 10 / 8 | Dense ranks the target Architects third, while its top two are animation/art-direction occupations. BM25+dense retains rank 3. Structured fusion does not guarantee a standalone gain is preserved. |

Dense-only Hit@5 is 3/60 versus BM25's 1/60, but this is weak evidence: the paired difference interval includes zero. Structured+dense has 5 Hit@5 gains and 8 losses versus structured. Three-way has no Hit@5 gains and 5 losses versus the M3 hybrid.

The fixed aggregation rewards each occupation's best task for every sentence, including negated, background, generic, and contradictory sentences. Averaging these maxima can dilute specific interests. Occupations with more task descriptions also have more opportunities for a high maximum. These are reasons to test aggregation/evidence variants later, not proof that any one mechanism explains all failures.

All task matches remain **provisional**, including the apparently meaningful ones. They are not asserted to support a recommendation. Semantic similarity does not resolve polarity, authenticity of familiarity, or contradictions. No actual extractor/tracker dialogue or probing benefit was evaluated.

Retain this reproducible baseline and its negative result. Diagnose aggregation and query evidence under controlled new variants before introducing approximate search or adaptive fusion. Preserve held-out data for final evaluation.
