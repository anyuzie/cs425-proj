# Structured oracle development baseline — failure review

Reviewed the five worst exact-target ranks, selected from all 60 development profiles. All five targets are eligible and present in the complete ranking. Inputs are synthetic `hidden_slots`, not extracted user interests. Text is ignored. Scores below are rounded; full precision is in `baseline/predictions.jsonl`.

| Profile | Target | Target rank / score | First-ranked occupation / score | NDCG@10 |
|---|---|---|---|---:|
| D035 | 29-1141.03 Critical Care Nurses | 103 / 4.789 | 25-1066.00 Psychology Teachers, Postsecondary / 7.430 | 0.0136 |
| D034 | 19-1031.00 Conservation Scientists | 94 / 1.801 | 29-1229.04 Physical Medicine and Rehabilitation Physicians / 3.863 | 0 |
| D037 | 25-9021.00 Farm and Home Management Educators | 92 / 3.831 | 25-1121.00 Art, Drama, and Music Teachers, Postsecondary / 7.836 | 0 |
| D059 | 29-1141.00 Registered Nurses | 85 / 4.273 | 19-3033.00 Clinical and Counseling Psychologists / 5.943 | 0 |
| D051 | 25-4012.00 Curators | 80 / 3.677 | 27-2012.04 Talent Directors / 7.001 | 0.0307 |

Observations and hypotheses (not established causal findings):

- D035 has strong investigative/social values (I=1.47, S=1.78). Psychology teaching outranks the nursing target under this broad interest formula. Rank 103 would have been lost if MRR were evaluated only on the default top 100.
- D034's moderately positive R/I/S and negative C do not uniquely identify conservation work. The top result is a medical occupation; NDCG=0 means no eligible labeled relation appears in the top ten.
- D037 combines S=1.17, A=0.56, and C=-1.65. Art/drama/music teaching scores much higher than the target. A shared education theme does not establish relevance under the supplied graded labels.
- D059 has S=1.87 and E=-1.0. The structured fit favors clinical/counseling psychology over nursing. The six-dimensional representation does not encode specific nursing activities.
- D051 has positive A/S/E/C. Talent direction outranks curatorial work. Occupational text and task distinctions are absent from the current scoring inputs.

The dot product rewards alignment and vector magnitude; it does not minimize distance to an occupation profile. Whether normalization, distance-based scoring, nonlinear scoring, or text retrieval helps is a future controlled experiment. These misses motivate evaluation, not an immediate scoring change. Synthetic profiles are noisy occupation-derived vectors, so an exact target is not guaranteed to be uniquely best even with oracle inputs.

No stale-evidence, dislike-retrieval, familiarity-filtering, or probing effect can be diagnosed here because retrieval and evidence policies are unimplemented. No helped/harmed treatment examples exist at M2; future paired comparisons must inspect both.
