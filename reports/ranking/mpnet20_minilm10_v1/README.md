# Sequential MPNet 20 → MiniLM 10 integration run

31 October 2026. Original real-model development/oracle run on 120 profiles and 378 eligible occupations, before extracting stage orchestration into the separate semantic pipeline. A destination-repository verification run checks identical output after that separation.

Order: weighted dot product top 100 → MPNet within baseline groups of 20 → MiniLM within groups of 10 formed from MPNet output. RIASEC values and baseline scores/ranks are preserved. Both model signals and intermediate ranks are exported; full-pool tail is unchanged.

| Permanent metric | Result |
|---|---:|
| hit_at_1 | 0.016667 |
| hit_at_5 | 0.125000 |
| hit_at_10 | 0.250000 |
| mrr | 0.101771 |
| ndcg_at_5 | 0.122811 |
| ndcg_at_10 | 0.160019 |
| recall_at_5 | 0.079658 |
| recall_at_10 | 0.157609 |
| recall_at_20 | 0.303874 |

The sequential configuration has Hit@5=.125 and MRR=.101771, below the earlier MiniLM groups-of-10 (.208333/.134000) and MPNet groups-of-10 (.175000/.133113) runs on identical inputs. It is integrated as a selectable experimental configuration; no quality improvement or automatic promotion is claimed. Recall@20 remains structurally equal to the dot-product baseline because no occupation can cross a baseline group-of-20 boundary.

All nine metrics and source/model/data provenance are saved. The consolidated CSV compares this run with the five other configurations; source manifests are preserved when imported to the destination repository. Results measure synthetic RIASEC-derived development labels, not independently reviewed semantic intent or actual-tracker accuracy.

See ../../../docs/SEQUENTIAL_RERANKING.md and ../../../docs/SEMANTIC_INTEGRATION.md for usage and separation of structured and semantic matching.
