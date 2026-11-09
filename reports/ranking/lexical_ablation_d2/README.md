# Lexical ablation under D2 title v3 (9 November 2026)

Arms: live default (`occupation_semantic_fusion_zscore_v1`); **D2 title v3 without lexical** (cosine + `occupation_dual_encoder_signed@finetuned_v2` + `dislike_demotion@title_v3`, `occupation_semantic_signed_v2_fusion_zscore_dislike_title_v3.json`); **D2 title v3 with lexical** (same + `bm25_signed@ignore_negative_v3`). The two D2 arms differ only in lexical matching. Task retrieval is not used (BM25 is leaked there).

| Benchmark | Metric | Live default | D2 title v3, no lexical | D2 title v3, with lexical |
|---|---|---|---|---|
| Personas, probing on (25) | MRR | .680 | .728 | **.770** |
| | Hit@5 | .80 | .80 | **.96** |
| | Hit@10 | .92 | .96 | .96 |
| | NDCG@10 | .539 | .591 | **.620** |
| | MRR conflict / agree | .639 / .733 | .613 / **.874** | **.780** / .758 |
| Personas, probing off | MRR | .653 | .749 | **.777** |
| | Hit@5 | .80 | .84 | **1.00** |
| | NDCG@10 | .538 | .586 | **.643** |
| Probes, reviewed (24; 22 with acceptable jobs) | acceptable MRR | .317 | .631 | **.676** |
| | Hit@5 / Hit@10 | .50 / .59 | .86 / .86 | **.91 / .91** |
| | avoided in top 10 | 2.79 | 0.17 | **0.08** |
| Development profiles, opening turn (120) | MRR | .064 | **.073** | .057 |
| | Hit@5 | .075 | **.092** | .042 |
| | NDCG@10 | .094 | **.102** | .077 |
| | Recall@50 | .391 | **.402** | .355 |

Paired differences, with lexical − without (MRR unless stated):
- Personas probing on: all +.042 [−.096, +.185]; conflict +.167 [−.005, +.350]; agree −.116 [−.318, +.086]. Probing off: all +.028; conflict **+.193 [+.018, +.375]**; agree −.182 [−.386, +.023].
- Development profiles: MRR −.016 [−.040, +.007]; **Hit@5 −.050 [−.092, −.017]; NDCG@10 −.025 [−.045, −.006]; Recall@50 −.047 [−.069, −.026]**.

Against the live default: without lexical, personas +.047 [−.034, +.139] (agree +.141 [+.000, +.310]), development MRR +.009 [−.004, +.025] (no metric worse); with lexical, personas +.090 [−.037, +.226], development Recall@50 −.036 [−.067, −.008].

Reading: lexical is a trade. It helps conflict personas (field named outright, so title-word matches) and the probes, and costs agree personas and the development profiles, where the labels follow the RIASEC profile. Without lexical, D2 title v3 improves on the live default on every benchmark here with no significant loss. Artifacts: `personas/`, `negation_probes/`, `dev_extractor_inputs/`.

## Added column: plain semantic + D2 title v3, no lexical

Arm `d2_title_v3_plain_semantic` (`occupation_semantic_fusion_zscore_dislike_title_v3.json`): cosine + `occupation_dual_encoder@finetuned_v1` (unchanged semantic) + `dislike_demotion@title_v3`, no lexical. Persona control reproduced exactly, so it is compared with the saved arms. Artifacts: `personas_plain_semantic/`, `negation_probes_plain_semantic/`, `dev_extractor_inputs/d2_title_v3_plain_semantic/`.

| Benchmark | Metric | Live default | Plain semantic + D2 | Signed v2 + D2, no lexical | Signed v2 + D2, with lexical |
|---|---|---|---|---|---|
| Personas, probing on | MRR | .680 | .680 | .728 | .770 |
| | Hit@5 / Hit@10 | .80 / .92 | .80 / .92 | .80 / .96 | .96 / .96 |
| | NDCG@10 | .539 | .551 | .591 | .620 |
| | MRR conflict / agree | .639 / .733 | .639 / .733 | .613 / .874 | .780 / .758 |
| Personas, probing off | MRR / Hit@5 / NDCG@10 | .653 / .80 / .538 | .653 / .80 / .548 | .749 / .84 / .586 | .777 / 1.00 / .643 |
| Probes (reviewed) | acceptable MRR / Hit@5 | .317 / .50 | .405 / .64 | .631 / .86 | .676 / .91 |
| | avoided in top 10 | 2.79 | 1.04 | 0.17 | 0.08 |
| Development profiles | MRR / Hit@5 | .064 / .075 | .065 / .075 | .073 / .092 | .057 / .042 |
| | NDCG@10 / Recall@50 | .094 / .391 | .094 / .393 | .102 / .402 | .077 / .355 |

- Plain semantic + D2 − live default: personas MRR +.000 (every best rank unchanged; NDCG@10 +.012); development profiles +.000. D2 alone barely moves these benchmarks; its effect shows on the probes (avoided 2.79 → 1.04).
- Signed v2 + D2 − plain semantic + D2 (no lexical in either): personas probing on +.047 [−.034, +.139], agree **+.141 [+.000, +.310]**; probing off +.095 [−.002, +.198], agree **+.272 [+.114, +.432]**, conflict −.044 [−.121, −.000]; development MRR +.008 [−.004, +.024]; probes MRR .405 → .631, avoided 1.04 → 0.17.
- Reading: with D2 in place, signed semantic (D1) still carries most of the gain. D2 enforces only dislikes whose words match a job title; removing dislike text from the semantic query is what stops it pulling the whole ranking towards rejected fields.
