# Cosine M1 with lexical and semantic retrieval

All three remaining combinations containing cosine M1 were evaluated successfully and appended once each to `reports/ranking/evaluation_history.csv`. Existing standalone and historical rows remain preserved. Presets are `configs/ranking/m1_cosine_bm25.json`, `m1_cosine_dense.json`, and `m1_cosine_three_way.json`.

The [pre-run protocol](FUSION_EXPERIMENT.md) fixes development/oracle inputs (60 profiles), 378 eligible occupations, raw query selection, unchanged BM25/MiniLM variants and equal-weight occupation-level RRF at k=60. No ranking code or metric definitions changed. No held-out data was used.

| Configuration | Hit@1 | Hit@5 | Hit@10 | MRR | NDCG@5 | NDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| Cosine M1 only (existing control) | .583333 | .883333 | .966667 | .713695 | .537513 | .528027 |
| Cosine M1 + BM25 | .133333 | .266667 | .400000 | .214472 | .166854 | .189500 |
| Cosine M1 + MiniLM | .083333 | .233333 | .466667 | .189187 | .153833 | .195790 |
| Cosine M1 + BM25 + MiniLM | .033333 | .116667 | .216667 | .101065 | .080535 | .100519 |

All combinations regress relative to cosine alone. Paired Hit@5 treatment-minus-control differences and 95% bootstrap intervals (2,000 samples, seed 425):

- BM25 addition: -.616667, interval [-.733333, -.500000]; 37 losses, zero gains.
- MiniLM addition: -.650000, interval [-.766667, -.516667]; 39 losses, zero gains.
- Both additions: -.766667, interval [-.866667, -.650000]; 46 losses, zero gains.

Replacing dot-product M1 with cosine improves Hit@5 within each matching fusion stack: .133333 → .266667 for BM25, .116667 → .233333 for MiniLM, and .050000 → .116667 for three-way fusion. These gains still leave every fused configuration substantially worse than cosine alone. Full metric comparisons, including uncertainty, are saved in [fusion_comparisons.json](fusion_comparisons.json); corresponding `*_paired.json` files retain per-profile changes.

Decision: retain all presets as experimental configurations; prefer cosine alone among these development/oracle conditions. Keep the production default unchanged pending tracker validation. These synthetic profiles favor structured RIASEC alignment by construction; this is not evidence that text retrieval is universally unhelpful. Diagnose weak retrieval signals and their equal influence in RRF before further fusion tuning.

Validation: shared evaluator completed all exports with no missing, ineligible or unranked targets. Identical input bytes were checked against the cosine-only control. All six paired comparisons passed compatibility checks, including historical matching dot-product stacks. Each new run ID occurs exactly once in central history with `m1=cosine_similarity@baseline_v1` and the enabled M3/M4 identities. Existing ranking correctness coverage remains the 172-test passing suite recorded in the standalone report; configuration-only experiments introduce no scoring changes.

Each run directory (`bm25/`, `dense/`, `three_way/`) retains exact configs, resolved parameters, source/data/model/cache fingerprints, predictions, metrics and latency. Implementation source is preserved in the existing `source_snapshot.zip`; each new configuration is also saved verbatim in its run. Reproduce with `python -m recommender.evaluate --ranking-config configs/ranking/m1_cosine_<bm25|dense|three_way>.json --output <fresh-directory>`.
