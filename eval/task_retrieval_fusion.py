"""Single-turn task-retrieval check for the #1 fusion variants (docs/SEMANTIC_LEDGER_PLAN.md §12.3).

Repeats the held-out task-retrieval protocol of `python -m training.occupation_semantic evaluate`
(reports/ranking/occupation_semantic_v1/evaluation.json) for fusion arms only: one task sentence per
query, RIASEC read from that sentence by the extractor, full 378 depth, binary multi-owner labels,
paired group bootstrap (2,000 resamples, seed 425). The control and semantic-only arms reproduce the
saved run, so the comparison is on identical inputs.

Formal task retrieval, not conversational career relevance. No parameter is chosen here: the
variants are fixed from first principles, and this run only reports them.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from recommender.models.ranker import RankingPipeline
from recommender.schemas import TYPES
from training.occupation_semantic import digest, metrics, provenance, read, write

ROOT = Path(__file__).resolve().parents[1]
ARMS = {
    "fusion": "configs/ranking/occupation_semantic_fusion_v1.json",  # control, rrf_k60
    "semantic": "configs/ranking/occupation_semantic_finetuned_v1.json",
    "fusion_tieavg": "configs/ranking/occupation_semantic_fusion_tieavg_v1.json",
    "fusion_zscore": "configs/ranking/occupation_semantic_fusion_zscore_v1.json",
}
CONTRASTS = [("fusion_tieavg", "fusion"), ("fusion_zscore", "fusion"), ("fusion_tieavg", "semantic"),
             ("fusion_zscore", "semantic"), ("fusion", "semantic")]
SAVED = ROOT / "reports/ranking/occupation_semantic_v1/evaluation.json"
SAVED_MODE = {"fusion": "fusion", "semantic": "finetuned"}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/occupation_semantic_v1.json")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or ROOT / "reports/ranking/fusion_variants_v1/task_retrieval" / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    if output.exists():
        parser.error("Output exists; choose a new directory")
    cfg = json.loads(args.config.read_text())
    data = read(ROOT / cfg["artifacts"] / "data.json")
    from pipeline.extractor import Extractor
    extractor = Extractor(ROOT / "models/extractor", ROOT / "data/labels.json")
    pipelines = {name: RankingPipeline(config=ROOT / path) for name, path in ARMS.items()}
    saved = json.loads(SAVED.read_text())["summary"]
    summary, all_rows = {}, {}
    for split in ("test", "occupation_heldout"):
        examples = [e for e in data["examples"] if e["split"] == split]
        if split == "occupation_heldout":
            examples = [{**e, "positives": sorted(set(e["positives"]) & set(data["heldout_codes"]))} for e in examples]
        rows, began = [], time.perf_counter()
        for j, e in enumerate(examples):
            reading = extractor.read(e["text"])
            v, obs = dict.fromkeys(TYPES, 0.), set()
            for s in reading.slots:
                v[s.slot] = s.value
                obs.add(s.slot)
            row = {"group": e["group"], "positives": e["positives"], "observed": sorted(obs), "modes": {}}
            for mode, pipe in pipelines.items():
                order = [o.code for o in pipe.rank(v, obs, [e["text"]], 378).occupations]
                row["modes"][mode] = {"metrics": metrics(order, set(e["positives"])), "top10": order[:10]}
            rows.append(row)
            if j % 200 == 0:
                print(split, j, "/", len(examples), flush=True)
        summary[split] = {"groups": len(rows), "seconds": time.perf_counter() - began, "modes": {}}
        for mode in ARMS:
            summary[split]["modes"][mode] = {k: float(np.mean([r["modes"][mode]["metrics"][k] for r in rows]))
                                             for k in rows[0]["modes"][mode]["metrics"]}
        summary[split]["reproduces_saved_mrr"] = {
            mode: abs(summary[split]["modes"][mode]["mrr"] - saved[split]["modes"][old]["mrr"]) < 1e-9
            for mode, old in SAVED_MODE.items()}
        rng = np.random.default_rng(cfg["seed"])
        for a, b in CONTRASTS:
            dif = np.array([r["modes"][a]["metrics"]["mrr"] - r["modes"][b]["metrics"]["mrr"] for r in rows])
            boot = [float(np.mean(rng.choice(dif, size=len(dif), replace=True))) for _ in range(2000)]
            summary[split][f"{a}_minus_{b}_mrr"] = {"difference": float(dif.mean()),
                                                    "ci95": np.quantile(boot, [.025, .975]).tolist()}
        all_rows[split] = rows
    output.mkdir(parents=True)
    write(output / "evaluation.json", {
        "summary": summary, "arms": ARMS,
        "fusion_identifiers": {name: p.fusion_identifier for name, p in pipelines.items()},
        "ranking_config_sha256": {name: digest(ROOT / path) for name, path in ARMS.items()},
        "source_sha256": {p: digest(ROOT / p) for p in ("recommender/models/fusion.py", "recommender/models/ranker.py",
                                                         "eval/task_retrieval_fusion.py")},
        "provenance": provenance(cfg), "split_manifest_sha256": digest(ROOT / cfg["artifacts"] / "split_manifest.json"),
        "ranking_depth": 378, "label_definition": "binary multi-owner task/context retrieval proxy; not graded career suitability",
        "quality_claim": "single-turn task retrieval only; development check for #1 fusion variants, nothing tuned"})
    write(output / "per_query.json", all_rows)
    print(json.dumps(summary, indent=2), flush=True)
    print("Saved", output)


if __name__ == "__main__":
    main()
