"""Dislike probes (docs/LEXICAL_MATCHING_PLAN.md, D1/D2) on data/eval/negation_probes_v1.jsonl.

Each probe's turns go to the ranker as user history. RIASEC comes from one extractor reading of
the joined turns (same protocol as eval/task_retrieval_fusion.py). Full 378 depth.

Metrics per probe, aggregated per split and kind:
- avoided_top10: number of avoided occupations in the top 10 (lower is better); avoided_any_top10: 0/1.
- mrr: 1 / rank of the best acceptable occupation (not for only_negative probes).
- acceptable_top10: share of acceptable occupations in the top 10.

The probe set is a DRAFT pending team review. Only the calibration split may inform a threshold.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from recommender.models.ranker import RankingPipeline
from recommender.schemas import TYPES

ROOT = Path(__file__).resolve().parents[1]
PROBES = ROOT / "data/eval/negation_probes_v1.jsonl"
ARMS = {
    "fusion_zscore": "configs/ranking/occupation_semantic_fusion_zscore_v1.json",  # live default
    "fusion_zscore_d1": "configs/ranking/occupation_semantic_signed_fusion_zscore_v1.json",
    "semantic": "configs/ranking/occupation_semantic_finetuned_v1.json",
    "semantic_signed": "configs/ranking/occupation_semantic_signed_finetuned_v1.json",
    "fusion_zscore_bm25_ignore_v2": "configs/ranking/occupation_semantic_fusion_zscore_bm25_signed_ignore_v2.json",
    "fusion_zscore_d1_bm25_ignore_v2": "configs/ranking/occupation_semantic_signed_fusion_zscore_bm25_signed_ignore_v2.json",  # candidate baseline
    "d2_title_v3_nolex": "configs/ranking/occupation_semantic_signed_v2_fusion_zscore_dislike_title_v3.json",  # D1 v2 + D2 title v3, no lexical
    "d2_title_v3_plain_semantic": "configs/ranking/occupation_semantic_fusion_zscore_dislike_title_v3.json",  # finetuned_v1 + D2 title v3, no lexical
    # routing v3 (D1 v2 + signed lexical v3), then D2 on top
    "routing_v3": "configs/ranking/occupation_semantic_signed_v2_fusion_zscore_bm25_signed_ignore_v3.json",
    **{f"routing_v3_d2_{v}": f"configs/ranking/occupation_semantic_signed_v2_fusion_zscore_bm25_signed_ignore_v3_dislike_{v}.json"
       for v in ("title_v1", "semantic_v1", "title_semantic_v1", "title_v2", "semantic_v2", "title_semantic_v2",
                 "title_v3", "semantic_v3", "title_semantic_v3")},
}
GROUP_KEYS = ("split", "kind")


def probe_metrics(codes, probe):
    position = {c: i for i, c in enumerate(codes, 1)}
    top10 = set(codes[:10])
    out = dict(avoided_top10=len(top10 & set(probe["avoided"])),
               avoided_any_top10=int(bool(top10 & set(probe["avoided"]))),
               avoided_best_rank=min((position[c] for c in probe["avoided"] if c in position), default=None))
    if probe["acceptable"]:
        best = min((position[c] for c in probe["acceptable"] if c in position), default=None)
        out.update(mrr=1 / best if best else 0.0, best_rank=best,
                   acceptable_top10=len(top10 & set(probe["acceptable"])) / len(probe["acceptable"]))
    return out


def summarise(rows):
    keys = ("avoided_top10", "avoided_any_top10", "mrr", "acceptable_top10")
    groups = {"all": rows}
    for k in GROUP_KEYS:
        for value in sorted({r[k] for r in rows}):
            groups[f"{k}={value}"] = [r for r in rows if r[k] == value]
    return {g: {"n": len(rs), **{m: (sum(r["metrics"][m] for r in rs if m in r["metrics"]) /
                                      max(1, sum(m in r["metrics"] for r in rs))) for m in keys}}
            for g, rs in groups.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or ROOT / "reports/ranking/negation_probes" / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    if output.exists():
        parser.error("Output exists; choose a new directory")
    from pipeline.extractor import Extractor
    extractor = Extractor(ROOT / "models/extractor", ROOT / "data/labels.json")
    pipelines = {name: RankingPipeline(ROOT / "data/occupations.csv", config=ROOT / path) for name, path in ARMS.items()}
    probes = [json.loads(line) for line in PROBES.read_text(encoding="utf-8").splitlines() if line.strip()]
    results = {name: [] for name in ARMS}
    for probe in probes:
        reading = extractor.read(" ".join(probe["turns"]))
        values, observed = dict.fromkeys(TYPES, 0.0), set()
        for s in reading.slots:
            values[s.slot], _ = s.value, observed.add(s.slot)
        for name, pipe in pipelines.items():
            response = pipe.rank(values, observed, list(probe["turns"]), pipe.eligible_count)
            codes = [o.code for o in response.occupations]
            results[name].append(dict(id=probe["id"], split=probe["split"], kind=probe["kind"], status=response.status,
                                      metrics=probe_metrics(codes, probe), top10=codes[:10]))
    summary = {name: summarise(rows) for name, rows in results.items()}
    output.mkdir(parents=True)
    (output / "results.json").write_text(json.dumps(dict(
        summary=summary, per_probe=results, arms=ARMS,
        probes_sha256=hashlib.sha256(PROBES.read_bytes()).hexdigest(),
        ranking_config_sha256={n: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for n, p in ARMS.items()},
        quality_claim="draft probe set pending team review; D1 has no tuned parameter"), indent=2) + "\n", encoding="utf-8")
    lines = ["| Arm | Group | n | avoided in top 10 | any avoided in top 10 | MRR | acceptable in top 10 |", "|---|---|---|---|---|---|---|"]
    for name, groups in summary.items():
        for g, s in groups.items():
            lines.append(f"| {name} | {g} | {s['n']} | {s['avoided_top10']:.2f} | {s['avoided_any_top10']:.2f} | {s['mrr']:.3f} | {s['acceptable_top10']:.2f} |")
    lines += ["", "| Probe | kind | " + " | ".join(f"{n} avoided/best" for n in ARMS) + " |", "|---|---|" + "---|" * len(ARMS)]
    for i, probe in enumerate(probes):
        cells = [f"{results[n][i]['metrics']['avoided_top10']}/{results[n][i]['metrics'].get('best_rank', '-')}" for n in ARMS]
        lines.append(f"| {probe['id']} | {probe['kind']} | " + " | ".join(cells) + " |")
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print("Saved", output)


if __name__ == "__main__":
    main()
