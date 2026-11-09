"""Calibrate D2's semantic threshold on the CALIBRATION negation probes only (docs/LEXICAL_MATCHING_PLAN.md).

Base ranking: routing-v3 candidate (`occupation_semantic_signed_v2_fusion_zscore_bm25_signed_ignore_v3.json`),
no post-fusion stage. For each threshold on the grid, D2 is applied to the base ranking with the
`semantic` and `title_or_semantic` matchers; `title` (no threshold) is reported alongside.

Selection rule (fixed before running): the HIGHEST threshold whose mean avoided-in-top-10 is within
0.1 of the best value on the grid and whose acceptable MRR is at least the base's. Higher thresholds
demote fewer occupations, so ties go to the more conservative setting. The test split is never read.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from eval.negation_probes import PROBES, probe_metrics
from recommender.configuration import load_parameters
from recommender.data.data_loader import load_eligible_occupations
from recommender.models.dislike_demotion import DislikeDemotion
from recommender.models.ranker import RankingPipeline
from recommender.schemas import TYPES

ROOT = Path(__file__).resolve().parents[1]
BASE = "configs/ranking/occupation_semantic_signed_v2_fusion_zscore_bm25_signed_ignore_v3.json"
GRID = [round(0.20 + 0.05 * i, 2) for i in range(11)]  # .20 ... .70
TOLERANCE = 0.1


def mean(rows, key):
    values = [r[key] for r in rows if key in r]
    return sum(values) / len(values) if values else None


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists")
    probes = [p for p in map(json.loads, PROBES.read_text(encoding="utf-8").splitlines()) if p["split"] == "calibration"]
    from pipeline.extractor import Extractor
    extractor = Extractor(ROOT / "models/extractor", ROOT / "data/labels.json")
    pipe = RankingPipeline(ROOT / "data/occupations.csv", config=ROOT / BASE)
    occupations = load_eligible_occupations(ROOT / "data/occupations.csv")
    base_rows, selected = [], {}
    for p in probes:
        values, observed = dict.fromkeys(TYPES, 0.0), set()
        for s in extractor.read(" ".join(p["turns"])).slots:
            values[s.slot] = s.value
            observed.add(s.slot)
        response = pipe.rank(values, observed, list(p["turns"]), pipe.eligible_count)
        codes = [o.code for o in response.occupations]
        selected[p["id"]] = codes
        base_rows.append(probe_metrics(codes, p))

    def run(matcher, threshold):
        params = deepcopy(load_parameters("dislike_demotion_title_semantic_v1_parameters"))
        params.update(matcher=matcher, semantic_threshold=threshold, calibration_only=True)
        stage = DislikeDemotion(occupations, params)
        rows = []
        for p in probes:
            matched = stage.matches(stage.targets(list(p["turns"])))
            codes = [c for c in selected[p["id"]] if c not in matched] + [c for c in selected[p["id"]] if c in matched]
            rows.append(dict(probe_metrics(codes, p), id=p["id"], kind=p["kind"], demoted=len(matched)))
        return rows

    summary = {"base": dict(avoided_top10=mean(base_rows, "avoided_top10"), mrr=mean(base_rows, "mrr"),
                            any=mean(base_rows, "avoided_any_top10"), acceptable_top10=mean(base_rows, "acceptable_top10"))}
    per = {}
    for matcher, grid in (("title", [None]), ("semantic", GRID), ("title_or_semantic", GRID)):
        for t in grid:
            rows = run(matcher, t if t is not None else 1.0)
            key = f"{matcher}@{t}" if t is not None else matcher
            per[key] = rows
            summary[key] = dict(avoided_top10=mean(rows, "avoided_top10"), mrr=mean(rows, "mrr"),
                                any=mean(rows, "avoided_any_top10"), acceptable_top10=mean(rows, "acceptable_top10"),
                                demoted_mean=mean(rows, "demoted"))
    choice = {}
    for matcher in ("semantic", "title_or_semantic"):
        keys = [f"{matcher}@{t}" for t in GRID]
        best = min(summary[k]["avoided_top10"] for k in keys)
        ok = [t for t in GRID if summary[f"{matcher}@{t}"]["avoided_top10"] <= best + TOLERANCE
              and summary[f"{matcher}@{t}"]["mrr"] >= summary["base"]["mrr"] - 1e-12]
        choice[matcher] = max(ok) if ok else None
    args.output.mkdir(parents=True)
    (args.output / "calibration.json").write_text(json.dumps(dict(
        rule=__doc__, grid=GRID, tolerance=TOLERANCE, base_config=BASE, split="calibration",
        probe_ids=[p["id"] for p in probes], summary=summary, choice=choice, per_probe=per), indent=2) + "\n", encoding="utf-8")
    print(f"{'setting':24} {'avoided@10':>10} {'any':>5} {'MRR':>6} {'acc@10':>6} {'demoted':>8}")
    for k, s in summary.items():
        print(f"{k:24} {s['avoided_top10']:10.2f} {s['any']:5.2f} {s['mrr']:6.3f} {s['acceptable_top10']:6.2f} {s.get('demoted_mean', 0) or 0:8.1f}")
    print("choice:", choice)


if __name__ == "__main__":
    main()
