"""Paired development evaluation of dot product and description reranking."""
import argparse
import json
from pathlib import Path

from .evaluate import run, save_json, digest
from .compare import paired_comparison
from .evaluation import relevance_labels
from ..data_loader import read_profiles, load_eligible_occupations


def export_recall(root):
    """Derive supplementary recall from saved predictions, without model inference."""
    for name in ("control", "treatment"):
        manifest = json.loads((root / name / "manifest.json").read_text(encoding="utf-8"))
        if any(digest(path) != sha for path, sha in manifest["dataset_sha256"].items()):
            raise ValueError("Recall export requires the original fingerprint-matching datasets")
    profiles = {p["id"]: p for p in read_profiles(Path("data/dev_profiles.jsonl"))}
    pool = {o.code for o in load_eligible_occupations(Path("data/occupations.csv"))}
    supplementary = {}
    for name, directory in (("control", root / "control"), ("treatment", root / "treatment")):
        predictions = {}
        for line in (directory / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            predictions.setdefault(row["profile_id"], []).append(row)
        rows = []
        for pid, predictions_for_profile in predictions.items():
            profile = profiles[pid]
            codes = [r["code"] for r in sorted(predictions_for_profile, key=lambda r: r["rank"])]
            relevant = set(relevance_labels(profile.get("target"), profile["related"], pool))
            rows.append({"profile_id": pid, **{
                f"recall_at_{k}": len(set(codes[:k]) & relevant) / len(relevant) if relevant else 0.
                for k in (5, 10, 20, 50)},
                "target_in_baseline_top100": profile.get("target") in codes[:100]})
        supplementary[name] = {"aggregate": {
            key: sum(r[key] for r in rows) / len(rows)
            for key in ("recall_at_5", "recall_at_10", "recall_at_20", "recall_at_50", "target_in_baseline_top100")}, "per_profile": rows}
        save_json(directory / "recall_metrics.json", supplementary[name])
    save_json(root / "supplementary_recall.json", supplementary)
    return supplementary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/ranking/description_top100"))
    parser.add_argument("--recall-only", action="store_true", help="Add recall exports to existing paired runs; do not rerun inference")
    args = parser.parse_args()
    root = args.output
    if args.recall_only:
        supplementary = export_recall(root)
        print(json.dumps({k: v["aggregate"] for k, v in supplementary.items()}, indent=2))
        return
    root.mkdir(parents=True)  # Never silently reuse stale experiment artifacts.
    control, treatment = root / "control", root / "treatment"
    for path, config in ((control, "configs/ranking/baseline_v1.json"),
                         (treatment, "configs/ranking/description_top100.json")):
        run(output=path, ranking_config=Path(config), history=root / "history.csv")
    comparison, diffs = paired_comparison(control, treatment)
    save_json(root / "comparison.json", comparison)
    save_json(root / "paired_diffs.json", diffs)
    supplementary = export_recall(root)
    profiles = {p["id"]: p for p in read_profiles(Path("data/dev_profiles.jsonl"))}
    # Inspect the largest five target-rank gains and losses without reproducing utterances.
    ordered = sorted(diffs, key=lambda d: (d["control_target_rank"] or 10000) -
                                        (d["treatment_target_rank"] or 10000), reverse=True)
    review = ["# Target-rank failure review", "", "Synthetic development/oracle fixtures; labels have not been independently reviewed.", ""]
    for label, cases in (("Largest gains", ordered[:5]), ("Largest losses", ordered[-5:])):
        review.extend(["## " + label, ""])
        for row in cases:
            pid = row["profile_id"]
            review.append(f"- {pid}; target {profiles[pid].get('target')}; rank {row['control_target_rank']} → {row['treatment_target_rank']}. Review semantic wording and relevance label before attributing cause.")
        review.append("")
    (root / "failure_review.md").write_text("\n".join(review), encoding="utf-8")
    print(json.dumps({"paired": comparison, "supplementary": {k: v["aggregate"] for k, v in supplementary.items()}}, indent=2))


if __name__ == "__main__":
    main()
