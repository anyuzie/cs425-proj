"""Run six text treatments and append to the existing comparison CSV."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from .evaluate import run, save_json, digest
from .evaluation import METRICS, aggregate, evaluate_ranking
from .history import dataset_fingerprint
from .compare import paired_comparison
from ..semantic.context_tasks import CONFIGURATIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-root", type=Path, required=True)
    args = parser.parse_args()
    root = Path("reports/ranking/context_tasks_v1")
    root.mkdir(parents=True, exist_ok=True)
    csv_path = Path("reports/ranking/semantic_experiments_consolidated.csv")
    original = csv_path.read_bytes()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields, old = reader.fieldnames, list(reader)
    labels = {p["id"]: p for p in map(json.loads, Path("data/dev_profiles.jsonl").read_text().splitlines())}
    # Re-evaluate saved predictions with the current evaluator, without giving labels
    # to the scorer. Require matching raw inputs before historical comparisons.
    for row in old:
        if row["experiment"].endswith("__context_tasks_v1"):
            raise ValueError("Treatments already appended; choose a new experiment version")
        directory = args.historical_root / row["report_path"]
        manifest = json.loads((directory / "manifest.json").read_text())
        for path, expected in manifest["dataset_sha256"].items():
            if digest(path) != expected:
                raise ValueError(f"Historical dataset changed: {path}")
        rankings = {}
        for prediction in map(json.loads, (directory / "predictions.jsonl").read_text().splitlines()):
            rankings.setdefault(prediction["profile_id"], []).append(prediction)
        if set(rankings) != set(labels):
            raise ValueError("Historical profile IDs differ")
        profiles = [{"profile_id": pid, **evaluate_ranking(
            [p["code"] for p in sorted(rankings[pid], key=lambda p: p["rank"])],
            labels[pid]["target"], labels[pid]["related"], set(manifest["pool"]))} for pid in labels]
        metrics = aggregate(profiles)
        if any(abs(metrics[key] - float(row[key])) > 1e-12 for key in METRICS):
            raise ValueError(f"Historical metric changed: {row['experiment']}")
        control = root / "historical_verified" / row["experiment"]
        control.mkdir(parents=True, exist_ok=False)
        save_json(control / "per_profile.json", profiles)
        save_json(control / "metrics.json", metrics)
        # Mark this as a re-evaluation, not a new encoder run.
        save_json(control / "provenance.json", {"source_report": row["report_path"],
            "predictions_sha256": digest(directory / "predictions.jsonl"),
            "operation": "recompute all ten metrics from frozen predictions using m2-v4"})
    base = {"components": {"m1": {"algorithm": "weighted_dot_product", "variant": "baseline_v1"},
                           "m3": None, "m4": None}, "fusion": None}
    history = root / "evaluation_history.csv"
    control_path = root / "weighted_dot_product_control"
    run(output=control_path, history=history, ranking_config=base)
    baseline = json.loads((control_path / "metrics.json").read_text())
    if any(abs(baseline[key] - float(old[0][key])) > 1e-12 for key in METRICS):
        raise ValueError("Baseline metrics no longer match")
    new_rows = []
    for name, parameters in CONFIGURATIONS.items():
        config = {**base, "reranking": parameters}
        config_path = Path("configs/ranking") / f"context_tasks_{name}_v1.json"
        save_json(config_path, config)
        report = root / name
        print(f"RUN {name}", flush=True)
        metrics = run(output=report, history=history, ranking_config=config_path)
        manifest = json.loads((report / "manifest.json").read_text())
        control_manifest = json.loads((control_path / "manifest.json").read_text())
        if manifest["inputs_sha256"] != control_manifest["inputs_sha256"]:
            raise ValueError("Treatment changed structured inputs")
        comparison, diffs = paired_comparison(control_path, report)
        save_json(report / "comparison_vs_dot_product.json", comparison)
        save_json(report / "paired_diffs_vs_dot_product.json", diffs)
        previous = json.loads((root / "historical_verified" / name / "per_profile.json").read_text())
        current = json.loads((report / "per_profile.json").read_text())
        previous = {r["profile_id"]: r for r in previous}
        save_json(report / "comparison_vs_description.json", {
            "operation": "m2-v4 re-evaluation of frozen description predictions vs new treatment",
            "mean_difference": {key: metrics[key] - sum(p[key] for p in previous.values()) / len(previous) for key in METRICS},
            "per_profile": [{"profile_id": p["profile_id"], **{key: p[key] - previous[p["profile_id"]][key] for key in METRICS}} for p in current]})
        source_row = next(r for r in old if r["experiment"] == name)
        row = {**source_row, "experiment": name + "__context_tasks_v1", **{key: metrics[key] for key in METRICS},
               "profile_count": metrics["profile_count"], "evaluator_version": "m2-v4", "metric_extension": "",
               "run_id": manifest["run_id"], "inputs_sha256": manifest["inputs_sha256"],
               "dataset_fingerprint": dataset_fingerprint(manifest["dataset_sha256"]), "report_path": report.as_posix()}
        new_rows.append(row)
        print(json.dumps({"experiment": name, **{key: metrics[key] for key in METRICS}}), flush=True)
    # Leave existing rows untouched; append only after all runs succeed.
    if csv_path.read_bytes() != original:
        raise ValueError("CSV changed during experiment")
    (root / "semantic_experiments_consolidated.before.csv").write_bytes(original)
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        csv.DictWriter(handle, fieldnames=fields).writerows(new_rows)
    save_json(root / "run_summary.json", {"original_csv_sha256": hashlib.sha256(original).hexdigest(),
        "treatments": new_rows, "historical_metrics_verified": len(old),
        "split": "development only; held-out profile files never loaded",
        "context": "occupation Description prose, not numerical O*NET Work Context ratings"})


if __name__ == "__main__":
    main()
