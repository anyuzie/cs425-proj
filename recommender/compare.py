"""Compare compatible saved runs using paired profile bootstrap intervals."""

import argparse
import csv
import json
from pathlib import Path
import random

from .evaluation import METRICS
from .evaluate import save_json


def paired_comparison(control: Path, treatment: Path, *, samples=2000, seed=425):
    def read(directory, name):
        return json.loads((directory / name).read_text(encoding="utf-8"))
    left, right = read(control, "manifest.json"), read(treatment, "manifest.json")
    for key in ("input_mode", "split", "dataset_sha256", "inputs_sha256", "pool", "eligibility",
                "ranking_depth", "metric_definitions", "evaluator_version"):
        if left[key] != right[key]:
            raise ValueError(f"incompatible paired runs: {key}")
    def indexed(directory):
        rows = read(directory, "per_profile.json")
        result = {r["profile_id"]: r for r in rows}
        if not rows or len(result) != len(rows):
            raise ValueError("empty or duplicate profile metrics")
        return result
    a, b = indexed(control), indexed(treatment)
    if a.keys() != b.keys():
        raise ValueError("paired runs must contain identical profile IDs")
    diffs = [{"profile_id": pid, **{m: b[pid][m] - a[pid][m] for m in METRICS},
              "control_target_rank": a[pid]["target_rank"],
              "treatment_target_rank": b[pid]["target_rank"]} for pid in sorted(a)]
    rng = random.Random(seed)
    bootstrap = {m: [] for m in METRICS}
    for _ in range(samples):
        selected = rng.choices(diffs, k=len(diffs))
        for m in METRICS:
            bootstrap[m].append(sum(row[m] for row in selected) / len(selected))
    summary = {"control_run_id": left["run_id"], "treatment_run_id": right["run_id"],
               "profiles": len(diffs), "bootstrap_samples": samples, "seed": seed,
               "method": "paired profile bootstrap; percentile 95% CI; treatment minus control",
               "metrics": {}}
    for m in METRICS:
        values = sorted(bootstrap[m])
        summary["metrics"][m] = {
            "mean_difference": sum(row[m] for row in diffs) / len(diffs),
            "ci95": [values[int(.025 * (samples - 1))], values[int(.975 * (samples - 1))]],
            "improved": sum(row[m] > 0 for row in diffs),
            "regressed": sum(row[m] < 0 for row in diffs),
        }
    return summary, diffs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("treatment", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError("comparison output already exists")
        summary, diffs = paired_comparison(args.control, args.treatment)
        args.output.mkdir(parents=True)
        save_json(args.output / "comparison.json", summary)
        with (args.output / "paired_diffs.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(diffs[0]))
            writer.writeheader()
            writer.writerows(diffs)
        print(json.dumps(summary, indent=2))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"M2 comparison FAILED: {error}\n")


if __name__ == "__main__":
    main()
