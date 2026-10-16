"""Run the frozen M2 structured oracle development baseline (no held-out access)."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
from time import perf_counter

from .data_loader import read_profiles
from .evaluation import aggregate, evaluate_ranking
from .schemas import OracleInput
from .structured import StructuredRanker


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(config_path: Path, output: Path) -> dict:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    # M2 freezes one explicit condition. Other modes require separate adapters/runs.
    expected = {
        "run_id": "m2_structured_oracle_dev_v1", "split": "development", "input_mode": "oracle",
        "occupations": "data/occupations.csv", "profiles": "data/dev_profiles.jsonl",
        "eligibility": "job_zone in [4,5] and complete supplied RIASEC z-scores",
        "algorithm": "signed dot product over all six oracle dimensions; score descending, code ascending",
        "ranking_depth": "full_pool", "seed": 425,
    }
    if config != expected:
        raise ValueError("M2 baseline requires the frozen development/oracle configuration")
    if config["split"] != "development" or config["input_mode"] != "oracle":
        raise ValueError("M2 supports development/oracle only")
    if Path(config["profiles"]) != Path("data/dev_profiles.jsonl"):
        raise ValueError("M2 must not access held-out profiles")
    if output.exists():
        raise ValueError("output already exists; preserve runs and choose a new directory")
    profiles = read_profiles(Path(config["profiles"]))
    ids = [p["id"] for p in profiles]
    if not ids or any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("empty profiles or invalid/duplicate profile IDs")
    start = perf_counter()
    ranker = StructuredRanker(Path(config["occupations"]))
    load_seconds = perf_counter() - start
    pool = ranker.eligible_codes
    snapshots, metrics, predictions, times = [], [], [], []
    for profile in profiles:
        inputs = OracleInput.from_profile(profile)
        snapshots.append({"profile_id": profile["id"], **asdict(inputs)})
        start = perf_counter()
        response = ranker.rank_oracle(inputs)
        times.append(perf_counter() - start)
        codes = [o.code for o in response.occupations]
        result = evaluate_ranking(codes, profile.get("target"), profile["related"], pool)
        metrics.append({"profile_id": profile["id"], "status": response.status, **result})
        for occupation in response.occupations:
            predictions.append({"run_id": config["run_id"], "profile_id": profile["id"],
                                "input_mode": "oracle", "target_code": profile.get("target"),
                                **asdict(occupation)})
    summary = aggregate(metrics)
    source = {str(p): digest(p) for p in sorted(Path("recommender").glob("*.py"))}
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    output.mkdir(parents=True)
    save_json(output / "config.json", config)
    save_json(output / "inputs.json", snapshots)
    save_json(output / "per_profile.json", metrics)
    save_json(output / "metrics.json", summary)
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    save_json(output / "latency.json", {
        "load_seconds": load_seconds, "preloaded_rank_median_seconds": statistics.median(times),
        "preloaded_rank_p95_seconds": sorted(times)[math_ceil_95(len(times)) - 1],
        "sample_count": len(times), "method": "one rank call per development profile; nearest-rank p95; excludes metrics/export",
    })
    save_json(output / "manifest.json", {
        "evaluator_version": "m2-v1", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": config["run_id"], "control_run_id": None,
        "hypothesis": "Establish the structured-only oracle development control before adding retrieval",
        "treatment": "none", "input_mode": "oracle", "split": "development",
        "git_revision": revision.stdout.strip(), "working_tree_dirty": bool(status.stdout.strip()),
        "source_sha256": source, "config_sha256": digest(output / "config.json"),
        "dataset_sha256": {config[k]: digest(config[k]) for k in ("profiles", "occupations")},
        "inputs_sha256": digest(output / "inputs.json"), "pool": sorted(pool),
        "eligibility": config["eligibility"], "ranking_depth": "full eligible pool; MRR not truncated at 100",
        "metric_definitions": "Hit@5 exact target; full-pool MRR; NDCG@10 gains 2^grade-1, target=4/short=3/long=2/supplemental=1, max grade per eligible code, zero IDCG=0; all profiles in denominator",
        "python": platform.python_version(), "dependencies": "standard library only",
        "model_revision": None, "task_index_fingerprint": None, "seed": config["seed"],
        "test_summary": "See reports/ranking/m2/DECISION.md for executed acceptance results",
        "command": f"python -m recommender.evaluate --config {config_path.as_posix()} --output <new-directory>",
    })
    return summary


def math_ceil_95(n):
    return (95 * n + 99) // 100


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/m2.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args.config, args.output), indent=2))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"M2 evaluation FAILED: {error}\n")


if __name__ == "__main__":
    main()
