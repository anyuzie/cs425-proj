"""Append-only evaluation registry; detailed artifacts remain authoritative."""

import argparse
from contextlib import contextmanager
import csv
import hashlib
import io
import json
import math
from pathlib import Path

from recommender.data.data_loader import read_profiles
from recommender.experiments.evaluation import METRICS, LEGACY_METRICS, RECALL_METRICS, aggregate, evaluate_ranking

DEFAULT_HISTORY = Path("reports/ranking/evaluation_history.csv")
FIELDS = ("run_id", "timestamp", "m1", "m3", "m4", "fusion", "description",
          "candidate_count", "profile_count", "hit_at_1", "hit_at_5", "hit_at_10",
          "mrr", "ndcg_at_5", "ndcg_at_10", *RECALL_METRICS, "config_hash", "dataset_fingerprint",
          "git_commit", "report_path")


def dataset_fingerprint(hashes: dict) -> str:
    """Hash the canonical path-to-SHA256 mapping, including every input dataset."""
    return hashlib.sha256(json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@contextmanager
def registry_lock(path):
    handle = path.open("x", encoding="utf-8")
    try:
        yield
    finally:
        handle.close()
        path.unlink()


def append_history(report: Path, *, history: Path = DEFAULT_HISTORY,
                   components: dict | None = None, metrics: dict | None = None,
                   description: str | None = None) -> bool:
    """Register a complete run; return False for an already registered run ID.

    Rankers list their algorithm IDs in manifest.components (m1/m3/m4/fusion).
    Optional overrides support verified imports of older saved runs.
    A sibling exclusive lock serializes writers; a busy registry raises an error.
    """
    report, history = Path(report), Path(history)
    for name in ("manifest.json", "config.json", "inputs.json", "predictions.jsonl",
                 "per_profile.json", "metrics.json", "latency.json"):
        if not (report / name).is_file():
            raise ValueError(f"incomplete evaluation: missing {name}")
    manifest = json.loads((report / "manifest.json").read_text(encoding="utf-8"))
    metrics = metrics if metrics is not None else json.loads((report / "metrics.json").read_text(encoding="utf-8"))
    components = components if components is not None else manifest["components"]
    for key in ("run_id", "timestamp_utc", "input_mode", "split", "eligibility",
                "evaluator_version", "metric_definitions", "pool", "dataset_sha256",
                "config_sha256", "git_revision"):
        if key not in manifest:
            raise ValueError(f"manifest missing {key}")
    if not manifest["run_id"] or not manifest["pool"] or metrics["profile_count"] < 1:
        raise ValueError("empty run ID, candidate pool, or profile set")
    for key in METRICS:
        if key not in metrics and key in RECALL_METRICS:
            legacy = manifest["evaluator_version"] not in {"m2-v3", "m2-v4"}
            before_recall50 = key == "recall_at_50" and manifest["evaluator_version"] != "m2-v4"
            if legacy or before_recall50:
                continue
        value = metrics[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"invalid metric: {key}")
    algorithms = {key: components.get(key, "") for key in ("m1", "m3", "m4", "fusion")}
    if any(not isinstance(v, str) or v.strip().lower() in {"none", "0", "n/a"} for v in algorithms.values()):
        raise ValueError("algorithm identifiers must be strings; unused components must be empty")
    row = {"run_id": manifest["run_id"], "timestamp": manifest["timestamp_utc"],
           **algorithms, "description": description if description is not None else manifest["hypothesis"],
           "candidate_count": len(manifest["pool"]), "profile_count": metrics["profile_count"],
           **{key: metrics.get(key, "") for key in METRICS}, "config_hash": manifest["config_sha256"],
           "dataset_fingerprint": dataset_fingerprint(manifest["dataset_sha256"]),
           "git_commit": manifest["git_revision"], "report_path": report.as_posix()}
    history.parent.mkdir(parents=True, exist_ok=True)
    lock = history.with_name(history.name + ".lock")
    with registry_lock(lock):
        exists = history.exists() and history.stat().st_size > 0
        if exists:
            with history.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                columns = reader.fieldnames
                old_rows = list(reader)
                if columns not in (list(FIELDS), [f for f in FIELDS if f != "recall_at_50"],
                                   [f for f in FIELDS if f not in RECALL_METRICS]):
                    raise ValueError("history CSV schema mismatch")
                if any(old["run_id"] == row["run_id"] for old in old_rows):
                    return False
            if columns != list(FIELDS):
                suffix = ".pre_recall50.bak" if "recall_at_20" in columns else ".pre_recall.bak"
                backup = history.with_suffix(history.suffix + suffix)
                if not backup.exists():
                    backup.write_bytes(history.read_bytes())
                buffer = io.StringIO(newline="")
                writer = csv.DictWriter(buffer, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(old_rows)
                temporary = history.with_suffix(history.suffix + ".migrating")
                temporary.write_text(buffer.getvalue(), encoding="utf-8", newline="")
                temporary.replace(history)
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        with history.open("a", newline="", encoding="utf-8") as handle:
            handle.write(buffer.getvalue())
        return True


def register_legacy_baseline(report: Path, *, history: Path = DEFAULT_HISTORY) -> bool:
    """Recompute added cutoffs from original predictions, verifying saved inputs."""
    report = Path(report)
    manifest = json.loads((report / "manifest.json").read_text(encoding="utf-8"))
    if manifest["run_id"] != "m2_structured_oracle_dev_v1" or manifest["input_mode"] != "oracle" or manifest["split"] != "development":
        raise ValueError("legacy import supports only the saved M2 baseline")
    for name, expected in manifest["dataset_sha256"].items():
        data = Path(name).read_bytes()
        # Git on Windows may check out historical LF text as CRLF. Accept only
        # an exact original hash after newline normalization, never content drift.
        if expected not in {hashlib.sha256(data).hexdigest(), hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()}:
            raise ValueError(f"dataset fingerprint mismatch: {name}")
    for filename, key in (("config.json", "config_sha256"), ("inputs.json", "inputs_sha256")):
        if hashlib.sha256((report / filename).read_bytes()).hexdigest() != manifest[key]:
            raise ValueError(f"artifact fingerprint mismatch: {filename}")
    predictions = {}
    for line in (report / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["run_id"] != manifest["run_id"]:
            raise ValueError("prediction run ID mismatch")
        predictions.setdefault(row["profile_id"], []).append(row)
    profiles = read_profiles(Path("data/dev_profiles.jsonl"))
    if set(predictions) != {p["id"] for p in profiles}:
        raise ValueError("prediction profile set mismatch")
    pool, rows = set(manifest["pool"]), []
    for profile in profiles:
        ranked = sorted(predictions[profile["id"]], key=lambda row: row["rank"])
        if [r["rank"] for r in ranked] != list(range(1, len(pool) + 1)):
            raise ValueError("incomplete full-pool predictions")
        rows.append(evaluate_ranking([r["code"] for r in ranked], profile["target"], profile["related"], pool))
    metrics = aggregate(rows)
    original = json.loads((report / "metrics.json").read_text(encoding="utf-8"))
    for key in (*LEGACY_METRICS, "profile_count"):
        if metrics[key] != original[key]:
            raise ValueError(f"saved baseline metric mismatch: {key}")
    return append_history(report, history=history, metrics=metrics,
                          components={"m1": "weighted_dot_product"},
                          description="M2 structured-only oracle development baseline; added cutoffs recomputed from saved predictions")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--legacy-baseline", action="store_true")
    args = parser.parse_args()
    register = register_legacy_baseline if args.legacy_baseline else append_history
    print("appended" if register(args.report, history=args.history) else "already registered")


if __name__ == "__main__":
    main()
