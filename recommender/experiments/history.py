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
from recommender.experiments.evaluation import METRICS, LEGACY_METRICS, aggregate, evaluate_ranking

DEFAULT_HISTORY = Path("reports/ranking/evaluation_history.csv")
FIELDS = ("run_id", "timestamp", "m1", "m3", "m4", "fusion", "description",
          "candidate_count", "profile_count", "hit_at_1", "hit_at_5", "hit_at_10",
          "mrr", "ndcg_at_5", "ndcg_at_10", "config_hash", "dataset_fingerprint",
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

    Future rankers use manifest.components with m1/m3/m4/fusion algorithm IDs.
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
        value = metrics[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"invalid metric: {key}")
    algorithms = {key: components.get(key, "") for key in ("m1", "m3", "m4", "fusion")}
    if any(not isinstance(v, str) or v.strip().lower() in {"none", "0", "n/a"} for v in algorithms.values()):
        raise ValueError("algorithm identifiers must be strings; unused components must be empty")
    row = {"run_id": manifest["run_id"], "timestamp": manifest["timestamp_utc"],
           **algorithms, "description": description if description is not None else manifest["hypothesis"],
           "candidate_count": len(manifest["pool"]), "profile_count": metrics["profile_count"],
           **{key: metrics[key] for key in METRICS}, "config_hash": manifest["config_sha256"],
           "dataset_fingerprint": dataset_fingerprint(manifest["dataset_sha256"]),
           "git_commit": manifest["git_revision"], "report_path": report.as_posix()}
    history.parent.mkdir(parents=True, exist_ok=True)
    lock = history.with_name(history.name + ".lock")
    with registry_lock(lock):
        exists = history.exists() and history.stat().st_size > 0
        if exists:
            with history.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames != list(FIELDS):
                    raise ValueError("history CSV schema mismatch")
                if any(old["run_id"] == row["run_id"] for old in reader):
                    return False
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        with history.open("a", newline="", encoding="utf-8") as handle:
            handle.write(buffer.getvalue())
        return True


# Datasets the frozen M2 baseline used that have since been replaced in place. The originals were moved,
# byte for byte, to these paths, so the baseline stays verifiable against its recorded hashes.
LEGACY_DATA_MOVES = {"data/dev_profiles.jsonl": "data/archive/dev_profiles_60.jsonl"}


def _same_text(data: bytes, expected: str) -> bool:
    """Exact content match, allowing only LF vs CRLF line endings.

    The frozen baseline was hashed on Windows (CRLF) while Git checks it out with LF elsewhere,
    and the reverse can happen for data files. Any other byte difference still fails.
    """
    lf = data.replace(b"\r\n", b"\n")
    return expected in {hashlib.sha256(v).hexdigest() for v in (data, lf, lf.replace(b"\n", b"\r\n"))}


def register_legacy_baseline(report: Path, *, history: Path = DEFAULT_HISTORY) -> bool:
    """Recompute added cutoffs from original predictions, verifying saved inputs."""
    report = Path(report)
    manifest = json.loads((report / "manifest.json").read_text(encoding="utf-8"))
    if manifest["run_id"] != "m2_structured_oracle_dev_v1" or manifest["input_mode"] != "oracle" or manifest["split"] != "development":
        raise ValueError("legacy import supports only the saved M2 baseline")
    verified = {}   # recorded dataset name -> the file whose bytes matched (original path or its archived copy)
    for name, expected in manifest["dataset_sha256"].items():
        def matches(path):
            return _same_text(Path(path).read_bytes(), expected)
        moved = LEGACY_DATA_MOVES.get(name)
        if matches(name):
            verified[name] = Path(name)
        elif moved and Path(moved).is_file() and matches(moved):
            verified[name] = Path(moved)
        else:
            raise ValueError(f"dataset fingerprint mismatch: {name}")
    for filename, key in (("config.json", "config_sha256"), ("inputs.json", "inputs_sha256")):
        if not _same_text((report / filename).read_bytes(), manifest[key]):
            raise ValueError(f"artifact fingerprint mismatch: {filename}")
    predictions = {}
    for line in (report / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["run_id"] != manifest["run_id"]:
            raise ValueError("prediction run ID mismatch")
        predictions.setdefault(row["profile_id"], []).append(row)
    profiles = read_profiles(verified.get("data/dev_profiles.jsonl", Path("data/dev_profiles.jsonl")))
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
