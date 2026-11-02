"""Derive additional metrics while preserving frozen historical experiment files."""
import hashlib
import json
from pathlib import Path

from .evaluation import evaluate_ranking
from ..data_loader import read_profiles


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_extension(report, manifest=None):
    report = Path(report)
    path = report / "recall50_extension.json"
    if not path.exists():
        return {}
    manifest = manifest or json.loads((report / "manifest.json").read_text(encoding="utf-8"))
    extra = json.loads(path.read_text(encoding="utf-8"))
    if (extra["source_manifest_sha256"] != digest(report / "manifest.json") or
            extra["predictions_sha256"] != digest(report / "predictions.jsonl") or
            extra["dataset_sha256"] != manifest["dataset_sha256"]):
        raise ValueError("Metric extension no longer matches its frozen experiment")
    return extra


def add_recall50(report):
    report = Path(report)
    manifest = json.loads((report / "manifest.json").read_text(encoding="utf-8"))
    for path, sha in manifest["dataset_sha256"].items():
        if digest(path) != sha:
            raise ValueError("Recall@50 requires original fingerprint-matching datasets")
    evaluation = json.loads((report / "evaluation_config.json").read_text(encoding="utf-8"))
    if evaluation["input_mode"] != "oracle":
        raise ValueError("Historical Recall@50 extension supports oracle profiles only")
    profiles = {p["id"]: p for p in read_profiles(Path(evaluation["profiles"]))}
    grouped = {}
    for line in (report / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        grouped.setdefault(row["profile_id"], []).append(row)
    if set(grouped) != set(profiles):
        raise ValueError("Saved predictions and original profiles differ")
    rows = []
    for pid, predictions in grouped.items():
        ordered = sorted(predictions, key=lambda r: r["rank"])
        if [r["rank"] for r in ordered] != list(range(1,len(ordered)+1)):
            raise ValueError("Invalid saved prediction ranks")
        p = profiles[pid]
        metrics = evaluate_ranking([r["code"] for r in ordered],p.get("target"),p["related"],set(manifest["pool"]))
        rows.append({"profile_id":pid,"recall_at_50":metrics["recall_at_50"]})
    extra = {"extension_version":"recall50-v1",
        "definition":"Macro recall over distinct eligible positive-grade labels at rank 50; empty relevance=0",
        "source_manifest_sha256":digest(report / "manifest.json"),
        "predictions_sha256":digest(report / "predictions.jsonl"),"dataset_sha256":manifest["dataset_sha256"],
        "aggregate":{"recall_at_50":sum(r["recall_at_50"] for r in rows)/len(rows)},"per_profile":rows}
    (report / "recall50_extension.json").write_text(json.dumps(extra,indent=2)+"\n",encoding="utf-8")
    return extra
