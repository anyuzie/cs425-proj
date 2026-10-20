"""Run with python -m eval.nli; tune on dev, freeze, then score test once."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from pipeline.nli import MODEL_NAME, NliConfirmer

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = tuple(i / 100 for i in range(40, 91))
BINARY = {"contradiction": 1, "elaboration": 0, "unrelated": 0}


def metrics(labels, scores, threshold):
    if len(labels) != len(scores) or not labels:
        raise ValueError("Expected equal, nonempty labels and scores")
    if any(y not in (0, 1) for y in labels) or any(
        not math.isfinite(s) or not 0 <= s <= 1 for s in scores
    ):
        raise ValueError("Expected binary labels and finite probabilities")
    tp = fp = tn = fn = 0
    for y, score in zip(labels, scores):
        predicted = score > threshold  # same strict boundary as System._confirmed
        tp += int(y == 1 and predicted)
        fp += int(y == 0 and predicted)
        tn += int(y == 0 and not predicted)
        fn += int(y == 1 and not predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": precision, "recall": recall,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "confusion_matrix": [[tn, fp], [fn, tp]], "count": len(labels)}


def select_threshold(labels, scores):
    sweep = [{"threshold": t, **metrics(labels, scores, t)} for t in THRESHOLDS]
    # Predeclared tie break: F1, precision, then highest threshold.
    best = max(sweep, key=lambda r: (r["f1"], r["precision"], r["threshold"]))
    return best["threshold"], sweep


def evaluate(rows, confirmer, freeze):
    """freeze(report) must persist dev selection before test inference starts."""
    if any(r["split"] not in {"dev", "test"} or r["label"] not in BINARY for r in rows):
        raise ValueError("Unexpected split or label")
    dev = [r for r in rows if r["split"] == "dev"]
    test = [r for r in rows if r["split"] == "test"]
    if not dev or not test:
        raise ValueError("Both dev and test splits are required")
    dev_scores = [confirmer.score(r["a"], r["b"]) for r in dev]
    threshold, sweep = select_threshold([BINARY[r["label"]] for r in dev], dev_scores)
    report = {"threshold": threshold, "comparison": "score > threshold",
              "selection": "dev F1; ties: precision, then highest threshold",
              "confusion_matrix_axes": "rows actual [negative, positive]; columns predicted [negative, positive]",
              "dev_sweep": sweep,
              "dev": metrics([BINARY[r["label"]] for r in dev], dev_scores, threshold)}
    freeze(report)
    test_scores = [confirmer.score(r["a"], r["b"]) for r in test]
    report["test"] = metrics([BINARY[r["label"]] for r in test], test_scores, threshold)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/nli_pairs.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/nli")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frozen_path = args.output / "threshold.json"
    result_path = args.output / "results.json"
    if frozen_path.exists() or result_path.exists():
        parser.error("Output already contains an evaluation; use a new output directory")
    raw = args.data.read_bytes()
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    confirmer = NliConfirmer(MODEL_NAME)
    metadata = {"model": MODEL_NAME, "data_sha256": hashlib.sha256(raw).hexdigest(),
                "label_order": confirmer.label_order,
                "model_commit": getattr(confirmer.model.config, "_commit_hash", None)}

    def freeze(report):
        frozen_path.write_text(json.dumps({**metadata, **report}, indent=2) + "\n", encoding="utf-8")

    report = evaluate(rows, confirmer, freeze)
    result_path.write_text(json.dumps({**metadata, **report}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("threshold", "dev", "test")}, indent=2))


if __name__ == "__main__":
    main()
