"""Export comparable semantic experiment metrics without changing saved runs."""
import argparse
import csv
import json
from pathlib import Path

from .evaluation import METRICS
from .history import dataset_fingerprint
from .metric_extensions import read_extension

DEFAULT_RUNS = (
    ("weighted_dot_product", "description_blocks_v1/control"),
    ("minilm_top100", "description_blocks_v1/top100"),
    ("minilm_groups5", "description_blocks_v1/blocks5"),
    ("minilm_groups10", "description_blocks_v1/blocks10"),
    ("mpnet_groups10", "mpnet_blocks10_v1/mpnet"),
    ("mpnet20_then_minilm10", "mpnet20_minilm10_v1"),
    ("minilm20_then_mpnet10", "minilm20_mpnet10_v1"),
    ("minilm_mpnet_equal_weights_groups10", "aggregate_minilm_mpnet10_v1"),
)


def consolidate(reports, output):
    rows, comparable = [], None
    for experiment, relative in DEFAULT_RUNS:
        report = reports / relative
        manifest = json.loads((report / "manifest.json").read_text(encoding="utf-8"))
        metrics = json.loads((report / "metrics.json").read_text(encoding="utf-8"))
        extension = read_extension(report, manifest)
        metrics.update(extension.get("aggregate", {}))
        selection = json.loads((report / "ranking_config.json").read_text(encoding="utf-8"))
        identity = {k: manifest[k] for k in ("dataset_sha256", "inputs_sha256", "pool", "input_mode",
                    "split", "evaluator_version", "metric_definitions")}
        if comparable is not None and identity != comparable:
            raise ValueError(f"Incompatible experiment inputs/metrics: {experiment}")
        comparable = identity
        reranking = selection.get("reranking", {})
        stages = reranking.get("stages", [reranking] if reranking else [])
        separator = " + " if "aggregation" in reranking else " -> "
        rows.append({"experiment": experiment, "structured_algorithm": selection["components"]["m1"]["algorithm"],
            "semantic_models": separator.join(stage["model_id"] for stage in stages),
            "block_sizes": str(reranking["block_size"]) if "aggregation" in reranking else " -> ".join(str(stage.get("block_size", 100)) for stage in stages),
            "semantic_aggregation": reranking.get("aggregation", ""),
            "semantic_weights": json.dumps(reranking.get("weights", {}), sort_keys=True) if "weights" in reranking else "",
            "model_revisions": separator.join(stage["model_revision"] for stage in stages),
            "profile_count": metrics["profile_count"], "candidate_count": len(manifest["pool"]),
            **{key: metrics[key] for key in METRICS},
            "input_mode": manifest["input_mode"], "split": manifest["split"],
            "evaluator_version": manifest["evaluator_version"],
            "metric_extension": "recall50-v1" if extension else "",
            "run_id": manifest["run_id"],
            "dataset_fingerprint": dataset_fingerprint(manifest["dataset_sha256"]),
            "inputs_sha256": manifest["inputs_sha256"], "report_path": report.as_posix()})
    # Preserve separately appended experiments when refreshing historical rows.
    if output.exists():
        with output.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(rows[0]):
                raise ValueError("Existing consolidated CSV schema differs")
            names = {row["experiment"] for row in rows}
            rows.extend(row for row in reader if row["experiment"] not in names)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=Path("reports/ranking"))
    parser.add_argument("--output", type=Path, default=Path("reports/ranking/semantic_experiments_consolidated.csv"))
    args = parser.parse_args()
    rows = consolidate(args.reports, args.output)
    print(f"Saved {len(rows)} comparable experiments and {len(METRICS)} metrics to {args.output}")


if __name__ == "__main__":
    main()
