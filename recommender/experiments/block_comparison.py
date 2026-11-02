"""Run matched oracle controls and description reranking in fixed rank blocks."""
import argparse
import json
from pathlib import Path

from .evaluate import run, save_json
from .compare import paired_comparison
from .history import append_history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    root = parser.parse_args().output
    root.mkdir(parents=True)
    results = {}
    for name, config in (("control", "baseline_v1"), ("top100", "description_top100"),
                         ("blocks5", "description_blocks5"), ("blocks10", "description_blocks10")):
        results[name] = run(output=root / name,
                            ranking_config=Path(f"configs/ranking/{config}.json"), history=root / "history.csv")
        append_history(root / name)
        if name != "control":
            summary, diffs = paired_comparison(root / "control", root / name)
            save_json(root / f"{name}_comparison.json", summary)
            save_json(root / f"{name}_paired_diffs.json", diffs)
    save_json(root / "metrics_summary.json", results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
