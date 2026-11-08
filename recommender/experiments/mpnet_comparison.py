"""Paired MiniLM versus MPNet comparison with identical ten-rank blocks."""
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
    for name, config in (("minilm", "description_blocks10"), ("mpnet", "description_mpnet_blocks10")):
        results[name] = run(output=root / name,
            ranking_config=Path(f"configs/ranking/{config}.json"), history=root / "history.csv")
        append_history(root / name)
    summary, diffs = paired_comparison(root / "minilm", root / "mpnet")
    save_json(root / "comparison.json", summary)
    save_json(root / "paired_diffs.json", diffs)
    save_json(root / "metrics_summary.json", results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
