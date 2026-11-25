"""Run the fixed mock tracker demo without touching evaluation profiles."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from time import perf_counter

from .structured import StructuredRanker


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--occupations", type=Path, default=Path("data/occupations.csv"))
    parser.add_argument("--config", type=Path, default=Path("configs/m1.json"))
    parser.add_argument("--output", type=Path, help="Optional reproducibility manifest and mock results")
    args = parser.parse_args()
    try:
        config_bytes = args.config.read_bytes()
        config = json.loads(config_bytes)
        request = config["request"]
        start = perf_counter()
        ranker = StructuredRanker(args.occupations)
        load_seconds = perf_counter() - start
        start = perf_counter()
        response = ranker.rank(request["interests"], set(request["observed_types"]),
                               request["user_sentences"], request["top_k"])
        rank_seconds = perf_counter() - start
        print(f"M1 structured demo | eligible={ranker.eligible_count} | status={response.status}")
        for occupation in response.occupations:
            print(f"{occupation.rank}. {occupation.code} | {occupation.title} | structured={occupation.signals['structured'].score:.6f}")
        if response.reason:
            print(response.reason)
        if args.output:
            revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
            status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=False)
            manifest = {
                "milestone": "M1", "input_mode": "mock_tracker", "config": config,
                "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                "occupations_sha256": hashlib.sha256(args.occupations.read_bytes()).hexdigest(),
                "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path("recommender").glob("*.py"))},
                "git_revision": revision.stdout.strip() or None, "working_tree_dirty": bool(status.stdout.strip()),
                "python": platform.python_version(), "dependencies": "standard library only",
                "eligible_count": ranker.eligible_count, "load_seconds": load_seconds,
                "rank_seconds": rank_seconds, "result": asdict(response),
                "command": "python -m recommender.cli --output reports/ranking/m1_demo.json",
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"M1 demo FAILED: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
