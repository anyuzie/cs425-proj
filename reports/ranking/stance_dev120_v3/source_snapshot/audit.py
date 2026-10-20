"""Run M0 schema audit; no ranking metrics or held-out text inspection."""

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

from .data_loader import audit_data


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--config", type=Path, default=Path("configs/m0.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/ranking/m0_audit.json"))
    args = parser.parse_args()
    try:
        config_bytes = args.config.read_bytes()
        config = json.loads(config_bytes)
        start = time.perf_counter()
        result = audit_data(args.data_dir, config["expected_counts"])
        elapsed = time.perf_counter() - start
        revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=False)
        result["manifest"] = {"milestone": "M0", "config": config, "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                              "python": platform.python_version(), "dependencies": "standard library only",
                              "git_revision": revision.stdout.strip() or None, "working_tree_dirty": bool(status.stdout.strip()),
                              "audit_seconds": elapsed, "command": "python -m recommender.audit"}
        result["manifest"]["source_sha256"] = {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path("recommender").glob("*.py"))
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"M0 audit PASS: {json.dumps(result['counts'], sort_keys=True)}")
        print(f"Saved {args.output}")
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(1, f"M0 audit FAILED: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
