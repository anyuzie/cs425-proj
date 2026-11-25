"""Verify the shared fine-tuned NLI files without requiring model dependencies."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def verify(checkpoint, manifest):
    errors = []
    for name, expected in manifest["files"].items():
        path = checkpoint / name
        if not path.is_file():
            errors.append(f"Missing {name}")
            continue
        if path.stat().st_size != expected["bytes"]:
            errors.append(f"Wrong file size: {name}")
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected["sha256"]:
            errors.append(f"Wrong SHA-256: {name}")
    if errors:
        raise ValueError("Checkpoint verification failed:\n" + "\n".join(errors))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/nli/probing_v2.json")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--smoke", action="store_true", help="Also load the model and run a sentence pair")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    checkpoint = args.checkpoint or ROOT / manifest["checkpoint"]
    try:
        verify(checkpoint, manifest)
        print(f"Verified {manifest['name']}: {len(manifest['files'])} files in {checkpoint}")
        if args.smoke:
            from pipeline.nli import NliConfirmer
            model = NliConfirmer(str(checkpoint.resolve()))
            scores = model.probabilities("I like solving engineering problems", "For work, I would rather work in healthcare")
            threshold = manifest["threshold"]
            print(json.dumps({"scores": scores, "threshold": threshold,
                              "confirmed": scores["contradiction"] > threshold}, indent=2))
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        parser.exit(1, f"{exc}\nSee README.md for checkpoint setup.\n")


if __name__ == "__main__":
    main()
