"""Real-checkpoint smoke test with explicitly mocked tracker, NLI and policy.

Run from repository root. Does not publish evaluation metrics or alter baseline_v1.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import fakes  # explicit smoke harness only; production app never imports tests
from pipeline import registry
from pipeline.interfaces import State
from pipeline.system import System


def sha256(path):
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def check(checkpoint, output, ranking_config):
    if output.exists():
        raise ValueError("output already exists; choose a fresh report path")
    config = json.loads((ROOT / "tests/test_config.json").read_text())
    config["components"].update(extractor="deberta", ranker="configured")
    config["paths"].update(extractor_checkpoint=str(checkpoint.resolve()), ranking_config=str(ranking_config.resolve()))
    parts = registry.build(config, ROOT / "data")
    names = json.loads((ROOT / "data/labels.json").read_text())["type_names"]
    system = System(**parts, type_names=names, input_mode="mock_tracker")
    state = State()
    system.start(state)
    turns = []
    for text in ("I enjoy building and repairing machines.",
                 "I dislike repairing machines and prefer investigating scientific problems.",
                 "I enjoy analysing data and solving research problems."):
        started = perf_counter()
        result = system.step(state, text)
        turns.append({"text": text, "elapsed_seconds": perf_counter() - started, "result": asdict(result)})
        codes = [o.code for o in state.ranking]
        if len(codes) != len(set(codes)) or not set(codes) <= system.ranker.eligible_codes:
            raise ValueError("invalid occupation ranking")
    if not any(s.value != 0 for s in state.slots.values()):
        raise ValueError("real extractor produced no nonzero preferences for smoke inputs; inspect checkpoint/head mapping")
    files = {p.relative_to(checkpoint).as_posix(): sha256(p) for p in sorted(checkpoint.rglob("*")) if p.is_file()}
    report = {"condition": "real extractor + mock tracker/NLI/policy", "input_mode": "mock_tracker",
        "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": files,
        "components": system.components, "resolved_ranking": system.ranker.resolved_config,
        "source_sha256": {str(p): sha256(p) for p in (ROOT / "pipeline").glob("*.py")},
        "turns": turns, "trace": system.trace(state),
        "quality_claim": "smoke/interface validation only; no relevance labels or accuracy metrics"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/extractor")
    parser.add_argument("--ranking-config", type=Path, default=ROOT / "configs/ranking.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/ranking/m5/extractor_smoke.json")
    args = parser.parse_args()
    try:
        check(args.checkpoint, args.output, args.ranking_config)
        print(f"M5 extractor smoke passed: {args.output}")
    except (OSError, ValueError, NotImplementedError) as error:
        parser.exit(1, f"M5 extractor smoke FAILED: {error}\n")
