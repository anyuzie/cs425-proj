"""Freeze the M5 ranking boundary using explicitly mocked upstream conversation fixtures.

Run from repository root. Never substitutes these fixtures for actual-tracker evidence.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import subprocess
import sys
from time import perf_counter
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from m5_fixture import capture, make_system
from recommender.evaluate import digest, run, save_json, math_ceil_95
from recommender.runtime_inputs import runtime_requests


def freeze(output):
    if output.exists():
        raise ValueError("freeze directory exists; preserve the accepted baseline")
    output.mkdir(parents=True)
    configurations = {
        "baseline_v1": Path("configs/ranking/baseline_v1.json"),
        "three_way_v1": Path("configs/ranking/m4_three_way.json"),
    }
    system = make_system(configurations["baseline_v1"])
    profiles, traces = capture(system)
    save_json(output / "integration_trace.json", traces)
    dataset = output / "mock_tracker_sessions.jsonl"
    dataset.write_text("".join(json.dumps(p, sort_keys=True) + "\n" for p in profiles), encoding="utf-8")
    config = json.loads(Path("configs/m2.json").read_text())
    config.update(run_id="m5_mock_tracker_dev", input_mode="mock_tracker", profiles=dataset.as_posix())
    save_json(output / "evaluation.json", config)
    benchmarks = {}
    for name, path in configurations.items():
        run(output / "evaluation.json", output / name, ranking_config=path)
        started = perf_counter()
        runtime = make_system(path)
        startup_seconds = perf_counter() - started
        _, integration_traces = capture(runtime)
        save_json(output / f"{name}_integration_trace.json", integration_traces)
        request = runtime_requests(profiles[0], "mock_tracker")[-1]
        runtime.ranker.rank(request.interests, request.observed_types, request.user_sentences, 100)
        samples = []
        for _ in range(100):
            started = perf_counter()
            runtime.ranker.rank(request.interests, request.observed_types, request.user_sentences, 100)
            samples.append(perf_counter() - started)
        benchmarks[name] = {"startup_seconds": startup_seconds, "samples": len(samples),
            "median_seconds": statistics.median(samples), "p95_seconds": sorted(samples)[math_ceil_95(len(samples))-1],
            "method": "100 repeated preloaded top-100 ranking calls after one warmup; fixed four-turn input; excludes extractor/tracker/policy/UI"}
    save_json(output / "latency.json", benchmarks)
    save_json(output / "baseline_contract.json", {
        "baseline": "baseline_v1", "default": "M1-only", "query_policy": "raw chronological user-only utterances",
        "structured_policy": "current tracker values and observed types; recompute each turn; no score accumulation or confidence reweighting",
        "policy_top_k": 100, "recommendation_top_k": 5, "max_tasks_per_occupation": 3,
        "task_status": "provisional until stance validation", "input_mode": "mock_tracker",
        "upstream_status": "production tracker/policy/NLI not implemented; extractor checkpoint unavailable",
        "latency_budget": {"status": "pending teammate agreement", "scope": "preloaded rank call; startup and end-to-end timings separate"},
        "quality_claim": "scripted interface acceptance only; illustrative labels are not actual-user relevance judgements"})
    sources = sorted(p for directory in ("recommender", "pipeline", "app", "tests", "configs", "scripts", "static")
                     for p in Path(directory).rglob("*") if p.is_file() and "__pycache__" not in p.parts
                     and p.suffix in {".py", ".json", ".yaml", ".js", ".html", ".css"})
    with zipfile.ZipFile(output / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            archive.write(path, path.as_posix())
    save_json(output / "baseline_manifest.json", {
        "baseline": "baseline_v1", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "source_sha256": {p.as_posix(): digest(p) for p in sources},
        "source_snapshot_sha256": digest(output / "source_snapshot.zip"),
        "artifact_sha256": {p.relative_to(output).as_posix(): digest(p) for p in output.rglob("*") if p.is_file()},
        "reproduce": "Restore source_snapshot.zip in a checkout with matching data/models/cache, then python scripts/freeze_ranking_baseline.py --output <new-directory>",
        "scope": "frozen ranking integration control; actual-team end-to-end acceptance and latency agreement pending"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/ranking/m5/frozen"))
    args = parser.parse_args()
    freeze(args.output)
