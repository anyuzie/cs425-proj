"""Evaluate oracle profiles or recorded multi-turn tracker snapshots (development only)."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import platform
import shlex
import statistics
import uuid
import subprocess
from time import perf_counter

from .configuration import DEFAULT_EVALUATION_CONFIG, DEFAULT_RANKING_CONFIG, load_configuration
from .data_loader import read_profiles
from .evaluation import aggregate, evaluate_ranking
from .history import DEFAULT_HISTORY, append_history
from .schemas import OracleInput
from .ranker import RankingPipeline
from .runtime_inputs import runtime_requests


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_evaluation(config: dict) -> dict:
    # Legacy bundled snapshots remain readable; current source files stay separate.
    evaluation = {key: value for key, value in config.items()
                  if key not in {"ranking", "resolved_ranking", "algorithm"}}
    if "algorithm" in config and config["algorithm"] != "signed dot product over all six oracle dimensions; score descending, code ascending":
        raise ValueError("ranking algorithm selection belongs in the ranking configuration")
    required = {"run_id", "split", "input_mode", "occupations", "profiles", "eligibility", "ranking_depth", "seed"}
    if set(evaluation) != required:
        raise ValueError(f"evaluation config requires exactly {sorted(required)}")
    if not isinstance(evaluation["run_id"], str) or not evaluation["run_id"] or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in evaluation["run_id"]):
        raise ValueError("evaluation run_id must contain only letters, digits, underscores or hyphens")
    if evaluation["split"] != "development" or evaluation["input_mode"] not in {"oracle", "actual_tracker", "mock_tracker"}:
        raise ValueError("evaluation requires development split and oracle, actual_tracker or mock_tracker input_mode")
    for key, expected in {
        "occupations": "data/occupations.csv",
        "eligibility": "job_zone in [4,5] and complete supplied RIASEC z-scores",
        "ranking_depth": "full_pool",
    }.items():
        if evaluation[key] != expected:
            raise ValueError(f"evaluation {key} must be {expected!r}")
    if evaluation["input_mode"] == "oracle" and evaluation["profiles"] != "data/dev_profiles.jsonl":
        raise ValueError("oracle evaluation profiles must be 'data/dev_profiles.jsonl'")
    if not isinstance(evaluation["profiles"], str) or not evaluation["profiles"]:
        raise ValueError("evaluation profiles must be a nonempty path")
    if type(evaluation["seed"]) is not int or evaluation["seed"] < 0:
        raise ValueError("evaluation seed must be a nonnegative integer")
    return evaluation


def default_output() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return Path("reports/ranking/experiments") / f"{timestamp}_{uuid.uuid4().hex}"


def run(config_path: Path = DEFAULT_EVALUATION_CONFIG, output: Path | None = None, *,
        history: Path = DEFAULT_HISTORY, ranking_config: Path | dict | None = None, registry=None) -> dict:
    config_path = Path(config_path)
    config, evaluation_source = load_configuration(config_path)
    if config_path.resolve() == DEFAULT_EVALUATION_CONFIG.resolve() and {"ranking", "resolved_ranking", "algorithm"} & config.keys():
        raise ValueError("configs/m2.json contains ranking settings; move algorithm/variant selections to configs/ranking.yaml")
    evaluation_config = validate_evaluation(config)
    ranking_source = None
    ranking_path = None
    if ranking_config is None and "ranking" in config:
        # Explicit compatibility for old bundled snapshots, not new default files.
        ranking_selection = config["ranking"]
    elif isinstance(ranking_config, dict):
        ranking_selection = ranking_config
    else:
        ranking_path = Path(ranking_config) if ranking_config is not None else DEFAULT_RANKING_CONFIG
        ranking_selection, ranking_source = load_configuration(ranking_path)
    output = Path(output) if output is not None else default_output()
    if output.exists():
        raise ValueError("output already exists; preserve runs and choose a new directory")
    run_id = f"{evaluation_config['run_id']}_{uuid.uuid4().hex}"
    start = perf_counter()
    ranker = RankingPipeline(Path(evaluation_config["occupations"]), config=ranking_selection, registry=registry)
    if "resolved_ranking" in config and config["resolved_ranking"] != ranker.resolved_config:
        raise ValueError("saved resolved variant configuration does not match registry")
    saved_config = {**evaluation_config, "ranking": ranker.selection_config, "resolved_ranking": ranker.resolved_config}
    load_seconds = perf_counter() - start
    profiles = read_profiles(Path(evaluation_config["profiles"]))
    ids = [p["id"] for p in profiles]
    if not ids or any(not isinstance(i, str) or not i for i in ids) or len(ids) != len(set(ids)):
        raise ValueError("empty profiles or invalid/duplicate profile IDs")
    pool = ranker.eligible_codes
    mode = evaluation_config["input_mode"]
    # Validate every runtime snapshot before any scoring or report creation.
    replay = {p["id"]: runtime_requests(p, mode) for p in profiles} if mode != "oracle" else {}
    snapshots, metrics, predictions, times = [], [], [], []
    turn_predictions = []
    fused_profiles = 0
    for profile in profiles:
        if mode == "oracle":
            inputs = OracleInput.from_profile(profile)
            snapshots.append({"profile_id": profile["id"], **asdict(inputs)})
            start = perf_counter()
            response = ranker.rank_oracle(inputs)
            times.append(perf_counter() - start)
        else:
            for turn, request in enumerate(replay[profile["id"]], 1):
                snapshot = {"profile_id": profile["id"], "turn": turn, "input_mode": mode,
                    "upstream": profile["upstream"], "interests": request.interests,
                    "observed_types": sorted(request.observed_types), "user_sentences": request.user_sentences}
                snapshots.append(snapshot)
                start = perf_counter()
                response = ranker.rank(request.interests, request.observed_types, request.user_sentences, len(pool))
                times.append(perf_counter() - start)
                turn_predictions.append({**snapshot, "status": response.status, "reason": response.reason,
                    "rank_seconds": times[-1], "occupations": [asdict(o) for o in response.occupations[:100]]})
        fused_profiles += int(any(o.rrf_score is not None for o in response.occupations))
        codes = [o.code for o in response.occupations]
        result = evaluate_ranking(codes, profile.get("target"), profile["related"], pool)
        metrics.append({"profile_id": profile["id"], "status": response.status, **result})
        for occupation in response.occupations:
            predictions.append({"run_id": run_id, "profile_id": profile["id"],
                                "input_mode": mode, "target_code": profile.get("target"),
                                **asdict(occupation)})
    summary = aggregate(metrics)
    source = {str(p): digest(p) for directory in ("recommender", "pipeline", "app")
              for p in sorted(Path(directory).glob("*.py"))}
    dense_metadata = ranker.component_metadata.get("m4", {})
    dense_identity = dense_metadata.get("identity", {})
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    output.mkdir(parents=True)
    save_json(output / "config.json", saved_config)  # Legacy-compatible reproduction bundle.
    save_json(output / "evaluation_config.json", evaluation_config)
    save_json(output / "ranking_config.json", ranker.selection_config)
    save_json(output / "resolved_ranking.json", ranker.resolved_config)
    source_directory = output / "config_sources"
    source_directory.mkdir()
    (source_directory / "evaluation.json").write_bytes(evaluation_source)
    configuration_sources = {"evaluation": {"path": config_path.as_posix(),
        "sha256": hashlib.sha256(evaluation_source).hexdigest(), "snapshot": "config_sources/evaluation.json"}}
    if ranking_source is not None:
        snapshot = f"config_sources/ranking{ranking_path.suffix.lower()}"
        (output / snapshot).write_bytes(ranking_source)
        configuration_sources["ranking"] = {"path": ranking_path.as_posix(),
            "sha256": hashlib.sha256(ranking_source).hexdigest(), "snapshot": snapshot}
    else:
        configuration_sources["ranking"] = {"path": "embedded config or in-memory mapping",
            "sha256": digest(output / "ranking_config.json"), "snapshot": "ranking_config.json"}
    save_json(output / "inputs.json", snapshots)
    if replay:
        save_json(output / "turn_predictions.json", turn_predictions)
    save_json(output / "per_profile.json", metrics)
    save_json(output / "metrics.json", summary)
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    save_json(output / "latency.json", {
        "load_seconds": load_seconds, "preloaded_rank_median_seconds": statistics.median(times),
        "preloaded_rank_p95_seconds": sorted(times)[math_ceil_95(len(times)) - 1],
        "sample_count": len(times), "method": "preloaded full-pool calls; each oracle profile or tracker turn; nearest-rank p95; excludes metrics/export",
    })
    save_json(output / "manifest.json", {
        "evaluator_version": "m2-v2", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id, "control_run_id": "m2_structured_oracle_dev_v1",
        "components": {**ranker.component_identifiers, "fusion": ranker.fusion_identifier},
        "component_metadata": ranker.component_metadata,
        "fused_profile_count": fused_profiles,
        "retrieval_policy": "No ranking cutoff before evaluation; lexical ranks contain positive matches only; unretrieved target MRR=0",
        "resolved_ranking": ranker.resolved_config,
        "algorithm_parameters": ranker.resolved_config["components"],
        "dataset_split": evaluation_config["split"],
        "hypothesis": f"Evaluate selected ranking on {mode} development inputs; final turn per session for tracker metrics",
        "query_policy": "raw chronological user-only utterances; no stance or lifecycle filtering",
        "evaluation_unit": "profile" if mode == "oracle" else "session final turn",
        "treatment": "See resolved_ranking for enabled algorithms and variants", "input_mode": mode, "split": "development",
        "git_revision": revision.stdout.strip(), "working_tree_dirty": bool(status.stdout.strip()),
        "source_sha256": source, "config_sha256": digest(output / "config.json"),
        "evaluation_config_sha256": digest(output / "evaluation_config.json"),
        "ranking_config_sha256": digest(output / "ranking_config.json"),
        "resolved_ranking_sha256": digest(output / "resolved_ranking.json"),
        "configuration_sources": configuration_sources,
        "dataset_sha256": {evaluation_config[k]: digest(evaluation_config[k]) for k in ("profiles", "occupations")},
        "inputs_sha256": digest(output / "inputs.json"), "pool": sorted(pool),
        "eligibility": evaluation_config["eligibility"], "ranking_depth": "full eligible pool; MRR not truncated at 100",
        "metric_definitions": "Hit@1/5/10 exact target; full-pool MRR; NDCG@5/10 gains 2^grade-1, target=4/short=3/long=2/supplemental=1, max grade per eligible code, zero IDCG=0; all profiles in denominator",
        "python": platform.python_version(), "dependencies": {"PyYAML": version("PyYAML"), **dense_identity.get("packages", {})},
        "model_revision": dense_identity.get("parameters", {}).get("model_revision"),
        "task_index_fingerprint": dense_metadata.get("artifact_sha256", {}).get("index.faiss"),
        "auxiliary_dataset_sha256": {dense_identity["parameters"]["tasks_path"]: dense_identity["tasks_sha256"]} if dense_identity else {},
        "seed": evaluation_config["seed"],
        "test_summary": "See milestone reports for executed acceptance results; runtime integration: reports/ranking/m5/DECISION.md",
        "command": f"python -m recommender.evaluate --config {shlex.quote((output / 'evaluation_config.json').as_posix())} --ranking-config {shlex.quote((output / 'ranking_config.json').as_posix())} --output <new-directory>",
    })
    append_history(output, history=history)
    return summary


def math_ceil_95(n):
    return (95 * n + 99) // 100


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_EVALUATION_CONFIG)
    parser.add_argument("--output", type=Path, help="Fresh report directory; defaults to reports/ranking/experiments/<UTC timestamp>_<UUID>")
    parser.add_argument("--ranking-config", type=Path, help="Ranking YAML/JSON; defaults to configs/ranking.yaml (or legacy embedded snapshot)")
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    args = parser.parse_args()
    try:
        output = args.output if args.output is not None else default_output()
        metrics = run(args.config, output, history=args.history, ranking_config=args.ranking_config)
        print(json.dumps({"report_path": output.as_posix(), "metrics": metrics}, indent=2))
    except (OSError, ValueError, KeyError, TypeError, NotImplementedError) as error:
        parser.exit(1, f"M2 evaluation FAILED: {error}\n")


if __name__ == "__main__":
    main()
