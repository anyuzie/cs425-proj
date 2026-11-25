"""Evaluate a challenger against the active selection, then publish improvements."""

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from recommender.configuration import load_configuration
from recommender.experiments.compare import paired_comparison
from recommender.experiments.evaluate import default_output, run, save_json
from recommender.experiments.evaluation import METRICS, aggregate
from recommender.experiments.history import DEFAULT_HISTORY, registry_lock

POLICIES = ("hit5_no_regression", "hit5", "pareto")


def improves(control, candidate, policy):
    if policy not in POLICIES:
        raise ValueError(f"unknown promotion policy: {policy}")
    for metrics in (control, candidate):
        for name in METRICS:
            value = metrics[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"invalid promotion metric: {name}")
    differences = {m: candidate[m] - control[m] for m in METRICS}
    # Ignore only floating-point roundoff; a tie never replaces the incumbent.
    nonregressing = all(d >= -1e-12 for d in differences.values())
    if policy == "hit5":
        return differences["hit_at_5"] > 1e-12
    if policy == "pareto":
        return nonregressing and any(d > 1e-12 for d in differences.values())
    return nonregressing and differences["hit_at_5"] > 1e-12


def atomic_write(path, content):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def evaluate_and_promote(evaluation, candidate, active, *, policy="hit5_no_regression",
                         output=None, history=DEFAULT_HISTORY):
    """Re-evaluate both selections under the same conditions; never edit variants.

    The active file is a mutable JSON deployment selection, not a baseline preset.
    An exclusive lock covers the whole round, and the reports and decision are
    written before the active file is atomically replaced. The decision records
    the intended bytes/hash so an interrupted publish can be checked later.
    """
    active, candidate = Path(active), Path(candidate)
    if active.suffix.lower() != ".json":
        raise ValueError("active selection must be a dedicated JSON file")
    if active.resolve() == candidate.resolve():
        raise ValueError("candidate and active selections must be separate files")
    if policy not in POLICIES:
        raise ValueError(f"unknown promotion policy: {policy}")
    output = Path(output) if output is not None else default_output()
    if output.exists():
        raise ValueError("round output already exists; choose a new directory")
    with registry_lock(active.with_suffix(active.suffix + ".lock")):
        incumbent, before = load_configuration(active)
        challenger, candidate_bytes = load_configuration(candidate)
        output.mkdir(parents=True)
        (output / "previous_active.json").write_bytes(before)
        (output / ("candidate_source" + candidate.suffix.lower())).write_bytes(candidate_bytes)
        control = run(evaluation, output / "control", history=history, ranking_config=incumbent)
        treatment = run(evaluation, output / "candidate", history=history, ranking_config=challenger)
        comparison, _ = paired_comparison(output / "control", output / "candidate")
        # Use validated per-profile aggregates, not caller-supplied summary claims.
        for directory, expected in (("control", control), ("candidate", treatment)):
            rows = json.loads((output / directory / "per_profile.json").read_text(encoding="utf-8"))
            if aggregate(rows) != expected:
                raise ValueError("saved per-profile metrics do not match the evaluation summary")
        accepted = improves(control, treatment, policy)
        # Lifecycle-filtered replay selections cannot be deployed to live System.
        deployable = challenger.get("evidence_filter", "raw") == "raw"
        accepted = accepted and deployable
        replacement = (json.dumps(challenger, indent=2, sort_keys=True) + "\n").encode("utf-8")
        decision = {
            "policy": policy, "promoted": False, "publication_status": "prepared" if accepted else "not_needed",
            "reason": "candidate_improved" if accepted else "live_system_requires_raw_evidence" if not deployable else "metric_rule_not_met",
            "active_path": str(active.resolve()), "candidate_path": str(candidate.resolve()),
            "previous_sha256": hashlib.sha256(before).hexdigest(),
            "selected_sha256": hashlib.sha256(replacement if accepted else before).hexdigest(),
            "control_metrics": control, "candidate_metrics": treatment,
            "comparison": comparison,
        }
        if active.read_bytes() != before:
            raise ValueError("active selection changed during evaluation; refusing promotion")
        save_json(output / "promotion.json", decision)
        if accepted:
            atomic_write(active, replacement)
            decision["promoted"] = True
            decision["publication_status"] = "published"
            atomic_write(output / "promotion.json", (json.dumps(decision, indent=2, sort_keys=True) + "\n").encode("utf-8"))
        return {"report_path": str(output), "metrics": treatment, "promotion": decision}
