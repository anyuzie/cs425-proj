"""Multi-turn fusion evaluation (#0, docs/SEMANTIC_LEDGER_PLAN.md §12.3-12.4).

Simulated dialogues with the real System on data/eval/fusion_personas_v1.jsonl. Six arms:
{fusion (RRF k=60), cosine only, semantic only} x {probing on, probing off}. Each arm uses its
ranker both inside the dialogue and for the final ranking.

Primary metric (fixed before any run, §12.4): MRR of the first acceptable occupation over the
full eligible pool. Secondary: Hit@5 and NDCG@10 against the acceptable set (binary gains).
Paired bootstrap CIs over personas, reported for all personas and for conflict vs agree.
Per-turn diagnostic: best acceptable rank under cosine, semantic and fusion for the same inputs.

EVALUATION ONLY: the personas must never be used to tune any component, weight or threshold.
"""
import argparse
from copy import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import subprocess

from eval.dialogue import DialogueUser
from pipeline.interfaces import State
from pipeline.policy import PROBE_REASON_CONF, ManualPolicy
from pipeline.system import System
from recommender.evaluation import evaluate_ranking
from recommender.models.ranker import RankingPipeline

ROOT = Path(__file__).resolve().parents[1]
PERSONAS = ROOT / "data/eval/fusion_personas_v1.jsonl"
RANKERS = {  # arm name -> versioned ranking configuration (no new variants here)
    "fusion": "configs/ranking/occupation_semantic_fusion_v1.json",  # = configs/ranking.yaml default
    "cosine": "configs/ranking/m1_cosine.json",
    "semantic": "configs/ranking/occupation_semantic_finetuned_v1.json",
}
PROBING = {"probing": False, "no_probing": True}  # arm suffix -> ManualPolicy(no_probing=...)
GROUPS = ("all", "conflict", "agree")
METRICS = ("mrr", "hit_at_5", "hit_at_10", "ndcg_at_10")
PRIMARY = "mrr"
BOOTSTRAP_SAMPLES, BOOTSTRAP_SEED = 10_000, 425


def acceptable_metrics(codes: list[str], acceptable: list[str]) -> dict:
    """Metrics against the persona's acceptable set; codes is the full-depth ranking."""
    position = {code: rank for rank, code in enumerate(codes, 1)}
    ranks = sorted(position[c] for c in acceptable if c in position)
    best = ranks[0] if ranks else None
    hits = set(codes[:10]) & set(acceptable)
    ideal = sum(1 / math.log2(r + 1) for r in range(1, min(10, len(acceptable)) + 1))
    dcg = sum(1 / math.log2(r + 1) for r, code in enumerate(codes[:10], 1) if code in hits)
    return dict(mrr=1 / best if best else 0.0, hit_at_5=int(best is not None and best <= 5),
                hit_at_10=int(best is not None and best <= 10), ndcg_at_10=dcg / ideal,
                best_rank=best, acceptable_ranks={c: position.get(c) for c in acceptable})


def best_rank(ranker, entry, acceptable) -> int | None:
    response = ranker.rank(entry["interests"], set(entry["observed_types"]), entry["user_sentences"],
                           ranker.eligible_count)
    ranks = [o.rank for o in response.occupations if o.code in acceptable]
    return min(ranks) if ranks else None


def simulate(system, persona, bank, diagnostics) -> dict:
    """eval.dialogue.simulate plus per-turn component diagnostics; System only sees reply text."""
    state, user = State(), DialogueUser(persona, bank)
    system.start(state)
    text, turns = user.opening(), []
    while not state.done:
        result = system.step(state, text)
        turns.append(dict(turn=state.turn, user=text, reply=result.reply, action=result.action,
                          slot=state.last_action.slot, profile=system.profile(state), top5=system.top5(state)))
        if state.turn > system.max_turns:
            raise RuntimeError("Policy exceeded the conversation budget")
        if not state.done:
            text = user.reply(state.last_action)
    interests = {t: state.slots[t].value if t in state.slots else 0.0 for t in "RIASEC"}
    response = system.ranker.rank(interests, set(state.slots), list(state.user_texts), system.ranker.eligible_count)
    codes = [o.code for o in response.occupations]
    if codes[:5] != [o.code for o in state.ranking[:5]]:
        raise RuntimeError("Final evaluation ranking differs from the displayed recommendations")
    acceptable = set(persona["acceptable"])
    per_turn = [dict(turn=e["turn"], observed_types=e["observed_types"],
                     **{name: best_rank(r, e, acceptable) for name, r in diagnostics.items()})
                for e in state.ranking_trace]
    return dict(id=persona["id"], field=persona["field"], conflict=persona["conflict"],
                turns=turns, per_turn_best_acceptable_rank=per_turn, ranking=codes[:50],
                recommendations=result.recommendations,
                metrics=acceptable_metrics(codes, persona["acceptable"]),
                target_metrics=evaluate_ranking(codes, persona["target"], persona["related"],
                                                system.ranker.eligible_codes))


def summarise(sessions: list[dict]) -> dict:
    out = {}
    for group in GROUPS:
        rows = [s for s in sessions if group == "all" or s["conflict"] == (group == "conflict")]
        out[group] = dict(n=len(rows), **{m: sum(s["metrics"][m] for s in rows) / len(rows) for m in METRICS}) if rows else dict(n=0)
    return out


def paired_bootstrap(a: list[float], b: list[float], seed=BOOTSTRAP_SEED, samples=BOOTSTRAP_SAMPLES) -> dict:
    """Mean of a - b with a percentile 95% CI, resampling personas with replacement."""
    if len(a) != len(b) or not a:
        raise ValueError("paired samples must be nonempty and equal length")
    diffs, rng = [x - y for x, y in zip(a, b)], random.Random(seed)
    means = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(samples))
    return dict(mean=sum(diffs) / len(diffs), ci95=[means[int(.025 * samples)], means[int(.975 * samples) - 1]], n=len(diffs))


def contrasts(arms: dict) -> dict:
    pairs = [(f"fusion_{p}", f"cosine_{p}") for p in PROBING] + [(f"fusion_{p}", f"semantic_{p}") for p in PROBING] \
        + [(f"semantic_{p}", f"cosine_{p}") for p in PROBING] + [(f"{r}_probing", f"{r}_no_probing") for r in RANKERS]
    out = {}
    for a, b in pairs:
        for group in GROUPS:
            keep = [i for i, s in enumerate(arms[a]["sessions"]) if group == "all" or s["conflict"] == (group == "conflict")]
            if not keep:
                continue
            for metric in METRICS:
                out.setdefault(f"{a}_minus_{b}", {}).setdefault(group, {})[metric] = paired_bootstrap(
                    [arms[a]["sessions"][i]["metrics"][metric] for i in keep],
                    [arms[b]["sessions"][i]["metrics"][metric] for i in keep])
    return out


def run(system, personas, bank, pipelines) -> dict:
    if not personas or len({p["id"] for p in personas}) != len(personas):
        raise ValueError("Expected nonempty personas with unique identifiers")
    arms = {}
    for ranker_name, pipeline in pipelines.items():
        for suffix, no_probing in PROBING.items():
            candidate = copy(system)
            candidate.ranker = pipeline
            candidate.policy = ManualPolicy(system.max_turns, no_probing=no_probing,
                                            probe_reason_conf=getattr(system.policy, "probe_reason_conf", PROBE_REASON_CONF))
            sessions = [simulate(candidate, p, bank, pipelines) for p in personas]
            arms[f"{ranker_name}_{suffix}"] = dict(summary=summarise(sessions), sessions=sessions)
    return dict(arms=arms, contrasts=contrasts(arms))


def report(results: dict) -> str:
    lines = ["# Fusion dialogue evaluation", "", f"Primary metric: {PRIMARY} of first acceptable occupation (§12.4).", "",
             "| Arm | Group | n | MRR | Hit@5 | Hit@10 | NDCG@10 |", "|---|---|---|---|---|---|---|"]
    for arm, value in results["arms"].items():
        for group, s in value["summary"].items():
            if s["n"]:
                lines.append(f"| {arm} | {group} | {s['n']} | {s['mrr']:.3f} | {s['hit_at_5']:.3f} | {s['hit_at_10']:.3f} | {s['ndcg_at_10']:.3f} |")
    lines += ["", "## Paired differences (MRR, 95% bootstrap CI)", "", "| Contrast | Group | Mean | CI |", "|---|---|---|---|"]
    for name, groups in results["contrasts"].items():
        for group, metrics in groups.items():
            d = metrics[PRIMARY]
            lines.append(f"| {name} | {group} | {d['mean']:+.3f} | [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] |")
    lines += ["", "## Per persona (best acceptable rank, final turn)", "",
              "| Persona | Field | Conflict | " + " | ".join(results["arms"]) + " |",
              "|---|---|---|" + "---|" * len(results["arms"])]
    first = next(iter(results["arms"].values()))["sessions"]
    for i, s in enumerate(first):
        ranks = [str(a["sessions"][i]["metrics"]["best_rank"]) for a in results["arms"].values()]
        lines.append(f"| {s['id']} | {s['field']} | {s['conflict']} | " + " | ".join(ranks) + " |")
    return "\n".join(lines) + "\n"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=ROOT / "configs/system.json",
                        help="Base System config; filler is forced to question-bank and the ranker is set per arm")
    parser.add_argument("--limit", type=int, default=None, help="First N personas (smoke runs only)")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    output = args.output or ROOT / "reports/ranking/dialogue_fusion_v1" / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    if output.exists():
        parser.error("Output exists; choose a new directory to preserve prior results")
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    config["components"]["filler"] = "question-bank"  # simulator answers are keyed to bank wording
    for key in ("extractor_checkpoint", "nli_model"):
        if not (ROOT / config["paths"][key]).exists():
            parser.error(f"{key} {config['paths'][key]!r} not found; fetch it or pass --config pointing at an available model")
    personas = [json.loads(line) for line in PERSONAS.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        personas = personas[:args.limit]
    bank = json.loads((ROOT / "data/question_bank.json").read_text())
    config["components"]["ranker"] = "configured"
    config["paths"]["ranking_config"] = RANKERS["fusion"]
    system = System.from_config(ROOT / "data", config)
    pipelines = {name: RankingPipeline(ROOT / "data/occupations.csv", config=ROOT / path) for name, path in RANKERS.items()}
    results = run(system, personas, bank, pipelines)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    source = [*sorted((ROOT / "pipeline").glob("*.py")), Path(__file__), ROOT / "eval/dialogue.py", ROOT / "data/eval/simulator.py"]
    checkpoints = {key: {p.name: sha256(p) for p in (ROOT / config["paths"][key]).glob("*")
                         if p.name in ("config.json", "model.safetensors")}
                   for key in ("extractor_checkpoint", "nli_model")}
    manifest = dict(
        condition="actual_tracker_dialogue_fusion_personas",
        quality_claim="none for --limit smoke runs" if args.limit else "development evidence for fusion choice (#0); personas never tuned on",
        persona_ids=[p["id"] for p in personas], personas_sha256=sha256(PERSONAS),
        base_config_path=str(config_path.relative_to(ROOT)) if config_path.is_relative_to(ROOT) else str(config_path),
        config=config, arms={f"{r}_{s}": dict(ranking_config=RANKERS[r], no_probing=n) for r in RANKERS for s, n in PROBING.items()},
        resolved_ranking={name: p.resolved_config for name, p in pipelines.items()},
        ranking_config_sha256={name: sha256(ROOT / path) for name, path in RANKERS.items()},
        git_revision=revision, git_dirty=dirty, python=platform.python_version(),
        source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in source},
        checkpoints=checkpoints, components=system.components,
        metric_definitions=dict(primary=PRIMARY, mrr="1 / rank of the first acceptable occupation, full eligible pool",
                                hit_at_k="first acceptable occupation within top k", ndcg_at_10="binary gains over the acceptable set",
                                target_metrics="existing M2 metrics against target + O*NET related, for continuity"),
        bootstrap=dict(samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED, unit="persona", ci="percentile 95%"),
        simulator="eval.dialogue.DialogueUser: question-bank CLARIFY answers by hidden level; resolutions for PROBE/CHALLENGE")
    output.mkdir(parents=True)
    for name, value in (("manifest", manifest), ("results", results)):
        (output / f"{name}.json").write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "REPORT.md").write_text(report(results), encoding="utf-8")
    print(report(results))
    print("Saved", output)


if __name__ == "__main__":
    main()
