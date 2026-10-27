"""Development-only full versus no-probing dialogue evaluation using real System."""
import argparse
from copy import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess

from data.eval.simulator import NAME, SimulatedUser
from pipeline.interfaces import State
from pipeline.policy import ManualPolicy
from pipeline.system import System
from recommender.data_loader import read_profiles
from recommender.evaluation import aggregate, evaluate_ranking

ROOT = Path(__file__).resolve().parents[1]


class DialogueUser(SimulatedUser):
    """Annotations stay in the simulated user; System receives only reply text."""

    def reply(self, action):
        kind = action.payload.get("retry", "CLARIFY") if action.kind == "ABSTAIN" else action.kind
        slot = action.slot
        variant = action.payload.get("clarify_index", 0)
        if kind == "PROBE":
            matching = any(plant.startswith("familiarity") and plant.rsplit(":", 1)[-1] == NAME[slot]
                           for plant in self.p.get("planted", []))
            if matching and self.p.get("latent_reveal"):
                return self.p["latent_reveal"]
            return self.p.get("resolutions", {}).get(slot, "I am not sure about that work.")
        if kind == "CHALLENGE":
            return self.p.get("resolutions", {}).get(slot, "I am not sure which statement is closer.")
        return self.answer(kind, slot, variant)


def simulate(system, profile, bank):
    state, user = State(), DialogueUser(profile, bank)
    system.start(state)
    text = user.opening()
    turns = []
    while not state.done:
        result = system.step(state, text)
        turns.append(dict(turn=state.turn, user=text, reply=result.reply, action=result.action,
                          slot=state.last_action.slot, retry=state.last_action.payload.get("retry"),
                          profile=system.profile(state), top5=system.top5(state)))
        if state.turn > system.max_turns:
            raise RuntimeError("Policy exceeded the conversation budget")
        if not state.done:
            text = user.reply(state.last_action)
    # Full depth is needed for MRR/NDCG@10; never feed target labels to the ranker.
    interests = {t: state.slots[t].value if t in state.slots else 0.0 for t in "RIASEC"}
    response = system.ranker.rank(interests, set(state.slots), list(state.user_texts), system.ranker.eligible_count)
    codes = [o.code for o in response.occupations]
    if codes[:5] != [o.code for o in state.ranking[:5]]:
        raise RuntimeError("Final evaluation ranking differs from the displayed recommendations")
    return dict(id=profile["id"], target=profile["target"], turns=turns,
                ranking=codes, recommendations=result.recommendations,
                metrics=evaluate_ranking(codes, profile["target"], profile["related"], system.ranker.eligible_codes))


def compare(system, profiles, bank):
    if not profiles or len({p["id"] for p in profiles}) != len(profiles):
        raise ValueError("Expected nonempty profiles with unique identifiers")
    results = {}
    for arm, no_probing in (("full", False), ("no_probing", True)):
        candidate = copy(system)
        candidate.policy = ManualPolicy(system.max_turns, no_probing=no_probing,
                                        probe_reason_conf=getattr(system.policy, "probe_reason_conf", 0.60))
        sessions = [simulate(candidate, profile, bank) for profile in profiles]
        results[arm] = dict(summary=aggregate([s["metrics"] for s in sessions]), sessions=sessions)
    delta = {key: results["full"]["summary"][key] - results["no_probing"]["summary"][key]
             for key in ("hit_at_5", "mrr", "ndcg_at_10")}
    return dict(arms=results, full_minus_no_probing=delta)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/system.json")
    parser.add_argument("--limit", type=int, default=5, help="First N development profiles; no held-out access")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("limit must be positive")
    output = args.output or ROOT / "reports/dialogue" / datetime.now(timezone.utc).strftime("dev_%Y%m%dT%H%M%S%fZ")
    if output.exists():
        parser.error("Output exists; choose a new directory to preserve prior results")
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    if config["components"]["filler"] != "question-bank":
        parser.error("Use question-bank wording for the controlled dialogue comparison")
    profiles_path = ROOT / "data/dev_profiles.jsonl"
    profiles = read_profiles(profiles_path)[:args.limit]
    bank = json.loads((ROOT / "data/question_bank.json").read_text())
    system = System.from_config(ROOT / "data", config)
    results = compare(system, profiles, bank)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    source = [*list((ROOT / "pipeline").glob("*.py")), Path(__file__)]
    checkpoints = {}
    for key in ("extractor_checkpoint", "nli_model"):
        directory = ROOT / config["paths"][key]
        checkpoints[key] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in directory.glob("*") if p.name in ("config.json", "model.safetensors")}
    manifest = dict(condition="actual_tracker_dialogue_development", quality_claim="none; functional smoke unless full evaluation is planned",
                    profile_ids=[p["id"] for p in profiles], config=config, git_revision=revision,
                    python=platform.python_version(), dataset_sha256=hashlib.sha256(profiles_path.read_bytes()).hexdigest(),
                    source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source},
                    checkpoints=checkpoints, components=system.components,
                    resolved_ranking=system.ranker.resolved_config,
                    metric_definitions="Existing M2 metrics: Hit@5, full-pool MRR, NDCG@10 with 2^grade-1 gains",
                    simulator="Reference graded CLARIFY replies; retry preserves action; familiarity reveal only for matching planted type")
    output.mkdir(parents=True)
    for name, value in (("manifest", manifest), ("results", results)):
        (output / f"{name}.json").write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({arm: result["summary"] for arm, result in results["arms"].items()}, indent=2))
    print("Full minus no_probing:", results["full_minus_no_probing"])
    print("Saved", output)


if __name__ == "__main__":
    main()
