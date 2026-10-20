"""Component name -> factory(config, data_dir). Components are imported lazily, inside their factory,
so the app starts without torch until a model part is built. Never import torch/transformers at the top of this file."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KINDS = ("extractor", "tracker", "rule", "confirmer", "ranker", "policy", "filler")


def _settings(cfg: dict) -> dict:
    return cfg.get("settings", {})


def _rule(cfg, d):
    from .rule import ContradictionRule
    return ContradictionRule()


def _ranker_structured(cfg, d):
    from recommender.models.structured import StructuredRanker  # the M1 baseline: no task quotes yet
    ranker = StructuredRanker(d / "occupations.csv")
    ranker.name = "structured"
    return ranker


def _ranker_configured(cfg, d):
    from recommender.models.ranker import RankingPipeline
    path = Path(cfg.get("paths", {}).get("ranking_config", "configs/ranking.yaml"))
    ranker = RankingPipeline(d / "occupations.csv", config=path if path.is_absolute() else ROOT / path)
    ranker.name = "configured"
    return ranker


def _filler_bank(cfg, d):
    from .templates import QuestionBankFiller
    qb = json.loads((d / "question_bank.json").read_text(encoding="utf-8"))
    names = json.loads((d / "labels.json").read_text(encoding="utf-8"))["type_names"]
    return QuestionBankFiller(qb, names)


# --- teammate parts: each imports its own file only when chosen in configs/system.json ---
def _extractor_deberta(cfg, d):
    from .extractor import DebertaExtractor  # pipeline/extractor.py
    return DebertaExtractor(ROOT / cfg["paths"]["extractor_checkpoint"], d / "labels.json")


def _tracker_decay(cfg, d):
    from .tracker import DecayTracker  # pipeline/tracker.py
    return DecayTracker()


def _confirmer_nli(cfg, d):
    from .nli import NliConfirmer  # pipeline/nli.py
    return NliConfirmer(cfg["paths"]["nli_model"])


def _policy_manual(cfg, d):
    from .policy import ManualPolicy  # pipeline/policy.py
    return ManualPolicy(_settings(cfg).get("max_turns", 10), _settings(cfg).get("no_probing", False))


REGISTRY: dict[str, dict] = {
    "extractor": {"deberta": _extractor_deberta},
    "tracker": {"decay": _tracker_decay},
    "rule": {"rule": _rule},
    "confirmer": {"nli": _confirmer_nli},
    "ranker": {"structured": _ranker_structured, "configured": _ranker_configured},
    "policy": {"manual": _policy_manual},
    "filler": {"question-bank": _filler_bank},
}


def build(config: dict, data_dir) -> dict:
    """Instantiate every component named in config['components']."""
    data_dir, chosen, out = Path(data_dir), config.get("components", {}), {}
    for kind in KINDS:
        name = chosen.get(kind)
        if name not in REGISTRY[kind]:
            raise ValueError(f"unknown {kind} {name!r} in config; valid {kind} names: {sorted(REGISTRY[kind])}")
    # Reject all invalid selections before loading any upstream checkpoints.
    for kind in KINDS:
        name = chosen[kind]
        out[kind] = REGISTRY[kind][name](config, data_dir)
    return out
