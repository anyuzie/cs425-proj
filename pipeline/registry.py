"""Component name -> factory(config, data_dir). Real models are imported lazily, inside their factory,
so stubs never need torch. Never import torch/transformers at the top of this file."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KINDS = ("extractor", "tracker", "rule", "confirmer", "ranker", "policy", "filler")


def _settings(cfg: dict) -> dict:
    return cfg.get("settings", {})


def _extractor_lexicon(cfg, d):
    from .stubs import LexiconExtractor
    return LexiconExtractor(d / "riasec_lexicon.json")


def _tracker_simple(cfg, d):
    from .stubs import SimpleTracker
    return SimpleTracker()


def _rule(cfg, d):
    from .stubs import ContradictionRule
    return ContradictionRule()


def _confirmer_passthrough(cfg, d):
    from .stubs import PassthroughConfirmer
    return PassthroughConfirmer()


def _ranker_structured(cfg, d):
    from .stubs import StructuredKeywordRanker
    return StructuredKeywordRanker(d / "occupations.csv", d / "tasks.csv")


def _policy_rule(cfg, d):
    from .stubs import RulePolicy
    return RulePolicy(_settings(cfg).get("max_turns", 10))


def _filler_bank(cfg, d):
    from .templates import QuestionBankFiller
    qb = json.loads((d / "question_bank.json").read_text(encoding="utf-8"))
    names = json.loads((d / "labels.json").read_text(encoding="utf-8"))["type_names"]
    return QuestionBankFiller(qb, names)


# --- real parts: each imports its own file only when chosen in configs/system.json ---
def _extractor_deberta(cfg, d):
    from .extractor import DebertaExtractor  # pipeline/extractor.py
    return DebertaExtractor(ROOT / cfg["paths"]["extractor_checkpoint"])


def _tracker_decay(cfg, d):
    from .tracker import DecayTracker  # pipeline/tracker.py
    return DecayTracker()


def _confirmer_nli(cfg, d):
    from .nli import NliConfirmer  # pipeline/nli.py
    return NliConfirmer(cfg["paths"]["nli_model"])


def _ranker_rrf(cfg, d):
    from recommender.fusion import RRFRanker  # recommender/fusion.py (+ bm25.py, dense.py)
    p = cfg["paths"]
    return RRFRanker(d / "occupations.csv", ROOT / p["bm25"], ROOT / p["faiss"], ROOT / p["faiss_owners"], p["embedder"])


def _policy_manual(cfg, d):
    from .policy import ManualPolicy  # pipeline/policy.py
    return ManualPolicy(_settings(cfg).get("max_turns", 10), _settings(cfg).get("no_probing", False))


REGISTRY: dict[str, dict] = {
    "extractor": {"lexicon-stub": _extractor_lexicon, "deberta": _extractor_deberta},
    "tracker": {"simple-stub": _tracker_simple, "decay": _tracker_decay},
    "rule": {"rule": _rule},
    "confirmer": {"passthrough-stub": _confirmer_passthrough, "nli": _confirmer_nli},
    "ranker": {"structured+keyword-stub": _ranker_structured, "rrf": _ranker_rrf},
    "policy": {"rule-stub": _policy_rule, "manual": _policy_manual},
    "filler": {"question-bank": _filler_bank},
}


def build(config: dict, data_dir) -> dict:
    """Instantiate every component named in config['components']."""
    data_dir, chosen, out = Path(data_dir), config.get("components", {}), {}
    for kind in KINDS:
        name = chosen.get(kind)
        if name not in REGISTRY[kind]:
            raise ValueError(f"unknown {kind} {name!r} in config; valid {kind} names: {sorted(REGISTRY[kind])}")
        out[kind] = REGISTRY[kind][name](config, data_dir)
    return out
