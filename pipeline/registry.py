"""Component name -> factory(config, data_dir). Real models are imported lazily, inside their factory,
so stubs never need torch. Never import torch/transformers at the top of this file."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KINDS = ("extractor", "tracker", "rule", "confirmer", "ranker", "policy", "filler")
DOC = "docs/INTEGRATION.md"


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


def _todo(kind: str, name: str, module: str):
    def factory(cfg, d):
        raise NotImplementedError(f"{kind} '{name}' is registered but {module} does not exist yet; "
                                  f"see {DOC} for the interface and how to implement it")
    return factory


REGISTRY: dict[str, dict] = {
    "extractor": {"lexicon-stub": _extractor_lexicon, "deberta": _todo("extractor", "deberta", "pipeline/extractor.py")},
    "tracker": {"simple-stub": _tracker_simple, "decay": _todo("tracker", "decay", "pipeline/tracker.py")},
    "rule": {"rule": _rule},
    "confirmer": {"passthrough-stub": _confirmer_passthrough, "nli": _todo("confirmer", "nli", "pipeline/nli.py")},
    "ranker": {"structured+keyword-stub": _ranker_structured, "rrf": _todo("ranker", "rrf", "recommender/fusion.py")},
    "policy": {"rule-stub": _policy_rule, "manual": _todo("policy", "manual", "pipeline/policy.py")},
    "filler": {"question-bank": _filler_bank},
}
# When a real module lands, replace its _todo with a lazy factory, e.g.
#   def _extractor_deberta(cfg, d):
#       from .extractor import DebertaExtractor   # import here, not at module top
#       return DebertaExtractor(ROOT / cfg["paths"]["extractor_checkpoint"])


def build(config: dict, data_dir) -> dict:
    """Instantiate every component named in config['components']."""
    data_dir, chosen, out = Path(data_dir), config.get("components", {}), {}
    for kind in KINDS:
        name = chosen.get(kind)
        if name not in REGISTRY[kind]:
            raise ValueError(f"unknown {kind} {name!r} in config; valid {kind} names: {sorted(REGISTRY[kind])}")
        out[kind] = REGISTRY[kind][name](config, data_dir)
    return out
