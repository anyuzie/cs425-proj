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


def _filler_t5(cfg, d):
    from .t5 import T5Filler
    qb = json.loads((d / "question_bank.json").read_text(encoding="utf-8"))
    names = json.loads((d / "labels.json").read_text(encoding="utf-8"))["type_names"]
    return T5Filler(qb, names, ROOT / cfg["paths"]["t5_checkpoint"])


def _filler_t5_v2(cfg, d):
    from .t5_v2 import T5V2Filler
    qb = json.loads((d / "question_bank.json").read_text(encoding="utf-8"))
    names = json.loads((d / "labels.json").read_text(encoding="utf-8"))["type_names"]
    return T5V2Filler(qb, names, ROOT / cfg["paths"]["t5_checkpoint"])


def _filler_t5_v3(cfg, d):
    from .t5_v3 import T5V3Filler
    qb = json.loads((d / "question_bank.json").read_text(encoding="utf-8"))
    names = json.loads((d / "labels.json").read_text(encoding="utf-8"))["type_names"]
    return T5V3Filler(qb, names, ROOT / cfg["paths"]["t5_checkpoint"])


# --- teammate parts: each imports its own file only when chosen in configs/system.json ---
def _extractor_deberta(cfg, d):
    from .extractor import DebertaExtractor  # pipeline/extractor.py
    return DebertaExtractor(ROOT / cfg["paths"]["extractor_checkpoint"], d / "labels.json")


def _tracker_decay(cfg, d):
    from .tracker import DecayTracker  # pipeline/tracker.py
    return DecayTracker(_settings(cfg).get("decay", 1.0))


def _confirmer_nli(cfg, d):
    from .nli import NliConfirmer  # pipeline/nli.py
    model = cfg["paths"]["nli_model"]
    local = ROOT / model
    if (Path(model).is_absolute() or model.replace("\\", "/").startswith("models/")) and not local.is_dir():
        raise FileNotFoundError(
            f"NLI checkpoint missing: {local}. Model weights are not included in Git. "
            "Copy the shared checkpoint folder here, then run "
            "python scripts/check_nli_checkpoint.py. See docs/NLI_DEPLOYMENT.md."
        )
    return NliConfirmer(str(local) if local.is_dir() else model)


def _policy_manual(cfg, d):
    from .policy import ManualPolicy  # pipeline/policy.py
    st = _settings(cfg)
    from .policy import PROBE_REASON_CONF
    return ManualPolicy(st.get("max_turns", 10), st.get("no_probing", False), st.get("probe_reason_conf", PROBE_REASON_CONF))


REGISTRY: dict[str, dict] = {
    "extractor": {"deberta": _extractor_deberta},
    "tracker": {"decay": _tracker_decay},
    "rule": {"rule": _rule},
    "confirmer": {"nli": _confirmer_nli},
    "ranker": {"structured": _ranker_structured, "configured": _ranker_configured},
    "policy": {"manual": _policy_manual},
    "filler": {"question-bank": _filler_bank, "t5": _filler_t5, "t5-v2": _filler_t5_v2, "t5-v3": _filler_t5_v3},
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
