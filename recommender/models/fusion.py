"""Occupation-level fusion of usable component rankings, separate from scoring.

Accepted fusion selections (each identifier keeps its meaning; add new variants, never edit one):

- ``{"algorithm": "rrf", "k": 60}``: equal-weight RRF, the control and live default (``rrf_k60``).
- ``{"algorithm": "rrf_tie_aware", "variant": "k60_v1"}``: the same RRF, but occupations with equal
  component scores share the average of the ranks they span, so ties are not broken by SOC code
  (docs/SEMANTIC_LEDGER_PLAN.md section 12.3, #1a).
- ``{"algorithm": "zscore_sum", "variant": "equal_v1"}``: per-query z-score of each component's raw
  scores, summed with equal weights, so a peaked component can beat a flat lead (#1b).
"""

from math import sqrt
from typing import Mapping, Protocol

from ..components import ComponentRanking, ComponentScore


class ReciprocalRankFusion:
    def __init__(self, k: int = 60):
        if type(k) is not int or k != 60:
            raise ValueError("baseline RRF requires k=60")
        self.k = k

    def fuse(self, rankings: Mapping[str, ComponentRanking]) -> ComponentRanking:
        scores = {}
        for key in sorted(rankings):
            ranking = rankings[key]
            ranking.__post_init__()
            if ranking.status != "usable":
                continue
            for item in ranking.occupations:
                scores[item.code] = scores.get(item.code, 0.0) + 1 / (self.k + item.rank)
        if not scores:
            return ComponentRanking("unusable", reason="No usable rankings for RRF.")
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return ComponentRanking("usable", tuple(ComponentScore(code, score, rank)
                                                 for rank, (code, score) in enumerate(ordered, 1)))


def _ordered(scores: dict[str, float]) -> ComponentRanking:
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return ComponentRanking("usable", tuple(ComponentScore(code, score, rank)
                                             for rank, (code, score) in enumerate(ordered, 1)))


def _usable(rankings: Mapping[str, ComponentRanking]):
    for key in sorted(rankings):
        ranking = rankings[key]
        ranking.__post_init__()
        if ranking.status == "usable":
            yield key, ranking


def average_ranks(ranking: ComponentRanking) -> dict[str, float]:
    """One-based ranks where exactly equal scores share the mean of the positions they occupy."""
    out, items, start = {}, ranking.occupations, 0
    while start < len(items):
        end = start
        while end + 1 < len(items) and items[end + 1].score == items[start].score:
            end += 1
        shared = (start + end) / 2 + 1
        for item in items[start:end + 1]:
            out[item.code] = shared
        start = end + 1
    return out


class TieAwareReciprocalRankFusion:
    """Equal-weight RRF on average ranks; otherwise identical to ``ReciprocalRankFusion``.

    An occupation missing from a component contributes nothing from it, as in the control.
    """

    def __init__(self, k: int = 60):
        if type(k) is not int or k != 60:
            raise ValueError("rrf_tie_aware k60_v1 requires k=60")
        self.k = k

    def fuse(self, rankings: Mapping[str, ComponentRanking]) -> ComponentRanking:
        scores = {}
        for _, ranking in _usable(rankings):
            for code, rank in average_ranks(ranking).items():
                scores[code] = scores.get(code, 0.0) + 1 / (self.k + rank)
        if not scores:
            return ComponentRanking("unusable", reason="No usable rankings for tie-aware RRF.")
        return _ordered(scores)


class ZScoreSumFusion:
    """Equal-weight sum of per-query z-scored component scores.

    Each usable component's raw scores are standardised over the occupations it ranked (population
    standard deviation). An occupation a component did not rank (e.g. cosine omits zero-norm
    occupations) gets that component's minimum z. A component whose scores are all equal carries no
    ordering information and contributes 0 to every occupation. Ties in the sum break by code.
    """

    EPSILON = 1e-12

    def fuse(self, rankings: Mapping[str, ComponentRanking]) -> ComponentRanking:
        standardised = []
        for _, ranking in _usable(rankings):
            values = [item.score for item in ranking.occupations]
            mean = sum(values) / len(values)
            std = sqrt(sum((v - mean) ** 2 for v in values) / len(values))
            if std <= self.EPSILON:
                z = {item.code: 0.0 for item in ranking.occupations}
            else:
                z = {item.code: (item.score - mean) / std for item in ranking.occupations}
            standardised.append(z)
        if not standardised:
            return ComponentRanking("unusable", reason="No usable rankings for z-score fusion.")
        codes = set().union(*standardised)
        floors = [min(z.values()) for z in standardised]
        return _ordered({code: sum(z.get(code, floor) for z, floor in zip(standardised, floors))
                         for code in codes})


class Fusion(Protocol):
    def fuse(self, rankings: Mapping[str, ComponentRanking]) -> ComponentRanking: ...


# Frozen fusion variants: selection -> (identifier, factory). Never change an entry's meaning.
FUSION_VARIANTS = {
    (("algorithm", "rrf"), ("k", 60)): ("rrf_k60", lambda: ReciprocalRankFusion(60)),
    (("algorithm", "rrf_tie_aware"), ("variant", "k60_v1")): (
        "rrf_tie_aware@k60_v1", lambda: TieAwareReciprocalRankFusion(60)),
    (("algorithm", "zscore_sum"), ("variant", "equal_v1")): ("zscore_sum@equal_v1", ZScoreSumFusion),
}


def _variant_key(config):
    if not isinstance(config, dict) or any(type(v) not in (str, int) for v in config.values()):
        return None
    return tuple(sorted(config.items()))


def validate_fusion(config):
    if config is not None and _variant_key(config) not in FUSION_VARIANTS:
        raise ValueError("fusion must be null or a registered variant: "
                         + ", ".join(identifier for identifier, _ in FUSION_VARIANTS.values()))


def fusion_identifier(config) -> str:
    validate_fusion(config)
    return "" if config is None else FUSION_VARIANTS[_variant_key(config)][0]


def build_fusion(config) -> Fusion:
    validate_fusion(config)
    if config is None:
        raise ValueError("fusion is disabled")
    return FUSION_VARIANTS[_variant_key(config)][1]()


def select_ranking(rankings: Mapping[str, ComponentRanking], config: dict | None,
                   fusion: Fusion | None = None) -> tuple[ComponentRanking, bool]:
    """Inactive signals never receive fabricated ranks or enter fusion."""
    usable = {key: ranking for key, ranking in rankings.items() if ranking.status == "usable"}
    if not usable:
        reasons = "; ".join(r.reason for r in rankings.values() if r.reason)
        return ComponentRanking("unusable", reason=reasons or "No scoring components are enabled."), False
    if len(usable) == 1:
        return next(iter(usable.values())), False
    if config is None:
        raise ValueError("multiple usable components require fusion configuration")
    if fusion is None:
        fusion = build_fusion(config)
    result = fusion.fuse(usable)
    if result.status != "usable":
        raise ValueError("fusion must return a usable occupation ranking")
    return result, True
