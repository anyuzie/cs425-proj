"""Equal-weight occupation-level reciprocal rank fusion, separate from scoring."""

from typing import Mapping, Protocol

from .components import ComponentRanking, ComponentScore


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


class Fusion(Protocol):
    def fuse(self, rankings: Mapping[str, ComponentRanking]) -> ComponentRanking: ...


def validate_fusion(config):
    if config is not None and config != {"algorithm": "rrf", "k": 60}:
        raise ValueError("fusion must be null or baseline rrf with k=60")


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
        validate_fusion(config)
        fusion = ReciprocalRankFusion(config["k"])
    result = fusion.fuse(usable)
    if result.status != "usable":
        raise ValueError("fusion must return a usable occupation ranking")
    return result, True
