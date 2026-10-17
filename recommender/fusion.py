"""Fusion boundary. RRF is specified for future work, not implemented here."""

from typing import Mapping, Protocol

from .components import ComponentRanking


class Fusion(Protocol):
    def fuse(self, rankings: Mapping[str, ComponentRanking]) -> ComponentRanking: ...


def validate_fusion(config):
    if config is not None and config != {"algorithm": "rrf", "k": 60}:
        raise ValueError("fusion must be null or the planned rrf with k=60")


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
        raise NotImplementedError("RRF fusion is planned but not implemented; only standalone ranking is supported at this gate")
    result = fusion.fuse(usable)
    if result.status != "usable":
        raise ValueError("fusion must return a usable occupation ranking")
    return result, True
