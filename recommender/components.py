"""Shared component contracts; no algorithm or orchestration dependencies."""

from dataclasses import dataclass
from math import isfinite
from typing import Literal, Mapping, Protocol
from .schemas import TaskEvidence


@dataclass(frozen=True)
class ComponentInput:
    """Validated upstream snapshot. Components consume only fields they need.

    None means the structured profile is unavailable; an empty observed set is
    a supplied profile with no observed dimensions. No evaluator labels cross here.
    ``weights`` is the tracker's per-observed-slot evidence mass, or None when the
    caller has none (oracle inputs, legacy callers); only weight-aware components read it.
    """

    interests: Mapping[str, float] | None = None
    observed_types: frozenset[str] = frozenset()
    user_sentences: tuple[str, ...] = ()
    weights: Mapping[str, float] | None = None


@dataclass(frozen=True)
class ComponentScore:
    code: str
    score: float
    rank: int


@dataclass(frozen=True)
class ComponentRanking:
    status: Literal["usable", "unavailable", "unusable"]
    occupations: tuple[ComponentScore, ...] = ()
    reason: str | None = None
    evidence: tuple[TaskEvidence, ...] = ()

    def __post_init__(self):
        if self.status not in {"usable", "unavailable", "unusable"}:
            raise ValueError("invalid component signal status")
        if self.status != "usable":
            if self.occupations or self.evidence or not self.reason:
                raise ValueError("inactive signal requires an empty ranking and reason")
            return
        if not isinstance(self.occupations, tuple) or not self.occupations:
            raise ValueError("usable signal requires a nonempty tuple of occupations")
        codes = set()
        for rank, item in enumerate(self.occupations, 1):
            if not isinstance(item, ComponentScore):
                raise ValueError("component output requires ComponentScore entries")
            if not isinstance(item.code, str) or not item.code.strip() or item.code in codes:
                raise ValueError("component output has empty or duplicate occupation codes")
            if isinstance(item.score, bool) or not isinstance(item.score, (int, float)) or not isfinite(item.score):
                raise ValueError("component scores must be finite numbers")
            if type(item.rank) is not int or item.rank != rank:
                raise ValueError("component ranks must be contiguous and one-based")
            codes.add(item.code)
        if list(self.occupations) != sorted(self.occupations, key=lambda item: (-item.score, item.code)):
            raise ValueError("component output must sort score descending, code ascending")
        for evidence in self.evidence:
            if not isinstance(evidence, TaskEvidence) or evidence.occupation_code not in codes or not evidence.task_id or not evidence.text.strip() or not isfinite(evidence.match_score):
                raise ValueError("task evidence must reference a ranked occupation and finite task match")

    def validate_pool(self, pool: set[str]) -> None:
        if any(item.code not in pool for item in self.occupations):
            raise ValueError("component recommendations outside eligible pool")


class ScoringComponent(Protocol):
    def score(self, inputs: ComponentInput) -> ComponentRanking: ...
