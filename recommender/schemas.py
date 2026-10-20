"""M0 contracts. No ranking or evidence-selection algorithms live here."""

from dataclasses import dataclass, field
from math import isfinite
from typing import Literal

TYPES = ("R", "I", "A", "S", "E", "C")


def validate_vector(values: dict[str, float], *, runtime: bool) -> None:
    if not isinstance(values, dict) or set(values) != set(TYPES):
        raise ValueError("interests must contain exactly R,I,A,S,E,C")
    for key, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
            raise ValueError(f"{key}: interest must be a finite number")
        if runtime and not -1 <= value <= 1:
            raise ValueError(f"{key}: runtime interest must be in [-1,1]")


@dataclass
class RankingRequest:
    interests: dict[str, float]
    observed_types: set[str]
    user_sentences: list[str]
    top_k: int = 100
    source_evidence: list[dict] | None = None

    def __post_init__(self) -> None:
        validate_vector(self.interests, runtime=True)
        if not isinstance(self.observed_types, set) or not self.observed_types <= set(TYPES):
            raise ValueError("observed_types must be a set of allowed RIASEC keys")
        if not isinstance(self.user_sentences, list) or any(not isinstance(s, str) for s in self.user_sentences):
            raise ValueError("user_sentences must be a list of user-only strings")
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k < 1:
            raise ValueError("top_k must be a positive integer")

    @property
    def has_structured_signal(self) -> bool:
        return any(self.interests[t] != 0 for t in self.observed_types)

    @property
    def has_text(self) -> bool:
        return any(s.strip() for s in self.user_sentences)


@dataclass
class OracleInput:
    """Offline synthetic z-scale vector; deliberately not a RankingRequest."""

    interests: dict[str, float]
    user_sentences: list[str]
    input_mode: Literal["oracle"] = field(default="oracle", init=False)

    def __post_init__(self) -> None:
        validate_vector(self.interests, runtime=False)
        if not isinstance(self.user_sentences, list) or any(not isinstance(s, str) for s in self.user_sentences):
            raise ValueError("oracle sentences must be a list of strings")

    @classmethod
    def from_profile(cls, profile: dict) -> "OracleInput":
        # Only scoring inputs cross this boundary: target/relevance labels do not.
        return cls(dict(profile["hidden_slots"]), [profile["text"]])


@dataclass
class Occupation:
    code: str
    title: str
    z_scores: dict[str, float]
    bm25_doc: str | None = None


@dataclass
class SignalScore:
    score: float
    rank: int


@dataclass
class TaskEvidence:
    task_id: str
    occupation_code: str
    text: str
    match_score: float
    status: Literal["supporting", "provisional"] = "provisional"


@dataclass
class RankedOccupation:
    code: str
    title: str
    rank: int
    occupation_z: dict[str, float]
    rrf_score: float | None = None
    signals: dict[str, SignalScore] = field(default_factory=dict)
    evidence: list[TaskEvidence] = field(default_factory=list)


@dataclass
class RankingResponse:
    """Proposed policy envelope around the minimum ranked-list contract."""

    status: Literal["ranked", "no_signal"]
    occupations: list[RankedOccupation] = field(default_factory=list)
    reason: str | None = None
    query_diagnostics: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in {"ranked", "no_signal"}:
            raise ValueError("invalid ranking status")
        if self.status == "no_signal" and (self.occupations or not self.reason):
            raise ValueError("no_signal requires an empty list and a reason")
        if self.status == "ranked" and not self.occupations:
            raise ValueError("ranked requires occupations")
