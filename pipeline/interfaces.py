"""Interfaces and shared dataclasses. State is keyed by RIASEC letters."""

from dataclasses import dataclass, field
from typing import Protocol

TYPES = ("R", "I", "A", "S", "E", "C")
MAX_TURNS = 10


@dataclass
class SlotReading:
    slot: str
    sign: int            # +1 or -1
    strength: float      # 0..1
    confidence: float
    reason: str          # enjoyment | familiarity | neither
    source: str          # source sentence

    @property
    def value(self) -> float:
        return self.sign * self.strength


@dataclass
class Reading:
    text: str
    slots: list[SlotReading] = field(default_factory=list)


@dataclass
class HistoryEntry:
    value: float
    confidence: float
    reason: str
    source: str
    turn: int


@dataclass
class SlotState:
    """Latest value per slot plus the history the checker needs."""

    value: float
    confidence: float
    reason: str
    source: str
    turn: int
    history: list[HistoryEntry] = field(default_factory=list)


@dataclass
class Flag:
    slot: str
    old: HistoryEntry
    new: HistoryEntry


@dataclass
class Action:
    kind: str                      # CLARIFY PROBE CHALLENGE ABSTAIN RECOMMEND
    slot: str | None = None
    why: str = ""
    payload: dict = field(default_factory=dict)


@dataclass
class State:
    turn: int = 0
    slots: dict[str, SlotState] = field(default_factory=dict)
    user_texts: list[str] = field(default_factory=list)
    last_action: Action | None = None
    last_reading_empty: bool = False
    flags: list[Flag] = field(default_factory=list)
    challenged: set[str] = field(default_factory=set)
    probed: set[str] = field(default_factory=set)
    settled: set[str] = field(default_factory=set)
    strikes: dict[str, int] = field(default_factory=dict)
    clarify_count: dict[str, int] = field(default_factory=dict)
    kind_count: dict[str, int] = field(default_factory=dict)
    actions: list[dict] = field(default_factory=list)
    done: bool = False


class Extractor(Protocol):
    name: str
    def read(self, text: str, expected_slot: str | None = None) -> Reading: ...


class Tracker(Protocol):
    name: str
    def update(self, state: State, reading: Reading) -> None: ...


class ContradictionChecker(Protocol):
    name: str
    def check(self, state: State) -> list[Flag]: ...


class Policy(Protocol):
    name: str
    def decide(self, state: State) -> Action: ...


class EvidenceProvider(Protocol):
    name: str
    def top_tasks(self, code: str, user_texts: list[str], n: int = 2) -> list[str]: ...
