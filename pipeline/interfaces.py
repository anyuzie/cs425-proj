from dataclasses import dataclass, field
from typing import Protocol

from recommender.schemas import RankingResponse

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
    reason_confidence: float = 1.0   # probability of `reason`; PROBE needs it above settings.probe_reason_conf

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
    reason_confidence: float = 1.0


@dataclass
class SlotState:
    """Latest value per slot plus the history the checker needs."""

    value: float
    confidence: float
    reason: str
    source: str
    turn: int
    history: list[HistoryEntry] = field(default_factory=list)
    reason_confidence: float = 1.0
    weight: float | None = None  # cumulative accepted confidence; None for legacy states


@dataclass
class Flag:
    slot: str
    old: HistoryEntry
    new: HistoryEntry
    score: float = 0.0   # confirmer's contradiction probability, set by System


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
    resolved: set[str] = field(default_factory=set)   # slots whose PROBE was answered; the rule skips them
    last_reply: str = ""                              # bot text of the previous turn (context for the extractor)
    strikes: dict[str, int] = field(default_factory=dict)
    clarify_count: dict[str, int] = field(default_factory=dict)
    kind_count: dict[str, int] = field(default_factory=dict)
    actions: list[dict] = field(default_factory=list)
    ranking: list = field(default_factory=list)       # this turn's top 100 RankedOccupations, set before the policy runs
    ranking_trace: list[dict] = field(default_factory=list)
    done: bool = False
    profile_history: list[dict[str, float]] = field(default_factory=list)
    conflict_candidates: list[Flag] | None = None
    merge_conflicts: bool = False      # control arm: merge opposite readings instead of holding them for a CHALLENGE
    revisited: set[str] = field(default_factory=set)            # settled types asked once more at the end
    unresolved_conflicts: list[str] = field(default_factory=list)  # challenges given up on (newer statement kept)
    profile_changed: bool = False      # set by the policy when it changed a slot this turn; System re-ranks
    turn_trace: dict = field(default_factory=dict)  # how this turn was handled, for the transcript log
    asked_tasks: set[tuple[str, str]] = field(default_factory=set)
    task_history: list[dict] = field(default_factory=list)  # question/source/reply plus actual extractor readings


class Extractor(Protocol):
    name: str
    def read(self, text: str, context: str | None = None, expected_slot: str | None = None) -> Reading:
        """context: the bot question the user is answering (None if the last turn was not a question)."""


class Tracker(Protocol):
    name: str
    def update(self, state: State, reading: Reading) -> None:
        """Merge the reading into state.slots/history, set state.last_reading_empty, and add the slot
        to state.resolved when the last action was a PROBE on it."""


class ContradictionRule(Protocol):
    name: str
    def candidates(self, state: State) -> list[Flag]: ...


class Confirmer(Protocol):
    name: str
    def score(self, old_src: str, new_src: str) -> float:
        """Contradiction probability 0..1 for two source sentences."""


class Ranker(Protocol):
    name: str
    eligible_count: int
    def rank(self, interests: dict, observed_types: set, user_sentences: list[str], top_k: int) -> RankingResponse:
        """Each RankedOccupation.evidence holds the TaskEvidence used as card quotes."""


class Policy(Protocol):
    name: str
    def decide(self, state: State) -> Action:
        """Returns the action with a complete filler payload (slot, index, clarify_index, quote fields);
        owns question-rotation counters (state.clarify_count, state.kind_count)."""


class Filler(Protocol):
    name: str
    def fill(self, action, payload: dict) -> str: ...
