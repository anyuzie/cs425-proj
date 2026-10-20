import re
from typing import Literal

from pydantic import BaseModel, field_validator

CODE_RE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
MESSAGE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class SessionRequest(BaseModel):
    participant_code: str | None = None
    debug: bool = True

    @field_validator("participant_code")
    @classmethod
    def _code(cls, v):
        if v is not None and not CODE_RE.fullmatch(v):
            raise ValueError("participant_code must match ^[A-Za-z0-9_-]{1,32}$")
        return v


class ChatRequest(BaseModel):
    session_id: str
    text: str
    message_id: str | None = None  # client-generated; a repeat returns the stored response without stepping

    @field_validator("message_id")
    @classmethod
    def _mid(cls, v):
        if v is not None and not MESSAGE_ID_RE.fullmatch(v):
            raise ValueError("message_id must match ^[A-Za-z0-9_-]{1,64}$")
        return v

    @field_validator("text")
    @classmethod
    def _text(cls, v):
        v = v.strip()
        if not 1 <= len(v) <= 2000:
            raise ValueError("text must be 1..2000 characters after strip")
        return v


class Rec(BaseModel):
    rank: int
    code: str
    title: str
    why: str
    tasks: list[str]
    task_evidence: list[dict] = []


class TraceProfileSlot(BaseModel):
    value: float | None
    confidence: float
    reason: str | None


class TraceAction(BaseModel):
    turn: int
    action: str
    why: str


class TraceTop(BaseModel):
    code: str
    title: str


class Trace(BaseModel):
    profile: dict[str, TraceProfileSlot]
    actions: list[TraceAction]
    top5: list[TraceTop]
    components: dict[str, str]
    input_mode: str = "unverified_tracker"
    ranking_trace: list[dict] = []
    resolved_ranking: dict | None = None


class QuotesNow(BaseModel):
    now: str


class QuotesChallenge(BaseModel):
    earlier: str
    earlier_turn: int
    now: str


class TurnResponse(BaseModel):
    session_id: str
    participant_code: str
    reply: str
    action: Literal["OPENING", "CLARIFY", "PROBE", "CHALLENGE", "ABSTAIN", "RECOMMEND"]
    turn: int
    max_turns: int
    done: bool
    recommendations: list[Rec] | None = None
    trace: Trace | None = None
    quotes: QuotesChallenge | QuotesNow | None = None


class Health(BaseModel):
    status: str
    eligible: int
    components: dict[str, str]
