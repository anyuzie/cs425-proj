"""Wires the components together. Server and CLI both use System."""

import json
from dataclasses import dataclass, replace
from pathlib import Path

from . import registry
from .interfaces import MAX_TURNS, TYPES, Action, State

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "system.json"
QUESTIONS = ("CLARIFY", "ABSTAIN", "PROBE", "CHALLENGE")

NO_SIGNAL_REPLY = ("I could not pick up enough about your preferences to suggest occupations. "
                   "Thanks for chatting, and please try again with a few more details.")


@dataclass
class TurnResult:
    reply: str
    action: str
    turn: int
    done: bool
    quotes: dict | None = None
    recommendations: list[dict] | None = None


class System:
    """Orchestrates the components; per-turn rules live in the components themselves."""

    def __init__(self, extractor, tracker, rule, confirmer, ranker, policy, filler, type_names,
                 max_turns=MAX_TURNS, confirm_threshold=0.6):
        self.extractor, self.tracker, self.rule, self.confirmer = extractor, tracker, rule, confirmer
        self.ranker, self.policy, self.filler, self.type_names = ranker, policy, filler, type_names
        self.max_turns, self.confirm_threshold = max_turns, confirm_threshold

    @classmethod
    def load(cls, data_dir, config_path=None) -> "System":
        d = Path(data_dir)
        config = json.loads(Path(config_path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
        c = registry.build(config, d)
        names = json.loads((d / "labels.json").read_text(encoding="utf-8"))["type_names"]
        st = config.get("settings", {})
        return cls(c["extractor"], c["tracker"], c["rule"], c["confirmer"], c["ranker"], c["policy"], c["filler"],
                   names, st.get("max_turns", MAX_TURNS), st.get("confirm_threshold", 0.6))

    @property
    def eligible(self) -> int:
        return self.ranker.eligible_count

    @property
    def components(self) -> dict:
        return {k: getattr(self, k).name for k in ("extractor", "tracker", "rule", "confirmer", "ranker",
                                                   "policy", "filler")}

    def start(self, state: State) -> TurnResult:
        reply = self.filler.fill("OPENING", {})
        state.last_action = Action("OPENING", None, "opening")
        state.last_reply = reply
        return TurnResult(reply, "OPENING", 0, False)

    def _rank(self, state: State, top_k: int):
        interests = {t: (state.slots[t].value if t in state.slots else 0.0) for t in TYPES}
        resp = self.ranker.rank(interests, set(state.slots), list(state.user_texts), top_k)
        return resp.occupations if resp.status == "ranked" else []

    def top5(self, state: State) -> list[dict]:
        return [{"code": o.code, "title": o.title} for o in self._rank(state, 5)]

    def _why(self, state: State) -> str:
        pos = sorted((t for t, s in state.slots.items() if s.value > 0), key=lambda t: -state.slots[t].value)[:2]
        if not pos:
            return "Ranked highest against the interests you described."
        return "Matches what you said about " + " and ".join(self.type_names[t].lower() for t in pos) + " work."

    def recommendations(self, state: State) -> list[dict]:
        why = self._why(state)
        return [{"rank": o.rank, "code": o.code, "title": o.title, "why": why,
                 "tasks": [e.text for e in o.evidence[:2]]} for o in self._rank(state, 5)]

    def _confirmed(self, state: State) -> list:
        flags = []
        for f in self.rule.candidates(state):
            score = self.confirmer.score(f.old.source, f.new.source)
            if score > self.confirm_threshold:
                flags.append(replace(f, score=score))
        return flags

    def step(self, state: State, text: str) -> TurnResult:
        state.turn += 1
        state.user_texts.append(text)
        last = state.last_action
        asked = last is not None and last.kind in QUESTIONS  # any slot question (ABSTAIN re-asks one)
        reading = self.extractor.read(text, state.last_reply if asked else None, last.slot if asked else None)
        self.tracker.update(state, reading)
        state.flags = self._confirmed(state)
        action = self.policy.decide(state)
        quotes, recs, done = None, None, False
        if action.kind == "RECOMMEND":
            recs = self.recommendations(state)
            done = True
            if recs:
                reply = self.filler.fill(action, action.payload)
            else:  # no usable signal at the end of the conversation
                recs, reply = None, NO_SIGNAL_REPLY
                action = Action("ABSTAIN", None, action.why + "; no structured signal")
        else:
            reply = self.filler.fill(action, action.payload)
        if action.kind == "CHALLENGE":
            quotes = {"earlier": action.payload["old_src"], "earlier_turn": action.payload["old_turn"],
                      "now": action.payload["new_src"]}
        elif action.kind == "PROBE":
            quotes = {"now": action.payload["new_src"]}
        state.last_action, state.last_reply = action, reply
        state.done = done or state.turn >= self.max_turns
        state.actions.append({"turn": state.turn, "action": action.kind, "why": action.why})
        return TurnResult(reply, action.kind, state.turn, state.done, quotes, recs)

    def profile(self, state: State) -> dict:
        return {t: ({"value": state.slots[t].value, "confidence": state.slots[t].confidence,
                     "reason": state.slots[t].reason} if t in state.slots
                    else {"value": None, "confidence": 0.0, "reason": None}) for t in TYPES}

    def trace(self, state: State) -> dict:
        return {"profile": self.profile(state), "actions": list(state.actions),
                "top5": self.top5(state), "components": self.components}
