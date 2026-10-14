"""Wires the components together. Server and CLI both use System."""

import json
from dataclasses import dataclass
from pathlib import Path

from recommender.data_loader import load_eligible_occupations
from recommender.structured import StructuredRanker

from . import templates
from .interfaces import MAX_TURNS, TYPES, Action, State
from .stubs import (KeywordEvidence, LexiconExtractor, RuleChecker, RulePolicy, SimpleTracker)

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
    def __init__(self, ranker, qb, type_names, extractor, tracker, checker, policy, evidence, max_turns=MAX_TURNS):
        self.ranker, self.qb, self.type_names = ranker, qb, type_names
        self.extractor, self.tracker, self.checker = extractor, tracker, checker
        self.policy, self.evidence = policy, evidence
        self.max_turns = max_turns

    @classmethod
    def load(cls, data_dir) -> "System":
        d = Path(data_dir)
        eligible = load_eligible_occupations(d / "occupations.csv")
        qb = json.loads((d / "question_bank.json").read_text(encoding="utf-8"))
        names = json.loads((d / "labels.json").read_text(encoding="utf-8"))["type_names"]
        return cls(StructuredRanker(d / "occupations.csv"), qb, names,
                   LexiconExtractor(d / "riasec_lexicon.json"), SimpleTracker(), RuleChecker(), RulePolicy(),
                   KeywordEvidence(d / "tasks.csv", {o.code for o in eligible}))

    @property
    def eligible(self) -> int:
        return self.ranker.eligible_count

    @property
    def components(self) -> dict:
        return {"extractor": self.extractor.name, "tracker": self.tracker.name, "checker": self.checker.name,
                "policy": self.policy.name, "evidence": self.evidence.name, "ranker": "structured"}

    def start(self, state: State) -> TurnResult:
        reply = templates.fill("OPENING", {}, self.qb, self.type_names)
        state.last_action = Action("OPENING", None, "opening")
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
                 "tasks": self.evidence.top_tasks(o.code, state.user_texts, 2)} for o in self._rank(state, 5)]

    def step(self, state: State, text: str) -> TurnResult:
        state.turn += 1
        state.user_texts.append(text)
        # A reply to any slot question (ABSTAIN re-asks one) is read as being about that slot.
        last = state.last_action
        expected = last.slot if last and last.kind in ("CLARIFY", "ABSTAIN", "PROBE", "CHALLENGE") else None
        self.tracker.update(state, self.extractor.read(text, expected))
        if last and last.kind == "PROBE" and last.slot:
            state.challenged.add(last.slot)  # the probe answer resolves the slot; a flip here is not a contradiction
        state.flags = self.checker.check(state)
        action = self.policy.decide(state)
        quotes, recs, done = None, None, False
        payload = dict(action.payload)
        payload["index"] = state.kind_count.get(action.kind, 0)
        if action.slot:
            payload["slot"] = action.slot
            payload["clarify_index"] = state.clarify_count.get(action.slot, 0)
        if action.kind == "RECOMMEND":
            recs = self.recommendations(state)
            done = True
            if recs:
                reply = templates.fill(action, payload, self.qb, self.type_names)
            else:  # no usable signal at the end of the conversation
                recs, reply = None, NO_SIGNAL_REPLY
                action = Action("ABSTAIN", None, action.why + "; no structured signal")
        else:
            reply = templates.fill(action, payload, self.qb, self.type_names)
        if action.kind == "CHALLENGE":
            quotes = {"earlier": action.payload["old_src"], "earlier_turn": action.payload["old_turn"],
                      "now": action.payload["new_src"]}
        elif action.kind == "PROBE":
            quotes = {"now": action.payload["new_src"]}
        if action.kind in ("CLARIFY", "ABSTAIN") and action.slot:
            state.clarify_count[action.slot] = state.clarify_count.get(action.slot, 0) + 1
        state.kind_count[action.kind] = state.kind_count.get(action.kind, 0) + 1
        state.last_action = action
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
