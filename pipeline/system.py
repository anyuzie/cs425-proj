"""Wires the components together. Server and CLI both use System."""

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import perf_counter

from . import registry
from .interfaces import MAX_TURNS, TYPES, Action, State
from .answers import read_answer
from .repair import needs_repair

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "system.json"
QUESTIONS = ("CLARIFY", "ABSTAIN", "PROBE", "CHALLENGE")
POLICY_TOP_K = 100  # ranking README: the dialogue policy consumes the top 100

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
                 max_turns=MAX_TURNS, confirm_threshold=0.6, input_mode="unverified_tracker"):
        self.extractor, self.tracker, self.rule, self.confirmer = extractor, tracker, rule, confirmer
        self.ranker, self.policy, self.filler, self.type_names = ranker, policy, filler, type_names
        self.max_turns, self.confirm_threshold = max_turns, confirm_threshold
        if getattr(ranker, "evidence_filter", "raw") != "raw":
            raise ValueError("Live System requires raw ranking queries; active_only requires "
                             "upstream lifecycle evidence snapshots and is supported by tracker replay only")
        if input_mode not in {"actual_tracker", "mock_tracker", "unverified_tracker"}:
            raise ValueError("invalid tracker input mode")
        if input_mode == "actual_tracker" and any(
                getattr(c, "name", "").startswith("test-") for c in (extractor, tracker)):
            raise ValueError("test components cannot be labelled actual_tracker")
        self.input_mode = input_mode

    @classmethod
    def load(cls, data_dir, config_path=None) -> "System":
        config = json.loads(Path(config_path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
        return cls.from_config(data_dir, config)

    @classmethod
    def from_config(cls, data_dir, config) -> "System":
        """Shared composition root for server, terminal and configured workflows."""
        d = Path(data_dir)
        c = registry.build(config, d)
        names = json.loads((d / "labels.json").read_text(encoding="utf-8"))["type_names"]
        st = config.get("settings", {})
        return cls(c["extractor"], c["tracker"], c["rule"], c["confirmer"], c["ranker"], c["policy"], c["filler"],
                   names, st.get("max_turns", MAX_TURNS), st.get("confirm_threshold", 0.6),
                   st.get("input_mode", "unverified_tracker"))

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
        started = perf_counter()
        resp = self.ranker.rank(interests, set(state.slots), list(state.user_texts), top_k)
        state.ranking_trace.append({"turn": state.turn, "input_mode": self.input_mode,
            "text": state.user_texts[-1] if state.user_texts else "",
            "interests": interests, "observed_types": sorted(state.slots),
            "user_sentences": list(state.user_texts), "status": resp.status, "reason": resp.reason,
            "rank_seconds": perf_counter() - started,
            "occupations": [asdict(o) for o in resp.occupations]})
        return resp.occupations if resp.status == "ranked" else []

    def top5(self, state: State) -> list[dict]:
        return [{"code": o.code, "title": o.title} for o in state.ranking[:5]]

    def _why(self, state: State) -> str:
        pos = sorted((t for t, s in state.slots.items() if s.value > 0), key=lambda t: -state.slots[t].value)[:2]
        if not pos:
            return "Ranked highest against the interests you described."
        return "Matches what you said about " + " and ".join(self.type_names[t].lower() for t in pos) + " work."

    def recommendations(self, state: State) -> list[dict]:
        why = self._why(state)
        return [{"rank": o.rank, "code": o.code, "title": o.title, "why": why,
                 "tasks": [e.text for e in o.evidence[:3]],
                 "task_evidence": [asdict(e) for e in o.evidence[:3]]} for o in state.ranking[:5]]

    def _confirmed(self, state: State) -> list:
        flags = []
        rule_state = state
        if state.conflict_candidates is not None:
            # Present pending tracker evidence without changing the teammate's rule.
            slots = {f.slot: replace(state.slots[f.slot], value=f.new.value,
                        confidence=f.new.confidence, reason=f.new.reason, source=f.new.source,
                        turn=state.turn, history=[f.old, f.new]) for f in state.conflict_candidates}
            rule_state = replace(state, slots=slots)
        for f in self.rule.candidates(rule_state):
            score = self.confirmer.score(f.old.source, f.new.source)
            if score > self.confirm_threshold:
                flags.append(replace(f, score=score))
        return flags

    def step(self, state: State, text: str) -> TurnResult:
        if state.done:
            raise ValueError("Conversation has ended; start a new session")
        state.turn += 1
        state.user_texts.append(text)
        last = state.last_action
        asked = last is not None and last.kind in QUESTIONS  # any slot question (ABSTAIN re-asks one)
        # The answer speaks for itself first; short CHALLENGE/PROBE answers are resolved by rule (pipeline/answers.py).
        reading = read_answer(self.extractor, state, last if asked else None, text)
        repair = asked and needs_repair(text)
        # The control arm listens to everything: with no_probing, opposite readings are merged, not held.
        state.merge_conflicts = bool(getattr(self.policy, "no_probing", False))
        self.tracker.update(state, reading)
        retry = last.payload.get("retry", "CLARIFY") if last and last.kind == "ABSTAIN" else last.kind if last else None
        if last and retry in ("CHALLENGE", "PROBE") and any(r.slot == last.slot for r in reading.slots):
            state.flags = [f for f in state.flags if f.slot != last.slot]
        # Confirmed conflicts survive neutral or unrelated answers until resolved.
        flags = {f.slot: f for f in state.flags}
        for flag in self._confirmed(state):
            flags.setdefault(flag.slot, flag)
        state.flags = list(flags.values())
        # Opposite readings the rule/NLI did not confirm are ordinary evidence: merge them (both arms alike).
        merge = getattr(self.tracker, "merge_unconfirmed", None)
        for candidate in state.conflict_candidates or []:
            if candidate.slot in flags:   # already held: keep the earlier statement, quote the latest opposite one
                flags[candidate.slot] = replace(flags[candidate.slot], new=candidate.new)
            elif merge:
                merge(state, candidate)
        state.flags = list(flags.values())
        state.profile_history.append({t: s.value for t, s in state.slots.items()})
        state.ranking = self._rank(state, POLICY_TOP_K)  # the policy reads it; the panel and cards reuse it
        state.profile_changed = False
        action = self.policy.decide(state)
        if state.profile_changed:   # the policy settled a stuck challenge: keep history, panel and cards consistent
            state.profile_history[-1] = {t: s.value for t, s in state.slots.items()}
            if state.ranking_trace and state.ranking_trace[-1].get("turn") == state.turn:
                state.ranking_trace.pop()   # replace this turn's trace row instead of adding a second one
            state.ranking = self._rank(state, POLICY_TOP_K)
        if action.kind == 'CLARIFY':
            from .t5_v2 import FOCUS
            action.payload['focus'] = FOCUS[action.slot][action.payload['clarify_index'] % 3]
        if action.kind == 'ABSTAIN' and repair:
            action.payload['repair'] = True
            action.payload['repair_count'] = last.payload.get('repair_count', 0) + 1
        if getattr(self.filler, "uses_context", False) and action.kind != "RECOMMEND":
            from .t5_v2 import KNOWN
            known = next((t for t, s in sorted(state.slots.items(), key=lambda item: -item[1].value)
                          if s.value > 0 and s.reason == "enjoyment" and t != action.slot), None)
            action.payload.update(last_user=text[:240], known_interest=KNOWN[known] if known else "")
        quotes, recs, done = None, None, False
        if action.kind == "RECOMMEND":
            recs = self.recommendations(state)
            done = True
            if recs:
                reply = self.filler.fill(action, action.payload)
            else:  # no usable signal at the end of the conversation
                recs, reply = None, NO_SIGNAL_REPLY
                action = Action("ABSTAIN", None, action.why + "; no usable ranking signal")
        else:
            reply = self.filler.fill(action, action.payload)
        quote_kind = action.payload.get("retry") if action.kind == "ABSTAIN" else action.kind
        if quote_kind == "CHALLENGE":
            quotes = {"earlier": action.payload["old_src"], "earlier_turn": action.payload["old_turn"],
                      "now": action.payload["new_src"]}
        elif quote_kind == "PROBE":
            quotes = {"now": action.payload["new_src"]}
        state.last_action, state.last_reply = action, reply
        state.done = done or state.turn >= self.max_turns
        state.actions.append({"turn": state.turn, "action": action.kind, "why": action.why})
        return TurnResult(reply, action.kind, state.turn, state.done, quotes, recs)

    def profile(self, state: State) -> dict:
        empty = {"value": None, "confidence": 0.0, "reason": None, "reason_confidence": 0.0}
        return {t: ({"value": s.value, "confidence": s.confidence, "reason": s.reason,
                     "reason_confidence": s.reason_confidence} if (s := state.slots.get(t)) else dict(empty))
                for t in TYPES}

    def trace(self, state: State) -> dict:
        return {"profile": self.profile(state), "actions": list(state.actions),
                "top5": self.top5(state), "components": self.components,
                "input_mode": self.input_mode, "ranking_trace": list(state.ranking_trace),
                "resolved_ranking": getattr(self.ranker, "resolved_config", None)}

    def tracker_record(self, state: State, *, profile_id: str, revision: str) -> dict:
        """Export inputs only; independent annotations must add target/related for evaluation."""
        if self.input_mode == "unverified_tracker":
            raise ValueError("declare actual_tracker or mock_tracker before exporting evaluation inputs")
        if not revision.strip():
            raise ValueError("upstream code/model revision is required")
        return {"id": profile_id, "input_mode": self.input_mode,
            "upstream": {"extractor": self.extractor.name, "tracker": self.tracker.name, "revision": revision},
            "turns": [{k: row[k] for k in ("text", "interests", "observed_types")}
                      for row in state.ranking_trace]}
