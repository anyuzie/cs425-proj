"""Deterministic actions over tracker state and teammate-confirmed conflicts."""

from statistics import pvariance

from .interfaces import MAX_TURNS, TYPES, Action, State


class PolicyBase:
    """Shared machinery. Subclasses implement _decide(state) and return an Action; decide() completes it."""

    name = "policy-base"

    def __init__(self, max_turns: int = MAX_TURNS, no_probing: bool = False) -> None:
        self.max_turns = max_turns
        self.no_probing = no_probing   # True = never CHALLENGE or PROBE (the evaluation's control arm)

    def _decide(self, state: State) -> Action:
        raise NotImplementedError

    def decide(self, state: State) -> Action:
        # Guaranteed here, not left to _decide: the conversation always ends with recommendations,
        # and the control arm can never probe (otherwise the full vs no_probing comparison is invalid).
        if state.turn >= self.max_turns:
            action = Action("RECOMMEND", None, f"turn {state.turn} reached max {self.max_turns}")
        else:
            action = self._decide(state)
            if self.no_probing and action.kind in ("CHALLENGE", "PROBE"):
                raise ValueError(f"policy returned {action.kind} while no_probing is on; skip rungs 1-2 in _decide")
        payload = dict(action.payload)
        payload["index"] = state.kind_count.get(action.kind, 0)
        if action.slot:
            payload["slot"] = action.slot
            payload["clarify_index"] = (action.payload.get('clarify_index', 0) if action.kind == 'ABSTAIN'
                                        else state.clarify_count.get(action.slot, 0))
        action.payload = payload
        if action.kind in ("CLARIFY", "ABSTAIN") and action.slot:
            state.clarify_count[action.slot] = state.clarify_count.get(action.slot, 0) + 1
        state.kind_count[action.kind] = state.kind_count.get(action.kind, 0) + 1
        return action


class ManualPolicy(PolicyBase):
    name = "manual"

    def _decide(self, state: State) -> Action:
        last = state.last_action
        pending = last and last.kind in ("CLARIFY", "CHALLENGE", "PROBE", "ABSTAIN")
        neutral_retry = pending and state.last_reading_empty
        probe = next((t for t in TYPES if t in state.slots and state.slots[t].reason == "familiarity"
                      and t not in state.probed), None)
        snapshots = state.profile_history[-3:]
        converged = len(snapshots) == 3 and all(
            t in a and t in b and abs(a[t] - b[t]) <= .1 + 1e-12
            for a, b in zip(snapshots, snapshots[1:]) for t in TYPES)
        if state.turn >= 5 and len(state.slots) == 6 and converged and not neutral_retry and (
                self.no_probing or (not state.flags and probe is None)):
            return Action("RECOMMEND", why="all six interests filled and stable over two updates")
        if neutral_retry:
            retry = last.payload.get("retry", "CLARIFY") if last.kind == "ABSTAIN" else last.kind
            payload = {**last.payload, "retry": retry, "previous_question": state.last_reply}
            if not (self.no_probing and retry in ("CHALLENGE", "PROBE")):
                return Action("ABSTAIN", last.slot, "no stance; rephrase the same question", payload)
        if not self.no_probing:
            if state.flags:
                flag = state.flags[0]
                return Action("CHALLENGE", flag.slot, "opposite signs confirmed by NLI",
                              {"old_src": flag.old.source, "old_turn": flag.old.turn, "new_src": flag.new.source})
            if probe:
                state.probed.add(probe)
                return Action("PROBE", probe, "familiarity-based reason, not yet probed",
                              {"new_src": state.slots[probe].source})
        unfilled = [t for t in TYPES if t not in state.slots]
        if unfilled:
            pool = state.ranking[:100]
            slot = max(unfilled, key=lambda t: pvariance([o.occupation_z[t] for o in pool]) if pool else 0)
            why = "unfilled interest with largest top-100 occupation variance"
        else:
            slot = min(TYPES, key=lambda t: state.slots[t].confidence)
            why = "refine least-confident interest while waiting for convergence or minimum turns"
        return Action("CLARIFY", slot, why)
