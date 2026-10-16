"""STEP 6 · Dialogue policy (full version of manual section 7).  STATUS: not built yet.

Switch on: configs/system.json -> "policy": "manual"
No model: plain Python, same input -> same output.

You only write ManualPolicy._decide(). PolicyBase.decide() (below) calls it, then adds the question-rotation
counters and the payload the template filler needs. tests/fakes.py SimplePolicy shows a minimal working
version of all five rungs.
"""

from .interfaces import MAX_TURNS, Action, State

PROBE_REASON_CONF = 0.60   # manual section 7: PROBE needs reason confidence above this


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
            payload["clarify_index"] = state.clarify_count.get(action.slot, 0)
        action.payload = payload
        if action.kind in ("CLARIFY", "ABSTAIN") and action.slot:
            state.clarify_count[action.slot] = state.clarify_count.get(action.slot, 0) + 1
        state.kind_count[action.kind] = state.kind_count.get(action.kind, 0) + 1
        return action


class ManualPolicy(PolicyBase):
    name = "manual"

    def __init__(self, max_turns: int = 10, no_probing: bool = False) -> None:
        raise NotImplementedError("pipeline/policy.py is not filled in yet (docs/INTEGRATION.md). "
                                  "Delete this line once the code below is written.")
        super().__init__(max_turns, no_probing)

    def _decide(self, state: State) -> Action:
        """Return one Action. Check top to bottom, first match wins:
          1 CHALLENGE  state.flags not empty           payload {"old_src", "old_turn", "new_src"}
          2 PROBE      state.slots[t].reason == "familiarity" and .reason_confidence > PROBE_REASON_CONF, not yet probed   payload {"new_src"}
          3 ABSTAIN    last action was a question and the reply had no stance; 2 strikes -> settle
          4 RECOMMEND  ready to recommend (every type filled or settled, profile converged, 5..10 turns)
          5 CLARIFY    ask the unfilled type with the highest variance
        Action(kind, slot_letter_or_None, why_text, payload). Update state.challenged / probed /
        settled / strikes yourself. state.ranking holds this turn's top 100 occupations (RankedOccupation,
        best first, each with occupation_z) for choosing which slot to ask about.
        PolicyBase.decide() already returns RECOMMEND at max_turns, so you don't need to.
        """
        # ================= PUT YOUR CODE HERE =================
        # Needed: highest-variance slot (not a fixed order), the PROBE_REASON_CONF check,
        # re-challenge / force-resolve after an ABSTAIN on a CHALLENGE,
        # a real convergence check, and skipping rungs 1-2 when self.no_probing is True.
        # ======================================================
        raise NotImplementedError
