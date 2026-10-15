"""STEP 6 · Dialogue policy (full version of manual section 7).  STATUS: not built yet.

Replaces: RulePolicy in pipeline/stubs.py
Switch on: configs/system.json -> "policy": "manual"
No model: plain Python, same input -> same output.

You only write _decide(). The question-rotation counters and the payload the template
filler needs are handled by RulePolicy.decide(), which this class inherits.
"""

from .interfaces import Action, State  # noqa: F401
from .stubs import RulePolicy


class ManualPolicy(RulePolicy):
    name = "manual"

    def __init__(self, max_turns: int = 10, no_probing: bool = False) -> None:
        raise NotImplementedError("pipeline/policy.py is not filled in yet (docs/INTEGRATION.md). "
                                  "Delete this line once the code below is written.")
        super().__init__(max_turns)
        self.no_probing = no_probing   # True = never CHALLENGE or PROBE (the evaluation's control arm)

    def _decide(self, state: State) -> Action:
        """Return one Action. Check top to bottom, first match wins:
          1 CHALLENGE  state.flags not empty           payload {"old_src", "old_turn", "new_src"}
          2 PROBE      reason "familiarity", reason confidence > 0.60, not yet probed   payload {"new_src"}
          3 ABSTAIN    last action was a question and the reply had no stance; 2 strikes -> settle
          4 RECOMMEND  ready to recommend (every type filled or settled, profile converged, 5..10 turns)
          5 CLARIFY    ask the unfilled type with the highest variance
        Action(kind, slot_letter_or_None, why_text, payload). Update state.challenged / probed /
        settled / strikes yourself. Start from RulePolicy._decide in stubs.py; it already does a
        simple version of all five.
        """
        # ================= PUT YOUR CODE HERE =================
        # differences from the stand-in: highest-variance slot (not fixed order), the 0.60 reason
        # confidence check, re-challenge / force-resolve after an ABSTAIN on a CHALLENGE,
        # a real convergence check, and skipping rungs 1-2 when self.no_probing is True.
        # ======================================================
        raise NotImplementedError
