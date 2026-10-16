"""STEP 2 · State tracker with recency decay.  STATUS: not built yet.

Replaces: SimpleTracker in pipeline/stubs.py (it just keeps the latest reading per type)
Switch on: configs/system.json -> "tracker": "decay"
No model: plain Python (manual section 5).
"""

from .interfaces import HistoryEntry, Reading, SlotState, State  # noqa: F401


class DecayTracker:
    name = "decay"

    def __init__(self) -> None:
        raise NotImplementedError("pipeline/tracker.py is not filled in yet (docs/INTEGRATION.md). "
                                  "Delete this line once the code below is written.")

    def update(self, state: State, reading: Reading) -> None:
        """Merge this turn's readings into state. Changes `state` in place, returns nothing.

        Must keep doing what SimpleTracker does (copy its code from stubs.py as a start):
          - state.last_reading_empty = not reading.slots
          - for each reading: append HistoryEntry(value, confidence, reason, source, state.turn)
            to that type's history and set state.slots[letter] = SlotState(...)
            (the source sentence MUST be kept: the quote cards and the NLI check use it)
          - if state.last_action is a PROBE on a type, add that letter to state.resolved
        """
        # ================= PUT YOUR CODE HERE =================
        # New part: weight older evidence down (recency decay) when computing the slot value,
        # merge weight = confidence. Keep value in -1..1 and confidence in 0..1.
        # ======================================================
        raise NotImplementedError
