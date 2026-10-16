"""STEP 3 · Contradiction rule (arithmetic, no model). Finds same-type readings with opposite signs."""

from .interfaces import Flag, State


class ContradictionRule:
    name = "rule"

    def candidates(self, state: State) -> list[Flag]:
        flags = []
        for slot, s in state.slots.items():
            # a flip right after a PROBE on the slot is the answer, not a contradiction
            if slot in state.challenged or slot in state.resolved or s.turn != state.turn or len(s.history) < 2:
                continue
            new = s.history[-1]
            for old in reversed(s.history[:-1]):
                if old.value * new.value < 0 and old.confidence > 0.4 and new.confidence > 0.4:
                    flags.append(Flag(slot, old, new))
                    break
        return flags
