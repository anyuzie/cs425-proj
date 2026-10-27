"""Confidence-weighted accepted preferences, raw evidence, and targeted resolutions."""

from math import isfinite

from .interfaces import TYPES, Flag, HistoryEntry, Reading, SlotState, State


class DecayTracker:
    """decay = how much one turn of age shrinks older evidence's weight: 1.0 = off (all turns equal), 0.9 = on."""
    name = "decay"

    def __init__(self, decay: float = 1.0) -> None:
        if isinstance(decay, bool) or not isinstance(decay, (int, float)) or not 0 < decay <= 1:
            raise ValueError("decay must be a number in (0, 1]")
        self.decay = float(decay)

    def update(self, state: State, reading: Reading) -> None:
        for r in reading.slots:
            if r.slot not in TYPES or r.sign not in (-1, 1) or r.reason not in ("enjoyment", "familiarity", "neither"):
                raise ValueError("Invalid extractor slot, sign or reason")
            for value in (r.strength, r.confidence, r.reason_confidence):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or not 0 <= value <= 1:
                    raise ValueError("Extractor strength and confidences must be finite numbers in [0, 1]")
            if not isinstance(r.source, str) or not r.source.strip():
                raise ValueError("Extractor source must be non-empty")
        state.last_reading_empty = not reading.slots
        state.conflict_candidates = []
        last = state.last_action
        kind = last.payload.get("retry", "CLARIFY") if last and last.kind == "ABSTAIN" else last.kind if last else None
        target = last.slot if last and kind in ("CHALLENGE", "PROBE") else None
        for r in reading.slots:
            entry = HistoryEntry(r.value, r.confidence, r.reason, r.source, state.turn, r.reason_confidence)
            old = state.slots.get(r.slot)
            history = (old.history if old else []) + [entry]
            state.challenged.discard(r.slot)
            state.resolved.discard(r.slot)
            if r.slot == target:
                state.resolved.add(r.slot)
                state.flags = [f for f in state.flags if f.slot != r.slot]
            elif old and old.value * r.value < 0 and not state.merge_conflicts:   # control arm merges instead
                previous = HistoryEntry(old.value, old.confidence, old.reason, old.source, old.turn, old.reason_confidence)
                state.conflict_candidates.append(Flag(r.slot, previous, entry))
                old.history = history
                continue
            value, confidence, weight = r.value, r.confidence, r.confidence
            if old and r.slot != target:
                old_weight = old.weight if old.weight is not None else old.confidence
                old_weight *= self.decay ** max(0, state.turn - old.turn)   # older evidence counts for less
                weight += old_weight
                if weight:
                    value = (old.value * old_weight + r.value * r.confidence) / weight
                    confidence = (old.confidence * old_weight + r.confidence ** 2) / weight
            state.slots[r.slot] = SlotState(value, confidence, r.reason, r.source,
                                            state.turn, history, r.reason_confidence, weight)
