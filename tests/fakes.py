"""Test-only components. Importing this module registers them in pipeline.registry (the app never knows them):
test-scripted, test-simple-tracker, test-passthrough, test-keyword-ranker, test-simple-policy.
They are also minimal working examples of each interface."""

import csv
import re
from pathlib import Path

from pipeline import registry
from pipeline.interfaces import (MAX_TURNS, TYPES, Action, HistoryEntry, Reading, SlotReading, SlotState,
                                 State)
from pipeline.policy import PROBE_REASON_CONF, PolicyBase
from recommender.schemas import RankingResponse, TaskEvidence
from recommender.structured import StructuredRanker

FILL_ORDER = ("C", "E", "A", "R", "S", "I")

# known sentence -> (slot, sign, strength, reason, reason_confidence); confidence is always 0.9
SCRIPT = {
    "I love working with machines and tools.": ("R", 1, 1.0, "enjoyment", 0.95),
    "I hate repairing machines, it is boring.": ("R", -1, 1.0, "enjoyment", 0.95),
    "I like machines but that's what I know, my dad has a workshop.": ("R", 1, 0.55, "familiarity", 0.9),
    "I love analysing data.": ("I", 1, 1.0, "enjoyment", 0.95),
    "My dad runs a sales company so that's what I know.": ("E", 1, 0.55, "familiarity", 0.9),
    "Honestly I grew up around it, I don't enjoy selling.": ("E", -1, 0.55, "enjoyment", 0.8),
    "My mum does accounts so I know it.": ("C", 1, 0.55, "familiarity", 0.60),   # exactly at the PROBE limit
}
ANSWERS = (("hmm", 0), ("not sure", 0), ("definitely", 1), ("first", 1), ("no way", -1), ("second", -1))


class ScriptedExtractor:
    name = "test-scripted"

    def read(self, text, context=None, expected_slot=None):
        slots = [SlotReading(s, sg, st, 0.9, why, sent, rc)
                 for sent, (s, sg, st, why, rc) in SCRIPT.items() if sent in text]
        low = text.lower()
        if not slots and expected_slot in TYPES:  # a reply to a CLARIFY question about expected_slot
            sign = next((sg for key, sg in ANSWERS if key in low), 0)
            if sign:
                slots = [SlotReading(expected_slot, sign, 1.0, 0.9, "enjoyment", text.strip())]
        return Reading(text, slots)


class SimpleTracker:
    name = "test-simple-tracker"

    def update(self, state: State, reading: Reading) -> None:
        state.last_reading_empty = not reading.slots
        last = state.last_action
        if last and last.kind == "PROBE" and last.slot:
            state.resolved.add(last.slot)  # the probe answer resolves the slot
        for r in reading.slots:
            entry = HistoryEntry(r.value, r.confidence, r.reason, r.source, state.turn, r.reason_confidence)
            old = state.slots.get(r.slot)
            history = (old.history if old else []) + [entry]
            state.slots[r.slot] = SlotState(r.value, r.confidence, r.reason, r.source, state.turn, history,
                                            r.reason_confidence)


class PassthroughConfirmer:
    name = "test-passthrough"

    def score(self, old_src: str, new_src: str) -> float:
        return 1.0


class SimplePolicy(PolicyBase):
    """Minimal version of all five rungs: fixed question order, no re-challenge, no convergence check."""

    name = "test-simple-policy"

    def _clarify(self, state: State) -> Action:
        free = [t for t in FILL_ORDER if t not in state.slots and t not in state.settled]
        if free:
            slot, why = free[0], f"{free[0]} unfilled"
        else:  # everything filled before turn 5: refine the weakest slot
            slot = min(FILL_ORDER, key=lambda t: abs(state.slots[t].value) if t in state.slots else 9)
            why = f"all slots filled or settled before turn 5; refining weakest slot {slot}"
        return Action("CLARIFY", slot, why, {"index": state.clarify_count.get(slot, 0)})

    def _decide(self, state: State) -> Action:
        has_signal = any(s.value != 0 for s in state.slots.values())
        # Hard stop so the conversation always ends at max_turns.
        if state.turn >= self.max_turns:
            return Action("RECOMMEND", None, f"turn {state.turn} reached max {self.max_turns}")
        # 1. CHALLENGE
        if state.flags:
            f = state.flags[0]
            state.challenged.add(f.slot)
            return Action("CHALLENGE", f.slot, f"{f.slot} flipped sign with confidence > 0.4",
                          {"old_src": f.old.source, "old_turn": f.old.turn, "new_src": f.new.source})
        # 2. PROBE
        for t in FILL_ORDER:
            s = state.slots.get(t)
            if s and s.reason == "familiarity" and s.reason_confidence > PROBE_REASON_CONF and t not in state.probed:
                state.probed.add(t)
                return Action("PROBE", t, f"{t} latest reading reason is familiarity (conf {s.reason_confidence:.2f})", {"new_src": s.source})
        # 3. ABSTAIN
        last = state.last_action
        if last and last.kind in ("CLARIFY", "PROBE", "CHALLENGE", "ABSTAIN") and state.last_reading_empty:
            if last.slot:
                state.strikes[last.slot] = state.strikes.get(last.slot, 0) + 1
                if state.strikes[last.slot] >= 2:
                    state.settled.add(last.slot)
            nxt = self._clarify(state)
            return Action("ABSTAIN", nxt.slot, f"no stance on {last.slot or 'any type'}; strikes "
                          f"{state.strikes.get(last.slot, 0) if last.slot else 0}", nxt.payload)
        # 4. RECOMMEND
        if has_signal and state.turn >= 5 and all(t in state.slots or t in state.settled for t in TYPES):
            return Action("RECOMMEND", None, "turn >= 5 and every slot filled or settled")
        # 5. CLARIFY
        return self._clarify(state)


STOPWORDS = frozenset("""a an and are as at be been but by can do does for from had has have he her his i if in into is it
its me my no not of on or our so than that the their them then there these they this those to up was we were what when
which who will with would you your am about also just very really like love enjoy hate lot get got""".split())


def _tokens(text: str) -> set[str]:
    return {w[:-1] if w.endswith("s") else w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOPWORDS}


class StructuredKeywordRanker:
    """StructuredRanker plus top task statements by word overlap with what the user said."""

    name = "test-keyword-ranker"

    def __init__(self, occupations_path: Path, tasks_path: Path, n_evidence: int = 3) -> None:
        self._ranker = StructuredRanker(occupations_path)
        self.eligible_count = self._ranker.eligible_count
        self._n = n_evidence
        eligible = self._ranker.eligible_codes
        self._tasks: dict[str, list[tuple[str, str, set[str]]]] = {}
        with Path(tasks_path).open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["code"] in eligible:
                    self._tasks.setdefault(row["code"], []).append((row["Task ID"], row["Task"], _tokens(row["Task"])))

    def _evidence(self, code: str, user: set[str]) -> list[TaskEvidence]:
        scored = [(len(toks & user), -i, tid, task) for i, (tid, task, toks) in enumerate(self._tasks.get(code, []))]
        scored = sorted((x for x in scored if x[0] > 0), reverse=True)[:self._n]
        return [TaskEvidence(tid, code, task, float(n)) for n, _, tid, task in scored]

    def rank(self, interests: dict, observed_types: set, user_sentences: list[str], top_k: int = 100) -> RankingResponse:
        resp = self._ranker.rank(interests, observed_types, user_sentences, top_k)
        user = _tokens(" ".join(user_sentences))
        for o in resp.occupations:
            o.evidence = self._evidence(o.code, user)
        return resp


R = registry.REGISTRY
R["extractor"]["test-scripted"] = lambda cfg, d: ScriptedExtractor()
R["tracker"]["test-simple-tracker"] = lambda cfg, d: SimpleTracker()
R["confirmer"]["test-passthrough"] = lambda cfg, d: PassthroughConfirmer()
R["ranker"]["test-keyword-ranker"] = lambda cfg, d: StructuredKeywordRanker(d / "occupations.csv", d / "tasks.csv")
R["policy"]["test-simple-policy"] = lambda cfg, d: SimplePolicy(cfg.get("settings", {}).get("max_turns", MAX_TURNS))
