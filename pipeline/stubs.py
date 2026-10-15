"""Stand-in components. Replace with the real extractor/tracker/NLI/policy behind the same interfaces."""

import csv
import json
import re
from pathlib import Path

from recommender.schemas import RankingResponse, TaskEvidence
from recommender.structured import StructuredRanker

from .interfaces import (MAX_TURNS, TYPES, Action, Flag, HistoryEntry, Reading, SlotReading,
                         SlotState, State)

POS = ["love", "enjoy", "like", "fun", "exciting", "interesting", "satisfying", "rewarding", "cool", "happy",
       "shiok", "great", "amazing", "favourite", "favorite", "passionate", "keen", "into", "prefer",
       "would rather", "sounds good", "fulfilling"]
NEG = ["hate", "dislike", "boring", "dull", "tedious", "sian", "nightmare", "torture", "awful", "drain",
       "can't stand", "meh", "not my thing", "not for me", "stressful", "uncomfortable", "rather not", "pass",
       "miserable"]
# Extra cues used only when answering a CLARIFY question about a known slot.
ANSWER_POS = ["definitely", "for sure", "absolutely", "yes", "yeah", "yep", "first one", "first", "sure"]
ANSWER_NEG = ["no way", "nope", "not really", "absolutely not", "definitely not", "second one", "second", "no"]
FAMILIARITY = ["that's what i know", "that is what i know", "my dad", "my mum", "my mom", "my parents",
               "always been around", "grew up", "internship"]
FAMILIARITY_RE = re.compile(r"\bdid the\b.{0,30}\btrack\b")
CONFIDENCE = 0.7
STRENGTHS = {"strong": 1.0, "mild": 0.55, "dontmind": 0.2}
FILL_ORDER = ("C", "E", "A", "R", "S", "I")
SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+|\n")


def words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def _has(low: str, phrases) -> bool:
    padded = f" {low} "
    return any(f" {p} " in padded for p in phrases)


def _count(low: str, phrases) -> int:
    return sum(low.count(p) for p in phrases)


class LexiconExtractor:
    name = "lexicon-stub"

    def __init__(self, lexicon_path: Path) -> None:
        lex = json.loads(Path(lexicon_path).read_text(encoding="utf-8"))
        self._terms = {t: {self._stem(x["term"]) for x in lex["types"][t][:60]} for t in TYPES}
        inten = lex.get("intensity", {})
        self._inten = {k: [p.lower() for p in inten.get(k, [])] for k in
                       ("strong_positive", "mild_positive", "strong_negative", "mild_negative", "dontmind")}
        self._negators = set(inten.get("negators", []))
        self._familiarity = FAMILIARITY + [p.lower() for p in inten.get("familiarity", [])]

    @staticmethod
    def _stem(x: str) -> str:
        return x[:-1] if x.endswith("s") else x

    def _is_familiar(self, text: str) -> bool:
        low = " ".join(words(text))
        return _has(low, self._familiarity) or bool(FAMILIARITY_RE.search(low))

    def _strength(self, low: str, sign: int) -> float:
        i = self._inten
        if _has(low, i["dontmind"]):
            return STRENGTHS["dontmind"]
        strong, mild = (i["strong_positive"], i["mild_positive"]) if sign > 0 else (i["strong_negative"], i["mild_negative"])
        if _has(low, strong):
            return STRENGTHS["strong"]
        if _has(low, mild):
            return STRENGTHS["mild"]
        return STRENGTHS["mild"]

    def _direction(self, w: list[str], low: str) -> int:
        """Port of lexicon_baseline.read() direction rule: +1, -1 or 0."""
        pos, neg = _count(low, POS), _count(low, NEG)
        for i, x in enumerate(w):  # "don't enjoy" flips a positive cue
            if x in self._negators and any(" ".join(w[i + 1:i + 3]).startswith(c) for c in POS):
                pos -= 1
                neg += 1
        if pos == neg == 0:
            return 0
        return 1 if pos > neg else -1

    def _stances(self, text: str) -> dict[str, tuple[int, float]]:
        w = words(text)
        low = " ".join(w)
        stems = {self._stem(x) for x in w}
        out = {}
        for t in TYPES:
            if not stems & self._terms[t]:
                continue
            sign = self._direction(w, low)
            if sign:
                out[t] = (sign, self._strength(low, sign))
        return out

    def _answer_stance(self, text: str) -> tuple[int, float] | None:
        w = words(text)
        low = " ".join(w)
        if _has(low, ["not sure", "idk", "dunno", "depends", "no idea", "hard to say"]):
            return None
        i = self._inten
        neg = (_has(low, NEG + ANSWER_NEG + i["strong_negative"] + i["mild_negative"])
               or self._direction(w, low) < 0)
        if neg:
            return -1, self._strength(low, -1)
        if _has(low, POS + ANSWER_POS + i["strong_positive"] + i["mild_positive"]) or self._direction(w, low) > 0:
            return 1, self._strength(low, 1)
        if _has(low, i["dontmind"]):
            return 1, STRENGTHS["dontmind"]
        return None

    def read(self, text: str, context: str | None = None, expected_slot: str | None = None) -> Reading:
        found: dict[str, tuple[int, float, str]] = {}
        sentences = [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]
        for sent in sentences:
            for t, (sign, strength) in self._stances(sent).items():
                found[t] = (sign, strength, sent)
        if not found:  # cue and term may sit in different sentences
            for t, (sign, strength) in self._stances(text).items():
                found[t] = (sign, strength, text.strip())
        if not found:  # "my dad runs a sales company" states a lean without a like/dislike cue
            for sent in sentences:
                if self._is_familiar(sent):
                    stems = {self._stem(x) for x in words(sent)}
                    for t in TYPES:
                        if stems & self._terms[t]:
                            found[t] = (1, STRENGTHS["mild"], sent)
        if not found and expected_slot in TYPES:
            stance = self._answer_stance(text)
            if stance:
                found[expected_slot] = (stance[0], stance[1], text.strip())
        slots = [SlotReading(t, sign, strength, CONFIDENCE,
                             "familiarity" if self._is_familiar(src) else "enjoyment", src)
                 for t, (sign, strength, src) in found.items()]
        return Reading(text, slots)


class SimpleTracker:
    name = "simple-stub"

    def update(self, state: State, reading: Reading) -> None:
        state.last_reading_empty = not reading.slots
        last = state.last_action
        if last and last.kind == "PROBE" and last.slot:
            state.resolved.add(last.slot)  # the probe answer resolves the slot
        for r in reading.slots:
            entry = HistoryEntry(r.value, r.confidence, r.reason, r.source, state.turn)
            old = state.slots.get(r.slot)
            history = (old.history if old else []) + [entry]
            state.slots[r.slot] = SlotState(r.value, r.confidence, r.reason, r.source, state.turn, history)


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


class PassthroughConfirmer:
    name = "passthrough-stub"

    def score(self, old_src: str, new_src: str) -> float:
        return 1.0


class RulePolicy:
    name = "rule-stub"

    def __init__(self, max_turns: int = MAX_TURNS) -> None:
        self.max_turns = max_turns

    def _clarify(self, state: State, prefix: str = "") -> Action:
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
            if s and s.reason == "familiarity" and t not in state.probed:
                state.probed.add(t)
                return Action("PROBE", t, f"{t} latest reading reason is familiarity", {"new_src": s.source})
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

    def decide(self, state: State) -> Action:
        action = self._decide(state)
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


STOPWORDS = frozenset("""a an and are as at be been but by can do does for from had has have he her his i if in into is it
its me my no not of on or our so than that the their them then there these they this those to up was we were what when
which who will with would you your am about also just very really like love enjoy hate lot get got""".split())


def _tokens(text: str) -> set[str]:
    return {w[:-1] if w.endswith("s") else w for w in re.findall(r"[a-z]+", text.lower()) if w not in STOPWORDS}


class StructuredKeywordRanker:
    """StructuredRanker plus top task statements by word overlap with what the user said."""

    name = "structured+keyword-stub"

    def __init__(self, occupations_path: Path, tasks_path: Path, n_evidence: int = 3) -> None:
        self._ranker = StructuredRanker(occupations_path)
        self.eligible_count = self._ranker.eligible_count
        self._n = n_evidence
        eligible = {o.code for o in self._ranker._occupations}
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
