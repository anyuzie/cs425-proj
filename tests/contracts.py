"""Interface contract mixins. Subclass one with your own make() (see docs/INTEGRATION.md):

    class TestMyExtractor(ExtractorContract, unittest.TestCase):
        def make(self):
            return MyExtractor(...)

Only interface invariants are checked here, never accuracy.
"""

import copy
import math
from pathlib import Path

from pipeline.interfaces import (TYPES, Action, Flag, HistoryEntry, Reading, SlotReading, SlotState, State)
from recommender.data_loader import load_eligible_occupations

DATA = Path(__file__).resolve().parent.parent / "data"
KINDS = ("CLARIFY", "PROBE", "CHALLENGE", "ABSTAIN", "RECOMMEND")
REASONS = {"enjoyment", "familiarity", "neither"}
ODD_TEXTS = ["", "   ", "\n\t", "x" * 20000, "word " * 3000, "I love 🎨🤖 and 🔧!!!", "我喜欢和机器一起工作。",
             "J'adore les machines. أحب الآلات", "hmm", "I love working with machines and tools.",
             "I hate repairing machines, it is boring."]


def make_state(turn: int = 1, slots: dict | None = None) -> State:
    """slots: {letter: [(value, source, turn), ...]} oldest first; the last entry is the latest reading."""
    st = State(turn=turn)
    for t, entries in (slots or {}).items():
        hist = [HistoryEntry(v, 0.7, "enjoyment", src, tn) for v, src, tn in entries]
        last = hist[-1]
        st.slots[t] = SlotState(last.value, last.confidence, last.reason, last.source, last.turn, hist)
    return st


def flipped_state() -> State:
    return make_state(2, {"R": [(1.0, "I love machines.", 1), (-1.0, "I hate machines.", 2)]})


class ExtractorContract:
    def make(self):
        raise NotImplementedError

    def check_reading(self, text, r):
        self.assertIsInstance(r, Reading)
        self.assertEqual(r.text, text)
        seen = set()
        for s in r.slots:
            self.assertIsInstance(s, SlotReading)
            self.assertIn(s.slot, TYPES)          # letters R/I/A/S/E/C, not full type names
            self.assertNotIn(s.slot, seen)
            seen.add(s.slot)
            self.assertIn(s.sign, (-1, 1))
            self.assertTrue(0 <= s.strength <= 1 and 0 <= s.confidence <= 1)
            self.assertIn(s.reason, REASONS)
            self.assertIsInstance(s.source, str)
            self.assertTrue(s.source.strip())
            self.assertIn(s.source.strip(), text)  # source sentences must be kept verbatim

    def test_odd_inputs_never_crash(self):
        ex = self.make()
        self.assertIsInstance(ex.name, str)
        for text in ODD_TEXTS:
            for context, slot in ((None, None), ("What do you enjoy?", "C"), ("", "nonsense")):
                self.check_reading(text, ex.read(text, context, slot))

    def test_empty_text_has_no_slots(self):
        ex = self.make()
        self.assertEqual(ex.read("").slots, [])
        self.assertEqual(ex.read("   ").slots, [])

    def test_keyword_and_positional_calls(self):
        ex = self.make()
        self.check_reading("I love machines", ex.read("I love machines", context=None, expected_slot=None))
        self.check_reading("hmm", ex.read("hmm", "Do you like paperwork?", "C"))


class TrackerContract:
    def make(self):
        raise NotImplementedError

    @staticmethod
    def reading(*slots):
        return Reading("t", [SlotReading(t, sg, st, 0.7, "enjoyment", f"src {t}") for t, sg, st in slots])

    def test_empty_reading(self):
        tr = self.make()
        self.assertIsInstance(tr.name, str)
        st = State(turn=1)
        tr.update(st, Reading(""))
        self.assertTrue(st.last_reading_empty)
        self.assertEqual(st.slots, {})

    def test_reading_lands_in_state(self):
        tr, st = self.make(), State(turn=1)
        tr.update(st, self.reading(("R", 1, 1.0), ("C", -1, 0.55)))
        self.assertFalse(st.last_reading_empty)
        for t, sign in (("R", 1), ("C", -1)):
            s = st.slots[t]
            self.assertIn(t, TYPES)
            self.assertTrue(-1 <= s.value <= 1 and 0 <= s.confidence <= 1)
            self.assertEqual(s.value > 0, sign > 0)
            self.assertEqual(s.turn, 1)
            self.assertEqual(s.history[-1].source, f"src {t}")  # source sentence kept for the rule/quotes
            self.assertEqual(s.history[-1].turn, 1)

    def test_history_grows_per_reading(self):
        tr, st = self.make(), State(turn=1)
        tr.update(st, self.reading(("R", 1, 1.0)))
        st.turn = 2
        tr.update(st, self.reading(("R", -1, 1.0)))
        self.assertEqual(len(st.slots["R"].history), 2)
        self.assertEqual(st.slots["R"].history[0].source, "src R")
        self.assertEqual(st.slots["R"].history[-1].turn, 2)

    def test_probe_answer_marks_slot_resolved(self):
        tr, st = self.make(), State(turn=2)
        st.last_action = Action("PROBE", "R")
        tr.update(st, self.reading(("R", -1, 1.0)))
        self.assertIn("R", st.resolved)
        st2 = State(turn=2)
        st2.last_action = Action("CLARIFY", "R")
        tr.update(st2, self.reading(("R", -1, 1.0)))
        self.assertNotIn("R", st2.resolved)


class RuleContract:
    def make(self):
        raise NotImplementedError

    def test_empty_state(self):
        self.assertEqual(self.make().candidates(State()), [])

    def test_flip_is_flagged_and_deterministic(self):
        rule = self.make()
        self.assertIsInstance(rule.name, str)
        a, b = rule.candidates(flipped_state()), rule.candidates(flipped_state())
        self.assertEqual(a, b)
        self.assertEqual([f.slot for f in a], ["R"])
        for f in a:
            self.assertIsInstance(f, Flag)
            self.assertIn(f.slot, TYPES)
            self.assertTrue(f.old.source and f.new.source)

    def test_no_flip_no_flag(self):
        st = make_state(2, {"R": [(1.0, "a", 1), (0.55, "b", 2)]})
        self.assertEqual(self.make().candidates(st), [])

    def test_challenged_and_resolved_slots_skipped(self):
        for field in ("challenged", "resolved"):
            st = flipped_state()
            getattr(st, field).add("R")
            self.assertEqual(self.make().candidates(st), [], field)


class ConfirmerContract:
    def make(self):
        raise NotImplementedError

    def test_score_range_and_odd_inputs(self):
        c = self.make()
        self.assertIsInstance(c.name, str)
        for a in ODD_TEXTS:
            for b in ODD_TEXTS[:6] + ODD_TEXTS[-2:]:
                s = c.score(a, b)
                self.assertIsInstance(s, float)
                self.assertTrue(math.isfinite(s) and 0.0 <= s <= 1.0, (a[:20], b[:20], s))

    def test_deterministic(self):
        c = self.make()
        a, b = "I love working with machines.", "I hate repairing machines."
        self.assertEqual(c.score(a, b), c.score(a, b))


class RankerContract:
    def make(self):
        raise NotImplementedError

    def eligible_codes(self) -> set[str]:
        return {o.code for o in load_eligible_occupations(DATA / "occupations.csv")}

    ZERO = dict.fromkeys(TYPES, 0.0)

    def check_response(self, resp, top_k):
        self.assertIn(resp.status, ("ranked", "no_signal"))
        if resp.status == "no_signal":
            self.assertEqual(resp.occupations, [])
            return
        occ = resp.occupations
        codes = [o.code for o in occ]
        self.assertTrue(1 <= len(occ) <= top_k)
        self.assertEqual(len(set(codes)), len(codes))
        self.assertLessEqual(set(codes), self.eligible_codes())
        self.assertEqual([o.rank for o in occ], list(range(1, len(occ) + 1)))
        for o in occ:
            self.assertTrue(o.title)
            for e in o.evidence:
                self.assertEqual(e.occupation_code, o.code)
                self.assertTrue(isinstance(e.text, str) and e.text)

    def test_ranked_shapes(self):
        r = self.make()
        self.assertIsInstance(r.name, str)
        self.assertEqual(r.eligible_count, len(self.eligible_codes()))
        interests = {**self.ZERO, "I": 0.55, "S": 1.0, "C": -0.55}
        for k in (1, 5, 50):
            self.check_response(r.rank(interests, {"I", "S", "C"}, ["I enjoy helping people."], k), k)

    def test_no_signal(self):
        r = self.make()
        resp = r.rank(self.ZERO, set(), [], 5)
        self.check_response(resp, 5)

    def test_odd_sentences_never_crash(self):
        r = self.make()
        for text in ODD_TEXTS:
            self.check_response(r.rank({**self.ZERO, "R": 1.0}, {"R"}, [text], 5), 5)
        self.check_response(r.rank({**self.ZERO, "R": 1.0}, {"R"}, [], 5), 5)

    def test_deterministic(self):
        r = self.make()
        args = ({**self.ZERO, "A": 1.0, "E": -0.55}, {"A", "E"}, ["I like drawing", "I dislike selling"], 5)
        a, b = r.rank(*args), r.rank(*args)
        self.assertEqual([(o.code, o.rank) for o in a.occupations], [(o.code, o.rank) for o in b.occupations])


# Payload keys each action needs, per pipeline/templates.py.
PAYLOAD_KEYS = {"CLARIFY": {"slot", "index", "clarify_index"}, "ABSTAIN": {"slot", "index", "clarify_index"},
                "CHALLENGE": {"old_src", "old_turn", "new_src", "index"}, "PROBE": {"new_src", "index"},
                "RECOMMEND": {"index"}}


class PolicyContract:
    def make(self):
        raise NotImplementedError

    def states(self):
        out = [make_state(1), make_state(3, {"C": [(1.0, "I like spreadsheets.", 3)]})]
        fam = make_state(2, {"E": [(0.55, "My dad runs sales.", 2)]})
        fam.slots["E"].reason = "familiarity"
        out.append(fam)
        flip = flipped_state()
        flip.flags = [Flag("R", flip.slots["R"].history[0], flip.slots["R"].history[1], 1.0)]
        out.append(flip)
        quiet = make_state(2, {"C": [(1.0, "x", 1)]})
        quiet.last_action, quiet.last_reading_empty = Action("CLARIFY", "E"), True
        out.append(quiet)
        out.append(make_state(10))
        full = make_state(6, {t: [(0.55, "s", 1)] for t in TYPES})
        out.append(full)
        return out

    def test_valid_action_and_payload(self):
        p = self.make()
        self.assertIsInstance(p.name, str)
        for st in self.states():
            a = p.decide(st)
            self.assertIn(a.kind, KINDS)
            self.assertTrue(PAYLOAD_KEYS[a.kind] <= set(a.payload), (a.kind, a.payload))
            if a.kind in ("CLARIFY", "ABSTAIN"):
                self.assertIn(a.slot, TYPES)
                self.assertEqual(a.payload["slot"], a.slot)
            if a.kind in ("CHALLENGE", "PROBE"):
                self.assertIn(a.slot, TYPES)
            for k in ("index", "clarify_index"):
                if k in a.payload:
                    self.assertIsInstance(a.payload[k], int)

    def test_deterministic(self):
        for st in self.states():
            a, b = self.make().decide(copy.deepcopy(st)), self.make().decide(copy.deepcopy(st))
            self.assertEqual((a.kind, a.slot, a.payload), (b.kind, b.slot, b.payload))

    def test_recommends_at_max_turns(self):
        self.assertEqual(self.make().decide(make_state(10, {"C": [(1.0, "s", 1)]})).kind, "RECOMMEND")


class FillerContract:
    def make(self):
        raise NotImplementedError

    def test_text_for_every_kind(self):
        f = self.make()
        self.assertIsInstance(f.name, str)
        payloads = {"CLARIFY": {"slot": "C"}, "ABSTAIN": {"slot": "E"},
                    "CHALLENGE": {"old_src": "I love X.", "old_turn": 1, "new_src": "I hate X."},
                    "PROBE": {"new_src": "My dad does it."}, "RECOMMEND": {}, "OPENING": {}}
        for kind, base in payloads.items():
            for index in (0, 1, 2, 7, 100):
                payload = {**base, "index": index, "clarify_index": index}
                for action in (kind, Action(kind, base.get("slot"), "", payload)):
                    text = f.fill(action, payload)
                    self.assertIsInstance(text, str)
                    self.assertTrue(text.strip(), (kind, index))
                    self.assertEqual(text, f.fill(action, payload))

    def test_every_slot_has_a_question(self):
        f = self.make()
        for t in TYPES:
            self.assertTrue(f.fill("CLARIFY", {"slot": t, "index": 0, "clarify_index": 0}).strip())
