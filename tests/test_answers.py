"""Reading answers to bot questions (pipeline/answers.py): the answer speaks for itself first, and short
CHALLENGE / PROBE answers are resolved by rule."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import fakes  # noqa: F401; register test-only perceptions
from contracts import DATA
from pipeline.answers import read_answer, short_answer
from pipeline.interfaces import Action, Flag, HistoryEntry, Reading, SlotReading, SlotState, State
from pipeline.system import System
from test_dialogue_integration import reading

TOOLS_CHALLENGE = ('Earlier you said "I love repairing machines and working with tools.", but just now '
                   '"I hate repairing machines and working with tools.". Which is closer?')


class LeakyExtractor:
    """Reads the answer correctly alone, but copies the question's topic (R) when the question is attached,
    like the trained model in scenario 11."""
    name = "leaky"

    def __init__(self):
        self.calls = []

    def read(self, text, context=None, expected_slot=None):
        self.calls.append(context)
        slots = []
        if "research" in text:
            slots.append(SlotReading("I", 1, 1.0, .99, "enjoyment", text))
        if context and expected_slot:
            slots.append(SlotReading(expected_slot, 1, .55, .9, "enjoyment", text))   # the leak
        return Reading(text, slots)


def challenge_state():
    old = HistoryEntry(1.0, .9, "enjoyment", "I love repairing machines and working with tools.", 1, .9)
    new = HistoryEntry(-1.0, .8, "enjoyment", "I hate repairing machines and working with tools.", 2, .9)
    state = State(turn=2, last_reply=TOOLS_CHALLENGE, flags=[Flag("R", old, new)])
    state.slots["R"] = SlotState(1.0, .9, "enjoyment", old.source, 1, [old, new], .9, .9)
    return state


class AnswerSpeaksForItselfFirst(unittest.TestCase):
    def test_full_sentence_is_read_alone_so_the_question_cannot_leak(self):
        ex = LeakyExtractor()
        r = read_answer(ex, challenge_state(), Action("CHALLENGE", "R"), "I enjoy analysing scientific research data.")
        self.assertEqual([(s.slot, s.sign) for s in r.slots], [("I", 1)])    # no fake R
        self.assertEqual(ex.calls, [None])                                    # the question was never attached

    def test_answer_with_no_stance_alone_uses_the_question(self):
        ex = LeakyExtractor()
        r = read_answer(ex, State(last_reply="Would you rather fix machines or write reports?"),
                        Action("CLARIFY", "R"), "Either is fine, I don't mind.")
        self.assertEqual([s.slot for s in r.slots], ["R"])
        self.assertEqual(ex.calls, [None, "Would you rather fix machines or write reports?"])

    def test_first_message_is_never_read_with_a_question(self):
        ex = LeakyExtractor()
        read_answer(ex, State(last_reply="Hi! Tell me a bit about yourself."), None, "I enjoy research")
        self.assertEqual(ex.calls, [None])


class ShortChallengeAnswers(unittest.TestCase):
    def pick(self, text):
        return short_answer("CHALLENGE", text, challenge_state(), "R")

    def test_first_keeps_the_earlier_statement(self):
        for text in ("the first one", "First", "the earlier one", "what I said before"):
            with self.subTest(text=text):
                (r,) = self.pick(text).slots
                self.assertEqual((r.slot, r.sign, r.source), ("R", 1, "I love repairing machines and working with tools."))

    def test_second_keeps_the_newer_statement(self):
        for text in ("the second one", "the latest", "what I just said", "the new one"):
            with self.subTest(text=text):
                (r,) = self.pick(text).slots
                self.assertEqual((r.slot, r.sign), ("R", -1))

    def test_ambiguous_long_or_unheld_answers_are_read_normally(self):
        self.assertIsNone(self.pick("not the first, the second"))
        self.assertIsNone(self.pick("honestly I think I prefer working on machines most days"))
        self.assertIsNone(short_answer("CHALLENGE", "the first one", State(), "R"))   # no contradiction held


class ShortProbeAnswers(unittest.TestCase):
    def answer(self, text):
        state = State()
        state.slots["E"] = SlotState(.55, .9, "familiarity", "My dad runs a sales company.", 1, [], .9, .9)
        return short_answer("PROBE", text, state, "E")

    def test_enjoyment_keeps_the_stance_as_enjoyment(self):
        for text in ("I enjoy it", "yes I like it", "No, I genuinely enjoy it", "I'd love that"):
            with self.subTest(text=text):
                (r,) = self.answer(text).slots
                self.assertEqual((r.sign, r.strength, r.reason), (1, .55, "enjoyment"))

    def test_familiarity_only_means_no_real_interest(self):
        for text in ("just what I know", "mainly experience", "I'm just used to it", "I don't enjoy it"):
            with self.subTest(text=text):
                (r,) = self.answer(text).slots
                self.assertEqual((r.value, r.reason), (0, "familiarity"))

    def test_mixed_or_unclear_answers_are_read_normally(self):
        for text in ("I like it but it's mostly habit", "hmm", "maybe"):
            with self.subTest(text=text):
                self.assertIsNone(self.answer(text))


class EndToEnd(unittest.TestCase):
    def setUp(self):
        config = json.loads((Path(__file__).parent / "test_config.json").read_text())
        config["components"].update(tracker="decay", policy="manual", ranker="structured")
        config["settings"]["input_mode"] = "mock_tracker"
        self.system = System.from_config(DATA, config)
        self.state = State()
        self.system.start(self.state)

    def step(self, item):
        with patch.object(self.system.extractor, "read", return_value=item):
            return self.system.step(self.state, item.text)

    def test_challenge_answered_with_the_second_one(self):
        self.step(reading("I like teaching", S=(1, .9)))
        self.assertEqual(self.step(reading("I dislike teaching", S=(-1, .8))).action, "CHALLENGE")
        result = self.step(reading("the second one"))
        self.assertNotIn(result.action, ("CHALLENGE", "ABSTAIN"))
        self.assertEqual(self.state.slots["S"].value, -1)
        self.assertFalse(self.state.flags)

    def test_probe_answered_with_just_what_i_know(self):
        self.assertEqual(self.step(reading("I studied sales", "familiarity", .9, E=(.7, .9))).action, "PROBE")
        result = self.step(reading("just what I know"))
        self.assertNotEqual(result.action, "ABSTAIN")                         # it was an answer, not "unsure"
        self.assertEqual(self.state.slots["E"].value, 0)                     # no real interest


if __name__ == "__main__":
    unittest.main()
