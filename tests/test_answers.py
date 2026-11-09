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
        for text in ("just what I know", "mainly experience", "I'm just used to it", "It is something I already know",
                     "I know it well", "I have a background in it", "I'm comfortable with it"):
            with self.subTest(text=text):
                (r,) = self.answer(text).slots
                self.assertEqual((r.value, r.reason), (0, "familiarity"))

    def test_not_enjoying_it_is_a_mild_dislike(self):
        for text in ("I don't enjoy it", "I don't really like it", "never liked it"):
            with self.subTest(text=text):
                (r,) = self.answer(text).slots
                self.assertEqual((r.sign, r.strength), (-1, .55))

    def test_mixed_or_unclear_answers_are_read_normally(self):
        for text in ("hmm", "maybe"):
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


class SecondReviewRegressions(unittest.TestCase):
    def probe(self, text):
        state = State()
        state.slots["C"] = SlotState(.6, .9, "familiarity", "x", 1, [], .9, .9)
        r = short_answer("PROBE", text, state, "C")
        return None if r is None else (r.slots[0].sign, r.slots[0].strength, r.slots[0].reason)

    def test_probe_negation_anywhere_in_the_clause_means_not_enjoying(self):
        for text in ("I wouldn't say I love it", "I can't say I enjoy it", "Not something I enjoy", "I dislike it"):
            with self.subTest(text=text):
                self.assertEqual(self.probe(text), (-1, .55, "enjoyment"))
        self.assertEqual(self.probe("I hate it"), (-1, 1.0, "enjoyment"))

    def test_probe_last_clause_wins(self):
        self.assertEqual(self.probe("I don't love it but I like it"), (1, .6, "enjoyment"))
        self.assertEqual(self.probe("I like it but it's mostly habit"), (1, 0.0, "familiarity"))

    def test_probe_past_or_hypothetical_is_not_judged(self):
        for text in ("I used to like it", "I'd like to try something else", "Nobody asked if I like it"):
            with self.subTest(text=text):
                self.assertIsNone(self.probe(text))
        self.assertEqual(self.probe("I'm just used to it"), (1, 0.0, "familiarity"))

    def test_knowing_plus_feeling_is_judged_on_the_feeling(self):
        self.assertEqual(self.probe("I know it well and I enjoy it")[2], "enjoyment")
        self.assertEqual(self.probe("I enjoy it, but I already know it"), (1, 0.0, "familiarity"))

    def test_more_challenge_phrasings(self):
        for text, sign in (("The latter.", -1), ("The former", 1), ("Earlier", 1), ("the earlier statement", 1),
                           ("the newer statement", -1), ("the newest", -1)):
            with self.subTest(text=text):
                (r,) = short_answer("CHALLENGE", text, challenge_state(), "R").slots
                self.assertEqual(r.sign, sign)
        self.assertIsNone(short_answer("CHALLENGE", "What I said last time", challenge_state(), "R"))

    def test_long_clarify_reply_never_borrows_the_question(self):
        ex = LeakyExtractor()
        text = "No, I'd rather spend my whole day doing research with my friends at the lab"
        r = read_answer(ex, State(last_reply="Do you like working with tools?"), Action("CLARIFY", "R"), text)
        self.assertEqual([s.slot for s in r.slots], ["I"])          # no leaked R

    def test_short_reply_read_with_question_becomes_a_statement(self):
        # "Yes" alone means nothing to the contradiction check or in a quote; restate it with its question.
        question = "Would you enjoy following clear procedures as part of your job?"
        r = read_answer(LeakyExtractor(), State(last_reply=question), Action("CLARIFY", "C"), "Yes")
        self.assertEqual([(s.slot, s.source) for s in r.slots],
                         [("C", "I like following clear procedures.")])
        self.assertEqual(r.text, "Yes")                               # the user's own text is unchanged

    def test_statement_forms(self):
        from pipeline.answers import as_statement
        self.assertEqual(as_statement("Would a job involving checking database entries appeal to you?", "Not really", -1),
                         "I do not like checking database entries.")
        self.assertEqual(as_statement("No worries. Imagine a job involving creating your own approach to a task; "
                                      "would that appeal to you?", "Yes it would", 1),
                         "I like creating my own approach to a task.")
        # either-or / several options: never invent which option was chosen; keep the student's own reply
        self.assertEqual(as_statement("Would you rather fix a broken machine or write a report?", "the first one", 1),
                         "the first one")
        self.assertEqual(as_statement("Would you enjoy fixing engines, or writing reports?", "the first", 1), "the first")
        # repair questions name the activity one sentence earlier
        self.assertEqual(as_statement("Imagine spending part of a working day sorting invoices. Would you enjoy that "
                                      "activity, dislike it, or feel unsure?", "no", -1), "I do not like sorting invoices.")
        self.assertEqual(as_statement("For example, imagine teaching a class at work. Would you enjoy that activity, "
                                      "dislike it, or feel unsure?", "yes", 1), "I like teaching a class.")
        # second person becomes first person throughout
        self.assertEqual(as_statement("Would a job involving helping customers who ask you for advice appeal to you?",
                                      "yes", 1), "I like helping customers who ask me for advice.")

    def test_unsure_clarify_reply_never_borrows_the_question(self):
        ex = LeakyExtractor()
        r = read_answer(ex, State(last_reply="Do you like working with tools?"), Action("CLARIFY", "R"),
                        "I really cannot decide between those two options right now")
        self.assertEqual(r.slots, [])
        self.assertEqual(ex.calls, [None])


class FinalReviewRegressions(unittest.TestCase):
    def test_mild_simulated_answer_keeps_the_asked_type(self):
        class DeskIsC:  # alone, "desk"/"report" reads as C; with the question it answers R
            name = "desk"
            def read(self, text, context=None, expected_slot=None):
                slots = [SlotReading("C", 1, .55, .9, "enjoyment", text)]
                if context:
                    slots.append(SlotReading(expected_slot, -1, .55, .9, "enjoyment", text))
                return Reading(text, slots)
        r = read_answer(DeskIsC(), State(last_reply="Hands-on with equipment, or information at a desk?"),
                        Action("CLARIFY", "R"), "I'd lean towards the desk one, slightly.")
        self.assertIn(("R", -1), [(s.slot, s.sign) for s in r.slots])

    def test_hedged_choice_still_pairs(self):
        ex = LeakyExtractor()
        r = read_answer(ex, State(last_reply="Hands-on or desk?"), Action("CLARIFY", "R"),
                        "Hmm not sure, maybe the hands-on one")
        self.assertEqual([s.slot for s in r.slots], ["R"])

    def test_into_it(self):
        state = State()
        state.slots["C"] = SlotState(.6, .9, "familiarity", "x", 1, [], .9, .9)
        (r,) = short_answer("PROBE", "Not really into it", state, "C").slots
        self.assertEqual(r.sign, -1)


class ReviewRegressions(unittest.TestCase):
    """Cases from the independent review that the first version got wrong."""

    def test_challenge_rule_ignores_now_before_and_negation(self):
        for text in ("I now enjoy hands-on work", "Not the first one", "The earlier one was a mistake",
                     "I still love it now", "I liked it before"):
            with self.subTest(text=text):
                self.assertIsNone(short_answer("CHALLENGE", text, challenge_state(), "R"))
        (r,) = short_answer("CHALLENGE", "No, the second one", challenge_state(), "R").slots
        self.assertEqual(r.sign, -1)

    def test_challenge_reply_with_its_own_stance_speaks_for_itself(self):
        class Says:  # alone, the reply states R+ itself
            name = "says"
            def read(self, text, context=None, expected_slot=None):
                return Reading(text, [SlotReading("R", 1, .9, .9, "enjoyment", text)])
        r = read_answer(Says(), challenge_state(), Action("CHALLENGE", "R"), "I now enjoy hands-on work")
        self.assertEqual([(s.slot, s.sign) for s in r.slots], [("R", 1)])

    def test_probe_rule_reads_local_negation_and_skips_dislikes(self):
        state = State()
        state.slots["C"] = SlotState(.6, .9, "familiarity", "x", 1, [], .9, .9)
        (r,) = short_answer("PROBE", "I do enjoy it, not just habit", state, "C").slots
        self.assertEqual((r.sign, r.reason), (1, "enjoyment"))
        state.slots["C"] = SlotState(-.6, .9, "familiarity", "x", 1, [], .9, .9)
        self.assertIsNone(short_answer("PROBE", "Actually I'd enjoy it", state, "C"))

    def test_clarify_answer_keeps_the_asked_type_alongside_another_stance(self):
        class Mixed:
            name = "mixed"
            def read(self, text, context=None, expected_slot=None):
                slots = [SlotReading("E", -1, .8, .9, "enjoyment", text)]
                if context:
                    slots.append(SlotReading(expected_slot, 1, .55, .9, "enjoyment", text))
                return Reading(text, slots)
        r = read_answer(Mixed(), State(last_reply="Would you rather help people or work alone?"),
                        Action("CLARIFY", "S"), "Yes, but I hate selling")
        self.assertEqual(sorted((s.slot, s.sign) for s in r.slots), [("E", -1), ("S", 1)])

    def test_clarify_statement_about_something_else_does_not_borrow_the_question(self):
        ex = LeakyExtractor()
        r = read_answer(ex, State(last_reply="Would you rather fix machines or write reports?"),
                        Action("CLARIFY", "R"), "I enjoy analysing scientific research data.")
        self.assertEqual([s.slot for s in r.slots], ["I"])
        self.assertEqual(ex.calls, [None])

    def test_challenge_and_probe_never_attach_the_question(self):
        for kind in ("CHALLENGE", "PROBE"):
            with self.subTest(kind=kind):
                ex = LeakyExtractor()
                read_answer(ex, challenge_state(), Action(kind, "R"), "hmm")
                self.assertEqual(ex.calls, [None])   # scenario 10: "hmm" after the challenge read as R+


if __name__ == "__main__":
    unittest.main()
