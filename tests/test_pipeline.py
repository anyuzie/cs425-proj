import unittest
from pathlib import Path

from pipeline.interfaces import State
from pipeline.stubs import KeywordEvidence, LexiconExtractor
from pipeline.system import System

DATA = Path(__file__).resolve().parent.parent / "data"


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system = System.load(DATA)

    def run_script(self, texts):
        state = State()
        self.system.start(state)
        return state, [self.system.step(state, t) for t in texts]

    def test_extractor_stance_strength_reason(self):
        ex = LexiconExtractor(DATA / "riasec_lexicon.json")
        r = ex.read("I love working with machines and tools.")
        self.assertEqual([(s.slot, s.sign, s.strength, s.confidence, s.reason) for s in r.slots],
                         [("R", 1, 1.0, 0.7, "enjoyment")])
        r = ex.read("I hate repairing machines, it is boring.")
        self.assertEqual((r.slots[0].sign, r.slots[0].strength), (-1, 1.0))
        r = ex.read("I like machines but that's what I know, my dad has a workshop.")
        self.assertEqual(r.slots[0].reason, "familiarity")
        self.assertEqual(ex.read("I study computer science in year two.").slots, [])

    def test_extractor_answers_to_clarify(self):
        ex = LexiconExtractor(DATA / "riasec_lexicon.json")
        r = ex.read("Definitely the first one", expected_slot="C")
        self.assertEqual((r.slots[0].slot, r.slots[0].sign), ("C", 1))
        r = ex.read("No way", expected_slot="E")
        self.assertEqual((r.slots[0].slot, r.slots[0].sign), ("E", -1))
        self.assertEqual(ex.read("Definitely the first one").slots, [])

    def test_familiarity_without_cue_is_mild_positive(self):
        ex = LexiconExtractor(DATA / "riasec_lexicon.json")
        r = ex.read("My dad runs a sales company so that's what I know.")
        self.assertEqual([(s.slot, s.sign, s.strength, s.reason) for s in r.slots],
                         [("E", 1, 0.55, "familiarity")])

    def test_reply_to_abstain_settles_reasked_slot(self):
        state, results = self.run_script(["I study computer science in year two.", "hmm", "no way"])
        self.assertEqual([r.action for r in results[1:]], ["ABSTAIN", "CLARIFY"])
        self.assertTrue(any(s.value < 0 for s in state.slots.values()))

    def test_probe_answer_resolves_without_challenge(self):
        state, results = self.run_script(["I love analysing data.",
                                          "My dad runs a sales company so that's what I know.",
                                          "Honestly I grew up around it, I don't enjoy selling."])
        self.assertEqual([r.action for r in results], ["CLARIFY", "PROBE", "CLARIFY"])
        self.assertLess(state.slots["E"].value, 0)

    def test_full_conversation_reaches_recommend(self):
        state = State()
        self.system.start(state)
        result = None
        while not state.done:
            result = self.system.step(state, "Definitely the first one")
            self.assertLessEqual(result.turn, 10)
        self.assertEqual(result.action, "RECOMMEND")
        self.assertIn(result.reply, self.system.qb["RECOMMEND"])
        self.assertEqual(len(result.recommendations), 5)
        self.assertEqual([r["rank"] for r in result.recommendations], [1, 2, 3, 4, 5])
        for rec in result.recommendations:
            self.assertLessEqual(len(rec["tasks"]), 2)
            self.assertTrue(rec["why"].startswith("Matches what you said about"))

    def test_challenge_on_planted_contradiction(self):
        _, results = self.run_script(["I love working with machines and tools.",
                                      "I hate repairing machines, it is boring."])
        r = results[1]
        self.assertEqual(r.action, "CHALLENGE")
        self.assertEqual(r.quotes, {"earlier": "I love working with machines and tools.", "earlier_turn": 1,
                                    "now": "I hate repairing machines, it is boring."})
        self.assertIn("I love working with machines and tools.", r.reply)

    def test_probe_on_familiarity(self):
        _, results = self.run_script(["I like machines but that's what I know, my dad has a workshop."])
        self.assertEqual(results[0].action, "PROBE")
        self.assertIn("quotes", results[0].__dict__)
        self.assertIn("now", results[0].quotes)

    def test_abstain_and_settle(self):
        state, results = self.run_script(["hmm", "hmm", "hmm"])
        self.assertEqual(results[0].action, "CLARIFY")
        self.assertEqual(results[1].action, "ABSTAIN")
        self.assertEqual(state.strikes["C"], 2)
        self.assertIn("C", state.settled)
        self.assertEqual(results[2].action, "ABSTAIN")

    def test_max_turns_without_signal_terminates(self):
        state = State()
        self.system.start(state)
        for _ in range(10):
            result = self.system.step(state, "hmm")
        self.assertTrue(result.done)
        self.assertIsNone(result.recommendations)

    def test_evidence_picks_overlapping_tasks(self):
        ev = KeywordEvidence(DATA / "tasks.csv", {"13-2031.00"})
        tasks = ev.top_tasks("13-2031.00", ["I like budget analysis and cost records"])
        self.assertEqual(len(tasks), 2)
        self.assertEqual(ev.top_tasks("13-2031.00", ["zzz qqq"]), [])
        self.assertEqual(ev.top_tasks("11-9999.00", ["budget"]), [])


if __name__ == "__main__":
    unittest.main()
