import json
import unittest
from pathlib import Path

from pipeline import registry
from pipeline.interfaces import Action, State
from pipeline.policy import PolicyBase
from fakes import StructuredKeywordRanker, _tokens as tokens  # also registers the test-* fakes
from pipeline.system import System

DATA = Path(__file__).resolve().parent.parent / "data"
# Behaviour tests pin the scripted test extractor and the test fakes, so switching configs/system.json to real models cannot break them.
TEST_CONFIG = Path(__file__).resolve().parent / "test_config.json"


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.system = System.load(DATA, TEST_CONFIG)

    def run_script(self, texts):
        state = State()
        self.system.start(state)
        return state, [self.system.step(state, t) for t in texts]

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
        qb = json.loads((DATA / "question_bank.json").read_text(encoding="utf-8"))
        self.assertIn(result.reply, qb["RECOMMEND"])
        self.assertEqual(len(result.recommendations), 5)
        self.assertEqual([r["rank"] for r in result.recommendations], [1, 2, 3, 4, 5])
        for rec in result.recommendations:
            self.assertLessEqual(len(rec["tasks"]), 3)
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

    def test_probe_needs_reason_confidence_above_limit(self):
        state, results = self.run_script(["My mum does accounts so I know it."])   # familiarity, confidence 0.60
        self.assertNotEqual(results[0].action, "PROBE")
        self.assertEqual(state.slots["C"].reason, "familiarity")
        self.assertEqual(state.slots["C"].reason_confidence, 0.60)

    def test_reason_confidence_reaches_state_and_profile(self):
        state, _ = self.run_script(["I like machines but that's what I know, my dad has a workshop."])
        self.assertEqual(state.slots["R"].reason_confidence, 0.9)
        self.assertEqual(state.slots["R"].history[-1].reason_confidence, 0.9)
        p = self.system.profile(state)
        self.assertEqual(p["R"]["reason_confidence"], 0.9)
        self.assertEqual(p["I"]["reason_confidence"], 0.0)

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
        ranker = StructuredKeywordRanker(DATA / "occupations.csv", DATA / "tasks.csv")
        code = "13-2031.00"
        tasks = [e.text for e in ranker._evidence(code, tokens("I like budget analysis and cost records"))]
        self.assertGreaterEqual(len(tasks), 2)
        self.assertEqual(ranker._evidence(code, tokens("zzz qqq")), [])
        self.assertEqual(ranker._evidence("11-9999.00", tokens("budget")), [])

    def test_ranker_fills_evidence(self):
        ranker = StructuredKeywordRanker(DATA / "occupations.csv", DATA / "tasks.csv")
        resp = ranker.rank({**dict.fromkeys("RIASEC", 0.0), "C": 1.0}, {"C"}, ["I like budget records"], 5)
        self.assertEqual(len(resp.occupations), 5)
        for o in resp.occupations:
            self.assertTrue(all(e.occupation_code == o.code for e in o.evidence))

    def test_structured_ranker_gives_cards_without_tasks(self):
        sys2 = System.load(DATA, TEST_CONFIG)
        sys2.ranker = registry.REGISTRY["ranker"]["structured"]({}, DATA)
        state = State()
        sys2.start(state)
        sys2.step(state, "I love analysing data.")
        recs = sys2.recommendations(state)
        self.assertEqual(len(recs), 5)
        self.assertTrue(all(r["tasks"] == [] for r in recs))

    def test_confirmer_threshold_drops_flag(self):
        class Never:
            name = "never"
            def score(self, a, b):
                return 0.5
        sys2 = System.load(DATA, TEST_CONFIG)
        sys2.confirmer = Never()
        state = State()
        sys2.start(state)
        sys2.step(state, "I love working with machines and tools.")
        r = sys2.step(state, "I hate repairing machines, it is boring.")
        self.assertNotEqual(r.action, "CHALLENGE")

    def test_unknown_component_lists_valid_names(self):
        import tempfile
        cfg = json.loads(TEST_CONFIG.read_text())
        cfg["components"]["extractor"] = "nope"
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(cfg, f)
        with self.assertRaisesRegex(ValueError, "deberta"):
            System.load(DATA, f.name)
        cfg["components"]["extractor"] = "test-scripted"
        # A part that is not filled in yet must stop startup with a pointer to the guide, not half-run.
        # Only remaining placeholders belong here. Implemented model loaders need
        # checkpoints; their controlled-output tests must not download models.
        for kind, name in [("tracker", "decay"), ("policy", "manual")]:
            one = json.loads(json.dumps(cfg))
            one["components"][kind] = name
            with open(f.name, "w") as g:
                json.dump(one, g)
            with self.subTest(kind=kind):
                try:
                    System.load(DATA, f.name)  # part already filled in: loading is fine
                except NotImplementedError as e:
                    self.assertIn("INTEGRATION.md", str(e))


class PolicyBaseGuarantees(unittest.TestCase):
    """What PolicyBase enforces so a teammate's _decide cannot break the conversation or the evaluation."""

    @classmethod
    def setUpClass(cls):
        cls.system = System.load(DATA, TEST_CONFIG)

    def test_always_recommends_at_max_turns_even_if_decide_never_does(self):
        class NeverRecommends(PolicyBase):
            def _decide(self, state):
                return Action("CLARIFY", "C", "keep asking")
        self.system.policy, policy = NeverRecommends(3), self.system.policy
        try:
            state = State()
            self.system.start(state)
            results = [self.system.step(state, "I love working with machines and tools.") for _ in range(3)]
        finally:
            self.system.policy = policy
        self.assertEqual(results[-1].action, "RECOMMEND")
        self.assertTrue(results[-1].done)
        self.assertEqual(len(results[-1].recommendations), 5)

    def test_no_probing_rejects_challenge_and_probe(self):
        class Probes(PolicyBase):
            def _decide(self, state):
                return Action("PROBE", "E", "probe", {"new_src": "x"})
        with self.assertRaisesRegex(ValueError, "no_probing"):
            Probes(10, no_probing=True).decide(State(turn=1))
        self.assertEqual(Probes(10).decide(State(turn=1)).kind, "PROBE")

    def test_policy_sees_this_turns_ranking(self):
        seen = []

        class Records(PolicyBase):
            def _decide(self, state):
                seen.append(list(state.ranking))
                return Action("CLARIFY", "C", "record")
        self.system.policy, policy = Records(10), self.system.policy
        try:
            state = State()
            self.system.start(state)
            self.system.step(state, "I love working with machines and tools.")
        finally:
            self.system.policy = policy
        self.assertEqual(len(seen[0]), 100)
        self.assertEqual([o.rank for o in seen[0][:3]], [1, 2, 3])
        self.assertEqual(self.system.top5(state), [{"code": o.code, "title": o.title} for o in seen[0][:5]])


if __name__ == "__main__":
    unittest.main()
