"""Mock mode (devtools/): the whole app runs without the trained models, and can never pass for a real run."""

import json
from pathlib import Path
import unittest

import devtools.mocks  # noqa: F401  registers mock-keyword / mock-nli
from contracts import DATA, ExtractorContract
from devtools.mocks import MockKeywordExtractor
from pipeline import registry
from pipeline.interfaces import State
from pipeline.system import System

CONFIG = Path(__file__).resolve().parent.parent / "configs/system.mock.json"


class TestMockExtractorContract(ExtractorContract, unittest.TestCase):
    def make(self):
        return MockKeywordExtractor(DATA / "riasec_lexicon.json")


class MockMode(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(CONFIG.read_text())
        self.system = System.from_config(DATA, self.config)
        self.state = State()
        self.system.start(self.state)

    def say(self, text):
        return self.system.step(self.state, text)

    def test_mock_config_uses_the_real_parts_except_the_models(self):
        self.assertEqual(self.system.components, {
            "extractor": "mock-keyword", "tracker": "decay", "rule": "rule", "confirmer": "mock-nli",
            "ranker": "configured", "policy": "manual", "filler": "question-bank"})
        self.assertEqual(self.system.input_mode, "mock_tracker")
        # Weight-free mock mode pins RIASEC cosine explicitly; the live default needs the semantic encoder.
        self.assertEqual(self.system.ranker.component_identifiers["m1"], "cosine_similarity@baseline_v1")

    def test_probe_and_challenge_flows_run(self):
        self.assertEqual(self.say("My dad runs a sales company so that's what I know.").action, "PROBE")
        self.say("just what I know")
        self.assertEqual(self.state.slots["E"].value, 0)
        self.say("I love working with machines and tools.")
        self.assertEqual(self.say("I hate repairing machines, it is boring.").action, "CHALLENGE")
        self.say("the second one")
        self.assertEqual(self.state.slots["R"].value, -1)

    def test_conversation_ends_with_recommendations(self):
        self.say("I love analysing data and solving research problems.")
        for _ in range(9):
            result = self.say("definitely")
            if result.done:
                break
        self.assertTrue(result.done)
        self.assertEqual(len(result.recommendations or []), 5)

    def test_mock_parts_can_never_be_labelled_a_real_run(self):
        self.config["settings"]["input_mode"] = "actual_tracker"
        with self.assertRaisesRegex(ValueError, "cannot be labelled actual_tracker"):
            System.from_config(DATA, self.config)

    def test_the_app_registry_only_knows_mocks_after_devtools_is_imported(self):
        source = (Path(registry.__file__)).read_text()
        self.assertNotIn("mock-", source)   # registered from devtools, never by the app itself


if __name__ == "__main__":
    unittest.main()
