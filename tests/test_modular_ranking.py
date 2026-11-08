"""Regression checks for package boundaries, component substitution and terminal I/O."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from app.terminal import run
from recommender.configuration import load_parameters, load_configuration
from recommender.models.ranker import RankingPipeline
from recommender.models.registry import default_registry
from recommender.schemas import OracleInput
from recommender.workflow import load_workflow


class ModularRankingTests(unittest.TestCase):
    def test_legacy_and_canonical_imports_share_implementation(self):
        import recommender.ranker as legacy
        import recommender.models.ranker as canonical
        self.assertIs(legacy, canonical)
        self.assertIs(legacy.RankingPipeline, RankingPipeline)

    def test_workflow_selects_active_configuration_and_preserves_baseline(self):
        paths = load_workflow(Path("configs/ranking/workflow.json"))
        candidate = RankingPipeline(config=paths["ranking"])
        baseline = RankingPipeline()
        self.assertEqual(paths["ranking"].name, "active.json")
        selected, _ = load_configuration(paths["ranking"])
        self.assertEqual(candidate.selection_config, selected)
        live, _ = load_configuration(Path("configs/ranking.yaml"))
        self.assertEqual(selected, live)
        self.assertEqual(candidate.component_identifiers["m1"], "cosine_similarity@baseline_v1")
        promotion, _ = load_configuration(paths["promotion"])
        challenger, _ = load_configuration(Path(promotion["candidate"]))
        self.assertEqual(challenger, live)
        self.assertEqual(baseline.component_identifiers["m1"], "weighted_dot_product@baseline_v1")
        inputs = OracleInput(dict(R=1., I=.5, A=0., S=-.5, E=0., C=.1), [])
        response = candidate.rank_oracle(inputs)
        self.assertEqual(len(response.occupations), candidate.eligible_count)
        self.assertTrue(all(o.rrf_score is None for o in response.occupations))

    def test_catalog_registers_new_settings_without_replacing_baseline(self):
        parameters = load_parameters("bm25_ranker_baseline_parameters")
        parameters["k1"] = 2.0
        catalog = {"variants": [{"component": "m3", "algorithm": "bm25",
                                "variant": "tuned_v1", "parameters": parameters}]}
        with patch("recommender.configuration.load_configuration", side_effect=lambda path:
                   (catalog, b"") if Path(path).name == "variants.json" else load_configuration(path)):
            registry = default_registry()
        new = registry.resolve("m3", {"algorithm": "bm25", "variant": "tuned_v1"})
        old = registry.resolve("m3", {"algorithm": "bm25", "variant": "baseline_v1"})
        self.assertEqual(new.parameters["k1"], 2.)
        self.assertEqual(old.parameters["k1"], 1.5)
        from recommender.data.data_loader import load_eligible_occupations
        from recommender.components import ComponentInput
        occupations = load_eligible_occupations(Path("data/occupations.csv"))
        query = ComponentInput(None, frozenset(), ("teach students science",))
        new_scores = registry.build(new, occupations).score(query)
        old_scores = registry.build(old, occupations).score(query)
        self.assertEqual(new_scores.status, "usable")
        self.assertNotEqual(new_scores.occupations, old_scores.occupations)

    def test_models_and_data_do_not_depend_on_evaluation_or_application(self):
        for directory in ("models", "data"):
            for path in (Path("recommender") / directory).glob("*.py"):
                for node in ast.walk(ast.parse(path.read_text())):
                    if isinstance(node, ast.ImportFrom):
                        self.assertFalse((node.module or "").startswith(
                            ("recommender.experiments", "app", "pipeline")), str(path))

    def test_terminal_uses_system_without_owning_conversation_logic(self):
        class FakeSystem:
            def start(self, state):
                return SimpleNamespace(reply="hello")

            def step(self, state, text):
                self.text = text
                state.done = True
                return SimpleNamespace(action="RECOMMEND", reply="done", recommendations=[
                    {"rank": 1, "title": "Teacher", "code": "test", "why": "fit", "tasks": ["Teach"]}])

        system = FakeSystem()
        output = []
        run(system, read=lambda _: "  science  ", write=lambda text="": output.append(text))
        self.assertEqual(system.text, "science")
        self.assertTrue(any("Teacher" in text for text in output))


if __name__ == "__main__":
    unittest.main()
