"""Modular architecture acceptance, with real saved pre-refactor predictions."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest

from recommender.components import ComponentInput, ComponentRanking, ComponentScore
from recommender.data_loader import load_eligible_occupations, read_profiles
from recommender.evaluate import run
from recommender.fusion import select_ranking
from recommender.m1 import BASELINE_PARAMETERS, WeightedDotProduct
from recommender.ranker import BASELINE_CONFIG, RankingPipeline
from recommender.registry import default_registry
from recommender.schemas import TYPES, OracleInput
from recommender.structured import StructuredRanker


class ArchitectureTests(unittest.TestCase):
    def test_exact_saved_oracle_scores_and_order(self):
        ranker = RankingPipeline(config=Path("configs/ranking.json"))
        saved = {}
        path = Path("reports/ranking/m2/baseline/predictions.jsonl")
        for line in path.read_text().splitlines():
            row = json.loads(line)
            saved.setdefault(row["profile_id"], []).append(row)
        for profile in read_profiles(Path("data/dev_profiles.jsonl")):
            actual = ranker.rank_oracle(OracleInput.from_profile(profile))
            expected = [{key: value for key, value in row.items()
                         if key not in {"run_id", "profile_id", "input_mode", "target_code"}}
                        for row in saved[profile["id"]]]
            self.assertEqual([asdict(o) for o in actual.occupations], expected)

    def test_runtime_legacy_formula_and_wrapper(self):
        pipeline, wrapper = RankingPipeline(), StructuredRanker()
        interests = dict.fromkeys(TYPES, 0.0) | {"I": 0.55, "S": 1.0, "C": -0.55}
        observed = {"I", "S", "C"}
        result = pipeline.rank(interests, observed, [], top_k=1000)
        self.assertEqual(result, wrapper.rank(interests, observed, [], top_k=1000))
        occupations = load_eligible_occupations(Path("data/occupations.csv"))
        expected = sorted([(sum(interests[t] * o.z_scores[t] for t in TYPES if t in observed), o.code)
                           for o in occupations], key=lambda item: (-item[0], item[1]))
        self.assertEqual([(o.signals["structured"].score, o.code) for o in result.occupations], expected)
        self.assertTrue(all(o.rrf_score is None and not o.evidence for o in result.occupations))

    def test_registry_is_deterministic_and_cannot_overwrite(self):
        registry = default_registry()
        selection = BASELINE_CONFIG["components"]["m1"]
        first = registry.resolve("m1", selection)
        self.assertEqual(first, registry.resolve("m1", selection))
        first.parameters["dimension_order"].reverse()
        self.assertEqual(registry.resolve("m1", selection).parameters, BASELINE_PARAMETERS)
        with self.assertRaisesRegex(ValueError, "modified"):
            registry.build(first, [])
        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register_variant("m1", "weighted_dot_product", "baseline_v1", {})
        with self.assertRaisesRegex(ValueError, "already registered"):
            registry.register_algorithm("m1", "weighted_dot_product", WeightedDotProduct)

    def test_unknown_selections_fail_before_data_loading(self):
        for key, selection, error in (
                ("m1", {"algorithm": "cosine_similarity", "variant": "baseline_v1"}, "unknown algorithm"),
                ("m1", {"algorithm": "weighted_dot_product", "variant": "missing"}, "unknown variant"),
                ("m3", {"algorithm": "unknown_bm25", "variant": "baseline_v1"}, "unknown algorithm"),
                ("m4", {"algorithm": "minilm", "variant": "baseline_v1"}, "unknown algorithm")):
            config = deepcopy(BASELINE_CONFIG)
            config["components"][key] = selection
            with self.subTest(selection=selection), self.assertRaisesRegex(ValueError, error):
                RankingPipeline(Path("does_not_exist.csv"), config=config)
        config = deepcopy(BASELINE_CONFIG)
        config["fusion"] = {"algorithm": "unknown"}
        with self.assertRaisesRegex(ValueError, "fusion"):
            RankingPipeline(config=config)

    def test_standalone_component_and_unavailable_signal(self):
        component = WeightedDotProduct(load_eligible_occupations(Path("data/occupations.csv")), BASELINE_PARAMETERS)
        self.assertEqual(component.score(ComponentInput(user_sentences=("Only text",))).status, "unavailable")
        self.assertEqual(component.score(ComponentInput(dict.fromkeys(TYPES, 0), frozenset(TYPES))).status, "unusable")
        self.assertEqual(component.score(ComponentInput(dict.fromkeys(TYPES, 0.5), frozenset(TYPES))).status, "usable")

    def test_disabled_components_and_empty_pipeline(self):
        config = {"components": dict.fromkeys(("m1", "m3", "m4")), "fusion": None}
        pipeline = RankingPipeline(config=config)
        self.assertEqual(pipeline.component_identifiers, {"m1": "", "m3": "", "m4": ""})
        result = pipeline.rank(dict.fromkeys(TYPES, 1), set(TYPES), ["text"])
        self.assertEqual(result.status, "no_signal")
        self.assertEqual(result.occupations, [])

    def test_output_contract_validation(self):
        for entries in (
                (ComponentScore("a", float("nan"), 1),),
                (ComponentScore("a", 1, 0),),
                (ComponentScore("a", 1, 1), ComponentScore("a", 0, 2)),
                (ComponentScore("b", 1, 1), ComponentScore("a", 1, 2)),
                (ComponentScore("a", 0, 1), ComponentScore("b", 1, 2))):
            with self.subTest(entries=entries), self.assertRaises(ValueError):
                ComponentRanking("usable", entries)
        with self.assertRaises(ValueError):
            ComponentRanking("unavailable", (ComponentScore("a", 1, 1),), "missing")
        with self.assertRaisesRegex(ValueError, "outside eligible"):
            ComponentRanking("usable", (ComponentScore("x", 1, 1),)).validate_pool({"a"})

    def test_unusable_signals_do_not_enter_fusion(self):
        usable = ComponentRanking("usable", (ComponentScore("a", -1, 1),))
        absent = ComponentRanking("unavailable", reason="no profile")
        unusable = ComponentRanking("unusable", reason="no query matches")
        self.assertEqual(select_ranking({"m1": usable, "m3": absent, "m4": unusable}, BASELINE_CONFIG["fusion"]), (usable, False))
        result, fused = select_ranking({"m3": unusable}, BASELINE_CONFIG["fusion"])
        self.assertEqual(result.status, "unusable")
        self.assertFalse(fused)
        result, fused = select_ranking({"m1": usable, "m3": usable}, BASELINE_CONFIG["fusion"])
        self.assertTrue(fused)
        self.assertEqual(result.occupations[0].score, 2 / 61)

    def test_select_registered_configurations_without_pipeline_changes(self):
        registry = default_registry()
        # Test-only alias: no new production variant or formula is introduced.
        registry.register_variant("m1", "weighted_dot_product", "test_alias", BASELINE_PARAMETERS)
        baseline = RankingPipeline(registry=registry)
        config = deepcopy(BASELINE_CONFIG)
        config["components"]["m1"]["variant"] = "test_alias"
        alias = RankingPipeline(config=config, registry=registry)
        inputs = (dict.fromkeys(TYPES, 0.5), set(TYPES), [])
        self.assertEqual(baseline.rank(*inputs), alias.rank(*inputs))
        self.assertEqual(alias.component_identifiers["m1"], "weighted_dot_product@test_alias")
        self.assertEqual(RankingPipeline(registry=registry).resolved_config, baseline.resolved_config)

    def test_evaluation_configuration_round_trip_and_history_preservation(self):
        central = Path("reports/ranking/evaluation_history.csv")
        before = central.read_bytes()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = root / "history.csv"
            history.write_bytes(before)
            first = run(Path("configs/m2.json"), root / "first", history=history,
                        ranking_config=Path("configs/ranking.json"))
            second = run(root / "first/config.json", root / "second", history=history)
            self.assertEqual(first, second)
            self.assertTrue(history.read_bytes().startswith(before))
            manifest = json.loads((root / "first/manifest.json").read_text())
            self.assertEqual(manifest["components"]["m1"], "weighted_dot_product@baseline_v1")
            self.assertEqual(manifest["components"]["fusion"], "")
            self.assertEqual(manifest["resolved_ranking"]["components"]["m1"]["parameters"], BASELINE_PARAMETERS)
            self.assertEqual(manifest["pool"], sorted(RankingPipeline().eligible_codes))
            config = json.loads((root / "first/config.json").read_text())
            config["resolved_ranking"]["components"]["m1"]["parameters"]["observed_only"] = False
            (root / "tampered.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "does not match"):
                run(root / "tampered.json", root / "invalid", history=history)
        self.assertEqual(before, central.read_bytes())


if __name__ == "__main__":
    unittest.main()
