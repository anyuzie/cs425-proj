"""Default-file workflow, safe YAML validation and configuration provenance."""

from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from recommender.configuration import load_configuration
from recommender.evaluate import default_output, run
from recommender.ranker import BASELINE_CONFIG


class ConfigurationTests(unittest.TestCase):
    def test_yaml_default_is_the_signed_lexical_semantic_dislike_selection(self):
        config, _ = load_configuration(Path("configs/ranking.yaml"))
        preset, _ = load_configuration(Path("configs/ranking/occupation_semantic_signed_v2_fusion_zscore_bm25_signed_ignore_v3_dislike_title_v3.json"))
        self.assertEqual(config, preset)
        self.assertEqual(config["components"]["m3"], {"algorithm": "bm25_signed", "variant": "ignore_negative_v3"})
        self.assertEqual(config["components"]["m4"], {"algorithm": "occupation_dual_encoder_signed", "variant": "finetuned_v2"})
        self.assertEqual(config["fusion"], {"algorithm": "zscore_sum", "variant": "equal_v1"})
        self.assertEqual(config["post_fusion"], {"algorithm": "dislike_demotion", "variant": "title_v3"})

    def test_zscore_default_remains_selectable_as_previous_default(self):
        previous, _ = load_configuration(Path("configs/ranking/occupation_semantic_fusion_zscore_v1.json"))
        self.assertEqual(previous["components"]["m4"], {"algorithm": "occupation_dual_encoder", "variant": "finetuned_v1"})
        self.assertNotIn("post_fusion", previous)
        system, _ = load_configuration(Path("configs/system.zscore.json"))
        self.assertEqual(system["paths"]["ranking_config"], "configs/ranking/occupation_semantic_fusion_zscore_v1.json")

    def test_rrf_fusion_remains_selectable_as_previous_default(self):
        rrf, _ = load_configuration(Path("configs/ranking/occupation_semantic_fusion_v1.json"))
        self.assertEqual(rrf["fusion"], {"algorithm": "rrf", "k": 60})
        system, _ = load_configuration(Path("configs/system.rrf.json"))
        self.assertEqual(system["paths"]["ranking_config"], "configs/ranking/occupation_semantic_fusion_v1.json")

    def test_cosine_remains_selectable_as_previous_default(self):
        cosine, _ = load_configuration(Path("configs/ranking/m1_cosine.json"))
        self.assertEqual(cosine["components"]["m1"], {"algorithm": "cosine_similarity", "variant": "baseline_v1"})
        system, _ = load_configuration(Path("configs/system.cosine.json"))
        self.assertEqual(system["paths"]["ranking_config"], "configs/ranking/m1_cosine.json")

    def test_dot_product_control_remains_the_programmatic_baseline(self):
        baseline, _ = load_configuration(Path("configs/ranking/baseline_v1.json"))
        self.assertEqual(baseline["components"], BASELINE_CONFIG["components"])

    def test_invalid_yaml_duplicate_keys_and_nonmapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ranking.yaml"
            for source in ("components: [", "components: null\ncomponents: {}", "- not-a-mapping",
                           "!!python/object/apply:os.system ['echo unsafe']"):
                path.write_text(source)
                with self.subTest(source=source), self.assertRaisesRegex(ValueError, "invalid configuration"):
                    load_configuration(path)
            path = Path(directory) / "config.json"
            path.write_text('{"seed": 1, "seed": 2}')
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_configuration(path)

    def test_missing_default_files_fail_without_history_or_data_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for target in ("evaluation", "ranking"):
                with self.subTest(target=target), patch("recommender.ranker.load_eligible_occupations") as loader:
                    kwargs = {"config_path": root / "missing.json"} if target == "evaluation" else {"ranking_config": root / "missing.yaml"}
                    with self.assertRaisesRegex(ValueError, "configuration file not found"):
                        run(output=root / "output", history=root / "history.csv", **kwargs)
                    loader.assert_not_called()
            self.assertFalse((root / "history.csv").exists())
            self.assertFalse((root / "output").exists())

    def test_invalid_variants_and_evaluation_settings_fail_before_execution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for field, value, message in (("algorithm", "missing", "unknown algorithm"), ("variant", "missing", "unknown variant")):
                config = deepcopy(BASELINE_CONFIG)
                config["components"]["m1"][field] = value
                path = root / "ranking.json"
                path.write_text(json.dumps(config))
                with patch("recommender.ranker.load_eligible_occupations") as loader:
                    with self.assertRaisesRegex(ValueError, message):
                        run(output=root / "output", history=root / "history.csv", ranking_config=path)
                    loader.assert_not_called()
            original = json.loads(Path("configs/m2.json").read_text())
            for field, value in (("seed", True), ("split", "held_out"), ("unexpected", 1)):
                path = root / "evaluation.json"
                path.write_text(json.dumps({**original, field: value}))
                with self.assertRaises(ValueError):
                    run(path, root / "output", history=root / "history.csv")
            self.assertFalse((root / "history.csv").exists())

    def test_default_evaluation_file_cannot_override_ranking_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "m2.json"
            config = json.loads(Path("configs/m2.json").read_text())
            config["ranking"] = BASELINE_CONFIG
            path.write_text(json.dumps(config))
            with patch("recommender.evaluate.DEFAULT_EVALUATION_CONFIG", path):
                with self.assertRaisesRegex(ValueError, "contains ranking settings"):
                    run(path, root / "output", history=root / "history.csv")
            self.assertFalse((root / "output").exists())




if __name__ == "__main__":
    unittest.main()
