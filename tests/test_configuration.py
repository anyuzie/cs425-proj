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
    def test_yaml_default_matches_accepted_baseline(self):
        config, _ = load_configuration(Path("configs/ranking.yaml"))
        self.assertEqual(config, BASELINE_CONFIG)

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

    def test_separate_snapshots_hashes_sources_and_reproduction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metrics = run(output=root / "first", history=root / "history.csv")
            manifest = json.loads((root / "first/manifest.json").read_text())
            evaluation = json.loads((root / "first/evaluation_config.json").read_text())
            self.assertNotIn("ranking", evaluation)
            self.assertNotIn("algorithm", evaluation)
            for filename, key in (("evaluation_config.json", "evaluation_config_sha256"),
                                  ("ranking_config.json", "ranking_config_sha256"),
                                  ("resolved_ranking.json", "resolved_ranking_sha256"), ("config.json", "config_sha256")):
                self.assertEqual(hashlib.sha256((root / "first" / filename).read_bytes()).hexdigest(), manifest[key])
            for entry in manifest["configuration_sources"].values():
                source = Path(entry["path"]).read_bytes()
                self.assertEqual((root / "first" / entry["snapshot"]).read_bytes(), source)
                self.assertEqual(hashlib.sha256(source).hexdigest(), entry["sha256"])
            self.assertEqual(metrics, run(root / "first/evaluation_config.json", root / "second",
                                         ranking_config=root / "first/ranking_config.json", history=root / "history.csv"))
            with (root / "history.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["m1"], "weighted_dot_product@baseline_v1")
            self.assertEqual(rows[0]["config_hash"], manifest["config_sha256"])

    def test_zero_argument_cli_and_automatic_reports(self):
        repo = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs").mkdir()
            (root / "data").mkdir()
            for path in ("configs/m2.json", "configs/ranking.yaml", "data/occupations.csv", "data/dev_profiles.jsonl"):
                shutil.copyfile(repo / path, root / path)
            # Source import via PYTHONPATH; all default files/artifacts belong to the fixture.
            import os
            environment = {**os.environ, "PYTHONPATH": str(repo)}
            completed = subprocess.run([sys.executable, "-m", "recommender.evaluate"], cwd=root,
                                       env=environment, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(completed.stdout)
            report = root / result["report_path"]
            self.assertTrue(report.is_dir())
            self.assertEqual(report.parent, root / "reports/ranking/experiments")
            expected_count = sum(bool(line.strip()) for line in
                                 (root / "data/dev_profiles.jsonl").read_text().splitlines())
            self.assertEqual(result["metrics"]["profile_count"], expected_count)
            with (root / "reports/ranking/evaluation_history.csv").open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["report_path"], result["report_path"])
        self.assertNotEqual(default_output(), default_output())


if __name__ == "__main__":
    unittest.main()
