"""Promotion decisions, failure isolation and real evaluator/workflow integration."""

import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from recommender.experiments.evaluation import METRICS
from recommender.experiments.promotion import evaluate_and_promote, improves
from recommender.experiments.evaluate import run
from recommender.workflow import main


class PromotionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.active = self.root / "active.json"
        self.active.write_bytes(Path("configs/ranking/baseline_v1.json").read_bytes())
        self.before = self.active.read_bytes()
        self.candidate = Path("configs/ranking/m1_cosine.json")

    def round(self, candidate=None, name="round"):
        return evaluate_and_promote(Path("configs/m2.json"), candidate or self.candidate,
                                    self.active, output=self.root / name, history=self.root / "history.csv")

    def test_rule_requires_hit5_gain_without_any_regression(self):
        control = dict.fromkeys(METRICS, .5)
        self.assertFalse(improves(control, control, "hit5_no_regression"))
        better = {**control, "hit_at_5": .6}
        self.assertTrue(improves(control, better, "hit5_no_regression"))
        for metric in METRICS:
            regressed = {**better, metric: .4}
            self.assertFalse(improves(control, regressed, "hit5_no_regression"))
        self.assertFalse(improves(control, {**control, "mrr": .6}, "hit5_no_regression"))

    def test_invalid_metrics_never_promote(self):
        control = dict.fromkeys(METRICS, .5)
        for value in (float("nan"), float("inf"), True, -1, 2):
            with self.subTest(value=value), self.assertRaises(ValueError):
                improves(control, {**control, "mrr": value}, "hit5_no_regression")

    def test_real_round_promotes_then_rejects_a_worse_challenger_and_tie(self):
        result = self.round()
        self.assertTrue(result["promotion"]["promoted"])
        self.assertEqual(json.loads(self.active.read_bytes()), json.loads(self.candidate.read_bytes()))
        self.assertEqual((self.root / "round/previous_active.json").read_bytes(), self.before)
        self.assertEqual(result["promotion"]["selected_sha256"], hashlib.sha256(self.active.read_bytes()).hexdigest())
        selected = self.active.read_bytes()
        worse = self.round(Path("configs/ranking/baseline_v1.json"), "worse")
        self.assertFalse(worse["promotion"]["promoted"])
        tie = self.round(name="tie")
        self.assertFalse(tie["promotion"]["promoted"])
        self.assertEqual(self.active.read_bytes(), selected)
        with (self.root / "history.csv").open(newline="", encoding="utf-8") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 6)

    def test_failed_evaluation_preserves_selection_and_releases_lock(self):
        with patch("recommender.experiments.promotion.run", side_effect=RuntimeError("model unavailable")):
            with self.assertRaisesRegex(RuntimeError, "model unavailable"):
                self.round()
        self.assertEqual(self.active.read_bytes(), self.before)
        self.assertFalse(self.active.with_suffix(".json.lock").exists())

    def test_incompatible_runs_cannot_promote(self):
        def changed_inputs(config, output, **kwargs):
            result = run(config, output, **kwargs)
            if output.name == "candidate":
                manifest_path = output / "manifest.json"
                manifest = json.loads(manifest_path.read_text())
                manifest["inputs_sha256"] = "different_inputs"
                manifest_path.write_text(json.dumps(manifest))
            return result
        with patch("recommender.experiments.promotion.run", side_effect=changed_inputs):
            with self.assertRaisesRegex(ValueError, "incompatible paired runs: inputs_sha256"):
                self.round()
        self.assertEqual(self.active.read_bytes(), self.before)

    def test_concurrent_round_is_rejected(self):
        lock = self.active.with_suffix(".json.lock")
        lock.write_text("other round")
        with self.assertRaises(FileExistsError):
            self.round()
        self.assertEqual(lock.read_text(), "other round")
        self.assertEqual(self.active.read_bytes(), self.before)

    def test_external_selection_change_is_not_overwritten(self):
        def edit_active(config, output, **kwargs):
            result = run(config, output, **kwargs)
            if output.name == "candidate":
                self.active.write_text('{"externally_changed": true}')
            return result
        with patch("recommender.experiments.promotion.run", side_effect=edit_active):
            with self.assertRaisesRegex(ValueError, "active selection changed"):
                self.round()
        self.assertEqual(json.loads(self.active.read_text()), {"externally_changed": True})

    def test_publish_failure_preserves_previous_bytes_and_prepared_audit(self):
        with patch("recommender.experiments.promotion.atomic_write", side_effect=OSError("write blocked")):
            with self.assertRaisesRegex(OSError, "write blocked"):
                self.round()
        self.assertEqual(self.active.read_bytes(), self.before)
        decision = json.loads((self.root / "round/promotion.json").read_text())
        self.assertFalse(decision["promoted"])
        self.assertEqual(decision["publication_status"], "prepared")

    def test_invalid_candidate_does_not_touch_active(self):
        bad = self.root / "bad.json"
        bad.write_text('{"components": {}}')
        with self.assertRaises(ValueError):
            self.round(bad)
        self.assertEqual(self.active.read_bytes(), self.before)

    def test_workflow_cli_uses_configured_candidate_and_explicit_override(self):
        policy = self.root / "promotion.json"
        policy.write_text(json.dumps({"candidate": str(self.candidate.resolve()), "policy": "hit5_no_regression"}))
        workflow = self.root / "workflow.json"
        workflow.write_text(json.dumps({"ranking": str(self.active), "evaluation": "configs/m2.json",
            "system": "configs/system.json", "preparation": "configs/ranking/preparation.json", "promotion": str(policy)}))
        for extra, expected in (([], self.candidate.resolve()), (["--ranking-config", "configs/ranking/m3_bm25.json"], Path("configs/ranking/m3_bm25.json").resolve())):
            with patch("sys.argv", ["workflow", "evaluate", "--config", str(workflow), *extra]), patch("recommender.experiments.promotion.evaluate_and_promote", return_value={}) as promote, patch("builtins.print"):
                main()
                self.assertEqual(promote.call_args.args[1], expected)
                self.assertEqual(promote.call_args.args[2], self.active)


if __name__ == "__main__":
    unittest.main()
