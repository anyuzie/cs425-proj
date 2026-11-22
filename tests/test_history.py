"""Persistent history and additional metric cutoff acceptance tests."""

import csv
import json
import io
import math
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from recommender.evaluate import run
from recommender.compare import paired_comparison
from recommender.evaluation import evaluate_ranking
from recommender.history import append_history, register_legacy_baseline


class CutoffTests(unittest.TestCase):
    def test_exact_hit_boundaries(self):
        pool = {str(i) for i in range(11)}
        for rank in (1, 5, 6, 10, 11):
            codes = [c for c in sorted(pool) if c != "0"]
            codes.insert(rank - 1, "0")
            result = evaluate_ranking(codes, "0", {}, pool)
            for k in (1, 5, 10):
                self.assertEqual(result[f"hit_at_{k}"], int(rank <= k))
            self.assertAlmostEqual(result["mrr"], 1 / rank)
            for k in (5, 10):
                self.assertAlmostEqual(result[f"ndcg_at_{k}"], 1 / math.log2(rank + 1) if rank <= k else 0)

    def test_graded_five_and_ten(self):
        related = {"primary_short": ["b"], "primary_long": ["c"], "supplemental": ["d"]}
        result = evaluate_ranking(list("befgacdh"), "a", related, set("abcdefgh"))
        ideal = 15 + 7 / math.log2(3) + 3 / math.log2(4) + 1 / math.log2(5)
        actual5 = 7 + 15 / math.log2(6)
        self.assertAlmostEqual(result["ndcg_at_5"], actual5 / ideal)
        self.assertAlmostEqual(result["ndcg_at_10"], (actual5 + 3 / math.log2(7) + 1 / math.log2(8)) / ideal)
        self.assertEqual(result["hit_at_5"], 1)
        empty = evaluate_ranking([], None, {}, set("a"))
        self.assertEqual(empty["ndcg_at_5"], 0)


class HistoryTests(unittest.TestCase):
    def test_legacy_import_rejects_content_drift_even_with_newline_tolerance(self):
        original = Path.read_bytes
        def changed(path):
            data = original(path)
            # drift the file the frozen baseline is verified against (its original 60 profiles, now archived)
            return data + b" " if path in (Path("data/dev_profiles.jsonl"), Path("data/archive/dev_profiles_60.jsonl")) else data
        with tempfile.TemporaryDirectory() as directory, patch.object(Path, "read_bytes", changed):
            history = Path(directory) / "history.csv"
            with self.assertRaisesRegex(ValueError, "dataset fingerprint mismatch"):
                register_legacy_baseline(Path("reports/ranking/m2/baseline"), history=history)
            self.assertFalse(history.exists())

    def test_success_duplicates_preservation_and_future_components(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history, report = root / "history.csv", root / "run"
            metrics = run(Path("configs/m2.json"), report, history=history)
            original = history.read_bytes()
            rows = list(csv.DictReader(io.StringIO(history.read_text(encoding="utf-8"))))
            self.assertEqual(len(rows), 1)
            self.assertEqual([rows[0][k] for k in ("m1", "m3", "m4")],  # configs/ranking.yaml default
                             ["cosine_similarity@baseline_v1", "bm25_signed@ignore_negative_v3",
                              "occupation_dual_encoder_signed@finetuned_v2"])
            self.assertEqual(rows[0]["fusion"], "zscore_sum@equal_v1")
            self.assertEqual(float(rows[0]["hit_at_1"]), metrics["hit_at_1"])
            self.assertFalse(append_history(report, history=history))
            self.assertEqual(history.read_bytes(), original)
            future = root / "future"
            shutil.copytree(report, future)
            manifest = json.loads((future / "manifest.json").read_text())
            manifest["run_id"] = "future_utility_fixture"
            manifest["components"] = {"m1": "weighted_dot_product", "m3": "bm25", "m4": "minilm", "fusion": "rrf_k60"}
            (future / "manifest.json").write_text(json.dumps(manifest))
            self.assertTrue(append_history(future, history=history))
            self.assertTrue(history.read_bytes().startswith(original))
            rows = list(csv.DictReader(io.StringIO(history.read_text(encoding="utf-8"))))
            self.assertEqual(rows[1]["fusion"], "rrf_k60")
            self.assertEqual(len(rows), 2)

    def test_failed_and_incomplete_runs_leave_history_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = root / "history.csv"
            register_legacy_baseline(Path("reports/ranking/m2/baseline"), history=history)
            original = history.read_bytes()
            with patch("recommender.evaluate.RankingPipeline.rank_oracle", side_effect=ValueError("scoring failed")):
                with self.assertRaisesRegex(ValueError, "scoring failed"):
                    run(Path("configs/m2.json"), root / "failed", history=history)
            with patch("recommender.evaluate.save_json", side_effect=OSError("export failed")):
                with self.assertRaises(OSError):
                    run(Path("configs/m2.json"), root / "export_failed", history=history)
            with self.assertRaises(ValueError):
                append_history(root / "export_failed", history=history)
            self.assertEqual(history.read_bytes(), original)

    def test_legacy_import_preserves_artifacts_and_skips_duplicates(self):
        baseline = Path("reports/ranking/m2/baseline")
        before = {p.name: p.read_bytes() for p in baseline.iterdir() if p.is_file()}
        with tempfile.TemporaryDirectory() as directory:
            history = Path(directory) / "history.csv"
            self.assertTrue(register_legacy_baseline(baseline, history=history))
            original = history.read_bytes()
            self.assertFalse(register_legacy_baseline(baseline, history=history))
            self.assertEqual(original, history.read_bytes())
        self.assertEqual(before, {p.name: p.read_bytes() for p in baseline.iterdir() if p.is_file()})

    def test_legacy_comparison_and_evaluator_version_guard(self):
        baseline = Path("reports/ranking/m2/baseline")
        summary, _ = paired_comparison(baseline, baseline, samples=20)
        self.assertEqual(set(summary["metrics"]), {"hit_at_5", "mrr", "ndcg_at_10"})
        self.assertEqual(summary["metrics"]["mrr"]["ci95"], [0, 0])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run(Path("configs/m2.json"), root / "new", history=root / "history.csv")
            with self.assertRaises(ValueError):
                paired_comparison(baseline, root / "new")


if __name__ == "__main__":
    unittest.main()
