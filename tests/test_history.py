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






if __name__ == "__main__":
    unittest.main()
