"""Hand-calculated metrics and M2 reproducibility acceptance tests."""

import json
import csv
import io
import math
from pathlib import Path
import tempfile
import unittest

from recommender.compare import paired_comparison
from recommender.evaluate import run
from recommender.evaluation import aggregate, evaluate_ranking, relevance_labels
from recommender.schemas import OracleInput
from recommender.structured import StructuredRanker


class MetricTests(unittest.TestCase):
    pool = set("abcdefghijk")

    def test_recall_counts_distinct_eligible_labels_and_empty_sets(self):
        pool = {str(i) for i in range(25)}
        related = {"primary_short": ["5", "5", "outside"], "supplemental": ["10", "20"]}
        result = evaluate_ranking([str(i) for i in range(25)], "0", related, pool)
        self.assertEqual(result["recall_at_5"], .25)
        self.assertEqual(result["recall_at_10"], .5)
        self.assertEqual(result["recall_at_20"], .75)
        empty = evaluate_ranking([], None, {}, pool)
        self.assertTrue(all(empty[f"recall_at_{k}"] == 0 for k in (5, 10, 20)))

    def score(self, codes, target="a", related=None):
        return evaluate_ranking(list(codes), target, related or {}, self.pool)

    def test_rank_one_six_absent(self):
        first = self.score("abcdef")
        self.assertEqual((first["hit_at_5"], first["mrr"], first["ndcg_at_10"]), (1, 1, 1))
        sixth = self.score("bcdefa")
        self.assertEqual(sixth["hit_at_5"], 0)
        self.assertAlmostEqual(sixth["mrr"], 1 / 6)
        self.assertAlmostEqual(sixth["ndcg_at_10"], 1 / math.log2(7))
        absent = self.score("bcdef")
        self.assertEqual((absent["mrr"], absent["ndcg_at_10"]), (0, 0))

    def test_graded_dcg(self):
        related = {"primary_short": ["b"], "primary_long": ["c"], "supplemental": ["d"]}
        self.assertEqual(self.score("abcd", related=related)["ndcg_at_10"], 1)
        ideal = 15 + 7 / math.log2(3) + 3 / math.log2(4) + 1 / math.log2(5)
        actual = 7 + 15 / math.log2(3) + 3 / math.log2(4) + 1 / math.log2(5)
        self.assertAlmostEqual(self.score("bacd", related=related)["ndcg_at_10"], actual / ideal)

    def test_max_grade_and_pool_filter(self):
        related = {"primary_short": ["a", "b", "b", "outside"], "supplemental": ["b", "c"]}
        self.assertEqual(relevance_labels("a", related, self.pool), {"a": 4, "b": 3, "c": 1})
        self.assertEqual(self.score("abc", related=related)["ndcg_at_10"], 1)

    def test_empty_ideal_and_target_diagnostics(self):
        missing = self.score("abc", target=None)
        invalid = self.score("abc", target="outside")
        self.assertEqual(missing["ndcg_at_10"], 0)
        self.assertTrue(missing["zero_idcg"])
        self.assertEqual(missing["target_status"], "missing")
        self.assertEqual(invalid["target_status"], "ineligible")
        summary = aggregate([missing, invalid])
        self.assertEqual(summary["profile_count"], 2)
        self.assertEqual(summary["missing_targets"], 1)
        self.assertEqual(summary["ineligible_targets"], 1)

    def test_duplicate_and_ineligible_predictions_rejected(self):
        for codes in (["a", "a"], ["outside"]):
            with self.assertRaises(ValueError):
                self.score(codes)

    def test_cutoffs(self):
        result = self.score("bcdefghijka")
        self.assertEqual(result["target_rank"], 11)
        self.assertEqual(result["ndcg_at_10"], 0)
        self.assertAlmostEqual(result["mrr"], 1 / 11)
        self.assertEqual(self.score("bcdea")["hit_at_5"], 1)

    def test_no_signal(self):
        result = self.score("")
        self.assertEqual(result["mrr"], 0)
        self.assertFalse(result["zero_idcg"])


class BaselineTests(unittest.TestCase):
    def test_history_adds_recall_columns_preserving_old_values_and_backup(self):
        from recommender.experiments.history import RECALL_METRICS, FIELDS
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = root / "history.csv"
            run(Path("configs/m2.json"), root / "first", history=history)
            rows = list(csv.DictReader(io.StringIO(history.read_text(encoding="utf-8"))))
            old_fields = [key for key in FIELDS if key not in RECALL_METRICS]
            with history.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=old_fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
            original = history.read_bytes()
            run(Path("configs/m2.json"), root / "second", history=history)
            self.assertEqual(history.with_suffix(".csv.pre_recall.bak").read_bytes(), original)
            new_rows = list(csv.DictReader(io.StringIO(history.read_text(encoding="utf-8"))))
            self.assertEqual(len(new_rows), 2)
            self.assertEqual({k: new_rows[0][k] for k in old_fields}, {k: rows[0][k] for k in old_fields})
            self.assertTrue(all(new_rows[0][k] == "" for k in RECALL_METRICS))
            self.assertTrue(all(new_rows[1][k] != "" for k in RECALL_METRICS))

    def test_oracle_scale_and_runtime_boundary(self):
        ranker = StructuredRanker()
        vector = dict.fromkeys("RIASEC", 0.0)
        vector["I"] = 2.5
        result = ranker.rank_oracle(OracleInput(vector, []))
        self.assertEqual(len(result.occupations), 378)
        for occupation in result.occupations:
            self.assertEqual(occupation.signals["structured"].score, 2.5 * occupation.occupation_z["I"])
        with self.assertRaises(ValueError):
            ranker.rank(vector, {"I"}, [])
        vector["I"] = 0
        self.assertEqual(ranker.rank_oracle(OracleInput(vector, [])).status, "no_signal")

    def test_reproducible_run_and_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a", Path(directory) / "b"
            first = run(Path("configs/m2.json"), a, history=Path(directory) / "history.csv")
            expected_count = len([line for line in Path("data/dev_profiles.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()])
            self.assertEqual(first["profile_count"], expected_count)
            self.assertEqual(first, run(Path("configs/m2.json"), b, history=Path(directory) / "history.csv"))
            for name in ("metrics.json", "inputs.json", "per_profile.json", "config.json"):
                self.assertEqual((a / name).read_bytes(), (b / name).read_bytes())
            # Run identity is unique; scores and ordering remain reproducible.
            def predictions(path):
                return [{k: v for k, v in json.loads(line).items() if k != "run_id"}
                        for line in (path / "predictions.jsonl").read_text().splitlines()]
            self.assertEqual(predictions(a), predictions(b))
            summary, diffs = paired_comparison(a, b, samples=100)
            self.assertEqual(len(diffs), expected_count)
            self.assertEqual(summary["metrics"]["mrr"]["ci95"], [0, 0])
            # Exercise a real paired difference while preserving comparison inputs.
            metrics = json.loads((b / "per_profile.json").read_text())
            for row in metrics:
                row["mrr"] += 0.1
            (b / "per_profile.json").write_text(json.dumps(metrics))
            summary, _ = paired_comparison(a, b, samples=100)
            self.assertAlmostEqual(summary["metrics"]["mrr"]["mean_difference"], 0.1)
            manifest = json.loads((b / "manifest.json").read_text())
            manifest["input_mode"] = "actual_tracker"
            (b / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "input_mode"):
                paired_comparison(a, b)
            with self.assertRaises(ValueError):
                run(Path("configs/m2.json"), a, history=Path(directory) / "history.csv")

    def test_held_out_config_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            config = json.loads(Path("configs/m2.json").read_text())
            config["profiles"] = "data/profiles.jsonl"
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                run(path, Path(directory) / "output")


if __name__ == "__main__":
    unittest.main()
