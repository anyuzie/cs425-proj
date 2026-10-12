import csv
import json
import tempfile
import unittest
from pathlib import Path

from recommender.data_loader import audit_data
from recommender.schemas import TYPES, OracleInput, RankingRequest, RankingResponse


class ContractTests(unittest.TestCase):
    def request(self, **kwargs):
        return RankingRequest(**({"interests": dict.fromkeys(TYPES, 0.0), "observed_types": set(), "user_sentences": []} | kwargs))

    def test_unknown_and_known_zero_remain_distinct(self):
        unknown, neutral = self.request(), self.request(observed_types={"R"})
        self.assertNotEqual(unknown.observed_types, neutral.observed_types)
        self.assertFalse(neutral.has_structured_signal)
        self.assertFalse(neutral.has_text)
        self.assertTrue(self.request(user_sentences=["I enjoy teaching"]).has_text)

    def test_invalid_interest_inputs(self):
        for value in (float("nan"), float("inf"), 1.1, -1.1, True, "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.request(interests=dict.fromkeys(TYPES, value))
        with self.assertRaises(ValueError):
            self.request(interests={"R": 0})

    def test_invalid_metadata(self):
        for kw in ({"observed_types": {"X"}}, {"user_sentences": [1]}, {"top_k": 0}, {"top_k": True}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                self.request(**kw)

    def test_oracle_preserves_scale_and_excludes_labels(self):
        profile = {"hidden_slots": dict.fromkeys(TYPES, 2.5), "text": "example", "target": "secret"}
        oracle = OracleInput.from_profile(profile)
        self.assertEqual(oracle.interests["R"], 2.5)
        self.assertEqual(oracle.input_mode, "oracle")
        self.assertFalse(hasattr(oracle, "target"))
        with self.assertRaises(ValueError):
            self.request(interests=oracle.interests)

    def test_explicit_no_signal(self):
        self.assertEqual(RankingResponse("no_signal", reason="No usable input").occupations, [])
        with self.assertRaises(ValueError):
            RankingResponse("no_signal")
        with self.assertRaises(ValueError):
            RankingResponse("ranked")


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.occupations = [{"code": "01-0001.00", "Title": "Fixture", "job_zone": "4", "has_riasec": "True", "bm25_doc": "fixture",
                             **{t: "3" for t in TYPES}, **{f"z_{t}": "0.25" for t in TYPES}}]
        self.tasks = [{"code": "01-0001.00", "Task ID": "1", "Task": "Fixture task"}]
        self.related = [{"code": "01-0001.00", "related_code": "01-0001.00", "tier": "Primary-Short", "rank": "1"}]
        self.write_csv("occupations.csv", self.occupations)
        self.write_csv("tasks.csv", self.tasks)
        self.write_csv("related_occupations.csv", self.related)
        (self.root / "labels.json").write_text(json.dumps({"types": list(TYPES)}))
        for name, ident in (("dev_profiles.jsonl", "D1"), ("profiles.jsonl", "P1")):
            self.write_profile(name, ident)

    def write_csv(self, name, rows):
        with (self.root / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def write_profile(self, name, ident, target="01-0001.00"):
        profile = {"id": ident, "text": "Synthetic fixture", "hidden_slots": dict.fromkeys(TYPES, 2), "target": target,
                   "related": {"primary_short": [], "primary_long": [], "supplemental": []}}
        (self.root / name).write_text(json.dumps(profile) + "\n")

    def test_valid_and_reproducible_audit(self):
        result = audit_data(self.root)
        self.assertEqual(result, audit_data(self.root))
        self.assertEqual(result["counts"]["eligible"], 1)
        self.assertFalse(result["splits"]["dev_profiles.jsonl"]["has_ordered_turns"])
        self.assertEqual(len(result["sha256"]), 6)

    def test_duplicate_codes(self):
        self.write_csv("occupations.csv", self.occupations * 2)
        with self.assertRaisesRegex(ValueError, "duplicate code"):
            audit_data(self.root)

    def test_broken_task_join(self):
        self.tasks[0]["code"] = "unknown"
        self.write_csv("tasks.csv", self.tasks)
        with self.assertRaisesRegex(ValueError, "unknown occupation"):
            audit_data(self.root)

    def test_missing_score(self):
        self.occupations[0]["z_R"] = ""
        self.write_csv("occupations.csv", self.occupations)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            audit_data(self.root)

    def test_nonfinite_score(self):
        self.occupations[0]["z_R"] = "nan"
        self.write_csv("occupations.csv", self.occupations)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            audit_data(self.root)

    def test_wrong_dimension_order(self):
        (self.root / "labels.json").write_text(json.dumps({"types": list(reversed(TYPES))}))
        with self.assertRaisesRegex(ValueError, "RIASEC order"):
            audit_data(self.root)

    def test_ineligible_target(self):
        self.write_profile("dev_profiles.jsonl", "D1", "unknown")
        with self.assertRaisesRegex(ValueError, "ineligible"):
            audit_data(self.root)

    def test_split_overlap(self):
        self.write_profile("profiles.jsonl", "D1")
        with self.assertRaisesRegex(ValueError, "overlap"):
            audit_data(self.root)

    def test_count_drift(self):
        with self.assertRaisesRegex(ValueError, "expected 378"):
            audit_data(self.root, {"eligible": 378})

    def test_missing_schema(self):
        self.tasks[0].pop("Task")
        self.write_csv("tasks.csv", self.tasks)
        with self.assertRaisesRegex(ValueError, "missing columns"):
            audit_data(self.root)


if __name__ == "__main__":
    unittest.main()
