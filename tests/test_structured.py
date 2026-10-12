import csv
import tempfile
import unittest
from pathlib import Path

from recommender.data_loader import load_eligible_occupations
from recommender.schemas import TYPES, RankingRequest
from recommender.structured import StructuredRanker


class StructuredTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "occupations.csv"
        # Reverse tied-code input order; dimensions beyond R/I should be ignored.
        self.rows = [self.row("02", 4, R=2, I=1, A=100),
                     self.row("01", 5, R=1, I=-1, A=-100),
                     self.row("03", 4, R=-1, I=1),
                     self.row("excluded", 3, R=1000)]
        self.write()

    def row(self, code, zone, **values):
        return {"code": code, "Title": f"Job {code}", "job_zone": str(zone),
                **{f"z_{t}": str(values.get(t, 0)) for t in TYPES}}

    def write(self):
        with self.path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def rank(self, **kwargs):
        args = {"interests": dict.fromkeys(TYPES, 0.0) | {"R": 1.0, "I": -0.5, "A": 1.0},
                "observed_types": {"R", "I"}, "user_sentences": [], "top_k": 100}
        return StructuredRanker(self.path).rank(**(args | kwargs))

    def test_hand_computed_positive_negative_and_unobserved(self):
        result = self.rank().occupations
        self.assertEqual([o.code for o in result], ["01", "02", "03"])
        self.assertEqual([o.signals["structured"].score for o in result], [1.5, 1.5, -1.5])
        self.assertEqual([o.rank for o in result], [1, 2, 3])
        self.assertTrue(all(o.rrf_score is None and not o.evidence for o in result))

    def test_sign_reversal(self):
        scores = dict.fromkeys(TYPES, 0.0) | {"R": -1.0}
        result = self.rank(interests=scores, observed_types={"R"}).occupations
        self.assertEqual([o.code for o in result], ["03", "01", "02"])
        self.assertEqual([o.signals["structured"].score for o in result], [1.0, -1.0, -2.0])

    def test_unknown_and_observed_neutral(self):
        zero = dict.fromkeys(TYPES, 0.0)
        self.assertNotEqual(RankingRequest(zero, set(), []).observed_types,
                            RankingRequest(zero, {"R"}, []).observed_types)
        for observed in (set(), {"R"}):
            with self.subTest(observed=observed):
                result = self.rank(interests=zero, observed_types=observed)
                self.assertEqual(result.status, "no_signal")
                self.assertEqual(result.occupations, [])
                self.assertTrue(result.reason)

    def test_nonzero_unobserved_does_not_make_signal(self):
        self.assertEqual(self.rank(observed_types=set()).status, "no_signal")

    def test_top_k_and_pool(self):
        self.assertEqual(len(self.rank(top_k=1).occupations), 1)
        self.assertEqual(len(self.rank(top_k=1000).occupations), 3)
        self.assertNotIn("excluded", [o.code for o in self.rank().occupations])

    def test_invalid_runtime_values(self):
        for value in (float("nan"), float("inf"), -float("inf"), 1.01, -1.01, True, "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.rank(interests=dict.fromkeys(TYPES, value))

    def test_invalid_keys_and_top_k(self):
        for kw in ({"interests": {"R": 1}}, {"observed_types": {"X"}},
                   {"interests": dict.fromkeys((*TYPES, "X"), 0)},
                   {"top_k": 0}, {"top_k": -1}, {"top_k": True}, {"top_k": 1.5}):
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                self.rank(**kw)

    def test_repeatability_and_defensive_output(self):
        ranker = StructuredRanker(self.path)
        args = (dict.fromkeys(TYPES, 0.5), set(TYPES), [])
        first = ranker.rank(*args)
        self.assertEqual(first, ranker.rank(*args))
        first.occupations[0].occupation_z["R"] = 999
        self.assertNotEqual(first, ranker.rank(*args))

    def test_duplicate_code_rejected(self):
        self.rows.append(self.rows[0])
        self.write()
        with self.assertRaisesRegex(ValueError, "duplicate code"):
            StructuredRanker(self.path)

    def test_invalid_occupation_scores(self):
        for value in ("", "nan", "inf", "not-a-number"):
            with self.subTest(value=value):
                self.rows[0]["z_R"] = value
                self.write()
                with self.assertRaises(ValueError):
                    StructuredRanker(self.path)

    def test_unrated_exclusion_and_preserved_z_scores(self):
        self.rows.append({"code": "unrated", "Title": "Unrated", "job_zone": "4", **{f"z_{t}": "" for t in TYPES}})
        self.write()
        occupations = load_eligible_occupations(self.path)
        self.assertEqual(len(occupations), 3)
        self.assertEqual(occupations[0].z_scores, {"R": 2.0, "I": 1.0, "A": 100.0, "S": 0.0, "E": 0.0, "C": 0.0})

    def test_missing_column(self):
        for row in self.rows:
            del row["z_C"]
        self.write()
        with self.assertRaisesRegex(ValueError, "missing columns"):
            StructuredRanker(self.path)


class RealPoolTests(unittest.TestCase):
    def test_verified_real_pool_and_repeatability(self):
        path = Path(__file__).resolve().parents[1] / "data" / "occupations.csv"
        self.assertTrue(path.is_file(), "M1 acceptance requires the supplied occupation CSV")
        ranker = StructuredRanker(path)
        self.assertEqual(ranker.eligible_count, 378)
        interests = dict.fromkeys(TYPES, 0.0) | {"I": 0.55, "S": 1.0, "C": -0.55}
        response = ranker.rank(interests, {"I", "S", "C"}, [], top_k=1000)
        self.assertEqual(response, ranker.rank(interests, {"I", "S", "C"}, [], top_k=1000))
        codes = [o.code for o in response.occupations]
        self.assertEqual(len(codes), len(set(codes)))
        self.assertEqual(set(codes), {o.code for o in load_eligible_occupations(path)})
        self.assertEqual(len(ranker.rank(interests, {"I", "S", "C"}, []).occupations), 100)


if __name__ == "__main__":
    unittest.main()
