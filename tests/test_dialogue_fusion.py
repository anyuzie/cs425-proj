"""Fusion dialogue evaluation: persona file, metrics, bootstrap and arm wiring without model downloads."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import fakes  # noqa: F401
from eval.dialogue_fusion import PERSONAS, RANKERS, acceptable_metrics, paired_bootstrap, report, run
from pipeline.interfaces import Reading, SlotReading
from pipeline.system import System
from recommender.data.data_loader import load_eligible_occupations
from recommender.models.ranker import RankingPipeline

ROOT = Path(__file__).resolve().parents[1]


class PersonaFileTests(unittest.TestCase):
    def test_personas_are_well_formed_and_in_pool(self):
        pool = {o.code for o in load_eligible_occupations(ROOT / "data/occupations.csv")}
        rows = [json.loads(line) for line in PERSONAS.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 25)
        self.assertEqual(len({r["id"] for r in rows}), 25)
        for r in rows:
            self.assertEqual(r["target"], r["acceptable"][0])
            self.assertLessEqual(set(r["acceptable"]), pool)
            self.assertEqual(set(r["hidden_slots"]), set("RIASEC"))
            self.assertEqual(set(r["resolutions"]), set("RIASEC"))
            self.assertEqual(r["conflict"], r["conflict_intended"])
        self.assertEqual(sum(r["conflict"] for r in rows), 14)


class MetricTests(unittest.TestCase):
    def test_first_acceptable_rank_drives_mrr_and_hits(self):
        codes = ["a", "b", "c", "d", "e", "f", "g"]
        m = acceptable_metrics(codes, ["g", "c"])
        self.assertEqual(m["best_rank"], 3)
        self.assertAlmostEqual(m["mrr"], 1 / 3)
        self.assertEqual((m["hit_at_5"], m["hit_at_10"]), (1, 1))
        self.assertEqual(m["acceptable_ranks"], {"g": 7, "c": 3})
        self.assertAlmostEqual(acceptable_metrics(["x", "a"], ["x", "a"])["ndcg_at_10"], 1.0)

    def test_rank_seven_is_a_hit_at_ten_but_not_at_five(self):  # the §12.1 case that motivated MRR
        m = acceptable_metrics([str(i) for i in range(1, 20)], ["7"])
        self.assertEqual((m["hit_at_5"], m["hit_at_10"]), (0, 1))
        self.assertAlmostEqual(m["mrr"], 1 / 7)

    def test_missing_acceptable_scores_zero(self):
        m = acceptable_metrics(["a", "b"], ["z"])
        self.assertEqual((m["mrr"], m["hit_at_10"], m["best_rank"]), (0.0, 0, None))

    def test_bootstrap_is_paired_and_deterministic(self):
        a, b = [1.0, .5, .2, 0.0], [.5, .5, .1, 0.0]
        first, second = paired_bootstrap(a, b, samples=500), paired_bootstrap(a, b, samples=500)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["mean"], .15)
        self.assertLessEqual(first["ci95"][0], first["mean"])
        self.assertGreaterEqual(first["ci95"][1], first["mean"])
        self.assertEqual(paired_bootstrap([1, 1], [1, 1], samples=50)["ci95"], [0.0, 0.0])
        with self.assertRaises(ValueError):
            paired_bootstrap([1], [1, 2])


class RunTests(unittest.TestCase):
    def test_six_arms_terminate_with_diagnostics_and_hidden_labels_stay_hidden(self):
        personas = [json.loads(line) for line in PERSONAS.read_text(encoding="utf-8").splitlines()][:2]
        bank = json.loads((ROOT / "data/question_bank.json").read_text())
        config = json.loads((Path(__file__).parent / "test_config.json").read_text())
        config["components"].update(tracker="decay", policy="manual", ranker="structured")
        system = System.from_config(ROOT / "data", config)
        cosine = RankingPipeline(ROOT / "data/occupations.csv", config=ROOT / RANKERS["cosine"])
        pipelines = dict.fromkeys(RANKERS, cosine)  # no semantic model in unit tests
        seen = []

        def extract(text, *args, **kwargs):
            seen.append(text)
            return Reading(text, [SlotReading(t, 1, .55, .9, "enjoyment", text) for t in "RIA"])

        with patch.object(system.extractor, "read", side_effect=extract):
            results = run(system, personas, bank, pipelines)
        self.assertEqual(set(results["arms"]), {f"{r}_{p}" for r in RANKERS for p in ("probing", "no_probing")})
        for arm in results["arms"].values():
            self.assertEqual(arm["summary"]["all"]["n"], 2)
            for session in arm["sessions"]:
                self.assertEqual(session["turns"][-1]["action"], "RECOMMEND")
                self.assertTrue(session["per_turn_best_acceptable_rank"])
                self.assertEqual(set(session["per_turn_best_acceptable_rank"][0]) - {"turn", "observed_types"}, set(RANKERS))
        self.assertIn("fusion_probing_minus_cosine_probing", results["contrasts"])
        for variant in ("fusion_tieavg", "fusion_zscore"):
            self.assertIn(f"{variant}_probing_minus_fusion_probing", results["contrasts"])
            self.assertIn(f"{variant}_no_probing_minus_semantic_no_probing", results["contrasts"])
        self.assertIn("fusion_zscore_bm25_probing_minus_fusion_zscore_probing", results["contrasts"])
        self.assertIn("fusion_zscore_bm25_no_probing_minus_semantic_no_probing", results["contrasts"])
        self.assertIn("| F01 |", report(results))
        subset = run(system, personas[:1], bank, {"fusion": cosine, "fusion_zscore": cosine}, {"probing": False})
        self.assertEqual(set(subset["arms"]), {"fusion_probing", "fusion_zscore_probing"})
        self.assertEqual(set(subset["contrasts"]), {"fusion_zscore_probing_minus_fusion_probing"})
        for p in personas:
            for code in p["acceptable"]:
                self.assertFalse(any(code in text for text in seen))


if __name__ == "__main__":
    unittest.main()
