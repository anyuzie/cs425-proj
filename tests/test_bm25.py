"""BM25 lexical retrieval and integration tests."""

from copy import deepcopy
import csv
import math
from pathlib import Path
import tempfile
import unittest
import json

from recommender.bm25_ranker import BASELINE_PARAMETERS, BM25Ranker, tokenize
from recommender.components import ComponentInput
from recommender.ranker import BASELINE_CONFIG, RankingPipeline
from recommender.schemas import TYPES, Occupation


class BM25Tests(unittest.TestCase):
    def component(self, documents):
        return BM25Ranker([Occupation(code, code, dict.fromkeys(TYPES, 0), text)
                           for code, text in documents], BASELINE_PARAMETERS)

    def test_exact_term_and_hand_computed_score(self):
        ranker = self.component([("repair", "engines engines"), ("nurse", "nursing patients")])
        result = ranker.score(ComponentInput(user_sentences=("ENGINES!",)))
        self.assertEqual([r.code for r in result.occupations], ["repair"])
        # N=2, df=1, tf=2, equal document lengths: IDF=ln(2), TF factor=10/7.
        self.assertAlmostEqual(result.occupations[0].score, math.log(2) * 10 / 7)

    def test_empty_stopword_and_out_of_vocabulary_are_unusable(self):
        ranker = self.component([("a", "engines"), ("b", "patients")])
        for text in ("", "  ! ?", "the and I am", "zzzznonexistent"):
            with self.subTest(text=text):
                result = ranker.score(ComponentInput(user_sentences=(text,)))
                self.assertEqual(result.status, "unusable")
                self.assertFalse(result.occupations)

    def test_stable_ties_and_no_zero_score_padding(self):
        ranker = self.component([("02", "engines"), ("01", "engines"), ("03", "patients")])
        result = ranker.score(ComponentInput(user_sentences=("engines",)))
        self.assertEqual([r.code for r in result.occupations], ["01", "02"])
        self.assertTrue(all(r.score > 0 for r in result.occupations))
        self.assertEqual(result, ranker.score(ComponentInput(user_sentences=("engines",))))

    def test_accumulated_user_text_repetition_and_negation_are_not_filtered(self):
        ranker = self.component([("a", "engines"), ("b", "patients")])
        single = ranker.score(ComponentInput(user_sentences=("engines",)))
        repeated = ranker.score(ComponentInput(user_sentences=("engines", "engines")))
        self.assertEqual(repeated.occupations[0].score, 2 * single.occupations[0].score)
        disliked = ranker.score(ComponentInput(user_sentences=("I dislike engines",)))
        self.assertEqual(disliked.occupations[0].code, "a")  # Documented raw-text limitation.
        self.assertEqual(tokenize("No, not NEVER; don't."), ["no", "not", "never", "don't"])

    def test_missing_documents_fail_only_when_bm25_enabled(self):
        with self.assertRaisesRegex(ValueError, "missing.*bm25_doc"):
            BM25Ranker([Occupation("a", "A", dict.fromkeys(TYPES, 0))], BASELINE_PARAMETERS)
        with self.assertRaisesRegex(ValueError, "no indexable"):
            self.component([("a", "the and")])

    def test_runtime_eligibility_text_only_fallback_and_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "occupations.csv"
            rows = [{"code": code, "Title": code, "job_zone": zone, "bm25_doc": doc,
                     **{f"z_{t}": 1 if t == "R" else 0 for t in TYPES}}
                    for code, zone, doc in (("a", 4, "engines"), ("b", 5, "patients"), ("x", 3, "engines engines"))]
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            config = deepcopy(BASELINE_CONFIG)
            config["components"]["m3"] = {"algorithm": "bm25", "variant": "baseline_v1"}
            pipeline = RankingPipeline(path, config=config)
            zero = dict.fromkeys(TYPES, 0)
            text_only = pipeline.rank(zero, set(), ["engines"], 100)
            self.assertEqual([o.code for o in text_only.occupations], ["a"])
            self.assertIsNone(text_only.occupations[0].rrf_score)
            self.assertEqual(set(text_only.occupations[0].signals), {"lexical"})
            structured_only = pipeline.rank(dict.fromkeys(TYPES, 1), set(TYPES), ["zzzznonexistent"], 100)
            self.assertTrue(all(o.rrf_score is None for o in structured_only.occupations))
            self.assertEqual(pipeline.rank(zero, set(), ["zzzznonexistent"]).status, "no_signal")
            both = pipeline.rank(dict.fromkeys(TYPES, 1), set(TYPES), ["engines"], 1)
            self.assertEqual(len(both.occupations), 1)
            self.assertAlmostEqual(both.occupations[0].rrf_score, 2 / 61)
            self.assertEqual(set(both.occupations[0].signals), {"structured", "lexical"})
            full = pipeline.rank(dict.fromkeys(TYPES, 1), set(TYPES), ["engines"], 100)
            self.assertEqual(len({o.code for o in full.occupations}), len(full.occupations))
            self.assertTrue({o.code for o in full.occupations} <= {"a", "b"})

    def test_evaluator_registers_hybrid_variants_and_real_fusion(self):
        from recommender.evaluate import run
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = run(Path("configs/m2.json"), root / "run", history=root / "history.csv",
                         ranking_config=Path("configs/ranking/m3_hybrid.json"))
            manifest = json.loads((root / "run/manifest.json").read_text())
            self.assertEqual(result["profile_count"], 120)   # dev set grown to 120 profiles
            self.assertEqual(manifest["components"]["m3"], "bm25@baseline_v1")
            self.assertEqual(manifest["components"]["fusion"], "rrf_k60")
            self.assertEqual(manifest["fused_profile_count"], 120)
            self.assertEqual(manifest["component_metadata"]["m3"]["documents"], 378)
            with (root / "history.csv").open(newline="") as handle:
                history = list(csv.DictReader(handle))
            self.assertEqual(history[0]["m3"], "bm25@baseline_v1")
            self.assertEqual(history[0]["fusion"], "rrf_k60")


if __name__ == "__main__":
    unittest.main()
