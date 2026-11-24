"""Semantic retrieval without negative clauses."""

from pathlib import Path
import unittest

from recommender.components import ComponentInput
from recommender.data.data_loader import load_eligible_occupations
from recommender.models.occupation_semantic_signed import PARAMETERS, SignedOccupationSemantic
from recommender.models.registry import default_registry
from recommender.occupation_semantic import OccupationSemantic

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT / PARAMETERS["index"]


@unittest.skipUnless((INDEX / "manifest.json").exists(), "fine-tuned semantic index not available locally")
class SignedSemanticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        occupations = load_eligible_occupations(ROOT / "data/occupations.csv")
        cls.signed = SignedOccupationSemantic(occupations, PARAMETERS)
        cls.base = OccupationSemantic(occupations, {"index": PARAMETERS["index"]})

    def query(self, *texts):
        return self.signed.filtered(texts)[0]

    def test_negative_and_skipped_clauses_are_deleted_rest_kept(self):
        self.assertEqual(self.query("i like helping others. but teaching, not as much"), ("i like helping others. but",))
        self.assertEqual(self.query("I love biology, i dont mind. I hate lab work"), ("I love biology, .",))
        self.assertEqual(self.query("i know. no"), ("i know.",))  # deletes the clause, not "no" inside "know"

    def test_history_without_dislikes_is_byte_identical(self):
        texts = ("I study psychology at NUS", "I really enjoy helping people one to one.")
        self.assertEqual(self.query(*texts), texts)
        inputs = ComponentInput(user_sentences=texts)
        self.assertEqual(self.signed.score(inputs), self.base.score(inputs))

    def test_early_stop_when_only_dislikes(self):
        result = self.signed.score(ComponentInput(user_sentences=("I hate teaching", "i dont mind")))
        self.assertEqual(result.status, "unusable")
        self.assertIn("early stop", result.reason)


    def test_diagnostics_list_removed_clauses(self):
        d = self.signed.query_diagnostics(ComponentInput(user_sentences=("i like helping others. but teaching, not as much",)))
        self.assertEqual([c["text"] for c in d["removed_clauses"]], ["teaching, not as much"])


class RegistryTests(unittest.TestCase):
    def test_registered_and_presets_exist(self):
        for variant in ("finetuned_v1", "finetuned_v2"):
            default_registry().resolve("m4", {"algorithm": "occupation_dual_encoder_signed", "variant": variant})
        self.assertTrue((ROOT / "configs/ranking/occupation_semantic_signed_v2_fusion_zscore_bm25_signed_ignore_v3_dislike_title_v3.json").exists())  # promoted default uses finetuned_v2

    def test_rejects_other_parameters(self):
        with self.assertRaises(ValueError):
            SignedOccupationSemantic([], {**PARAMETERS, "index": "elsewhere"})


if __name__ == "__main__":
    unittest.main()
