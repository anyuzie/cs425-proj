"""L1 signed lexical matching: clause routing, early stop, ignore vs filter (docs/LEXICAL_MATCHING_PLAN.md)."""

from pathlib import Path
import unittest

from recommender.bm25_ranker import BASELINE_PARAMETERS, BM25Ranker
from recommender.components import ComponentInput
from recommender.models.bm25_signed_ranker import (FILTER_PARAMETERS, FILTER_V2_PARAMETERS, IGNORE_PARAMETERS,
                                                  IGNORE_V2_PARAMETERS, SignedBM25Ranker)
from recommender.models.ranker import RankingPipeline
from recommender.models.registry import default_registry
from recommender.schemas import TYPES, Occupation

ROOT = Path(__file__).resolve().parents[1]
DOCS = [("music", "music composer scores harmony"), ("nurse", "nursing patients care hospital"),
        ("therapy", "music therapy patients"), ("sales", "sales selling customers")]


def occupations():
    return [Occupation(code, code, dict.fromkeys(TYPES, 0), text) for code, text in DOCS]


def ask(ranker, *texts):
    return ranker.score(ComponentInput(user_sentences=texts))


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.ranker = SignedBM25Ranker(occupations(), IGNORE_PARAMETERS)

    def routes(self, text):
        return [(c["source_span"], c["route"]) for c in self.ranker.clauses([text])]

    def test_explicit_neutral_negative_and_skipped(self):
        self.assertEqual(self.routes("I love composing music for films"), [("I love composing music for films", "positive")])
        self.assertEqual({r for _, r in self.routes("I've been writing music since I was twelve")}, {"neutral"})
        self.assertEqual(self.routes("I hate working with patients")[0][1], "negative")
        self.assertEqual(self.routes("I'd rather avoid sales")[0][1], "negative")      # extractor miss, caught by the cue
        self.assertEqual(self.routes("Teaching is not for me")[0][1], "negative")
        self.assertEqual(self.routes("Either is fine, I don't mind")[-1][1], "skipped")
        self.assertEqual(self.routes("not sure, maybe")[0][1], "skipped")
        self.assertEqual(self.routes("It doesn't matter to me")[0][1], "skipped")
        self.assertEqual(self.routes("maybe something with music")[0][1], "neutral")   # uncertain, no negation cue


class ApostropheLessTests(unittest.TestCase):
    CASES = ["i dont want to work with patients", "cant draw at all tho", "teaching isnt for me",
             "i wouldnt enjoy sales", "nah, not really", "dont care much for science stuff"]

    def route(self, parameters, text):
        return [c["route"] for c in SignedBM25Ranker(occupations(), parameters).clauses([text])]

    def test_v2_treats_apostrophe_less_contractions_as_negation(self):
        for parameters in (IGNORE_V2_PARAMETERS, FILTER_V2_PARAMETERS):
            for text in self.CASES:
                with self.subTest(text=text):
                    self.assertNotIn("neutral", self.route(parameters, text))
                    self.assertNotIn("positive", self.route(parameters, text))

    def test_v2_apostrophe_less_non_preference_is_skipped(self):
        for text in ("i dont mind", "doesnt matter to me", "idk"):
            with self.subTest(text=text):
                self.assertEqual(self.route(IGNORE_V2_PARAMETERS, text), ["skipped"])

    def test_v1_unchanged(self):
        self.assertEqual(self.route(IGNORE_PARAMETERS, "cant draw at all tho"), ["neutral"])
        self.assertEqual(self.route(IGNORE_PARAMETERS, "i don't want to work with patients"), ["negative"])

    def test_v2_keeps_positive_words_containing_cue_letters(self):
        # word boundaries: "cantonese", "nothingness"-style substrings must not trigger
        self.assertEqual(self.route(IGNORE_V2_PARAMETERS, "I love teaching cantonese"), ["positive"])


class ScoringTests(unittest.TestCase):
    def test_early_stop_without_positive_or_neutral_clause(self):
        for parameters in (IGNORE_PARAMETERS, FILTER_PARAMETERS):
            result = ask(SignedBM25Ranker(occupations(), parameters), "I hate working with patients")
            self.assertEqual(result.status, "unusable")
            self.assertIn("early stop", result.reason)

    def test_without_negatives_equals_baseline_bm25(self):
        baseline = BM25Ranker(occupations(), BASELINE_PARAMETERS)
        for parameters in (IGNORE_PARAMETERS, FILTER_PARAMETERS):
            signed = SignedBM25Ranker(occupations(), parameters)
            self.assertEqual(ask(signed, "I love music and harmony"), ask(baseline, "I love music and harmony"))

    def test_negative_clause_never_adds_score(self):
        signed = SignedBM25Ranker(occupations(), IGNORE_PARAMETERS)
        result = ask(signed, "I love music", "I don't want to work with patients")
        baseline = BM25Ranker(occupations(), BASELINE_PARAMETERS)
        self.assertEqual(result, ask(baseline, "I love music"))
        self.assertNotIn("nurse", [o.code for o in result.occupations])
        self.assertIn("nurse", [o.code for o in ask(baseline, "I love music", "I don't want to work with patients").occupations])

    def test_filter_removes_occupations_matching_the_dislike_more(self):
        texts = ("I love music", "I hate dealing with patients and therapy")
        ignore = ask(SignedBM25Ranker(occupations(), IGNORE_PARAMETERS), *texts)
        filtered_ranker = SignedBM25Ranker(occupations(), FILTER_PARAMETERS)
        filtered = ask(filtered_ranker, *texts)
        self.assertEqual({o.code for o in ignore.occupations}, {"music", "therapy"})
        self.assertEqual([o.code for o in filtered.occupations], ["music"])
        self.assertEqual(filtered_ranker.last_trace["removed"], ["therapy"])
        self.assertEqual([o.rank for o in filtered.occupations], [1])

    def test_filter_keeps_occupation_matching_likes_more(self):
        result = ask(SignedBM25Ranker(occupations(), FILTER_PARAMETERS), "I love music therapy and harmony", "I hate sales")
        self.assertEqual({o.code for o in result.occupations}, {"music", "therapy"})

    def test_rejects_unregistered_parameters(self):
        with self.assertRaises(ValueError):
            SignedBM25Ranker(occupations(), {**IGNORE_PARAMETERS, "negative_handling": "penalise"})


class RegistryTests(unittest.TestCase):
    def test_variants_registered_and_presets_resolve(self):
        registry = default_registry()
        for variant in ("ignore_negative_v1", "filter_negative_v1", "ignore_negative_v2", "filter_negative_v2", "ignore_negative_v3"):
            self.assertTrue(registry.resolve("m3", {"algorithm": "bm25_signed", "variant": variant}))
        self.assertTrue((ROOT / "configs/ranking/occupation_semantic_signed_v2_fusion_zscore_bm25_signed_ignore_v3_dislike_title_v3.json").exists())  # promoted default uses ignore_negative_v3


if __name__ == "__main__":
    unittest.main()
