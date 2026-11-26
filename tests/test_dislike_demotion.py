"""Routing v3 and the post-fusion dislike stage."""

from pathlib import Path
import unittest

from recommender.components import ComponentRanking, ComponentScore
from recommender.configuration import load_parameters
from recommender.data.data_loader import load_eligible_occupations
from recommender.models.clause_routing import ClauseRouter
from recommender.models.dislike_demotion import VARIANTS, DislikeDemotion, stem, stems_match, validate_post_fusion
from recommender.models.ranker import RankingPipeline
from recommender.schemas import TYPES, Occupation

ROOT = Path(__file__).resolve().parents[1]
V2 = load_parameters("bm25_signed_ignore_negative_v2_parameters")
V3 = load_parameters("bm25_signed_ignore_negative_v3_parameters")


def routes(parameters, text):
    return [(c["source_span"], c["route"]) for c in ClauseRouter(parameters).clauses([text])]


class RoutingV3Tests(unittest.TestCase):
    def test_scope_split_keeps_the_liked_part(self):
        self.assertEqual(routes(V3, "I enjoy cooking, just not baking"), [("I enjoy cooking", "neutral"), ("just not baking", "negative")])
        self.assertEqual(routes(V3, "I like chemistry, except organic chemistry")[1], ("except organic chemistry", "negative"))
        self.assertEqual(routes(V3, "I'm into history just not wars")[0], ("I'm into history", "neutral"))

    def test_no_split_when_tail_is_only_the_cue(self):
        self.assertEqual(routes(V3, "teaching, not as much"), [("teaching, not as much", "negative")])
        self.assertEqual(routes(V3, "I love maths, no doubt"), [("I love maths, no doubt", "positive")])

    def test_anything_but(self):
        self.assertEqual(routes(V3, "Everything but retail. I like books"),
                         [("Everything", "skipped"), ("retail", "negative"), ("I like books", "neutral")])

    def test_avoid_only_in_first_person(self):
        self.assertEqual(routes(V3, "I want to help people avoid debt")[0][1], "neutral")
        self.assertEqual(routes(V3, "I try to avoid public speaking")[0][1], "negative")
        self.assertEqual(routes(V3, "I'd rather avoid meetings")[0][1], "negative")

    def test_v2_unchanged(self):
        self.assertEqual(routes(V2, "I enjoy cooking, just not baking"), [("I enjoy cooking, just not baking", "negative")])
        self.assertEqual(routes(V2, "I want to help people avoid debt")[0][1], "negative")

    def test_pieces_are_substrings_in_order(self):
        text = "I love biology, just not lab work. I hate sales"
        cursor = 0
        for clause in ClauseRouter(V3).clauses([text]):
            cursor = text.index(clause["source_span"], cursor) + len(clause["source_span"])


class StemTests(unittest.TestCase):
    def test_matches(self):
        for a, b in (("teaching", "teachers"), ("nursing", "nurses"), ("judge", "judges"), ("accounting", "accountants"),
                     ("music", "musicians"), ("manage", "managers"), ("sales", "sales")):
            with self.subTest(a=a, b=b):
                self.assertTrue(stems_match(stem(a), stem(b)))

    def test_non_matches(self):
        for a, b in (("cars", "care"), ("construction", "conservation"), ("lab", "labor")):
            with self.subTest(a=a, b=b):
                self.assertFalse(stems_match(stem(a), stem(b)))


def occupation(code, title):
    return Occupation(code, title, dict.fromkeys(TYPES, 0), title)


class TitleStageTests(unittest.TestCase):
    def setUp(self):
        self.occupations = [occupation("01", "Psychology Teachers"), occupation("02", "School Psychologists"),
                            occupation("03", "Registered Nurses"), occupation("04", "Sales Managers")]
        self.stage = DislikeDemotion(self.occupations, VARIANTS["title_v1"])
        self.ranking = ComponentRanking("usable", tuple(ComponentScore(c, s, r) for r, (c, s) in
                                                        enumerate([("01", 3.0), ("02", 2.0), ("03", 1.0), ("04", 0.5)], 1)))

    def test_targets(self):
        self.assertEqual([t["target"] for t in self.stage.targets(["i like helping others. but teaching, not as much"])], ["teaching"])
        self.assertEqual(self.stage.targets(["not a people person", "i dont really want to do that"]), [])

    def test_demotes_below_all_kept_and_keeps_order(self):
        result, diagnostics = self.stage.apply(self.ranking, ["I study psychology. but teaching, not as much. Nursing is not for me"])
        self.assertEqual([o.code for o in result.occupations], ["02", "04", "01", "03"])
        self.assertEqual(diagnostics["demoted"], 2)
        self.assertEqual([o.rank for o in result.occupations], [1, 2, 3, 4])

    def test_no_dislike_leaves_ranking_identical(self):
        result, diagnostics = self.stage.apply(self.ranking, ["I study psychology and love helping people"])
        self.assertIs(result, self.ranking)
        self.assertEqual(diagnostics["demoted"], 0)


class TwoTierTests(unittest.TestCase):
    """v2: mild wording lowers the score; strong wording moves below every other job."""

    def setUp(self):
        self.occupations = [occupation(c, t) for c, t in (("01", "Psychology Teachers"), ("02", "School Psychologists"),
                                                           ("03", "Registered Nurses"), ("04", "Sales Managers"),
                                                           ("05", "Mental Health Counselors"))]
        self.stage = DislikeDemotion(self.occupations, VARIANTS["title_v2"])
        # Teachers far ahead of the rest; counselors just behind school psychologists.
        self.ranking = ComponentRanking("usable", tuple(ComponentScore(c, s, r) for r, (c, s) in enumerate(
            [("01", 10.0), ("02", 3.0), ("03", 2.9), ("05", 2.8), ("04", 0.0)], 1)))

    def test_strength(self):
        grade = lambda text: [t["strength"] for t in self.stage.targets([text])]
        self.assertEqual(grade("but teaching, not as much"), ["mild"])
        self.assertEqual(grade("I'd rather not work with patients"), ["mild"])
        self.assertEqual(grade("Teaching is not for me"), ["strong"])
        self.assertEqual(grade("I hate sales"), ["strong"])
        self.assertEqual(grade("Everything but retail"), ["strong"])

    def test_mild_lowers_score_but_a_big_lead_survives(self):
        result, d = self.stage.apply(self.ranking, ["but teaching, not as much"])
        self.assertEqual(result.occupations[0].code, "01")  # 10 - 1 SD is still first
        self.assertAlmostEqual(result.occupations[0].score, 10.0 - d["mild_penalty"])
        self.assertGreater(d["mild_penalty"], 0)

    def test_strong_moves_below_everything(self):
        result, _ = self.stage.apply(self.ranking, ["Teaching is not for me"])
        self.assertEqual(result.occupations[-1].code, "01")

    def test_mild_can_reorder_close_neighbours(self):
        result, _ = self.stage.apply(self.ranking, ["nursing, not really"])
        codes = [o.code for o in result.occupations]
        self.assertLess(codes.index("05"), codes.index("03"))

    def test_strong_overrides_mild_for_the_same_job(self):
        result, _ = self.stage.apply(self.ranking, ["teaching, not as much", "teaching is not for me"])
        self.assertEqual(result.occupations[-1].code, "01")

    def test_v3_penalty_is_larger(self):
        _, d2 = self.stage.apply(self.ranking, ["but teaching, not as much"])
        _, d3 = DislikeDemotion(self.occupations, VARIANTS["title_v3"]).apply(self.ranking, ["but teaching, not as much"])
        self.assertAlmostEqual(d3["mild_penalty"], 1.5 * d2["mild_penalty"])

    def test_v1_still_binary(self):
        result, _ = DislikeDemotion(self.occupations, VARIANTS["title_v1"]).apply(self.ranking, ["but teaching, not as much"])
        self.assertEqual(result.occupations[-1].code, "01")


class ConfigTests(unittest.TestCase):
    def test_validation(self):
        self.assertEqual(validate_post_fusion({"algorithm": "dislike_demotion", "variant": "title_v1"}), "dislike_demotion@title_v1")
        with self.assertRaises(ValueError):
            validate_post_fusion({"algorithm": "dislike_demotion", "variant": "nope"})

    def test_semantic_variants_are_calibrated(self):
        for name in ("semantic_v1", "title_semantic_v1", "semantic_v2", "title_semantic_v2", "semantic_v3", "title_semantic_v3"):
            self.assertTrue(VARIANTS[name]["calibrated"])
            self.assertIsInstance(VARIANTS[name]["semantic_threshold"], float)

    def test_pipeline_without_post_fusion_unchanged(self):
        from recommender.configuration import load_configuration
        if not (ROOT / "models/occupation_semantics_v1/finetuned_index/manifest.json").exists():
            self.skipTest("semantic index not available locally")
        best, _ = load_configuration(ROOT / "configs/ranking/occupation_semantic_signed_v2_fusion_zscore_bm25_signed_ignore_v3_dislike_title_v3.json")
        base = {k: v for k, v in best.items() if k != "post_fusion"}
        a, b = RankingPipeline(config=base), RankingPipeline(config=best)
        args = ({**dict.fromkeys(TYPES, 0.0), "I": .6, "S": .4}, {"I", "S"}, ["I study psychology and love helping people"], 50)
        self.assertEqual([o.code for o in a.rank(*args).occupations], [o.code for o in b.rank(*args).occupations])
        profile = ({**dict.fromkeys(TYPES, 0.0), "I": .6, "S": .4}, {"I", "S"})
        strong = b.rank(*profile, ["i study psychology", "i like helping others. but teaching is not for me"], 378)
        top10 = [o.title for o in strong.occupations[:10]]
        self.assertFalse(any("Teacher" in t for t in top10), top10)  # strong wording: below every other job
        history = ["i study psychology", "i like helping others. but teaching, not as much"]
        mild, plain = b.rank(*profile, history, 378), a.rank(*profile, history, 378)
        score = lambda r: next(o.rrf_score for o in r.occupations if o.title.startswith("Psychology Teachers"))
        self.assertLess(score(mild), score(plain))  # mild wording: lower score, not removed
        self.assertEqual(mild.query_diagnostics["dislike_demotion"]["targets"][0]["strength"], "mild")


if __name__ == "__main__":
    unittest.main()
