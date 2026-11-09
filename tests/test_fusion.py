"""Hand-calculated occupation-level one-based RRF fixtures."""

import unittest

from recommender.components import ComponentRanking, ComponentScore
from recommender.fusion import (FUSION_VARIANTS, ReciprocalRankFusion, TieAwareReciprocalRankFusion,
                                ZScoreSumFusion, average_ranks, build_fusion, fusion_identifier,
                                select_ranking, validate_fusion)

TIE = {"algorithm": "rrf_tie_aware", "variant": "k60_v1"}
ZSCORE = {"algorithm": "zscore_sum", "variant": "equal_v1"}


def ranking(codes):
    return ComponentRanking("usable", tuple(ComponentScore(code, -rank, rank) for rank, code in enumerate(codes, 1)))


class FusionTests(unittest.TestCase):
    def test_hand_calculation_missing_occupations_and_ties(self):
        result = ReciprocalRankFusion().fuse({"m1": ranking(["a", "b", "c"]), "m3": ranking(["b", "a", "d"])})
        self.assertEqual([r.code for r in result.occupations], ["a", "b", "c", "d"])
        self.assertEqual([r.rank for r in result.occupations], [1, 2, 3, 4])
        self.assertAlmostEqual(result.occupations[0].score, 1 / 61 + 1 / 62)
        self.assertAlmostEqual(result.occupations[2].score, 1 / 63)
        self.assertEqual(result.occupations[2].score, result.occupations[3].score)

    def test_reordering_signals_and_raw_score_magnitudes_do_not_change_fusion(self):
        first = {"m1": ranking(["b", "a"]), "m3": ranking(["a", "b"])}
        self.assertEqual(ReciprocalRankFusion().fuse(first), ReciprocalRankFusion().fuse(dict(reversed(list(first.items())))))
        other = ComponentRanking("usable", (ComponentScore("b", 10000, 1), ComponentScore("a", -10000, 2)))
        self.assertEqual(ReciprocalRankFusion().fuse(first), ReciprocalRankFusion().fuse({**first, "m1": other}))

    def test_missing_signals_bypass_fusion(self):
        usable = ranking(["a"])
        absent = ComponentRanking("unusable", reason="no match")
        self.assertEqual(select_ranking({"m1": usable, "m3": absent}, {"algorithm": "rrf", "k": 60}), (usable, False))
        with self.assertRaisesRegex(ValueError, "require fusion"):
            select_ranking({"m1": usable, "m3": usable}, None)
        with self.assertRaises(ValueError):
            ReciprocalRankFusion(0)



def scored(pairs):
    pairs = sorted(pairs, key=lambda p: (-p[1], p[0]))
    return ComponentRanking("usable", tuple(ComponentScore(c, s, r) for r, (c, s) in enumerate(pairs, 1)))


class FusionVariantTests(unittest.TestCase):
    def test_registered_identifiers_and_rejections(self):
        self.assertEqual(fusion_identifier({"algorithm": "rrf", "k": 60}), "rrf_k60")
        self.assertEqual(fusion_identifier(TIE), "rrf_tie_aware@k60_v1")
        self.assertEqual(fusion_identifier(ZSCORE), "zscore_sum@equal_v1")
        self.assertEqual(fusion_identifier(None), "")
        self.assertEqual(len(FUSION_VARIANTS), 3)
        for bad in ({"algorithm": "rrf", "k": 10}, {"algorithm": "rrf", "k": 60.0}, {"algorithm": "rrf", "k": True},
                    {"algorithm": "zscore_sum"}, {"algorithm": "zscore_sum", "variant": "equal_v2"},
                    {**ZSCORE, "weights": [1, 2]}, {"algorithm": "rrf_tie_aware", "variant": "k60_v1", "k": 60}, []):
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "fusion"):
                validate_fusion(bad)
        self.assertIsInstance(build_fusion(TIE), TieAwareReciprocalRankFusion)
        self.assertIsInstance(build_fusion(ZSCORE), ZScoreSumFusion)

    def test_control_output_is_unchanged_through_select_ranking(self):
        rankings = {"m1": scored([("a", 1.0), ("b", 1.0), ("c", .5)]), "m4": scored([("c", .9), ("b", .2), ("a", .1)])}
        result, fused = select_ranking(rankings, {"algorithm": "rrf", "k": 60})
        self.assertTrue(fused)
        self.assertEqual(result, ReciprocalRankFusion().fuse(rankings))

    def test_average_ranks_share_the_mean_position(self):
        self.assertEqual(average_ranks(scored([("a", 1.0), ("b", 1.0), ("c", 1.0), ("d", .2), ("e", .2), ("f", 0)])),
                         {"a": 2, "b": 2, "c": 2, "d": 4.5, "e": 4.5, "f": 6})

    def test_tie_aware_rrf_hand_calculation(self):
        # m1 ties a, b, c (one observed type); the control would order them by code.
        m1, m4 = scored([("a", 1.0), ("b", 1.0), ("c", 1.0), ("d", .1)]), scored([("c", .9), ("d", .5), ("a", .1), ("b", 0)])
        result = TieAwareReciprocalRankFusion().fuse({"m1": m1, "m4": m4})
        expected = {"a": 1 / 62 + 1 / 63, "b": 1 / 62 + 1 / 64, "c": 1 / 62 + 1 / 61, "d": 1 / 64 + 1 / 62}
        self.assertEqual([r.code for r in result.occupations], ["c", "a", "b", "d"])
        for item in result.occupations:
            self.assertAlmostEqual(item.score, expected[item.code])
        # Without ties it equals the control exactly.
        plain = {"m1": ranking(["a", "b", "c"]), "m3": ranking(["b", "a", "d"])}
        self.assertEqual(TieAwareReciprocalRankFusion().fuse(plain), ReciprocalRankFusion().fuse(plain))

    def test_zscore_lets_a_peaked_component_beat_a_flat_lead(self):
        # The section 12.1 shape: flat cosine favours x, peaked semantic favours y; RRF picks the overlap.
        # y is mid-pool under the flat ranking (rank 13 of 43) and far ahead under the peaked one.
        filler = [f"f{i:02d}" for i in range(40)]
        flat = scored([("x", .714), ("o", .70), ("y", .60)] + [(c, .69 - i * .01) for i, c in enumerate(filler)])
        peaked = scored([("y", .50), ("o", .30), ("x", .14)] + [(c, .13 - i * .001) for i, c in enumerate(filler)])
        rankings = {"m1": flat, "m4": peaked}
        self.assertEqual([r.code for r in ReciprocalRankFusion().fuse(rankings).occupations[:2]], ["x", "o"])
        self.assertEqual(ZScoreSumFusion().fuse(rankings).occupations[0].code, "y")

    def test_zscore_hand_calculation_missing_and_degenerate(self):
        a = scored([("p", 3.0), ("q", 1.0)])           # z: p +1, q -1
        b = scored([("p", 0.0), ("q", 2.0), ("r", 4.0)])  # mean 2, std sqrt(8/3)
        sd = (8 / 3) ** .5
        result = {r.code: r.score for r in ZScoreSumFusion().fuse({"m1": a, "m4": b}).occupations}
        self.assertAlmostEqual(result["p"], 1 - 2 / sd)
        self.assertAlmostEqual(result["q"], -1 + 0)
        self.assertAlmostEqual(result["r"], -1 + 2 / sd)  # r missing from m1 gets m1's minimum z
        flat = scored([("p", .5), ("q", .5), ("r", .5)])
        only_b = {r.code: r.score for r in ZScoreSumFusion().fuse({"m1": flat, "m4": b}).occupations}
        self.assertAlmostEqual(only_b["r"], 2 / sd)
        self.assertAlmostEqual(only_b["p"], -2 / sd)
        # Order invariance and scale invariance of a component.
        scaled = scored([("p", 300.0), ("q", 100.0)])
        self.assertEqual([r.code for r in ZScoreSumFusion().fuse({"m1": a, "m4": b}).occupations],
                         [r.code for r in ZScoreSumFusion().fuse({"m4": b, "m1": scaled}).occupations])

    def test_variants_skip_inactive_signals(self):
        absent = ComponentRanking("unusable", reason="no match")
        for fusion in (TieAwareReciprocalRankFusion(), ZScoreSumFusion()):
            self.assertEqual(fusion.fuse({"m1": absent}).status, "unusable")
            self.assertEqual(fusion.fuse({"m1": absent, "m4": ranking(["a", "b"])}).occupations[0].code, "a")


class PipelineFusionWiringTests(unittest.TestCase):
    def test_pipeline_uses_selected_variant_and_reports_its_identifier(self):
        from recommender.models.ranker import RankingPipeline
        base = {"components": {"m1": {"algorithm": "cosine_similarity", "variant": "baseline_v1"},
                               "m3": {"algorithm": "bm25", "variant": "baseline_v1"}, "m4": None}}
        interests = {"R": -.5, "I": .8, "A": .6, "S": -.2, "E": -.4, "C": -.3}
        sentences = ["I love researching music history and composing."]
        orders = {}
        for name, fusion in (("rrf_k60", {"algorithm": "rrf", "k": 60}), ("rrf_tie_aware@k60_v1", TIE),
                             ("zscore_sum@equal_v1", ZSCORE)):
            pipeline = RankingPipeline(config={**base, "fusion": fusion})
            self.assertEqual(pipeline.fusion_identifier, name)
            response = pipeline.rank(interests, set(interests), sentences, 20)
            self.assertEqual(response.status, "ranked")
            self.assertTrue(all(o.rrf_score is not None for o in response.occupations))
            orders[name] = [o.code for o in response.occupations]
        self.assertNotEqual(orders["rrf_k60"], orders["zscore_sum@equal_v1"])
        with self.assertRaisesRegex(ValueError, "fusion"):
            RankingPipeline(config={**base, "fusion": {"algorithm": "zscore_sum", "variant": "tuned"}})


if __name__ == "__main__":
    unittest.main()
