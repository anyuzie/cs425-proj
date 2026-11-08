"""Hand-calculated occupation-level one-based RRF fixtures."""

import unittest

from recommender.components import ComponentRanking, ComponentScore
from recommender.fusion import ReciprocalRankFusion, select_ranking


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


if __name__ == "__main__":
    unittest.main()
