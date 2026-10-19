"""Cosine mathematics, partial observations and registered pipeline behavior."""
from copy import deepcopy
from math import sqrt
from pathlib import Path
import unittest

from recommender.components import ComponentInput
from recommender.cosine import BASELINE_PARAMETERS, CosineSimilarity
from recommender.ranker import RankingPipeline
from recommender.schemas import Occupation, OracleInput, TYPES


def occupation(code, **values):
    return Occupation(code, code, dict.fromkeys(TYPES, 0.0) | values)


def inputs(observed=('R', 'I'), **values):
    return ComponentInput(dict.fromkeys(TYPES, 0.0) | values, frozenset(observed))


class CosineTests(unittest.TestCase):
    def ranker(self, *occupations):
        return CosineSimilarity(list(occupations), deepcopy(BASELINE_PARAMETERS))

    def test_hand_computed_signed_scores_and_code_ties(self):
        ranker = self.ranker(occupation('b', R=2, I=2), occupation('a', R=1, I=1),
                             occupation('c', R=-1, I=0), occupation('d', R=0, I=1))
        result = ranker.score(inputs(R=1))
        self.assertEqual([o.code for o in result.occupations], ['a', 'b', 'd', 'c'])
        for item, expected in zip(result.occupations, [1/sqrt(2), 1/sqrt(2), 0, -1]):
            self.assertAlmostEqual(item.score, expected)

    def test_unknown_and_observed_neutral_affect_norm_differently(self):
        ranker = self.ranker(occupation('a', R=1, I=10, A=100))
        self.assertEqual(ranker.score(inputs(observed=('R',), R=1, I=-1)).occupations[0].score, 1)
        self.assertAlmostEqual(ranker.score(inputs(R=1)).occupations[0].score, 1/sqrt(101))

    def test_zero_vectors_and_missing_profile_have_explicit_status(self):
        ranker = self.ranker(occupation('zero'), occupation('valid', R=1))
        self.assertEqual(ranker.score(ComponentInput()).status, 'unavailable')
        for request in (inputs(), inputs(observed=(), R=1), inputs(observed=('I',), R=1)):
            self.assertEqual(ranker.score(request).status, 'unusable')
        result = ranker.score(inputs(R=1))
        self.assertEqual([o.code for o in result.occupations], ['valid'])
        self.assertEqual(self.ranker(occupation('zero')).score(inputs(R=1)).status, 'unusable')

    def test_scaling_and_sign_reversal(self):
        ranker = self.ranker(occupation('a', R=3, I=4), occupation('b', R=-4, I=3))
        first = ranker.score(inputs(R=0.5, I=-0.25))
        scaled = ranker.score(inputs(R=5, I=-2.5))
        reversed_scores = {o.code: o.score for o in ranker.score(inputs(R=-0.5, I=0.25)).occupations}
        for a, b in zip(first.occupations, scaled.occupations):
            self.assertEqual(a.code, b.code)
            self.assertAlmostEqual(a.score, b.score)
            self.assertAlmostEqual(a.score, -reversed_scores[a.code])

    def test_pool_snapshot_and_parameter_validation(self):
        item = occupation('a', R=1)
        ranker = self.ranker(item)
        before = ranker.score(inputs(R=1))
        item.z_scores['R'] = -1
        self.assertEqual(before, ranker.score(inputs(R=1)))
        with self.assertRaisesRegex(ValueError, 'documented parameters'):
            CosineSimilarity([item], BASELINE_PARAMETERS | {'observed_only': False})

    def test_real_pool_registry_runtime_and_oracle(self):
        ranker = RankingPipeline(config=Path('configs/ranking/m1_cosine.json'))
        vector = dict.fromkeys(TYPES, 0.0) | {'I': 0.5, 'S': 1.0, 'C': -0.5}
        first = ranker.rank(vector, set(TYPES), [], top_k=1000)
        self.assertEqual(first, ranker.rank(vector, set(TYPES), [], top_k=1000))
        self.assertEqual(len(first.occupations), 378)
        self.assertTrue(all(-1 <= o.signals['structured'].score <= 1 and o.rrf_score is None
                            for o in first.occupations))
        self.assertEqual(len(ranker.rank(vector, set(TYPES), [], top_k=5).occupations), 5)
        oracle = ranker.rank_oracle(OracleInput({t: v*3 for t, v in vector.items()}, []))
        self.assertEqual([o.code for o in first.occupations], [o.code for o in oracle.occupations])


if __name__ == '__main__':
    unittest.main()
