"""Reverse sequence partitions MiniLM output and reports MPNet as final signal."""
from copy import deepcopy
import unittest

from test_sequential_reranking import Encoder
from recommender.models.ranker import RankingPipeline, BASELINE_CONFIG
from recommender.models.sequential_description_config import REVERSE_PARAMETERS
from recommender.schemas import OracleInput


class ReverseTests(unittest.TestCase):
    def test_reverse_sequence_and_final_alias(self):
        config = deepcopy(BASELINE_CONFIG)
        config.update(fusion=None, reranking=REVERSE_PARAMETERS)
        pipeline = RankingPipeline(config=config)
        inputs = OracleInput(dict(R=1., I=.5, A=0., S=-.5, E=0., C=.1), ["query"])
        baseline = RankingPipeline().rank_oracle(inputs)
        before = [o.code for o in baseline.occupations]
        first, second = pipeline._rerankers
        self.assertEqual([s.parameters["model_id"] for s in pipeline._rerankers],
            ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/all-mpnet-base-v2"])
        scores1 = {first.descriptions[c]: i/101 for i,c in enumerate(before[:100])}
        scores2 = {first.descriptions[c]: -i/101 for i,c in enumerate(before[:100])}
        first.encoder, second.encoder = Encoder(scores1, 384), Encoder(scores2, 768)
        def order(codes, size, scores):
            return [c for start in range(0,len(codes),size) for c in sorted(
                codes[start:start+size],key=lambda c:(-scores[first.descriptions[c]],c))]
        intermediate=order(before[:100],20,scores1)
        expected=order(intermediate,10,scores2)+before[100:]
        response=pipeline.rank_oracle(inputs)
        self.assertEqual([o.code for o in response.occupations],expected)
        self.assertNotEqual(expected[:100],order(before[:100],10,scores2))
        for o in response.occupations[:100]:
            self.assertEqual(o.signals["stage_minilm"].rank,intermediate.index(o.code)+1)
            self.assertEqual(o.signals["stage_mpnet"].rank,o.rank)
            self.assertEqual(o.signals["semantic"],o.signals["semantic_mpnet"])
        short=pipeline.rank(inputs.interests,set(inputs.interests),inputs.user_sentences,3)
        self.assertEqual([o.code for o in short.occupations],expected[:3])
        self.assertEqual(len(first.encoder.calls),3)
        self.assertEqual(len(second.encoder.calls),3)
