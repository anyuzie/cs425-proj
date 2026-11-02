"""Aggregation uses shared candidates, combines raw scores once, and preserves RIASEC."""
from copy import deepcopy
import unittest

from recommender.models.ranker import RankingPipeline, BASELINE_CONFIG
from recommender.semantic.aggregation_config import AGGREGATED_PARAMETERS
from recommender.schemas import OracleInput


class AggregationTests(unittest.TestCase):
    def pipeline(self):
        config = deepcopy(BASELINE_CONFIG)
        config.update(fusion=None, reranking=deepcopy(AGGREGATED_PARAMETERS))
        return RankingPipeline(config=config)

    def test_shared_inputs_weighted_scores_order_invariance_and_tail(self):
        inputs = OracleInput(dict(R=1., I=.5, A=0., S=-.5, E=0., C=.1), ["query"])
        before = RankingPipeline().rank_oracle(inputs)
        codes = [o.code for o in before.occupations]
        pipeline = self.pipeline()
        values = {"minilm": {c: .9-(i%10)*.1 for i,c in enumerate(codes[:100])},
                  "mpnet": {c: ((i+3)%10)/10-.5 for i,c in enumerate(codes[:100])}}
        calls = []
        for stage in pipeline._rerankers:
            name = stage.parameters["algorithm"].removesuffix("_descriptions")
            def score(candidates, utterances, name=name):
                calls.append((name, list(candidates), list(utterances)))
                return {c: values[name][c] for c in candidates}
            stage.score = score
            stage.rerank = lambda *a, **k: self.fail("Individual models must not reorder candidates")
        combined = {c: .5*values["minilm"][c]+.5*values["mpnet"][c] for c in codes[:100]}
        expected = [c for start in range(0,100,10) for c in sorted(codes[start:start+10],
                    key=lambda c: (-combined[c], c))] + codes[100:]
        response = pipeline.rank_oracle(inputs)
        self.assertEqual([o.code for o in response.occupations], expected)
        self.assertEqual(calls[0][1:], calls[1][1:])
        self.assertEqual(calls[0][1], codes[:100])
        original_signals = {o.code: o.signals["structured"] for o in before.occupations}
        for o in response.occupations[:100]:
            self.assertEqual(o.signals["semantic_aggregate"].score, combined[o.code])
            self.assertEqual(o.signals["semantic"], o.signals["semantic_aggregate"])
            self.assertEqual(o.signals["stage_aggregate"].rank, o.rank)
            self.assertEqual(o.signals["structured"], original_signals[o.code])
            self.assertNotIn("stage_minilm", o.signals)
            self.assertNotIn("stage_mpnet", o.signals)
        self.assertTrue(all(set(o.signals)=={"structured"} for o in response.occupations[100:]))
        pipeline._rerankers.reverse()
        self.assertEqual([o.code for o in pipeline.rank_oracle(inputs).occupations], expected)
        short = pipeline.rank(inputs.interests,set(inputs.interests),inputs.user_sentences,3)
        self.assertEqual([o.code for o in short.occupations],expected[:3])
        partial = deepcopy(before)
        partial.occupations = partial.occupations[:23]
        result = pipeline._semantic_pipeline.rerank(partial, inputs.user_sentences)
        self.assertEqual(set(o.code for o in result.occupations[20:]),set(codes[20:23]))
        self.assertEqual(len(pipeline.component_metadata["reranking"]["stages"]),2)

    def test_blank_and_no_structured_signal_skip_scoring(self):
        pipeline = self.pipeline()
        for stage in pipeline._rerankers:
            stage.score = lambda *a: self.fail("Should not score")
        inputs = OracleInput(dict.fromkeys("RIASEC",1.),[" "])
        self.assertEqual(pipeline.rank_oracle(inputs), RankingPipeline().rank_oracle(inputs))
        self.assertEqual(pipeline.rank_oracle(OracleInput(dict.fromkeys("RIASEC",0.),["query"])).status,"no_signal")

    def test_ties_and_frozen_weights(self):
        pipeline = self.pipeline()
        for stage in pipeline._rerankers:
            stage.score = lambda codes, utterances: dict.fromkeys(codes,-.25)
        inputs = OracleInput(dict.fromkeys("RIASEC",1.),["query"])
        before = RankingPipeline().rank_oracle(inputs)
        after = pipeline.rank_oracle(inputs)
        for start in range(0,100,10):
            self.assertEqual([o.code for o in after.occupations[start:start+10]],
                             sorted(o.code for o in before.occupations[start:start+10]))
        params=deepcopy(AGGREGATED_PARAMETERS)
        params["weights"]["minilm"] = -.5
        config=deepcopy(BASELINE_CONFIG)
        config.update(fusion=None,reranking=params)
        with self.assertRaisesRegex(ValueError,"frozen"):
            RankingPipeline(config=config)
