"""Sequential stages partition the previous order, preserving both score traces."""
from copy import deepcopy
import unittest
import numpy as np

from recommender.models.ranker import RankingPipeline, BASELINE_CONFIG
from recommender.models.sequential_description_config import SEQUENTIAL_PARAMETERS
from recommender.schemas import OracleInput


class Encoder:
    def __init__(self, scores, dimension):
        self.scores, self.dimension, self.calls = scores, dimension, []

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        values = np.zeros((len(texts), self.dimension), dtype="float32")
        for i, text in enumerate(texts):
            score = self.scores.get(text, 1.)
            values[i, 0], values[i, 1] = score, np.sqrt(1 - score * score)
        return values


class SequentialTests(unittest.TestCase):
    def pipeline(self):
        config = deepcopy(BASELINE_CONFIG)
        config.update(fusion=None, reranking=deepcopy(SEQUENTIAL_PARAMETERS))
        return RankingPipeline(config=config)

    def test_order_boundaries_traces_cache_and_display_limit(self):
        inputs = OracleInput(dict(R=1., I=.5, A=0., S=-.5, E=0., C=.1), ["query"])
        baseline = RankingPipeline().rank_oracle(inputs)
        before = [o.code for o in baseline.occupations]
        pipeline = self.pipeline()
        mpnet, minilm = pipeline._rerankers
        scores1 = {mpnet.descriptions[c]: i / 101 for i, c in enumerate(before[:100])}
        scores2 = {minilm.descriptions[c]: -i / 101 for i, c in enumerate(before[:100])}
        mpnet.encoder, minilm.encoder = Encoder(scores1, 768), Encoder(scores2, 384)
        def order(codes, size, scores):
            return [c for start in range(0, len(codes), size)
                    for c in sorted(codes[start:start+size], key=lambda c: (-scores[mpnet.descriptions[c]], c))]
        intermediate = order(before[:100], 20, scores1)
        expected = order(intermediate, 10, scores2) + before[100:]
        result = pipeline.rank_oracle(inputs)
        after = [o.code for o in result.occupations]
        self.assertEqual(after, expected)
        self.assertNotEqual(after[:100], order(before[:100], 10, scores2))
        for start in range(0, 100, 10):
            self.assertEqual(set(after[start:start+10]), set(intermediate[start:start+10]))
        originals = {o.code: o.signals["structured"] for o in baseline.occupations}
        for occupation in result.occupations[:100]:
            self.assertEqual(occupation.signals["structured"], originals[occupation.code])
            self.assertEqual(occupation.signals["stage_mpnet"].rank, intermediate.index(occupation.code)+1)
            self.assertEqual(occupation.signals["stage_minilm"].rank, occupation.rank)
            self.assertEqual(occupation.signals["semantic"], occupation.signals["semantic_minilm"])
            self.assertIn("semantic_mpnet", occupation.signals)
        self.assertTrue(all(set(o.signals) == {"structured"} for o in result.occupations[100:]))
        short = pipeline.rank(inputs.interests, set(inputs.interests), inputs.user_sentences, 3)
        self.assertEqual([o.code for o in short.occupations], expected[:3])
        self.assertEqual(len(mpnet.encoder.calls), 3)
        self.assertEqual(len(minilm.encoder.calls), 3)
        self.assertEqual(pipeline.resolved_config["reranking"], SEQUENTIAL_PARAMETERS)
        self.assertEqual(len(pipeline.component_metadata["reranking"]["stages"]), 2)
        # The semantic pipeline is callable without any RIASEC values or signals.
        opaque = deepcopy(baseline)
        for occupation in opaque.occupations:
            occupation.occupation_z = {}
            occupation.signals = {}
        independent = pipeline._semantic_pipeline.rerank(opaque, inputs.user_sentences)
        self.assertEqual([o.code for o in independent.occupations], expected)

    def test_blank_and_absent_structured_signal_skip_both_encoders(self):
        pipeline = self.pipeline()
        class NeverEncoder:
            def encode(self, *args, **kwargs):
                raise AssertionError("Encoder should not be invoked")
        for stage in pipeline._rerankers:
            stage.encoder = NeverEncoder()
        inputs = OracleInput(dict.fromkeys("RIASEC", 1.), [" "])
        before = RankingPipeline().rank_oracle(inputs)
        after = pipeline.rank_oracle(inputs)
        self.assertEqual(before, after)
        self.assertEqual(pipeline.rank_oracle(OracleInput(dict.fromkeys("RIASEC", 0.), ["query"])).status, "no_signal")

    def test_modified_or_reversed_stages_rejected(self):
        for mutate in (lambda p: p["stages"].reverse(),
                       lambda p: p["stages"][0].update(block_size=10)):
            params = deepcopy(SEQUENTIAL_PARAMETERS)
            mutate(params)
            config = deepcopy(BASELINE_CONFIG)
            config.update(fusion=None, reranking=params)
            with self.assertRaisesRegex(ValueError, "frozen"):
                RankingPipeline(config=config)


if __name__ == "__main__":
    unittest.main()
