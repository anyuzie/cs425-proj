"""Contract checks for the candidate gate and description-only encoder inputs."""
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from recommender.models.description_reranker import PARAMETERS, VARIANTS
from recommender.models.ranker import RankingPipeline, BASELINE_CONFIG
from recommender.schemas import OracleInput


class Encoder:
    def __init__(self, dimension=384):
        self.calls = []
        self.dimension = dimension

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        vectors = np.zeros((len(texts), self.dimension), dtype="float32")
        vectors[:, 0] = 1
        return vectors


class DescriptionTests(unittest.TestCase):
    def test_model_loader_uses_selected_identity_and_dimension(self):
        for variant in ("blocks10_v1", "mpnet_blocks10_v1"):
            parameters = VARIANTS[variant]
            config = deepcopy(BASELINE_CONFIG)
            config.update(fusion=None, reranking=parameters)
            pipeline = RankingPipeline(config=config)
            encoder = Encoder(parameters["dimension"])
            encoder.eval = lambda: None
            calls = []
            def load(model_id, **kwargs):
                calls.append((model_id, kwargs))
                return encoder
            torch = SimpleNamespace(set_num_threads=lambda n: None, manual_seed=lambda n: None,
                                    use_deterministic_algorithms=lambda value: None)
            with patch.dict("sys.modules", {"torch": torch,
                "sentence_transformers": SimpleNamespace(SentenceTransformer=load)}):
                values = pipeline._reranker._encode(["query"])
                pipeline._reranker._encode(["again"])
            self.assertEqual(values.shape, (1, parameters["dimension"]))
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0], parameters["model_id"])
            self.assertEqual(calls[0][1]["revision"], parameters["model_revision"])
            self.assertEqual(encoder.max_seq_length, 256)

    def test_rank_blocks_cannot_exchange_candidates(self):
        inputs = OracleInput(dict(R=1., I=.5, A=0., S=-.5, E=0., C=.1), ["query"])
        before = [o.code for o in RankingPipeline().rank_oracle(inputs).occupations]
        for variant, size in (("blocks5_v1", 5), ("blocks10_v1", 10), ("mpnet_blocks10_v1", 10)):
            config = deepcopy(BASELINE_CONFIG)
            config.update(fusion=None, reranking=VARIANTS[variant])
            pipeline = RankingPipeline(config=config)
            pipeline._reranker.encoder = Encoder(VARIANTS[variant]["dimension"])
            after = [o.code for o in pipeline.rank_oracle(inputs).occupations]
            for start in range(0, 100, size):
                self.assertEqual(after[start:start+size], sorted(before[start:start+size]))
            self.assertEqual(after[100:], before[100:])
            short = pipeline.rank(inputs.interests, set(inputs.interests), inputs.user_sentences, 3)
            self.assertEqual([o.code for o in short.occupations], after[:3])

    def test_gate_description_only_tail_and_cache(self):
        config = deepcopy(BASELINE_CONFIG)
        config.update(fusion=None, reranking=PARAMETERS)
        pipeline = RankingPipeline(config=config)
        encoder = Encoder()
        pipeline._reranker.encoder = encoder
        inputs = OracleInput(dict(R=1., I=.5, A=0., S=-.5, E=0., C=.1), ["user utterance"])
        baseline = RankingPipeline().rank_oracle(inputs)
        result = pipeline.rank_oracle(inputs)
        before = [o.code for o in baseline.occupations]
        after = [o.code for o in result.occupations]
        self.assertEqual(set(before[:100]), set(after[:100]))
        self.assertEqual(before[100:], after[100:])
        self.assertEqual(after[:100], sorted(before[:100]))
        self.assertEqual(encoder.calls[0], [pipeline._reranker.descriptions[c] for c in before[:100]])
        self.assertEqual(encoder.calls[1], ["user utterance"])
        self.assertTrue(all(o.rrf_score is None for o in result.occupations))
        pipeline.rank_oracle(inputs)
        self.assertEqual(len(encoder.calls), 3)  # Query only on repeat.
        short = pipeline.rank(inputs.interests, set(inputs.interests), inputs.user_sentences, 5)
        self.assertEqual([o.code for o in short.occupations], after[:5])
        inputs.user_sentences = [" "]
        self.assertEqual([o.code for o in pipeline.rank_oracle(inputs).occupations], before)

    def test_other_structured_algorithms_rejected(self):
        config = deepcopy(BASELINE_CONFIG)
        config.update(fusion=None, reranking=PARAMETERS)
        config["components"]["m1"]["algorithm"] = "cosine_similarity"
        with self.assertRaises(ValueError):
            RankingPipeline(config=config)

    def test_invalid_embeddings_fail_and_no_profile_does_not_encode(self):
        config = deepcopy(BASELINE_CONFIG)
        config.update(fusion=None, reranking=PARAMETERS)
        pipeline = RankingPipeline(config=config)
        encoder = Encoder()
        pipeline._reranker.encoder = encoder
        empty = OracleInput(dict.fromkeys("RIASEC", 0.), ["some text"])
        self.assertEqual(pipeline.rank_oracle(empty).status, "no_signal")
        self.assertEqual(encoder.calls, [])
        class BadEncoder:
            def encode(self, texts, **kwargs):
                return np.zeros((len(texts), 384))
        pipeline._reranker.encoder = BadEncoder()
        with self.assertRaisesRegex(ValueError, "normalized"):
            pipeline.rank_oracle(OracleInput(dict.fromkeys("RIASEC", 1.), ["some text"]))


if __name__ == "__main__":
    unittest.main()
