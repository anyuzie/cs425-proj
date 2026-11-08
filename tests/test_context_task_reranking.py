"""Contracts for raw-text inputs, grouping, preserved RIASEC and full-pool evaluation."""
from copy import deepcopy
from pathlib import Path
import unittest
import numpy as np

from recommender.models.ranker import RankingPipeline
from recommender.schemas import OracleInput
from recommender.semantic.context_tasks import CONFIGURATIONS, ContextTaskReranker, TEXT_POLICY
from recommender.models.description_reranker import VARIANTS


class Tokenizer:
    def encode(self, text, add_special_tokens=True):
        return text.split()

    def decode(self, tokens, **kwargs):
        return " ".join(tokens)


class Encoder:
    tokenizer = Tokenizer()

    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        vectors = np.zeros((len(texts), 384), dtype="float32")
        vectors[:, 0] = 1.
        return vectors


class ContextTaskTests(unittest.TestCase):
    def setUp(self):
        self.config = {"components": {"m1": {"algorithm": "weighted_dot_product", "variant": "baseline_v1"},
                                      "m3": None, "m4": None}, "fusion": None}
        self.inputs = OracleInput(dict(R=1., I=.5, A=0., S=-.5, E=0., C=.1), ["query"])

    def test_raw_text_only_complete_task_coverage_and_chunk_boundaries(self):
        encoder = Encoder()
        scorer = ContextTaskReranker(Path("data/occupations.csv"), {"11-1011.00"},
                                    parameters=VARIANTS["blocks5_v1"], encoder=encoder)
        text = scorer.descriptions["11-1011.00"]
        self.assertIn("Context:\nDetermine and formulate policies", text)
        self.assertIn("Tasks:\n", text)
        self.assertIn("Direct or coordinate an organization's financial", text)
        for excluded in ("z_R", "normalised_rating", "probabilities", "hidden_slots", "target_profile"):
            self.assertNotIn(excluded, text)
        chunks = scorer._chunks(text)
        self.assertEqual(" ".join(chunks).split(), text.split())
        self.assertTrue(all(len(chunk.split()) <= TEXT_POLICY["chunk_tokens"] for chunk in chunks))
        scores = scorer.score(["11-1011.00"], [" query "])
        self.assertAlmostEqual(scores["11-1011.00"], 1.)
        self.assertEqual(encoder.calls[0], chunks)
        self.assertEqual(encoder.calls[1], ["query"])
        self.assertEqual(scorer.score(["11-1011.00"], [" "]), {})
        self.assertEqual(len(encoder.calls), 2)
        with self.assertRaises(ValueError):
            scorer.score(["not_eligible"], ["query"])

    def test_all_six_variants_gate_groups_and_preserve_structured_scores(self):
        before = RankingPipeline(config=self.config).rank_oracle(self.inputs)
        codes = [o.code for o in before.occupations]
        for name, parameters in CONFIGURATIONS.items():
            with self.subTest(name=name):
                pipeline = RankingPipeline(config={**self.config, "reranking": parameters})
                calls = []
                for stage in pipeline._rerankers:
                    def score(candidates, utterances):
                        calls.append(list(candidates))
                        return {code: float(i) for i, code in enumerate(candidates)}
                    stage.score = score
                after = pipeline.rank_oracle(self.inputs)
                ordered = [o.code for o in after.occupations]
                self.assertTrue(all(c == codes[:100] for c in calls[:1]))
                self.assertEqual(ordered[100:], codes[100:])
                stages = parameters.get("stages", [parameters])
                size = 20 if len(stages) == 2 and "weights" not in parameters else parameters["block_size"]
                for start in range(0, 100, size):
                    self.assertEqual(set(ordered[start:start+size]), set(codes[start:start+size]))
                signals = {o.code: o.signals["structured"] for o in before.occupations}
                self.assertTrue(all(o.signals["structured"] == signals[o.code] for o in after.occupations))
                self.assertTrue(all(o.rrf_score is None for o in after.occupations))
                if len(stages) == 2 and "weights" not in parameters:
                    first_size = stages[0]["block_size"]
                    expected = [c for start in range(0, 100, first_size)
                                for c in reversed(codes[start:start+first_size])]
                    self.assertEqual(calls[1], expected)
                broken = deepcopy(parameters)
                broken["text_policy"]["precomputed_embeddings"] = True
                with self.assertRaises(ValueError):
                    RankingPipeline(config={**self.config, "reranking": broken})

    def test_chunk_mean_is_not_renormalized(self):
        scorer = ContextTaskReranker(Path("data/occupations.csv"), {"11-1011.00"},
                                    parameters=VARIANTS["blocks5_v1"], encoder=Encoder())
        # Mean of orthogonal normalized chunks is (.5,.5); its cosine-mean
        # against (1,0) is .5, not the .707 result of renormalizing that mean.
        scorer.cache["11-1011.00"] = np.zeros(384, dtype="float32")
        scorer.cache["11-1011.00"][:2] = .5
        self.assertEqual(scorer.score(["11-1011.00"], ["query"]), {"11-1011.00": .5})


if __name__ == "__main__":
    unittest.main()
