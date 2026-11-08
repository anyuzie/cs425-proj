"""Contracts for sentence-to-task nearest-neighbour reranking."""
from copy import deepcopy
from pathlib import Path
import unittest

import numpy as np

from recommender.models.description_reranker import VARIANTS
from recommender.models.ranker import RankingPipeline
from recommender.schemas import OracleInput
from recommender.semantic.clause_tasks import (BASE_POLICY, CONFIGURATIONS, SIGNED_POLICY,
                                               ClauseTaskReranker, split_sentences)

CODE = "11-1011.00"


class MappedEncoder:
    """Returns fixed unit vectors per text; unknown text maps to axis 0."""

    def __init__(self, mapping=None, dimension=384):
        self.mapping, self.dimension, self.calls = mapping or {}, dimension, []

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        vectors = np.zeros((len(texts), self.dimension), dtype="float32")
        for row, text in enumerate(texts):
            vectors[row, self.mapping.get(text, 0)] = 1.
        return vectors


def unit(*pairs, dimension=384):
    vector = np.zeros(dimension, dtype="float32")
    for index, value in pairs:
        vector[index] = value
    return vector / np.linalg.norm(vector)


class ClauseTaskTests(unittest.TestCase):
    def setUp(self):
        self.config = {"components": {"m1": {"algorithm": "weighted_dot_product", "variant": "baseline_v1"},
                                      "m3": None, "m4": None}, "fusion": None}
        self.inputs = OracleInput(dict(R=1., I=.5, A=0., S=-.5, E=0., C=.1),
                                  ["I like fixing code. I hate selling things!"])

    def scorer(self, policy=BASE_POLICY, mapping=None):
        return ClauseTaskReranker(Path("data/occupations.csv"), {CODE}, parameters=VARIANTS["blocks10_v1"],
                                  policy=policy, encoder=MappedEncoder(mapping))

    def test_splitter_matches_frozen_m4_splitter(self):
        text = ["I like code.  Really!\nmaybe not?", "...", "plain clause"]
        self.assertEqual(split_sentences(text), ["I like code.", "Really!", "maybe not?", "plain clause"])
        try:
            from recommender.models.dense_ranker import sentences
        except ImportError:
            self.skipTest("faiss unavailable")
        self.assertEqual(split_sentences(text), sentences(text))

    def test_max_over_tasks_then_mean_over_sentences(self):
        scorer = self.scorer(mapping={"I like fixing code.": 1, "I hate selling things!": 2})
        tasks = len(scorer.tasks[CODE])
        matrix = np.zeros((tasks, 384), dtype="float32")
        matrix[:, 3] = 1.                       # irrelevant tasks
        matrix[1] = unit((1, .9), (3, .1))     # strong match for sentence 1
        matrix[2] = unit((2, .5), (3, .5))     # weaker match for sentence 2
        scorer.task_cache[CODE] = matrix
        scores = scorer.score([CODE], self.inputs.user_sentences)
        first, second = float(matrix[1, 1]), float(matrix[2, 2])
        self.assertAlmostEqual(scores[CODE], (first + second) / 2, places=6)
        matches = scorer.last_matches[CODE]
        self.assertEqual([m["task_id"] for m in matches], list(scorer.tasks[CODE])[1:3])
        self.assertEqual([m["sign"] for m in matches], [1, 1])

    def test_signed_policy_flips_only_explicit_negatives(self):
        mapping = {"I like fixing code.": 1, "I hate selling things!": 1, "Maybe I don't mind sales.": 1}
        signed = self.scorer(SIGNED_POLICY, mapping)
        signed.task_cache[CODE] = np.tile(unit((1, 1.)), (len(signed.tasks[CODE]), 1))
        text = ["I like fixing code. I hate selling things! Maybe I don't mind sales."]
        self.assertAlmostEqual(signed.score([CODE], text)[CODE], (1 - 1 + 1) / 3, places=6)
        self.assertEqual([m["sign"] for m in signed.last_matches[CODE]], [1, -1, 1])
        unsigned = self.scorer(BASE_POLICY, mapping)
        unsigned.task_cache[CODE] = signed.task_cache[CODE]
        self.assertAlmostEqual(unsigned.score([CODE], text)[CODE], 1., places=6)

    def test_only_raw_task_text_and_sentences_are_encoded(self):
        scorer = self.scorer()
        scorer.score([CODE], [" one.  two "])
        self.assertEqual(scorer.encoder.calls[0], list(scorer.tasks[CODE].values()))
        self.assertEqual(scorer.encoder.calls[1], ["one.", "two"])
        scorer.score([CODE], ["again"])
        self.assertEqual(len(scorer.encoder.calls), 3)  # task vectors reused
        self.assertEqual(scorer.score([CODE], ["  ...  "]), {})
        with self.assertRaises(ValueError):
            scorer.score(["not_eligible"], ["query"])
        with self.assertRaises(ValueError):
            ClauseTaskReranker(Path("data/occupations.csv"), {CODE}, parameters=VARIANTS["blocks10_v1"],
                               policy={**BASE_POLICY, "preference": "other"})

    def test_all_variants_keep_groups_tail_and_structured_signals(self):
        before = RankingPipeline(config=self.config).rank_oracle(self.inputs)
        codes = [o.code for o in before.occupations]
        for name, parameters in CONFIGURATIONS.items():
            with self.subTest(name=name):
                pipeline = RankingPipeline(config={**self.config, "reranking": parameters})
                stage = pipeline._rerankers[0]
                self.assertIsInstance(stage, ClauseTaskReranker)
                self.assertEqual(stage.policy, parameters["text_policy"])
                self.assertEqual(stage.parameters["block_size"], 10)
                calls = []

                def score(candidates, utterances):
                    calls.append(list(candidates))
                    return {code: float(i) for i, code in enumerate(candidates)}
                stage.score = score
                after = pipeline.rank_oracle(self.inputs)
                ordered = [o.code for o in after.occupations]
                self.assertEqual(calls, [codes[:100]])
                self.assertEqual(ordered[100:], codes[100:])
                for start in range(0, 100, 10):
                    self.assertEqual(ordered[start:start + 10], list(reversed(codes[start:start + 10])))
                signals = {o.code: o.signals["structured"] for o in before.occupations}
                self.assertTrue(all(o.signals["structured"] == signals[o.code] for o in after.occupations))
                self.assertEqual(pipeline.resolved_config["reranking"], parameters)
                broken = deepcopy(parameters)
                broken["text_policy"]["precomputed_embeddings"] = True
                with self.assertRaises(ValueError):
                    RankingPipeline(config={**self.config, "reranking": broken})

    def test_reranker_diagnostics(self):
        from recommender.experiments.clause_tasks import reranker_diagnostics
        control = [{"profile_id": p, "target_rank": r} for p, r in
                   (("a", 7), ("b", 3), ("c", 15), ("d", 250), ("e", None))]
        treatment = [{"profile_id": p, "target_rank": r} for p, r in
                     (("a", 2), ("b", 9), ("c", 11), ("d", 250), ("e", None))]
        d = reranker_diagnostics(control, treatment)
        self.assertEqual(d["targets_in_reranked_top100"], 3)
        self.assertAlmostEqual(d["hit_at_5_ceiling"], 2 / 5)
        self.assertAlmostEqual(d["first_block_top5_conversion"], .5)
        self.assertAlmostEqual(d["mean_within_block_position"], (2 + 9 + 1) / 3)
        self.assertEqual((d["target_moved_up"], d["target_moved_down"], d["target_unchanged"]), (2, 1, 0))
        self.assertAlmostEqual(d["mean_rank_change"], (5 - 6 + 4) / 3)

    def test_variant_identities_are_distinct_and_frozen(self):
        names = [p["variant"] for p in CONFIGURATIONS.values()]
        self.assertEqual(len(set(names)), 4)
        self.assertEqual(CONFIGURATIONS["minilm_groups10"]["model_id"], VARIANTS["blocks10_v1"]["model_id"])
        self.assertEqual(CONFIGURATIONS["mpnet_groups10_signed"]["text_policy"]["preference"],
                         "explicit_negative_sign_v1")


if __name__ == "__main__":
    unittest.main()
