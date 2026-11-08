import unittest
from pathlib import Path
from unittest.mock import Mock
import numpy as np
from recommender.utterance.context_matching import ContextMatchingPipeline, preference, route_clause_pipelines

class ContextTests(unittest.TestCase):
    def test_polarity(self):
        for text, sign in [("I like working outdoors", 1), ("I dislike working outdoors", -1),
                           ("I don't like noise", -1), ("Quiet work", 1),
                           ("I am not sure yet", None), ("I don't dislike outdoors", None),
                           ("I like outdoors but dislike noise", None), ("I do not work outdoors", None)]:
            with self.subTest(text=text):
                self.assertEqual(preference(text)["sign"], sign)
        self.assertEqual(preference("I dislike working outdoors")["target"], "working outdoors")

    def test_projection_and_cached_attributes(self):
        encoder = Mock()
        encoder.encode.side_effect = lambda texts, **kw: np.tile([1., 0.], (len(texts), 1))
        path = Path(__file__).resolve().parents[1] / "data/work_context/onet_31_0/compact/context_dimensions.csv"
        pipeline = ContextMatchingPipeline(path, encoder=encoder)
        positive = pipeline.process("I like outdoors")
        negative = pipeline.process("I dislike outdoors")
        self.assertEqual(positive["context_vector"], [1.]*57)
        self.assertEqual(positive["context_vector"], negative["context_vector"])
        self.assertEqual(negative["preference"]["sign"], -1)
        self.assertEqual(encoder.encode.call_count, 3)
        self.assertEqual(len(pipeline.metadata["dimension_ids"]), 57)

    def test_independent_routes(self):
        task, context = Mock(), Mock()
        for t, c in [(False, False), (True, False), (False, True), (True, True)]:
            task.reset_mock(); context.reset_mock()
            result = route_clause_pipelines("text", dict(task_score=.8 if t else .1, context_score=.8 if c else .1, route_task=t, route_context=c),
                                           task_pipeline=task, context_pipeline=context)
            self.assertEqual(set(result), ({"task"} if t else set()) | ({"context"} if c else set()))
            self.assertEqual(task.process.call_count, int(t))
            self.assertEqual(context.process.call_count, int(c))
        self.assertEqual(route_clause_pipelines("text", dict(task_score=.8, context_score=.8, route_task=True, route_context=True)), {})

    def test_shared_score_floor_and_boundary(self):
        for task_score, context_score, skipped in [(.29, .29, True), (.0, .299999, True),
                                                   (.3, .1, False), (.1, .3, False), (.1, .9, False)]:
            task, context = Mock(), Mock()
            result = route_clause_pipelines("text", dict(task_score=task_score, context_score=context_score,
                route_task=True, route_context=True), task_pipeline=task, context_pipeline=context)
            self.assertEqual(result == {}, skipped)
            self.assertEqual(task.process.call_count, int(not skipped))
            self.assertEqual(context.process.call_count, int(not skipped))

    def test_reference_and_informal_negation(self):
        self.assertEqual(preference("i dont really want to do it as my job")["sign"], -1)
        self.assertEqual(preference("I don't really like noise")["sign"], -1)
        pipeline = self.make_projection(.2)
        for text in ["I prefer that environment", "I dont really want to do it as my job"]:
            result = pipeline.process(text)
            self.assertEqual(result["matching_status"], "unresolved_reference")
            self.assertEqual(result["top_contexts"], [])
        result = pipeline.process("It is important to me to work outdoors")
        self.assertTrue(result["matching_usable"])

    def make_projection(self, similarity):
        encoder = Mock()
        def encode(texts, **kwargs):
            if len(texts) == 57:
                return np.tile([1., 0.], (57, 1))
            return np.array([[similarity, np.sqrt(1-similarity**2)] for _ in texts])
        encoder.encode.side_effect = encode
        return ContextMatchingPipeline(Path(__file__).resolve().parents[1] /
            "data/work_context/onet_31_0/compact/context_dimensions.csv", encoder=encoder)

    def test_no_forced_matches_and_cutoff_boundary(self):
        pipeline = self.make_projection(.2)
        for text in ["i like to bake", "i only bake for friends"]:
            self.assertEqual(pipeline.process(text)["top_contexts"], [])
        self.assertEqual(len(self.make_projection(.5).process("a workplace condition")["top_contexts"]), 3)
        self.assertEqual(self.make_projection(.499).process("a workplace condition")["top_contexts"], [])

    def test_office_evidence_and_unsupported_conditions(self):
        pipeline = self.make_projection(.8)
        text = "for work, i just want to be in the office everyday"
        result = pipeline.process(text)
        self.assertEqual([a["context_id"] for a in result["top_contexts"]], ["4.C.2.a.1.a"])
        evidence = result["top_contexts"][0]
        self.assertEqual(text[evidence["source_start"]:evidence["source_end"]], "office")
        for text in ["I like quiet workplaces", "I prefer remote work", "I like outdoors but dislike noise"]:
            self.assertEqual(pipeline.process(text)["top_contexts"], [])
