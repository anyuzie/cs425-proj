import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock
from recommender.utterance.context_occupations import ContextOccupationMatcher
from app.ranking_terminal import show_clause_scores

class ContextOccupationTests(unittest.TestCase):
    def test_masks_zero_weighted_evidence_and_order(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            def write(name, rows):
                with (root/name).open("w", newline="", encoding="utf-8") as handle:
                    csv.writer(handle).writerows(rows)
            write("o.csv", [["code", "Title"], ["a", "A"], ["b", "B"], ["c", "C"], ["d", "D"]])
            write("v.csv", [["occupation_code", "x", "y"], ["a", "1", "0"], ["b", "0", "0"], ["c", "", ""], ["d", "1", ""]])
            write("m.csv", [["occupation_code", "x", "y"], ["a", "1", "1"], ["b", "1", "1"], ["c", "0", "0"], ["d", "1", "0"]])
            matcher = ContextOccupationMatcher(root/"v.csv", root/"m.csv", root/"o.csv", ["x", "y"])
            matches = matcher.match({"top_contexts": [dict(context_id="x", context="X", similarity=.75), dict(context_id="y", context="Y", similarity=.25)]})
            self.assertEqual([m["code"] for m in matches], ["d", "a", "b"])
            self.assertEqual(matches[0]["coverage"], .75)
            self.assertEqual(matches[1]["association_score"], .75)
            self.assertEqual(matches[2]["association_score"], 0.)
            self.assertEqual(matcher.match({"top_contexts": []}), [])
            with self.assertRaises(ValueError):
                ContextOccupationMatcher(root/"v.csv", root/"m.csv", root/"o.csv", ["y", "x"])

    def test_per_clause_display_and_negative_interpretation(self):
        classifier, pipeline, matcher = Mock(), Mock(), Mock()
        classifier.predict.return_value = [dict(task_score=.1, context_score=.9, route_task=False, route_context=True),
                                           dict(task_score=.9, context_score=.1, route_task=True, route_context=False)]
        pipeline.process.return_value = dict(preference=dict(sign=-1, polarity="negative", rule="explicit_negative", target="outdoors"), context_vector=[0.]*57, top_contexts=[])
        matcher.match.return_value = [dict(code="a", title="Outdoor occupation", association_score=.8, coverage=1., evidence=[])]
        output = StringIO()
        with redirect_stdout(output):
            show_clause_scores(SimpleNamespace(clauses=[SimpleNamespace(text="I dislike outdoors"), SimpleNamespace(text="repair machinery")]), classifier, pipeline, matcher)
        pipeline.process.assert_called_once_with("I dislike outdoors")
        matcher.match.assert_called_once()
        self.assertIn("potential preference conflicts", output.getvalue())
        self.assertIn("Outdoor occupation", output.getvalue())
