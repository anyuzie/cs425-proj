import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import redirect_stdout
from io import StringIO
import unittest
from app.utterance_collection import UtteranceCollection
from pipeline.interfaces import Reading
from recommender.schemas import RankingResponse, RankedOccupation, SignalScore
from recommender.utterance import UtteranceProcessingPipeline
import app.ranking_terminal as terminal


class CollectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.processor = UtteranceProcessingPipeline()

    def test_terminal_success_failure_and_original_input(self):
        response = RankingResponse("ranked", [RankedOccupation(
            "test", "Test occupation", 1, {}, signals={"structured": SignalScore(.5, 1)})])
        ranker = SimpleNamespace(resolved_config={"test": True},
            component_identifiers={"m1": "test"}, eligible_count=1)
        text = "I repair machines outdoors"
        result = (Reading(text), {"R": .5}, {"R"}, response)
        with TemporaryDirectory() as folder:
            output = StringIO()
            with patch.object(terminal, "TaskContextClassifier", return_value=SimpleNamespace(task_threshold=.5, context_threshold=.5, predict=lambda texts: [dict(task_score=.8, context_score=.7, route_task=True, route_context=True) for _ in texts])), patch.object(terminal, "ContextOccupationMatcher", return_value=SimpleNamespace(metadata={}, match=lambda context: [])), patch.object(terminal, "ContextMatchingPipeline", return_value=SimpleNamespace(metadata={"dimension_ids": []}, process=lambda text: {"preference": {"sign": 1, "polarity": "positive", "rule": "test", "target": text}, "context_vector": [0.]*57, "top_contexts": []})), patch.object(terminal, "RankingPipeline", return_value=ranker), patch.object(terminal, "Extractor"), patch.object(terminal, "UtteranceProcessingPipeline", return_value=self.processor), patch.object(terminal, "rank_utterance", side_effect=[result, RuntimeError("inference failed")]), patch("builtins.input", side_effect=["  " + text + "  ", "second entry", "", "x"]), patch("sys.argv", ["terminal", "--clause-splitter", "spacy", "--collection-dir", folder]), redirect_stdout(output):
                self.assertEqual(terminal.main(), 0)
            rows = [json.loads(line) for line in next(Path(folder).glob("*.jsonl")).read_text(encoding="utf-8").splitlines()]
            self.assertEqual([r["event"] for r in rows], ["session", "submitted", "completed", "submitted", "failed"])
            self.assertEqual(rows[1]["raw_text"], "  " + text + "  ")
            self.assertEqual(rows[1]["entry_id"], rows[2]["entry_id"])
            self.assertEqual(rows[3]["entry_id"], rows[4]["entry_id"])
            predictions = rows[2]["predictions"]
            self.assertEqual(predictions["clauses"][0]["outcome"], "mixed")
            self.assertEqual(predictions["ranking"]["occupations"][0]["signals"]["structured"]["score"], .5)
            self.assertEqual(rows[2]["annotation_status"], "unlabelled")
            self.assertIn("task_score=0.800000 | context_score=0.700000", output.getvalue())
            self.assertIn("route_task=true | route_context=true", output.getvalue())
            self.assertLess(output.getvalue().index("Test occupation"), output.getvalue().index("clauses:"))

    def test_sessions_do_not_overwrite_and_pending_survives(self):
        with TemporaryDirectory() as folder:
            first = UtteranceCollection(folder, {})
            first.submitted("pending", "pending")
            second = UtteranceCollection(folder, {})
            self.assertNotEqual(first.path, second.path)
            self.assertEqual(len(first.path.read_text().splitlines()), 2)
