from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock
from app.ranking_terminal import show_clause_scores


class TerminalClauseScoresTests(TestCase):
    def test_exact_clauses_and_order(self):
        texts = ["I dislike repairing machinery", "prefer working outdoors"]
        processed = SimpleNamespace(clauses=[SimpleNamespace(text=t) for t in texts])
        scores = [dict(task_score=.8, context_score=.2, route_task=True, route_context=False),
                  dict(task_score=.1, context_score=.9, route_task=False, route_context=True)]
        classifier = Mock()
        classifier.predict.return_value = scores
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(show_clause_scores(processed, classifier), scores)
        classifier.predict.assert_called_once_with(texts)
        self.assertIn("task_score=0.800000 | context_score=0.200000", output.getvalue())
        self.assertLess(output.getvalue().index("1. I dislike"), output.getvalue().index("2. prefer"))

    def test_empty_and_mismatched_predictions(self):
        classifier = Mock()
        with redirect_stdout(StringIO()):
            self.assertEqual(show_clause_scores(SimpleNamespace(clauses=[]), classifier), [])
        classifier.predict.assert_not_called()
        classifier.predict.return_value = []
        with self.assertRaises(ValueError):
            show_clause_scores(SimpleNamespace(clauses=[SimpleNamespace(text="repair machinery")]), classifier)
