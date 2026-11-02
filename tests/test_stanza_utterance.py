import unittest
from copy import deepcopy
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import Mock
import torch
from recommender.utterance.stanza_pipeline import StanzaUtteranceProcessingPipeline
from recommender.schemas import RankingRequest
from app.ranking_terminal import show_clause_scores


class StanzaUtteranceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.pipeline = StanzaUtteranceProcessingPipeline()

    def check(self, text, expected):
        result = self.pipeline.split(text)
        self.assertEqual([c.text for c in result.clauses], expected)
        for clause in result.clauses:
            self.assertEqual(clause.text, text[clause.start:clause.end])
        for left, right in zip(result.clauses, result.clauses[1:]):
            self.assertLessEqual(left.end, right.start)
        return result

    def test_user_regression_without_sentence_punctuation(self):
        expected = ["i find it funny when i see someone fall down",
                    "sometimes i feel like that makes me quite evil",
                    "i like to care for people",
                    "i occasionally also volunteer at the old folks home"]
        self.check(", ".join(expected), expected)

    def test_complements_relative_clauses_and_negation(self):
        for text in ["I find it funny when I see someone fall down",
                     "Sometimes I feel like that makes me quite evil",
                     "I want to care for people", "I help people become independent",
                     "I like jobs that allow me to work outdoors",
                     "I do not enjoy repairing machinery and working outdoors",
                     "I don't like caring for people", "I repair machines and install equipment",
                     "I work with patients and families", "Outdoor work"]:
            with self.subTest(text=text):
                self.check(text, [text])

    def test_explicit_subjects_and_sentences(self):
        self.check("I like caring for people but I dislike working nights.",
                   ["I like caring for people", "I dislike working nights"])
        self.check("I repair machines. I work outdoors.", ["I repair machines", "I work outdoors"])
        self.check("I like quiet workplaces but the noise bothers me.",
                   ["I like quiet workplaces", "the noise bothers me"])

    def test_input_and_classifier_contract(self):
        request = RankingRequest(dict(R=1., I=0., A=0., S=0., E=0., C=0.), {"R"}, ["I like caring for people."])
        before = deepcopy(request)
        processed = self.pipeline.process(request)[0]
        self.assertEqual(request, before)
        classifier = Mock()
        classifier.predict.return_value = [dict(task_score=.9, context_score=.2, route_task=True, route_context=False)]
        with redirect_stdout(StringIO()):
            show_clause_scores(processed, classifier)
        classifier.predict.assert_called_once_with(["I like caring for people"])
        self.assertEqual(self.pipeline.split("  ").clauses, ())
        with self.assertRaises(TypeError):
            self.pipeline.split(None)
