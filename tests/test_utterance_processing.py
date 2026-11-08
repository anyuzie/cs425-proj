import unittest
from copy import deepcopy
from recommender.utterance import UtteranceProcessingPipeline
from recommender.schemas import RankingRequest

class UtteranceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = UtteranceProcessingPipeline()

    def texts(self, text):
        result = self.pipeline.split(text)
        for clause in result.clauses:
            self.assertEqual(clause.text, text[clause.start:clause.end])
        return [c.text for c in result.clauses]

    def test_coordination_and_subject(self):
        result = self.pipeline.split("I repair machines and install electrical equipment.")
        self.assertEqual([c.text for c in result.clauses], ["I repair machines", "install electrical equipment"])
        self.assertEqual(result.clauses[1].inherited_subject, "I")

    def test_objects_complements_and_context_stay(self):
        for text in ["I work with patients and families", "I want to teach children", "I repair machines outdoors"]:
            self.assertEqual(self.texts(text), [text])

    def test_sentences_fragments_empty(self):
        self.assertEqual(self.texts("I repair machines. I teach children."), ["I repair machines", "I teach children"])
        self.assertEqual(self.texts("Outdoor work"), ["Outdoor work"])
        self.assertEqual(self.texts("  "), [])

    def test_negation(self):
        self.assertEqual(self.texts("I do not repair machines, but I teach children."), ["I do not repair machines", "I teach children"])

    def test_combined_input_is_unchanged(self):
        request = RankingRequest(dict(R=1., I=0., A=0., S=-.5, E=0., C=0.), {"R", "S"}, ["I repair machines."])
        before = deepcopy(request)
        results = self.pipeline.process(request)
        self.assertEqual(request, before)
        self.assertEqual(results[0].clauses[0].text, "I repair machines")

    def test_missing_parser_rejected(self):
        import spacy
        with self.assertRaises(ValueError):
            UtteranceProcessingPipeline(nlp=spacy.blank("en"))

    def test_classification_outcomes_and_offsets(self):
        examples = {
            "I repair machines": (True, False, "activity"),
            "I prefer outdoor work": (False, True, "job nature"),
            "I repair machines outdoors": (True, True, "mixed"),
            "Something technical": (False, False, "uncertain"),
            "I help people become independent": (True, False, "activity"),
            "I dislike repairing machines": (True, False, "activity"),
            "I do not repair machines": (True, False, "activity"),
        }
        for text, expected in examples.items():
            with self.subTest(text=text):
                clause = self.pipeline.split(text).clauses[0]
                self.assertEqual((clause.activity_detected, clause.job_nature_detected, clause.outcome), expected)
                for evidence in clause.activity_evidence + clause.job_nature_evidence:
                    self.assertEqual(text[evidence.start:evidence.end], evidence.text)

    def test_labels_stay_with_corresponding_clause(self):
        result = self.pipeline.split("I repair machines. I prefer outdoor work.")
        self.assertEqual([c.outcome for c in result.clauses], ["activity", "job nature"])

    def test_caregiving_and_noise_regression(self):
        text = "i like taking care of the elderly and kids, but high noise level over stimulates me"
        clauses = self.pipeline.split(text).clauses
        self.assertTrue(any(c.activity_detected for c in clauses))
        self.assertTrue(any(c.job_nature_detected for c in clauses))
        activity = [e.text for c in clauses for e in c.activity_evidence]
        nature = [e.text for c in clauses for e in c.job_nature_evidence]
        self.assertIn("taking care of the elderly and kids", activity)
        self.assertIn("high noise level", nature)
        for clause in clauses:
            for evidence in clause.activity_evidence + clause.job_nature_evidence:
                self.assertEqual(text[evidence.start:evidence.end], evidence.text)

    def test_multiword_activity_not_generic_take(self):
        for text in ["I take care of patients", "I look after children"]:
            self.assertTrue(self.pipeline.split(text).clauses[0].activity_detected)
        self.assertFalse(self.pipeline.split("I take a break").clauses[0].activity_detected)
