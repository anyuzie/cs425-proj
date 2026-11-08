"""Verify the audit distinguishes real generation, fallback and deterministic text."""
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from eval.t5_wording import AuditedT5, profiles
from pipeline.templates import QuestionBankFiller


class WordingAuditTests(unittest.TestCase):
    def setUp(self):
        data = Path(__file__).resolve().parents[1]/'data'
        self.filler = AuditedT5.__new__(AuditedT5)
        QuestionBankFiller.__init__(self.filler,json.loads((data/'question_bank.json').read_text()),
                                   json.loads((data/'labels.json').read_text())['type_names'])
        self.filler.tokenizer = Mock(return_value=SimpleNamespace(input_ids=[1,2,3]))
        self.filler.records = []

    def test_generated_template_is_still_model_output(self):
        payload = {'slot':'S'}
        template = QuestionBankFiller.fill(self.filler,'CLARIFY',payload)
        with patch('pipeline.t5.T5Filler.generate',return_value=template):
            self.assertEqual(self.filler.fill('CLARIFY',payload),template)
        self.assertEqual(self.filler.records[-1]['source'],'t5')

    def test_rejected_and_repeated_generation_is_fallback(self):
        for raw,payload in [('a bad response',{'slot':'S'}),
                            ('Would you enjoy helping other people learn?',{'slot':'S','retry':'CLARIFY','previous_question':'Would you enjoy helping other people learn?'})]:
            kind = 'ABSTAIN' if 'retry' in payload else 'CLARIFY'
            with patch('pipeline.t5.T5Filler.generate',return_value=raw):
                output = self.filler.fill(kind,payload)
            self.assertEqual(output,QuestionBankFiller.fill(self.filler,kind,payload))
            self.assertEqual(self.filler.records[-1]['source'],'fallback')

    def test_opening_bypasses_generation(self):
        with patch('pipeline.t5.T5Filler.generate') as generate:
            self.filler.fill('OPENING',{})
            generate.assert_not_called()
        self.assertEqual(self.filler.records[-1]['source'],'deterministic')

    def test_personas_have_complete_bounded_preferences(self):
        people = profiles()
        self.assertEqual(len(people),len({p['id'] for p in people}))
        for person in people:
            self.assertEqual(set(person['intended_preferences']),set('RIASEC'))
            self.assertTrue(all(-1 <= x <= 1 for x in person['intended_preferences'].values()))


if __name__ == '__main__':
    unittest.main()
