"""Acceptance expectations must report model failures rather than hide them."""
import json
from pathlib import Path
import unittest

from eval.scenario_tests import expectation_errors


class ScenarioTests(unittest.TestCase):
    def result(self, value=None, **updates):
        return dict(action='CLARIFY', done=False, recommendations=None,
                    trace={'profile': {'I': {'value': value}}}, **updates)

    def test_expected_sign_and_neutral_failure_are_detected(self):
        self.assertFalse(expectation_errors(self.result(.5), {'slot':'I','sign':1}))
        self.assertTrue(expectation_errors(self.result(-.5), {'slot':'I','sign':1}))
        self.assertTrue(expectation_errors(self.result(), {'slot':'I','sign':1}))
        self.assertFalse(expectation_errors(self.result(), {'empty_profile':True}))
        self.assertTrue(expectation_errors(self.result(.5), {'empty_profile':True}))

    def test_action_completion_and_recommendations_are_checked(self):
        result = self.result()
        expected = {'action':'ABSTAIN','done':True,'recommendations':5}
        self.assertEqual(len(expectation_errors(result, expected)), 3)
        self.assertTrue(expectation_errors(result, {'not_action':'CLARIFY'}))
        result.update(action='ABSTAIN',done=True,recommendations=[{}]*5)
        self.assertFalse(expectation_errors(result, expected))
        self.assertTrue(expectation_errors(result, {'no_recommendations':True}))

    def test_fixture_ids_and_messages_are_valid(self):
        cases = json.loads((Path(__file__).parent/'conversation_cases.json').read_text(encoding='utf-8'))
        self.assertEqual(len(cases), len({c['id'] for c in cases}))
        for case in cases:
            self.assertTrue(case['manual'])
            self.assertTrue(1 <= len(case['messages']) <= 10)
            self.assertTrue(all(1 <= len(m.strip()) <= 2000 for m in case['messages']))


if __name__ == '__main__':
    unittest.main()
