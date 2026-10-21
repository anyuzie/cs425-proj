import unittest
import json

from eval.t5_compare import legacy_prompt, test_summary


class ComparisonTests(unittest.TestCase):
    def test_legacy_adapter_removes_only_new_fields(self):
        fields=dict(action='PROBE',type='Social',reference='question',context=['original quote'],retry=True,
                    previous_question='previous question',style='scenario',focus='helping people',last_user='hmm',known_interest='research')
        result=json.loads(legacy_prompt(fields).split(': ',1)[1])
        self.assertEqual(result['context'],fields['context'])
        self.assertEqual(result['previous_question'],fields['previous_question'])
        self.assertEqual(result['style'],'direct')
        self.assertNotIn('focus',result)

    def test_summary_separates_surface_validity_and_exact_match(self):
        rows=[dict(action='CLARIFY',surface_valid=True,exact_target=False,repeated_previous=False,raw='A valid paraphrase?'),
              dict(action='ABSTAIN',surface_valid=False,exact_target=False,repeated_previous=True,raw='A repeated question?')]
        result=test_summary(rows)
        self.assertEqual(result['surface_valid'],1)
        self.assertEqual(result['exact_target'],0)
        self.assertEqual(result['repeated_previous'],1)


if __name__=='__main__':unittest.main()
