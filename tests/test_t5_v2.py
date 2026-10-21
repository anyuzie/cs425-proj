import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from pipeline.t5_v2 import T5V2Filler
from pipeline.templates import QuestionBankFiller
from eval.t5_wording import AuditedT5V2


class V2Tests(unittest.TestCase):
    def setUp(self):
        data=Path(__file__).resolve().parents[1]/'data'
        self.filler=T5V2Filler.__new__(T5V2Filler)
        QuestionBankFiller.__init__(self.filler,json.loads((data/'question_bank.json').read_text()),
                                   json.loads((data/'labels.json').read_text())['type_names'])
        self.filler.generate=Mock(return_value='Would you enjoy helping other people learn at work?')

    def test_prompt_has_bounded_explicit_subject_and_context(self):
        payload=dict(slot='S',last_user='I enjoy research.',known_interest='research',index=1)
        self.filler.fill('CLARIFY',payload)
        fields=json.loads(self.filler.generate.call_args.args[0].split(': ',1)[1])
        self.assertEqual(fields['style'],'scenario')
        self.assertEqual(fields['known_interest'],'research')
        self.assertIn('colleague',fields['focus'])
        self.assertEqual(fields['last_user'],payload['last_user'])

    def test_quotes_and_choice_contract_preserved(self):
        payload=dict(slot='R',old_src='I love tools.',new_src='I hate tools.',old_turn=1)
        self.filler.generate.return_value='Which statement is closer to the work you would enjoy?'
        out=self.filler.fill('CHALLENGE',payload)
        self.assertIn(payload['old_src'],out)
        self.assertIn(payload['new_src'],out)
        self.filler.generate.return_value='Would you enjoy practical work as part of your day?'
        self.assertEqual(self.filler.fill('CHALLENGE',payload),QuestionBankFiller.fill(self.filler,'CHALLENGE',payload))

    def test_repeated_retry_falls_back_without_changing_action(self):
        payload=dict(slot='S',retry='CLARIFY',previous_question=self.filler.generate.return_value)
        self.assertEqual(self.filler.fill('ABSTAIN',payload),QuestionBankFiller.fill(self.filler,'ABSTAIN',payload))

    def test_prompt_field_leakage_is_not_displayed(self):
        self.filler.generate.return_value='Direct or last_user: Would you like leading a team at work?'
        payload={'slot':'E'}
        self.assertEqual(self.filler.fill('CLARIFY',payload),QuestionBankFiller.fill(self.filler,'CLARIFY',payload))

    def test_style_prefix_and_probe_without_experience_are_rejected(self):
        payload={'slot':'S','new_src':'I studied teaching.'}
        self.filler.generate.return_value='Direct, would you enjoy teaching people at work?'
        self.assertEqual(self.filler.fill('CLARIFY',payload),QuestionBankFiller.fill(self.filler,'CLARIFY',payload))
        self.filler.generate.return_value='Would you enjoy teaching people throughout your working day?'
        self.assertEqual(self.filler.fill('PROBE',payload),QuestionBankFiller.fill(self.filler,'PROBE',payload))

    def test_v3_does_not_mix_background_with_question_target(self):
        from pipeline.t5_v3 import T5V3Filler
        filler=T5V3Filler.__new__(T5V3Filler)
        QuestionBankFiller.__init__(filler,self.filler.qb,self.filler.type_names)
        fields=filler.fields('CLARIFY',dict(slot='A',known_interest='persuading people',last_user='I like sales.'))
        self.assertEqual(fields['known_interest'],'')
        self.assertEqual(fields['last_user'],'')
        self.assertIn('creating',fields['focus'])
        retry=filler.fields('ABSTAIN',dict(slot='A',retry='CLARIFY',question_style=fields['style']))
        self.assertNotEqual(fields['style'],retry['style'])
        self.assertEqual(fields['focus'],retry['focus'])

    def test_opening_and_recommendation_bypass_model(self):
        self.filler.fill('OPENING',{})
        self.filler.fill('RECOMMEND',{})
        self.filler.generate.assert_not_called()

    def test_audit_uses_v2_prompt_and_preserves_v2_name(self):
        self.assertTrue(issubclass(AuditedT5V2,T5V2Filler))
        self.assertEqual(AuditedT5V2.name,'t5-v2')

    def test_system_passes_short_context_from_accepted_positive_state(self):
        import fakes
        from pipeline.system import System
        from pipeline.tracker import DecayTracker
        from pipeline.policy import ManualPolicy
        from pipeline.interfaces import State, Reading, SlotReading
        data=Path(__file__).resolve().parents[1]/'data'
        cfg=json.loads((Path(__file__).parent/'test_config.json').read_text())
        system=System.from_config(data,cfg)
        system.tracker=DecayTracker();system.policy=ManualPolicy()
        system.filler.uses_context=True
        state=State();system.start(state)
        text='I enjoy research. '+ 'detail '*50
        reading=Reading(text,[SlotReading('I',1,.8,.9,'enjoyment','I enjoy research.'),
                              SlotReading('E',-1,.9,.9,'enjoyment','I dislike sales.')])
        with patch.object(system.extractor,'read',return_value=reading):
            system.step(state,text)
        self.assertEqual(state.last_action.payload['known_interest'],'research')
        self.assertEqual(state.last_action.payload['last_user'],text[:240])

    def test_familiarity_is_not_presented_as_an_enjoyed_interest(self):
        import fakes
        from pipeline.system import System
        from pipeline.tracker import DecayTracker
        from pipeline.policy import ManualPolicy
        from pipeline.interfaces import State, Reading, SlotReading
        data=Path(__file__).resolve().parents[1]/'data'
        cfg=json.loads((Path(__file__).parent/'test_config.json').read_text())
        system=System.from_config(data,cfg)
        system.tracker=DecayTracker();system.policy=ManualPolicy(no_probing=True)
        system.filler.uses_context=True
        state=State();system.start(state)
        text='My course involved research.'
        item=Reading(text,[SlotReading('I',1,.8,.9,'familiarity',text)])
        with patch.object(system.extractor,'read',return_value=item):
            system.step(state,text)
        self.assertEqual(state.last_action.payload['known_interest'],'')


if __name__=='__main__':unittest.main()
