"""Simulator routing and paired development evaluation without model downloads."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import fakes  # noqa: F401
from eval.dialogue import DialogueUser, compare
from pipeline.interfaces import Action, Reading, SlotReading
from pipeline.system import System

DATA = Path(__file__).resolve().parents[1] / "data"


class DialogueEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.bank = json.loads((DATA/'question_bank.json').read_text())
        self.profile = dict(id='unit',text='opening preference',target='15-1221.00',
                            hidden_slots=dict.fromkeys('RIASEC',1),related={},
                            resolutions=dict.fromkeys('RIASEC','I enjoy this work.'),
                            planted=['familiarity:Enterprising'],latent_reveal='Genuine hidden preference.')

    def test_retry_preserves_original_action_and_reveal_is_scoped(self):
        user = DialogueUser(self.profile,self.bank)
        self.assertEqual(user.reply(Action('PROBE','E')), 'Genuine hidden preference.')
        self.assertNotEqual(user.reply(Action('PROBE','S')), 'Genuine hidden preference.')
        self.assertEqual(user.reply(Action('ABSTAIN','E',payload={'retry':'PROBE'})), 'Genuine hidden preference.')
        self.assertEqual(user.reply(Action('ABSTAIN','E',payload={'retry':'CHALLENGE'})), 'I enjoy this work.')

    def test_clarify_retry_uses_asked_variant(self):
        user = DialogueUser(self.profile,self.bank)
        expected = self.bank['ANSWERS']['Investigative'][1]['pos']
        self.assertEqual(user.reply(Action('ABSTAIN','I',payload={'retry':'CLARIFY','clarify_index':1})),expected)

    def test_both_arms_terminate_and_annotations_do_not_enter_extractor(self):
        config=json.loads((Path(__file__).parent/'test_config.json').read_text())
        config['components'].update(tracker='decay',policy='manual',ranker='structured')
        system=System.from_config(DATA,config)
        seen=[]
        def extract(text,*args):
            self.assertIsInstance(text,str)
            seen.append(text)
            return Reading(text,[SlotReading(t,1,.55,.9,'enjoyment',text) for t in 'RIASEC'])
        with patch.object(system.extractor,'read',side_effect=extract):
            result=compare(system,[self.profile],self.bank)
        self.assertEqual(set(result['arms']),{'full','no_probing'})
        for arm in result['arms'].values():
            session=arm['sessions'][0]
            self.assertEqual(session['turns'][-1]['action'],'RECOMMEND')
            self.assertEqual(len(session['recommendations']),5)
            self.assertLessEqual(len(session['turns']),10)
            self.assertEqual(arm['summary']['profile_count'],1)
        self.assertFalse(system.policy.no_probing)
        self.assertNotIn(self.profile['target'],seen)
