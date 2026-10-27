import unittest
from unittest.mock import Mock

from pipeline.interfaces import State, Action
from pipeline.policy import ManualPolicy
from pipeline.repair import needs_repair, repair_question


class RepairTests(unittest.TestCase):
    def test_retry_keeps_facet_and_choice(self):
        state=State(turn=2,last_reading_empty=True,
                    last_action=Action('CLARIFY','S',payload={'clarify_index':1}),
                    clarify_count={'S':2})
        action=ManualPolicy().decide(state)
        self.assertEqual(action.payload['clarify_index'],1)
        self.assertIn('repairing a broken machine',repair_question(
            {'slot':'R','focus':'repairing a broken machine'}))
        self.assertIn('Which statement',repair_question(
            {'slot':'R','retry':'CHALLENGE','old_src':'A','new_src':'B'}))
        self.assertIn('undecided',repair_question({'slot':'S','repair_count':2}))
        self.assertTrue(needs_repair('Leave it undecided for now.'))
        self.assertIn('Imagine',repair_question({'slot':'C','retry':'PROBE','new_src':'I studied accounting.'}))

    def test_system_passes_extractor_reading_to_tracker_unchanged(self):
        import json
        from pathlib import Path
        import fakes
        from pipeline.system import System
        from pipeline.interfaces import Reading, SlotReading
        from pipeline.tracker import DecayTracker
        root=Path(__file__).resolve().parents[1]
        system=System.from_config(root/'data',json.loads((root/'tests/test_config.json').read_text()))
        system.tracker=DecayTracker();system.policy=ManualPolicy()
        state=State(last_action=Action('CLARIFY','S'),last_reply='Would you enjoy teaching?')
        text='Could you give me a concrete example?'
        item=Reading(text,[SlotReading('S',1,.55,.99,'enjoyment',text)])
        system.extractor.read=Mock(return_value=item)
        original=system.tracker.update
        system.tracker.update=Mock(wraps=original)
        system.step(state,text)
        self.assertIs(system.tracker.update.call_args.args[1],item)
        self.assertEqual(state.slots['S'].value,.55)
        # the answer alone already had a stance, so the bot's question is not attached (pipeline/answers.py)
        system.extractor.read.assert_called_once_with(text,None,None)


if __name__=='__main__':unittest.main()
