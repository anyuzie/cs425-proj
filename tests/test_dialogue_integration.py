"""Production tracker/policy/filler with controlled perceptions and real ranking."""
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from contracts import DATA, PolicyContract, TrackerContract
import fakes  # noqa: F401; register test-only perceptions
from pipeline.interfaces import Action, Reading, SlotReading, State
from pipeline.policy import ManualPolicy
from pipeline.system import System
from pipeline.templates import QuestionBankFiller
from pipeline.tracker import DecayTracker
from pipeline.t5 import T5Filler


def reading(text="a statement", reason="enjoyment", reason_conf=1.0, **slots):
    return Reading(text, [SlotReading(t, 1 if value >= 0 else -1, abs(value), confidence,
                                     reason, text, reason_conf) for t, (value, confidence) in slots.items()])


class TestProductionTracker(TrackerContract, unittest.TestCase):
    def make(self):
        return DecayTracker()


class TestProductionPolicy(PolicyContract, unittest.TestCase):
    def make(self):
        return ManualPolicy()


class DialogueIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((Path(__file__).parent / 'test_config.json').read_text())
        self.config['components'].update(tracker='decay', policy='manual', ranker='structured')
        self.config['settings']['input_mode'] = 'mock_tracker'
        self.system = System.from_config(DATA, self.config)
        self.state = State()
        self.system.start(self.state)

    def step(self, item):
        with patch.object(self.system.extractor, 'read', return_value=item):
            return self.system.step(self.state, item.text)

    def test_cumulative_merge_and_zero_weight(self):
        self.step(reading(I=(1, .9)))
        self.step(reading(I=(.5, .6)))
        self.step(reading(I=(.2, .3)))
        self.assertAlmostEqual(self.state.slots['I'].value, (1*.9+.5*.6+.2*.3)/1.8)
        self.assertAlmostEqual(self.state.slots['I'].weight, 1.8)
        self.step(reading(R=(.8, 0)))
        self.step(reading(R=(.3, 0)))
        self.assertEqual(self.state.slots['R'].value, .3)

    def test_conflict_persists_retries_resolves_and_can_recur(self):
        self.step(reading('I like teaching', S=(1, .9)))
        result = self.step(reading('I dislike teaching', S=(-1, .9)))
        self.assertEqual(result.action, 'CHALLENGE')
        self.assertEqual(self.state.slots['S'].value, 1)
        result = self.step(reading('hmm'))
        self.assertEqual(result.action, 'ABSTAIN')
        self.assertEqual(self.state.last_action.payload['retry'], 'CHALLENGE')
        self.assertIn('I like teaching', result.reply)
        self.assertIn('I dislike teaching', result.reply)
        self.assertEqual(len(self.state.flags), 1)
        result = self.step(reading('I like research', I=(.5, .8)))
        self.assertEqual(result.action, 'CHALLENGE')
        result = self.step(reading('I prefer not to teach', S=(-.8, .9)))
        self.assertNotEqual(result.action, 'CHALLENGE')
        self.assertFalse(self.state.flags)
        self.assertEqual(self.state.slots['S'].value, -.8)
        self.assertEqual(self.step(reading('I enjoy teaching after all', S=(.8, .9))).action, 'CHALLENGE')

    def test_familiarity_probe_needs_reason_confidence_and_targeted_answer(self):
        # only probe when the reason head is fairly sure it is familiarity (> 0.60)
        result = self.step(reading('I studied sales', reason='familiarity', reason_conf=.6, E=(.7, 0)))
        self.assertNotEqual(result.action, 'PROBE')
        self.state = State()
        self.system.start(self.state)
        result = self.step(reading('I studied sales', reason='familiarity', reason_conf=.9, E=(.7, 0)))
        self.assertEqual(result.action, 'PROBE')
        result = self.step(reading('not sure'))
        self.assertIn('I studied sales', result.reply)
        self.assertEqual(self.state.last_action.payload['retry'], 'PROBE')
        self.step(reading('I would dislike sales', E=(-1, .5), I=(.5, .7)))
        self.assertEqual(self.state.slots['E'].value, -1)
        self.assertIn('I', self.state.slots)
        self.assertFalse(self.state.flags)

    def test_tracker_candidates_use_unchanged_teammate_rule(self):
        for confidence, expected in ((.4, 'CLARIFY'), (.9, 'CHALLENGE')):
            with self.subTest(confidence=confidence):
                self.state=State()
                self.system.start(self.state)
                self.step(reading('I enjoy teaching', S=(1,confidence)))
                result=self.step(reading('I dislike teaching', S=(-1,confidence)))
                self.assertEqual(result.action,expected)

    def test_no_signal_retries_once_per_type_and_stops_at_ten(self):
        actions = []
        for _ in range(10):
            result = self.step(reading('hmm'))
            actions.append((result.action, self.state.last_action.slot))
        self.assertTrue(result.done)
        self.assertFalse(self.state.slots)                  # nothing invented
        self.assertTrue(self.state.settled)                 # unsure types marked as no clear preference
        self.assertIsNone(result.recommendations)           # still no recommendations without a signal
        for (a, slot), (b, nxt) in zip(actions, actions[1:]):
            self.assertFalse(a == b == 'ABSTAIN' and slot == nxt, actions)   # never the same retry twice
        with self.assertRaises(ValueError):
            self.step(reading('more'))

    def test_minimum_turns_and_two_updates_convergence(self):
        full = reading(**{t:(.5,.9) for t in 'RIASEC'})
        for _ in range(4):
            self.assertFalse(self.step(full).done)
        result = self.step(full)
        self.assertEqual(result.action, 'RECOMMEND')
        self.assertEqual(len(result.recommendations), 5)
        self.assertEqual(len(self.state.profile_history), 5)

    def test_variance_selects_unfilled_type(self):
        self.step(reading(I=(.7,.9)))
        from types import SimpleNamespace
        self.state.ranking = [SimpleNamespace(occupation_z=dict.fromkeys('RIASEC',0)),
                              SimpleNamespace(occupation_z={t: 10 if t=='A' else 0 for t in 'RIASEC'})]
        self.assertEqual(self.system.policy.decide(self.state).slot, 'A')

    def test_no_probing_control_skips_both_interventions(self):
        self.system.policy = ManualPolicy(no_probing=True)
        self.assertEqual(self.step(reading(reason='familiarity', E=(1,.8))).action, 'CLARIFY')
        self.assertEqual(self.step(reading(E=(-1,.8))).action, 'CLARIFY')

    def test_invalid_reading_rejected(self):
        with self.assertRaises(ValueError):
            self.system.tracker.update(State(), reading(I=(.5,float('nan'))))


class T5IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.filler = T5Filler.__new__(T5Filler)
        bank = json.loads((DATA/'question_bank.json').read_text())
        names = json.loads((DATA/'labels.json').read_text())['type_names']
        QuestionBankFiller.__init__(self.filler, bank, names)
        self.filler.generate = Mock(return_value='Which statement better describes the work you want?')

    def test_quotes_preserved_and_recommendation_bypasses_model(self):
        payload = dict(slot='S', old_src='I like teaching "beginners".', new_src='I dislike teaching.')
        text = self.filler.fill('CHALLENGE', payload)
        self.assertIn(payload['old_src'],text)
        self.assertIn(payload['new_src'],text)
        self.filler.generate.reset_mock()
        self.filler.fill('RECOMMEND',{})
        self.filler.fill('OPENING',{})
        self.filler.generate.assert_not_called()

    def test_invalid_or_repeated_generation_falls_back(self):
        payload = dict(slot='I', retry='CLARIFY', previous_question='Would you enjoy analysing evidence?')
        self.filler.generate.return_value = payload['previous_question']
        expected = QuestionBankFiller.fill(self.filler,'ABSTAIN',payload)
        self.assertEqual(self.filler.fill('ABSTAIN',payload), expected)
        self.filler.generate.return_value = 'Here is a job recommendation.'
        self.assertEqual(self.filler.fill('ABSTAIN',payload), expected)


if __name__ == '__main__':
    unittest.main()
