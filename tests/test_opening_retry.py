import json
import unittest
from pathlib import Path

from pipeline.interfaces import Action, State
from pipeline.policy import ManualPolicy
from pipeline.templates import OPENING_RETRY, QuestionBankFiller

ROOT = Path(__file__).resolve().parents[1]


class OpeningRetryTests(unittest.TestCase):
    def test_empty_opening_reply_gets_one_retry(self):
        state = State(turn=1, last_reading_empty=True, user_texts=['hmm'],
                      last_action=Action('OPENING', None, 'opening'))
        action = ManualPolicy().decide(state)
        self.assertEqual((action.kind, action.slot, action.payload['retry']), ('ABSTAIN', None, 'OPENING'))

    def test_empty_answer_to_the_retry_moves_on_to_clarify(self):
        state = State(turn=2, last_reading_empty=True, user_texts=['hmm', 'no idea'],
                      last_action=Action('ABSTAIN', None, payload={'retry': 'OPENING'}))
        self.assertEqual(ManualPolicy().decide(state).kind, 'CLARIFY')

    def test_opening_reply_with_an_interest_is_not_retried(self):
        state = State(turn=1, last_reading_empty=False, user_texts=['I like drawing'],
                      last_action=Action('OPENING', None, 'opening'))
        self.assertNotEqual(ManualPolicy().decide(state).kind, 'ABSTAIN')

    def test_control_arm_gets_the_same_retry(self):
        state = State(turn=1, last_reading_empty=True, user_texts=['hmm'],
                      last_action=Action('OPENING', None, 'opening'))
        self.assertEqual(ManualPolicy(no_probing=True).decide(state).payload['retry'], 'OPENING')

    def test_retry_wording_is_the_template_for_every_filler(self):
        from pipeline.t5_v3 import T5V3Filler
        qb = json.loads((ROOT / 'data/question_bank.json').read_text(encoding='utf-8'))
        names = json.loads((ROOT / 'data/labels.json').read_text(encoding='utf-8'))['type_names']
        payload = {'retry': 'OPENING', 'index': 0}
        self.assertEqual(QuestionBankFiller(qb, names).fill('ABSTAIN', payload), OPENING_RETRY)
        t5 = T5V3Filler.__new__(T5V3Filler)   # no model needed: T5 must not be called for this reply
        QuestionBankFiller.__init__(t5, qb, names)
        t5.generate = lambda prompt: self.fail('T5 was called for the opening retry')
        self.assertEqual(t5.fill('ABSTAIN', dict(payload)), OPENING_RETRY)

    def test_conversation_asks_for_an_example_after_an_empty_opening(self):
        import fakes  # noqa: F401  (registers the test components)
        from pipeline.system import System
        config = json.loads((ROOT / 'tests/test_config.json').read_text())
        system = System.from_config(ROOT / 'data', config)
        system.policy = ManualPolicy()
        state = State()
        system.start(state)
        first = system.step(state, 'hmm, not sure')
        self.assertEqual((first.action, first.reply), ('ABSTAIN', OPENING_RETRY))
        second = system.step(state, 'still not sure')
        self.assertEqual(second.action, 'CLARIFY')


if __name__ == '__main__':
    unittest.main()
