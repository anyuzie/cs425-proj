import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

from pipeline.interfaces import Action, Flag, HistoryEntry, Reading, SlotReading, SlotState, State
from pipeline.policy import ManualPolicy
from pipeline.task_questions import load_tasks, select_task
from pipeline.templates import QuestionBankFiller
from pipeline.t5_v3 import T5V3Filler

ROOT = Path(__file__).resolve().parents[1]


class TaskQuestionsTests(unittest.TestCase):
    def setUp(self):
        self.tasks = load_tasks(ROOT/'data/tasks.csv', ROOT/'configs/question_tasks.json')
        self.task = next(t for t in self.tasks if t['task_id'] == '1070')
        self.bank = json.loads((ROOT/'data/question_bank.json').read_text())
        self.names = json.loads((ROOT/'data/labels.json').read_text())['type_names']

    def state(self, turn=3):
        return State(turn=turn, ranking=[
            SimpleNamespace(code='11-9041.00', occupation_z={t:0 for t in 'RIASEC'}),
            SimpleNamespace(code='other', occupation_z={t:4 if t=='S' else 0 for t in 'RIASEC'})])

    def test_catalog_uses_real_tasks_and_covers_six_interests(self):
        self.assertEqual(len(self.tasks),23)
        self.assertEqual({t['slot'] for t in self.tasks},set('RIASEC'))
        self.assertEqual(self.task['text'],'Present and explain proposals, reports, or findings to clients.')
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.json'
            path.write_text(json.dumps({'tasks':[dict(self.task,task_id='missing')]}))
            with self.assertRaises(ValueError):load_tasks(ROOT/'data/tasks.csv',path)

    def test_selection_waits_for_context_and_does_not_repeat(self):
        state=self.state(2)
        self.assertIsNone(select_task(self.tasks,state,'S'))
        state.turn=3
        self.assertEqual(select_task(self.tasks,state,'S'),self.task)
        self.assertIsNone(select_task(self.tasks,state,'S'))
        self.assertIsNone(select_task(self.tasks,self.state(),'A'))

    def test_selection_does_not_use_occupation_outside_top_100(self):
        state=self.state()
        state.ranking=[SimpleNamespace(code='other')]*100+state.ranking[:1]
        self.assertIsNone(select_task(self.tasks,state,'S'))

    def test_policy_and_retry_preserve_same_task_and_facet(self):
        state=self.state();policy=ManualPolicy(task_questions=self.tasks)
        first=policy.decide(state)
        self.assertEqual((first.kind,first.slot),('CLARIFY','S'))
        self.assertEqual(first.payload['task'],self.task)
        state.last_action=first;state.last_reading_empty=True;state.user_texts=['hmm'];state.turn+=1
        retry=policy.decide(state)
        self.assertEqual(retry.kind,'ABSTAIN')
        self.assertEqual(retry.payload['task'],first.payload['task'])
        self.assertEqual(retry.payload['clarify_index'],first.payload['clarify_index'])
        self.assertEqual(len(state.asked_tasks),1)

    def test_challenge_and_probe_keep_priority_over_task_selection(self):
        state=self.state();policy=ManualPolicy(task_questions=self.tasks)
        a=HistoryEntry(1,.9,'enjoyment','I like teaching.',1)
        b=HistoryEntry(-1,.9,'enjoyment','I dislike teaching.',2)
        state.flags=[Flag('S',a,b)]
        self.assertEqual(policy.decide(state).kind,'CHALLENGE')
        self.assertFalse(state.asked_tasks)
        state.flags=[];state.slots={'R':SlotState(.55,.9,'familiarity','I used tools.',1,reason_confidence=.9)}
        self.assertEqual(policy.decide(state).kind,'PROBE')
        self.assertFalse(state.asked_tasks)

    def test_generic_fallback_when_no_matching_tasks(self):
        state=self.state();state.ranking=state.ranking[1:]
        action=ManualPolicy(task_questions=self.tasks).decide(state)
        self.assertEqual(action.kind,'CLARIFY')
        self.assertNotIn('task',action.payload)

    def test_familiarity_probe_keeps_the_task_context(self):
        state=self.state()
        text='I was trained to explain proposals to clients.'
        state.slots['S']=SlotState(.55,.9,'familiarity',text,2,reason_confidence=.9)
        state.task_history=[{'slot':'S','reply':text,'task':dict(self.task)}]
        action=ManualPolicy(task_questions=self.tasks).decide(state)
        self.assertEqual(action.kind,'PROBE')
        self.assertEqual(action.payload['task'],self.task)

    def test_legacy_t5_also_rejects_activity_drift(self):
        from pipeline.t5 import T5Filler
        filler=T5Filler.__new__(T5Filler)
        QuestionBankFiller.__init__(filler,self.bank,self.names)
        filler.generate=Mock(return_value='Would you enjoy helping or teaching people at work?')
        self.assertIn(self.task['activity'],filler.fill('CLARIFY',{'slot':'S','task':self.task}))

    def test_model_receives_task_and_cannot_change_the_activity(self):
        filler=T5V3Filler.__new__(T5V3Filler)
        QuestionBankFiller.__init__(filler,self.bank,self.names)
        payload={'slot':'S','task':dict(self.task)}
        filler.generate=Mock(return_value='Would you enjoy helping or teaching people at work?')
        delivered=filler.fill('CLARIFY',payload)
        fields=json.loads(filler.generate.call_args.args[0].split(': ',1)[1])
        self.assertEqual(fields['focus'],self.task['activity'])
        self.assertIn(self.task['text'],fields['context'])
        self.assertIn(self.task['activity'],delivered)
        raw=f'Would you enjoy {self.task["activity"]} during your working day?'
        filler.generate.return_value=raw
        self.assertEqual(filler.fill('CLARIFY',payload),raw)
        payload.update(retry='CLARIFY',previous_question=raw,repair=True)
        self.assertIn(self.task['activity'],filler.fill('ABSTAIN',payload))

    def test_recommendation_does_not_claim_missing_task_evidence(self):
        filler=QuestionBankFiller(self.bank,self.names)
        reply=filler.fill('RECOMMEND',{'has_task_evidence':False,'uncertain_slots':['S']})
        self.assertNotIn('tasks',reply)
        self.assertIn('uncertain',reply)

    def test_task_history_keeps_actual_readings_not_intended_target(self):
        import fakes
        from pipeline.system import System
        system=System.from_config(ROOT/'data',json.loads((ROOT/'tests/test_config.json').read_text()))
        system.policy=ManualPolicy()
        state=State(last_action=Action('CLARIFY','S',payload={'task':dict(self.task)}),
                    last_reply='Would you enjoy explaining proposals to clients?')
        text='I enjoy analysing scientific data and investigating difficult research problems on my own.'
        item=Reading(text,[SlotReading('I',1,.55,.9,'enjoyment',text)])
        with patch.object(system.extractor,'read',return_value=item):system.step(state,text)
        self.assertIn('I',state.slots)
        self.assertNotIn('S',state.slots)
        entry=state.task_history[0]
        self.assertEqual(entry['task'],self.task)
        self.assertEqual(entry['question'],'Would you enjoy explaining proposals to clients?')
        self.assertEqual(entry['readings'][0]['slot'],'I')

    def test_accepted_evidence_clears_stale_uncertainty(self):
        from pipeline.tracker import DecayTracker
        state=State(turn=3,settled={'S'},unresolved_conflicts=['S'])
        DecayTracker().update(state,Reading('I enjoy teaching.',[SlotReading('S',1,.55,.9,'enjoyment','I enjoy teaching.')]))
        self.assertNotIn('S',state.settled)
        self.assertNotIn('S',state.unresolved_conflicts)

    def test_task_yes_no_is_retained_without_forcing_a_global_type(self):
        import fakes
        from pipeline.system import System
        from pipeline.tracker import DecayTracker
        for text,sign in [('Yes, I would.',1),('No, I would not.',-1)]:
            system=System.from_config(ROOT/'data',json.loads((ROOT/'tests/test_config.json').read_text()))
            system.policy=ManualPolicy(task_questions=self.tasks);system.tracker=DecayTracker()
            state=State(turn=3,last_action=Action('CLARIFY','S',payload={'task':dict(self.task)}),
                        last_reply='Would you enjoy helping clients understand proposals?')
            with patch.object(system.extractor,'read',return_value=Reading(text)):
                result=system.step(state,text)
            self.assertEqual(state.task_history[-1]['preference'],sign)
            self.assertNotIn('S',state.slots)
            self.assertEqual(result.action,'CLARIFY')
            self.assertNotIn('broader_task_check',state.last_action.payload)
            self.assertNotIn('More broadly',result.reply)
            self.assertIn('unfilled interest with largest top-100 occupation variance',state.last_action.why)

    def test_partial_profile_at_turn_cap_is_marked_uncertain(self):
        state=State(turn=10,slots={'S':SlotState(.55,.9,'enjoyment','I like teaching.',1)})
        action=ManualPolicy().decide(state)
        self.assertEqual(set(action.payload['uncertain_slots']),set('RIAEC'))

    def test_task_only_answer_does_not_reask_an_already_known_interest(self):
        slots={t:SlotState(.55,.9,'enjoyment','An explicit preference.',1) for t in 'RIASEC'}
        snapshot=dict.fromkeys('RIASEC',.55)
        state=State(turn=5,slots=slots,profile_history=[dict(snapshot) for _ in range(3)],
            last_reading_empty=True,last_action=Action('CLARIFY','S',payload={'task':self.task}),
            task_history=[{'turn':5,'slot':'S','task':self.task,'reply':'No.','preference':-1,'readings':[]}])
        action=ManualPolicy(task_questions=self.tasks).decide(state)
        self.assertEqual(action.kind,'RECOMMEND')
        self.assertEqual(state.slots['S'].value,.55)

    def test_task_from_stale_ranking_is_not_presented(self):
        import fakes
        from pipeline.system import System
        from pipeline.tracker import DecayTracker
        system=System.from_config(ROOT/'data',json.loads((ROOT/'tests/test_config.json').read_text()))
        system.policy=ManualPolicy(task_questions=self.tasks);system.tracker=DecayTracker()
        old=HistoryEntry(.9,.9,'enjoyment','I like tools.',1)
        new=HistoryEntry(-.9,.9,'enjoyment','I dislike tools.',2)
        state=State(turn=3,slots={'R':SlotState(.9,.9,old.reason,old.source,1)},
            flags=[Flag('R',old,new)],strikes={'R':1},
            last_action=Action('ABSTAIN','R',payload={'retry':'CHALLENGE','old_src':old.source,'new_src':new.source,'old_turn':1}))
        before=self.state().ranking
        after=[SimpleNamespace(code='new',title='New occupation',occupation_z=dict.fromkeys('RIASEC',0))]
        with patch.object(system.extractor,'read',return_value=Reading('I am not sure.')), patch.object(system,'_rank',side_effect=[before,after]):
            system.step(state,'I am not sure.')
        self.assertNotIn('task',state.last_action.payload)
        self.assertNotIn((self.task['occupation_code'],self.task['task_id']),state.asked_tasks)
        self.assertIn('refreshed top 100',state.last_action.why)


if __name__=='__main__':unittest.main()
