import csv
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from annotation_tool.occupation_review import OccupationReviewStore, ReviewSave, create_occupation_app, eligible


class OccupationReviewTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.path = self.root/'all.csv'
        self.batch = self.root/'batch.csv'
        fields=['clause_id','source_text','occupation_code','occupation_title','source_field','task_id',
            'start','end','text','proposed_category','proposed_activity','proposed_job_nature','proposal_reason',
            'split_review_note','review_flags','reviewer_text','reviewer_activity','reviewer_job_nature',
            'reviewer_notes','review_status']
        with self.path.open('w',encoding='utf-8-sig',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields)
            writer.writeheader()
            for i,(start,end) in enumerate([(0,6),(7,15)]):
                writer.writerow(dict(clause_id=f'OP{i}',source_text='Repair outdoors',occupation_code='1',
                    occupation_title='Example',source_field='Description',start=start,end=end,
                    text='Repair outdoors'[start:end],proposed_category='activity',proposed_activity='1',
                    proposed_job_nature='0',review_status='pending'))
        self.batch.write_text('clause_id\nOP0\n',encoding='utf-8')
        self.store=OccupationReviewStore(self.path,self.batch)

    def request(self,category='both',ids=None,clauses=None,revision=None,confirmed=True):
        return ReviewSave(revision=revision or self.store.snapshot()['revision'],ids=ids or ['OP0'],
            clauses=clauses or [dict(start=0,end=6,text='Repair',category=category,notes='Checked')],
            reviewer='reviewer',confirmed=confirmed)

    def test_labels_reload_backup_and_other_row(self):
        before=self.store.read()[0][1]
        original=self.path.read_bytes()
        result=self.store.save(self.request())
        self.assertEqual(Path(result['backup_path']).read_bytes(),original)
        rows=OccupationReviewStore(self.path,self.batch).snapshot()['rows']
        self.assertTrue(eligible(rows[0]))
        self.assertEqual((rows[0]['reviewer_activity'],rows[0]['reviewer_job_nature']),('1','1'))
        self.assertEqual(rows[0]['proposed_category'],'activity')
        self.assertTrue(rows[0]['in_first_batch'])
        self.assertFalse(rows[1]['in_first_batch'])
        self.assertTrue(all(rows[1][k]==v for k,v in before.items()))

    def test_neither_is_resolved_uncertain_is_not(self):
        self.store.save(self.request('neither'))
        row=self.store.read()[0][0]
        self.assertTrue(eligible(row))
        self.assertEqual((row['reviewer_activity'],row['reviewer_job_nature']),('0','0'))
        self.store.save(self.request('uncertain'))
        row=self.store.read()[0][0]
        self.assertFalse(eligible(row))
        self.assertEqual((row['reviewer_activity'],row['reviewer_job_nature']),('',''))

    def test_merge_then_split_stable_lineage(self):
        result=self.store.save(self.request(ids=['OP0','OP1'],clauses=[dict(start=0,end=15,text='Repair outdoors',category='both')]))
        merged=result['saved_ids'][0]
        rows=self.store.snapshot()['rows']
        self.assertFalse(eligible(rows[0]))
        self.assertEqual(rows[0]['active'],'false')
        self.assertTrue(rows[-1]['in_first_batch'])
        result=self.store.save(self.request(ids=[merged],clauses=[dict(start=0,end=6,text='Repair',category='activity'),
            dict(start=7,end=15,text='outdoors',category='job_nature')]))
        ids=result['saved_ids']
        reloaded=self.store.snapshot()['rows']
        self.assertEqual([r['clause_id'] for r in reloaded[-2:]],ids)
        self.assertTrue(all(eligible(r) and r['in_first_batch'] for r in reloaded[-2:]))
        self.assertTrue(all('OP0' in r['replaces_ids'] for r in reloaded[-2:]))

    def test_stale_and_confirmation(self):
        request=self.request()
        self.store.save(request)
        before=self.path.read_bytes()
        with self.assertRaises(FileExistsError):self.store.save(request)
        with self.assertRaises(ValueError):self.store.save(self.request(confirmed=False))
        self.assertEqual(before,self.path.read_bytes())

    def test_reviewer_identity_required(self):
        before=self.path.read_bytes()
        request=self.request()
        request.reviewer='   '
        with self.assertRaisesRegex(ValueError,'reviewer name'):
            self.store.save(request)
        self.assertEqual(before,self.path.read_bytes())

    def test_prepared_clauses_can_share_source_provenance(self):
        rows,fields,_=self.store.read()
        fields.append('preparation_status')
        for row in rows:
            row.update(start='0',end='15',preparation_status='assistant_checked')
        with self.path.open('w',encoding='utf-8-sig',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields)
            writer.writeheader();writer.writerows(rows)
        result=self.store.save(self.request(clauses=[dict(start=0,end=15,text='Workers repair equipment.',category='activity')]))
        self.assertEqual(result['saved_ids'],['OP0'])
        self.assertEqual(result['progress']['reviewed'],1)
        self.assertEqual(result['progress']['pending'],1)
        self.assertEqual(result['rows'][1]['review_status'],'pending')

    def test_replace_failure_preserves_csv(self):
        before=self.path.read_bytes()
        with patch('annotation_tool.occupation_review.os.replace',side_effect=OSError('disk error')):
            with self.assertRaises(OSError):self.store.save(self.request())
        self.assertEqual(before,self.path.read_bytes())
        self.assertFalse(list(self.root.glob('*.lock')))
        self.assertFalse(list(self.root.glob('*.tmp')))
        self.assertEqual(len(list(self.root.glob('*.bak'))),1)

    def test_invalid_overlap_and_cross_source(self):
        with self.assertRaises(ValueError):
            self.store.save(self.request(clauses=[dict(start=0,end=15,text='Repair outdoors',category='both')]))
        with self.assertRaises(ValueError):
            self.store.save(self.request(category='invalid'))
        with self.assertRaises(ValueError):
            self.store.save(self.request(clauses=[dict(start=0,end=100,text='bad',category='activity')]))
        rows, fields, _ = self.store.read()
        rows[1]['source_field'] = 'Task'
        with self.path.open('w',encoding='utf-8-sig',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError,'same occupation and source'):
            self.store.save(self.request(ids=['OP0','OP1']))

    def test_backup_failure_preserves_csv(self):
        before=self.path.read_bytes()
        with patch('annotation_tool.occupation_review.Path.write_bytes',side_effect=OSError('backup error')):
            with self.assertRaises(OSError):self.store.save(self.request())
        self.assertEqual(before,self.path.read_bytes())
        self.assertFalse(list(self.root.glob('*.lock')))

    def test_launch_modes_keep_separate_defaults(self):
        import annotation_tool.__main__ as command
        with patch('sys.argv',['annotation_tool','serve']), patch('uvicorn.run') as run:
            command.main()
            self.assertEqual(run.call_args.kwargs['port'],8010)
        with patch('sys.argv',['annotation_tool','serve-occupations','--occupation-csv',str(self.path),
                              '--occupation-batch',str(self.batch)]), patch('uvicorn.run') as run:
            command.main()
            self.assertEqual(run.call_args.kwargs['port'],8011)

    def test_routes_are_separate_and_csv_updates(self):
        client=TestClient(create_occupation_app(self.path,self.batch))
        self.assertEqual(client.get('/').status_code,200)
        self.assertEqual(client.get('/api/entries').status_code,404)
        result=client.post('/api/occupation/save',json=self.request('neither').model_dump())
        self.assertEqual(result.status_code,200)
        self.assertEqual(client.get('/api/occupation/rows').json()['progress']['reviewed'],1)


if __name__=='__main__':unittest.main()
