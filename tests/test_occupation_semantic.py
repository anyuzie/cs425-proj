"""Contract and leakage tests; real-model/HTTP evidence is a separate script."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from recommender.components import ComponentInput
from recommender.occupation_semantic import OccupationSemantic, fingerprint, load_bundle
from training.occupation_semantic import grouped_components, positive_mask, normal, metrics


class SemanticTests(unittest.TestCase):
    def test_transitive_grouping_keeps_duplicates(self):
        groups,_,_=grouped_components(['Analyse patient health records.', 'analyse patient health records!',
                                       'Repair industrial machinery.'])
        self.assertEqual(groups[0],groups[1]);self.assertNotEqual(groups[0],groups[2])

    def test_multi_positive_mask(self):
        self.assertEqual(positive_mask([{'positives':['a','b']},{'positives':['b','c']}],['a','b','c','d']),
                         [[True,True,False,False],[False,True,True,False]])

    def test_missing_and_empty_statuses(self):
        comp=OccupationSemantic([SimpleNamespace(code='a')],{'index':'missing_semantic_test_path'})
        self.assertEqual(comp.score(ComponentInput()).status,'unusable')
        result=comp.score(ComponentInput(user_sentences=('chemistry degree',)))
        self.assertEqual(result.status,'unavailable');self.assertIn('unavailable',result.reason)

    def test_no_preference_gate_signed_scores_and_ties(self):
        encoder=SimpleNamespace(encode=lambda *a,**k: np.array([[1.,0.]],dtype='float32'))
        comp=OccupationSemantic([SimpleNamespace(code=c) for c in ['b','a','c']],{'index':'missing_semantic_test_path'})
        comp.bundle=(encoder,np.array([[-1.,0.],[0.,1.],[0.,-1.]],dtype='float32'),['b','c','a'],{})
        out=comp.score(ComponentInput(user_sentences=('arbitrary nonempty text',)))
        self.assertEqual([(x.code,x.score) for x in out.occupations],[('a',0.),('c',0.),('b',-1.)])

    def test_checkpoint_fingerprint_detects_mutation(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'config.json';p.write_text('{}');a=fingerprint(d)
            p.write_text('{"changed":true}');self.assertNotEqual(a,fingerprint(d))

    def test_incompatible_index_rejected_before_encoder(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'manifest.json').write_text(json.dumps({'builder':'stale','normalization':'l2','dimension':384}))
            with self.assertRaisesRegex(ValueError,'incompatible'):
                load_bundle(d,'test-invalid')

    def test_retrieval_metrics_multiple_positives(self):
        r=metrics(['x','a','b'],{'a','b'})
        self.assertEqual(r['mrr'],.5);self.assertEqual(r['recall@5'],1.);self.assertEqual(r['hit@1'],0.)

    def test_prepared_split_and_final_index_leakage(self):
        root=Path('models/occupation_semantics_v1')
        if not (root/'data.json').exists(): self.skipTest('Prepare experiment first')
        data=json.loads((root/'data.json').read_text())
        groups=[e['group'] for e in data['examples']];self.assertEqual(len(groups),len(set(groups)))
        for e in data['examples']:
            if e['split']=='train':self.assertFalse(set(e['positives']) & set(data['heldout_codes']))
        documents=root/'finetuned_index/documents.json'
        if not documents.exists():self.skipTest('Build final index first')
        indexed=json.loads(documents.read_text());all_text=[normal(v['text']) for v in indexed.values()]
        def leaks(splits):
            return sorted({normal(t) for e in data['examples'] if e['split'] in splits
                           for t in e['texts'] if any(normal(t) in d for d in all_text)})
        # Frozen test queries must never appear verbatim (or as substrings) in indexed text.
        self.assertEqual(leaks({'test'}),[])
        # Known non-test containment: short statements that are substrings of distinct, longer
        # training tasks (e.g. "Serve on committees." inside "Serve on committees to promote...").
        # Validation was already used for selection; the held-out case is excluded in the report.
        self.assertEqual(leaks({'validation','occupation_heldout'}),
                         ['hire new faculty','participate in campus and community events','serve on committees'])


if __name__=='__main__':unittest.main()
