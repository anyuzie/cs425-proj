import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
try:from task_context import route_scores,TaskContextClassifier
except ImportError:from recommender.utterance.task_context import route_scores,TaskContextClassifier


class TaskContextTests(unittest.TestCase):
    def test_independent_routing(self):
        for a,b,task,context in [(.9,.8,True,True),(.1,.2,False,False),(.9,.1,True,False),(.1,.9,False,True)]:
            result=route_scores(a,b)
            self.assertEqual((result['route_task'],result['route_context']),(task,context))
            self.assertEqual(result['task_score'],a);self.assertEqual(result['context_score'],b)
        self.assertTrue(route_scores(.5,.5)['route_task'])
        result=route_scores(.7,.7,.8,.6)
        self.assertEqual((result['route_task'],result['route_context']),(False,True))

    def test_rejects_invalid_scores_and_thresholds(self):
        for value in (float('nan'),float('inf'),-.1,1.1,True):
            with self.assertRaises(ValueError):route_scores(value,.5)
        for value in (0,1,float('nan'),True):
            with self.assertRaises(ValueError):route_scores(.5,.5,context_threshold=value)

    def test_real_bert_sigmoid_order_and_checkpoint_validation(self):
        import torch
        from transformers import BertConfig,BertForSequenceClassification,BertTokenizer
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'vocab.txt').write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nrepair\noutdoors\n',encoding='utf-8')
            tokenizer=BertTokenizer(vocab_file=str(root/'vocab.txt'));tokenizer.save_pretrained(root)
            config=BertConfig(vocab_size=7,hidden_size=8,num_hidden_layers=1,num_attention_heads=2,
                intermediate_size=16,num_labels=2,id2label={0:'task',1:'context'},problem_type='multi_label_classification')
            model=BertForSequenceClassification(config)
            with torch.no_grad():
                model.classifier.weight.zero_();model.classifier.bias.copy_(torch.tensor([2.,1.]))
            model.save_pretrained(root)
            with self.assertRaises(ValueError):TaskContextClassifier(root)
            model.config.task_context_classifier={'trained':True,'max_length':32,'thresholds':[.5,.5]}
            model.save_pretrained(root)
            classifier=TaskContextClassifier(root)
            result=classifier.predict(['repair outdoors'])[0]
            self.assertAlmostEqual(result['task_score'],.880797,places=5)
            self.assertAlmostEqual(result['context_score'],.731059,places=5)
            self.assertTrue(result['route_task'] and result['route_context'])
            self.assertEqual(classifier.predict([]),[])
            with self.assertRaises(ValueError):classifier.predict('raw string')
            with self.assertRaises(ValueError):classifier.predict([''])

    def test_real_dataset_isolation_and_source_targets(self):
        # Installed tests use the frozen dataset, local tests use staging data.
        folder=Path(__file__).resolve().parent/'data'
        if not folder.exists():folder=Path(__file__).resolve().parents[1]/'data/task_context_source_v1'
        def read(name):return [json.loads(line) for line in (folder/name).read_text(encoding='utf-8').splitlines()]
        train,validation=read('train.jsonl'),read('validation.jsonl')
        normalize=lambda s:' '.join(''.join(c if c.isalnum() else ' ' for c in s.casefold()).split())
        self.assertFalse({normalize(r['text']) for r in train}&{normalize(r['text']) for r in validation})
        for key in ('occupation_code','context_id'):
            self.assertFalse({r[key] for r in train if key in r}&{r[key] for r in validation if key in r})
        for row in train+validation:
            expected={'task':1,'context':0} if row['source']=='tasks.csv' else {'task':0,'context':1}
            self.assertEqual(row['labels'],expected)
        self.assertEqual(len(train),14065);self.assertEqual(len(validation),3435)

if __name__=='__main__':unittest.main()
