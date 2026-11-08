import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from recommender.occupation_classifier import OccupationAnnotations, OccupationClauseClassifier
from training.occupation_classifier import load_examples
from app.ranking_terminal import rank_utterance, show_result


class OccupationClassificationTests(unittest.TestCase):
    def test_occupation_prose_preserves_shared_arguments(self):
        class NLP:
            def __call__(self,text):
                return SimpleNamespace(sents=[SimpleNamespace(text=text,start_char=0)])
        # The runtime occupation path must not invoke conversational splitting.
        splitter=SimpleNamespace(nlp=NLP(),split=lambda text: self.fail('utterance splitter used for occupation prose'))
        classifier=SimpleNamespace(predict=lambda texts:[{'labels':{'activity':True,'job_nature':False},
            'probabilities':{'activity':.9,'job_nature':.1}} for text in texts])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'occupations.csv'
            path.write_text('code,Description\n1,"Install, maintain, and repair machinery."\n',encoding='utf-8')
            rows=OccupationAnnotations(path,None,splitter,classifier).get('1')
            self.assertEqual(len(rows),1)
            self.assertEqual(rows[0]['text'],'Install, maintain, and repair machinery.')

    def test_real_hugging_face_bce_sigmoid_and_missing_checkpoint(self):
        import torch
        from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(ValueError):
                OccupationClauseClassifier(root)
            (root / "vocab.txt").write_text(
                "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nrepair\noutdoors\n", encoding="utf-8")
            tokenizer = BertTokenizer(vocab_file=str(root / "vocab.txt"))
            tokenizer.save_pretrained(root)
            config = BertConfig(vocab_size=7, hidden_size=8, num_hidden_layers=1,
                num_attention_heads=2, intermediate_size=16, num_labels=2,
                problem_type="multi_label_classification",
                id2label={0: "activity", 1: "job_nature"})
            model = BertForSequenceClassification(config).eval()
            with torch.no_grad():
                model.classifier.weight.zero_()
                model.classifier.bias.copy_(torch.tensor([2., -2.]))
            encoded = tokenizer(["repair outdoors"], return_tensors="pt")
            targets = torch.tensor([[1., 0.]])
            result = model(**encoded, labels=targets)
            self.assertTrue(torch.allclose(result.loss,
                torch.nn.functional.binary_cross_entropy_with_logits(result.logits, targets)))
            model.save_pretrained(root)
            with self.assertRaises(ValueError):
                OccupationClauseClassifier(root)
            # This tiny synthetic checkpoint tests plumbing only, not model quality.
            model.config.occupation_classifier = {"trained": True, "max_length": 32}
            model.save_pretrained(root)
            classifier = OccupationClauseClassifier(root)
            prediction = classifier.predict(["repair outdoors"])[0]
            self.assertEqual(prediction["labels"], {"activity": True, "job_nature": False})
            self.assertAlmostEqual(prediction["probabilities"]["activity"], .880797, places=5)

    def test_sources_both_labels_offsets_and_cache(self):
        class Classifier:
            calls = 0
            def predict(self, texts):
                self.calls += 1
                return [{"labels": {"activity": True, "job_nature": True},
                         "probabilities": {"activity": .9, "job_nature": .8}} for text in texts]
        splitter = SimpleNamespace(split=lambda text: SimpleNamespace(
            clauses=[SimpleNamespace(text=text, start=0, end=len(text))] if text else []))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "occupations.csv").write_text('code,Description\n1,Repair outdoors\n', encoding="utf-8")
            (root / "tasks.csv").write_text('code,Task ID,Task\n1,7,Install machinery\n', encoding="utf-8")
            classifier = Classifier()
            annotations = OccupationAnnotations(root / "occupations.csv", root / "tasks.csv", splitter, classifier)
            rows = annotations.get("1")
            self.assertEqual([r["source_field"] for r in rows], ["Description", "Task:7"])
            self.assertTrue(all(all(r["labels"].values()) for r in rows))
            self.assertEqual(rows[0]["end"], len(rows[0]["text"]))
            self.assertIs(rows, annotations.get("1"))
            self.assertEqual(classifier.calls, 1)

    def test_terminal_only_cuts_display(self):
        extractor = SimpleNamespace(read=lambda text: SimpleNamespace(slots=[]))
        occupations = [SimpleNamespace(code=str(i), title=f"Occupation{i}", rank=i+1,
                                       rrf_score=None, signals={}) for i in range(15)]
        response = SimpleNamespace(status="ok", occupations=occupations)
        class Ranker:
            def rank_with_evidence(self, *args, **kwargs):
                self.top_k = kwargs["top_k"]
                return response
        ranker = Ranker()
        result = rank_utterance("hello", extractor, ranker)
        annotations = SimpleNamespace(get=lambda code: [{"text": "Repair outdoors", "source_field": "Description",
            "labels": {"activity": True, "job_nature": True},
            "probabilities": {"activity": .9, "job_nature": .9}}])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            show_result(*result, occupation_annotations=annotations)
        self.assertEqual(ranker.top_k, 50)
        self.assertEqual(len(response.occupations), 15)
        self.assertIn("Top 10 occupations", out.getvalue())
        self.assertNotIn("Occupation10", out.getvalue())
        self.assertEqual(out.getvalue().count("Repair outdoors"), 20)

    def test_requires_both_reviewed_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "examples.jsonl"
            path.write_text('{"text":"Outdoor work","labels":{"activity":0}}', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_examples(path)


if __name__ == "__main__":
    unittest.main()
