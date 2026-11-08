import importlib.util
import unittest

# Same rule as tests/test_dense.py: these need the optional ranking packages.
if not all(importlib.util.find_spec(name) for name in ("numpy", "faiss")):
    raise unittest.SkipTest("install requirements-ranking.txt for task-preference fixtures")

import numpy as np  # noqa: E402

from recommender.components import ComponentInput  # noqa: E402
from recommender.task_preferences import TaskPreferenceExtractor  # noqa: E402
from recommender.stance_config import PARAMETERS, POSITIVE_PARAMETERS  # noqa: E402
from recommender.stance_ranker import StanceTaskRanker  # noqa: E402
from recommender.dense_ranker import Task  # noqa: E402


class StanceTests(unittest.TestCase):
    def extract(self, text):
        return TaskPreferenceExtractor().extract((text,))

    def usable(self, text):
        return [(e.activity_text, e.polarity) for e in self.extract(text) if e.status == 'usable']

    def test_negation(self):
        self.assertEqual(self.usable("I don't enjoy filling forms."), [('filling forms', 'negative')])
        self.assertEqual(self.usable("I don't hate filling forms."), [])
        self.assertEqual(self.usable("I've never tried filling forms."), [])

    def test_contrast(self):
        self.assertEqual(self.usable('I enjoy fixing machines but dislike filling forms.'),
                         [('fixing machines', 'positive'), ('filling forms', 'negative')])

    def test_background_uncertainty_and_negated_action(self):
        for text in ["My dad enjoys fixing machines.", "I study mechanical engineering.",
                     "I have worked with machines.", "I might enjoy fixing machines.",
                     "I am not sure I like fixing machines.", "I enjoy not filling forms.",
                     "I like it.", "I enjoy fixing machines and hate filling forms."]:
            self.assertEqual(self.usable(text), [], text)

    def test_weak_and_backward(self):
        self.assertEqual(self.usable('I quite like fixing machines.'), [('fixing machines', 'positive')])
        self.assertEqual(self.usable('Fixing machines is my favourite thing.'), [('Fixing machines', 'positive')])

    def test_conflict(self):
        rows = self.extract('I enjoy filling forms. I dislike filling forms.')
        self.assertTrue(all(e.status == 'unresolved' for e in rows))

    def scorer(self, parameters=PARAMETERS, zero=False):
        class Encoder:
            def encode(self, texts):
                return np.array([[0, 0] if zero else [1, 0] for _ in texts], dtype='float32')
        tasks = [Task('a', '1', 'forms'), Task('b', '2', 'machines')]
        return StanceTaskRanker(tasks, np.eye(2, dtype='float32'), Encoder(), parameters)

    def test_positive_negative_and_missing_candidate(self):
        scorer = self.scorer()
        positive = scorer.score(ComponentInput(user_sentences=('I enjoy filling forms.',)))
        negative = scorer.score(ComponentInput(user_sentences=('I dislike filling forms.',)))
        self.assertEqual([(r.code, r.score) for r in positive.occupations], [('a', 1), ('b', 0)])
        self.assertEqual([(r.code, r.score) for r in negative.occupations], [('b', 0), ('a', -1)])
        self.assertNotIn('missing', {r.code for r in positive.occupations})
        self.assertEqual(scorer.last_trace['top_matches'][1]['matches'][0]['polarity'], 'negative')

    def test_empty_positive_only_zero_guard_and_ties(self):
        self.assertEqual(self.scorer().score(ComponentInput()).status, 'unusable')
        self.assertEqual(self.scorer(POSITIVE_PARAMETERS).score(ComponentInput(user_sentences=('I dislike filling forms.',))).status, 'unusable')
        with self.assertRaisesRegex(ValueError, 'normalized'):
            self.scorer(zero=True).score(ComponentInput(user_sentences=('I enjoy filling forms.',)))
        s=self.scorer()
        s.embeddings[:]=np.array([1,0])
        s.index.reset();s.index.add(s.embeddings)
        self.assertEqual([r.code for r in s.score(ComponentInput(user_sentences=('I enjoy filling forms.',))).occupations], ['a','b'])

    def test_no_structured_or_annotation_reads(self):
        class OnlyText:
            user_sentences = ('I enjoy filling forms.',)
            def __getattr__(self, key):
                raise AssertionError('Forbidden input: '+key)
        self.assertEqual(self.scorer().score(OnlyText()).status, 'usable')


if __name__ == '__main__':
    unittest.main()
