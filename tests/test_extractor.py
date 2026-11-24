"""Extractor's conversion (read) with load/predict faked. Tests our code, not the model."""

import hashlib
import random
import unittest

from contracts import DATA, ExtractorContract
from pipeline.extractor import Extractor


class FakeModel(Extractor):
    """predict() returns `table[sentence]` if given, else valid probabilities derived from the sentence hash."""

    def __init__(self, table=None):
        self.table, self.calls = table or {}, []
        super().__init__("nowhere", DATA / "labels.json")

    def load(self, checkpoint_dir):
        pass

    def predict(self, context, sentence):
        self.calls.append((context, sentence))
        if sentence in self.table:
            return self.table[sentence]
        rng = random.Random(int(hashlib.md5(sentence.encode()).hexdigest(), 16))
        def soft(n):
            v = [rng.random() + 0.01 for _ in range(n)]
            return [x / sum(v) for x in v]
        return {"direction": [soft(3) for _ in range(6)], "strength": [rng.random() for _ in range(6)],
                "reason": soft(3)}


def out(direction=None, strength=None, reason=(0.7, 0.2, 0.1)):
    """direction: {letter: (neutral, positive, negative)}; unlisted types are neutral."""
    d = direction or {}
    return {"direction": [list(d.get(t, (1.0, 0.0, 0.0))) for t in "RIASEC"],
            "strength": list(strength or [0.5] * 6), "reason": list(reason)}




class TestExtractorRead(unittest.TestCase):
    def test_sentence_split_and_short_piece_join(self):
        s = Extractor.sentences
        self.assertEqual(s(""), [])
        self.assertEqual(s("  \n "), [])
        self.assertEqual(s("I really enjoy building things with my hands. I also like to draw pictures all day! "
                           "Why not?\nI love solving hard problems with other people."),
                         ["I really enjoy building things with my hands.",
                          "I also like to draw pictures all day! Why not?",
                          "I love solving hard problems with other people."])
        self.assertEqual(s("Yes. I enjoy fixing machines at the weekend. Really."),
                         ["Yes.", "I enjoy fixing machines at the weekend. Really."])   # first piece has nothing before it
        self.assertEqual(s("one two three four five six seven. a b c d e f. one two three four five six seven."),
                         ["one two three four five six seven. a b c d e f.", "one two three four five six seven."])








if __name__ == "__main__":
    unittest.main()
