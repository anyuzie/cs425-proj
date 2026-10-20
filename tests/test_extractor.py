"""DebertaExtractor's conversion (read) with load/predict faked. Tests our code, not the model."""

import hashlib
import random
import unittest
from types import SimpleNamespace

from contracts import DATA, ExtractorContract
from pipeline.extractor import DebertaExtractor


class FakeDeberta(DebertaExtractor):
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


class TestDebertaContract(ExtractorContract, unittest.TestCase):
    def make(self):
        return FakeDeberta()


class TestDebertaRead(unittest.TestCase):
    def test_sentence_split_and_short_piece_join(self):
        s = DebertaExtractor.sentences
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

    def test_empty_text(self):
        ex = FakeDeberta()
        r = ex.read("   ")
        self.assertEqual((r.text, r.slots, ex.calls), ("   ", [], []))

    def test_mapping_neutral_sign_reason(self):
        sent = "I love building machines but I dislike paperwork."
        ex = FakeDeberta({sent: out({"R": (0.1, 0.8, 0.1), "C": (0.05, 0.15, 0.8), "A": (0.9, 0.05, 0.05)},
                                    [0.9, 0.1, 0.2, 0.3, 0.4, 0.35], reason=(0.1, 0.75, 0.15))})
        r = ex.read(sent, "What do you like?", "C")
        self.assertEqual(ex.calls, [("What do you like?", sent)])
        got = {s.slot: (s.sign, s.strength, s.confidence, s.reason, s.source, s.reason_confidence) for s in r.slots}
        self.assertEqual(got, {"R": (1, 0.9, 0.8, "familiarity", sent, 0.75),
                               "C": (-1, 0.35, 0.8, "familiarity", sent, 0.75)})

    def test_duplicate_type_keeps_highest_confidence(self):
        a, b = "I enjoy repairing engines at the weekend.", "I truly hate fixing machines in a hot garage."
        ex = FakeDeberta({a: out({"R": (0.2, 0.7, 0.1)}), b: out({"R": (0.05, 0.05, 0.9)})})
        r = ex.read(f"{a} {b}")
        self.assertEqual([(s.slot, s.sign, s.confidence, s.source) for s in r.slots], [("R", -1, 0.9, b)])

    def test_values_clamped(self):
        sent = "I enjoy repairing engines at the weekend."
        ex = FakeDeberta({sent: out({"R": (0.0, 1.2, 0.0)}, [1.5] * 6, reason=(1.1, 0.0, 0.0))})
        s = ex.read(sent).slots[0]
        self.assertEqual((s.strength, s.confidence, s.reason_confidence), (1.0, 1.0, 1.0))

    def test_shape_validation(self):
        good = out()
        bad = {"not a dict": [1],
               "direction": {**good, "direction": good["direction"][:5]},
               "strength": {**good, "strength": [0.5] * 5},
               "reason": {**good, "reason": [0.5, 0.5]},
               "direction row": {**good, "direction": [[1.0, 0.0]] * 6}}
        for key, value in bad.items():
            with self.subTest(key=key):
                ex = FakeDeberta({"I enjoy repairing engines at the weekend.": value})
                with self.assertRaisesRegex(ValueError, key.split()[0] if key != "not a dict" else "dict"):
                    ex.read("I enjoy repairing engines at the weekend.")
        missing = {k: v for k, v in good.items() if k != "reason"}
        with self.assertRaisesRegex(ValueError, "missing key 'reason'"):
            FakeDeberta({"I enjoy repairing engines at the weekend.": missing}).read("I enjoy repairing engines at the weekend.")

    def test_missing_checkpoint_points_to_guide(self):
        # Checked before torch is imported, so this runs without torch installed.
        with self.assertRaisesRegex(FileNotFoundError, "INTEGRATION.md"):
            DebertaExtractor("nowhere", DATA / "labels.json")


def model_labels(order=("negative", "neutral", "positive")):
    """id2label shaped like the trained checkpoint: 18 direction, 3 reason, 24 strength labels."""
    labels = [f"{t}:{c}" for t in "RIASEC" for c in order] + ["enjoyment", "familiarity", "neither"]
    labels += [f"{t}:strength:{k}" for t in "RIASEC" for k in ("inactive", "dontmind", "mild", "strong")]
    return {i: lab for i, lab in enumerate(labels)}


class ModelOrderDeberta(FakeDeberta):
    """Like the real checkpoint: predict() returns probabilities in the model's id2label order."""

    def __init__(self, table=None, id2label=None):
        self.id2label = id2label or model_labels()
        super().__init__(table)

    def load(self, checkpoint_dir):
        self.model = SimpleNamespace(config=SimpleNamespace(id2label=self.id2label))


class TestModelLabelOrder(unittest.TestCase):
    S = "I enjoy building and repairing machines."

    def test_reads_the_models_class_order(self):
        neutral = [0.1, 0.8, 0.1]                                  # negative, neutral, positive
        result = {"direction": [[0.05, 0.10, 0.85]] + [neutral] * 5,   # R positive in the model's order
                  "strength": [1.0, 0, 0, 0, 0, 0], "reason": [0.9, 0.05, 0.05]}
        ex = ModelOrderDeberta({self.S: result})
        self.assertEqual(ex.direction_names, {0: "negative", 1: "neutral", 2: "positive"})
        slots = ex.read(self.S).slots
        self.assertEqual([(r.slot, r.sign, r.strength, r.confidence, r.reason) for r in slots],
                         [("R", 1, 1.0, 0.85, "enjoyment")])   # positive stays positive; neutral types dropped

    def test_falls_back_to_labels_json_without_model_labels(self):
        self.assertEqual(FakeDeberta().direction_names, {0: "neutral", 1: "positive", 2: "negative"})

    def test_rejects_inconsistent_order_between_types(self):
        labels = model_labels()
        labels[3], labels[4] = labels[4], labels[3]                # type I: neutral before negative
        with self.assertRaisesRegex(ValueError, "different class order"):
            ModelOrderDeberta(id2label=labels)


if __name__ == "__main__":
    unittest.main()
