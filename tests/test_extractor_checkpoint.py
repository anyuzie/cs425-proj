"""Extractor checkpoint adapter regressions; tensor fixtures do not claim model accuracy."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from pipeline.extractor import Extractor


class CheckpointTests(unittest.TestCase):
    def test_import_does_not_load_model_dependencies(self):
        result = subprocess.run([sys.executable, "-c",
            "import pipeline.extractor, sys; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_checkpoint_explains_config_path(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "paths.extractor_checkpoint"):
                Extractor(Path(directory), Path("data/labels.json"))

    def test_unlabelled_checkpoint_heads_are_rejected(self):
        class Fixture(Extractor):
            def load(self, checkpoint_dir):
                pass
        extractor = Fixture(Path("unused"), Path("data/labels.json"))
        with self.assertRaisesRegex(ValueError, "id2label"):
            extractor._configure_heads(SimpleNamespace(id2label={i: f"LABEL_{i}" for i in range(45)}))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "torch required for head adapter fixture")
    def test_positive_negative_neutral_and_inference_context(self):
        import torch
        test = self
        calls = []

        class Batch(dict):
            def to(self, device):
                return self

        def tokenizer(*args, **kwargs):
            calls.append((args, kwargs))
            return Batch(input_ids=torch.tensor([[1, 2]]))

        class Model:
            def __call__(self, **kwargs):
                test.assertFalse(torch.is_grad_enabled())
                # Saved teammate checkpoint order is negative, neutral, positive.
                direction = torch.tensor([[0., 0., 9.], [9., 0., 0.]] + [[0., 9., 0.]] * 4)
                strength = torch.tensor([[0., 0., 0., 9.]] * 6)
                return SimpleNamespace(logits=torch.cat((direction.flatten(), torch.tensor([9., 0., 0.]), strength.flatten())).unsqueeze(0))

        class Fixture(Extractor):
            def load(self, checkpoint_dir):
                self.device, self.tok, self.model = "cpu", tokenizer, Model()
                self.cfg = {"max_len": 128, "T_slot": 1., "T_reason": 1.}
                names = ([f"{t}:{name}" for t in "RIASEC" for name in ("negative", "neutral", "positive")]
                         + ["enjoyment", "familiarity", "neither"]
                         + [f"{t}:strength:{name}" for t in "RIASEC" for name in ("inactive", "dontmind", "mild", "strong")])
                self._configure_heads(SimpleNamespace(id2label=dict(enumerate(names))))

        extractor = Fixture(Path("unused"), Path("data/labels.json"))
        reading = extractor.read("I enjoy machines but dislike research.", "What work do you enjoy?")
        slots = {s.slot: s for s in reading.slots}
        self.assertEqual(set(slots), {"R", "I"})
        self.assertEqual(slots["R"].value, 1.)
        self.assertEqual(slots["I"].value, -1.)
        self.assertEqual(calls[0][0], ("What work do you enjoy?", "I enjoy machines but dislike research."))
        self.assertIs(calls[0][1]["truncation"], True)
        from m5_fixture import make_system
        from pipeline.interfaces import State
        system = make_system()
        system.extractor = extractor
        state = State()
        system.step(state, "I enjoy machines but dislike research.")
        self.assertEqual(state.ranking_trace[0]["interests"]["R"], 1.)
        self.assertEqual(state.ranking_trace[0]["interests"]["I"], -1.)
        self.assertEqual(len(state.ranking), 100)

