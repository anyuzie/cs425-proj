"""Training safeguards and objective tests; no model downloads."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from contextlib import redirect_stdout
import io

from scripts.finetune_nli import binary_loss, binary_metrics, read_data, compare

try:
    import torch
except ImportError:
    torch = None


class FineTuningTests(unittest.TestCase):
    def test_generated_data_and_existing_holdout_are_disjoint(self):
        root = Path(__file__).resolve().parents[1]
        rows = read_data(root / "data/nli_natural_v1.jsonl")
        synthetic = [r for r in rows if r.get("source") == "generated_probing_v1"]
        self.assertEqual(len(synthetic), 500)
        self.assertEqual(sum(r["label"] == "contradiction" for r in synthetic), 250)
        self.assertEqual(len({(r["a"], r["b"]) for r in synthetic}), 500)
        for split, expected in (("train", 400), ("dev", 50), ("test", 50)):
            self.assertEqual(sum(r["split"] == split for r in synthetic), expected)
        self.assertTrue(all(r["split"] == "train" for r in synthetic
                            if r["group_id"] == "probing_activity_00"))

    def test_reversed_pair_leakage_rejected(self):
        rows = []
        for split in ("train", "dev", "test"):
            for label in ("contradiction", "elaboration"):
                rows.append(dict(id=f"{split}_{label}", split=split, label=label,
                                 a=f"{split} {label} a", b=f"{split} {label} b"))
        rows[-1].update(a=rows[0]["b"], b=rows[0]["a"])
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in rows))
            with self.assertRaisesRegex(ValueError, "crosses splits"):
                read_data(path)

    def test_accuracy_uses_strict_runtime_boundary(self):
        result = binary_metrics([1, 0], [.6, .6], .6)
        self.assertEqual(result["accuracy"], .5)
        self.assertEqual(result["confusion_matrix"], [[1, 0], [1, 0]])

    def test_comparison_exports_matching_rows_without_training_inference(self):
        rows = []
        for split in ("dev", "test"):
            for label in ("contradiction", "elaboration"):
                rows.append(dict(id=f"{split}_{label}", split=split, label=label,
                                 a=label, b="second sentence"))

        class FakeConfirmer:
            def __init__(self, path):
                self.path = path

            def probabilities(self, a, b):
                p = .95 if a == "contradiction" else .1
                return {"contradiction": p, "entailment": (1-p)/2, "neutral": (1-p)/2}

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            args = SimpleNamespace(output=Path(directory) / "report", base="stock",
                                   tuned=Path(directory) / "checkpoint")
            with patch("scripts.finetune_nli.NliConfirmer", FakeConfirmer), \
                 patch("scripts.finetune_nli.provenance", return_value={"data_sha256": "test"}), \
                 redirect_stdout(io.StringIO()):
                compare(args, rows)
            report = json.loads((args.output / "comparison.json").read_text())
            self.assertEqual(report["metrics"]["tuned"]["test"]["threshold_0.60"]["accuracy"], 1.)
            self.assertEqual(report["test_changes"]["unchanged"], 2)
            self.assertTrue((args.output / "predictions.csv").exists())
            self.assertTrue((args.output / "thresholds.json").exists())

    @unittest.skipIf(torch is None, "torch unavailable")
    def test_loss_pushes_contradiction_in_correct_direction_for_reordered_head(self):
        logits = torch.tensor([[0., 0., 0.], [0., 0., 0.]], requires_grad=True)
        loss = binary_loss(logits, torch.tensor([1, 0]), 2)
        loss.backward()
        self.assertLess(logits.grad[0, 2].item(), 0)
        self.assertGreater(logits.grad[1, 2].item(), 0)
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
