"""Deployment selection and checkpoint verification without loading real models."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pipeline import registry
from scripts.check_nli_checkpoint import verify

ROOT = Path(__file__).resolve().parents[1]


class NliDeploymentTests(unittest.TestCase):
    def test_real_presets_use_versioned_checkpoint(self):
        for path in (ROOT / "configs").glob("system*.json"):
            cfg = json.loads(path.read_text(encoding="utf-8"))
            if cfg["components"]["confirmer"] == "nli":
                self.assertEqual(cfg["paths"]["nli_model"], "models/nli-probing-v1/checkpoint", str(path))

    def test_missing_local_checkpoint_never_becomes_hub_lookup(self):
        with patch("pipeline.nli.NliConfirmer") as load:
            with self.assertRaisesRegex(FileNotFoundError, "check_nli_checkpoint.py"):
                registry._confirmer_nli({"paths": {"nli_model": "models/missing-test-checkpoint"}}, None)
            load.assert_not_called()

    def test_explicit_stock_model_still_supported(self):
        with patch("pipeline.nli.NliConfirmer") as load:
            registry._confirmer_nli({"paths": {"nli_model": "cross-encoder/nli-deberta-v3-base"}}, None)
            load.assert_called_once_with("cross-encoder/nli-deberta-v3-base")

    def test_verifier_detects_corrupted_and_missing_files(self):
        content = b"checkpoint fixture"
        manifest = {"files": {"weights": {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}}}
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            folder = Path(directory)
            with self.assertRaisesRegex(ValueError, "Missing"):
                verify(folder, manifest)
            (folder / "weights").write_bytes(content)
            verify(folder, manifest)
            (folder / "weights").write_bytes(b"x" * len(content))
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                verify(folder, manifest)


if __name__ == "__main__":
    unittest.main()
