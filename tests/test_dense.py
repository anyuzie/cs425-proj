"""M4 aggregation/cache correctness; fake embeddings only in isolated fixtures."""

from copy import deepcopy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys
from unittest.mock import patch

from recommender.components import ComponentInput
from recommender.dense_config import BASELINE_PARAMETERS
from recommender.schemas import Occupation, TYPES

AVAILABLE = all(importlib.util.find_spec(name) for name in ("numpy", "faiss", "sentence_transformers", "torch"))
if AVAILABLE:
    import numpy as np
    from recommender.dense_ranker import DenseRanker, Task, load_tasks, sentences, MiniLMEncoder
    from recommender.build_dense import build


class FixtureEncoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        mapping = {"first": [1, 0], "second": [0, 1], "third": [.6, .8]}
        return np.array([mapping[text.rstrip(".!?")] for text in texts], dtype="float32")


@unittest.skipUnless(AVAILABLE, "install requirements-ranking.txt for dense fixtures")
class DenseTests(unittest.TestCase):
    def test_disabled_dense_does_not_import_model_dependencies(self):
        code = "from recommender.ranker import RankingPipeline; import sys; RankingPipeline(); assert 'torch' not in sys.modules; assert 'faiss' not in sys.modules"
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_evidence_cannot_reference_an_unranked_occupation(self):
        from recommender.components import ComponentRanking, ComponentScore
        from recommender.schemas import TaskEvidence
        with self.assertRaisesRegex(ValueError, "ranked occupation"):
            ComponentRanking("usable", (ComponentScore("a", 1, 1),),
                             evidence=(TaskEvidence("1", "outside", "task", .5),))

    def test_paired_dense_runs_reject_changed_task_corpus(self):
        from recommender.compare import paired_comparison
        from recommender.evaluation import METRICS
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {key: "same" for key in ("input_mode", "split", "dataset_sha256", "inputs_sha256", "pool", "eligibility", "ranking_depth", "metric_definitions", "evaluator_version")}
            row = {"profile_id": "p", "target_rank": 1, **dict.fromkeys(METRICS, 1.0)}
            for mode, tasks in (("a", "original"), ("b", "changed")):
                (root / mode).mkdir()
                (root / mode / "manifest.json").write_text(json.dumps({**manifest, "run_id": mode, "auxiliary_dataset_sha256": {"tasks.csv": tasks}}))
                (root / mode / "per_profile.json").write_text(json.dumps([row]))
            with self.assertRaisesRegex(ValueError, "auxiliary_dataset"):
                paired_comparison(root / "a", root / "b", samples=10)

    def occupations(self):
        return [Occupation(code, code, dict.fromkeys(TYPES, 0)) for code in "abc"]

    def test_hand_aggregation_exact_search_and_missing_task_occupation(self):
        encoder = FixtureEncoder()
        tasks = [Task("a", "1", "first"), Task("a", "2", "second"), Task("b", "3", "third")]
        ranker = DenseRanker(self.occupations(), tasks, [[1, 0], [0, 1], [.6, .8]], encoder)
        result = ranker.score(ComponentInput(user_sentences=("first. second",)))
        self.assertEqual([item.code for item in result.occupations], ["a", "b"])
        self.assertAlmostEqual(result.occupations[0].score, 1)
        self.assertAlmostEqual(result.occupations[1].score, .7, places=6)
        self.assertTrue(all(e.status == "provisional" for e in result.evidence))
        self.assertEqual({(e.occupation_code, e.task_id) for e in result.evidence}, {("a", "1"), ("a", "2"), ("b", "3")})
        self.assertEqual(encoder.calls, [["first.", "second"]])

    def test_empty_text_does_not_encode_and_query_never_reembeds_corpus(self):
        encoder = FixtureEncoder()
        ranker = DenseRanker(self.occupations(), [Task("a", "1", "corpus text")], [[1, 0]], encoder)
        for text in ("", " \n.!?"):
            self.assertEqual(ranker.score(ComponentInput(user_sentences=(text,))).status, "unusable")
        self.assertEqual(encoder.calls, [])
        ranker.score(ComponentInput(user_sentences=("first",)))
        ranker.score(ComponentInput(user_sentences=("second",)))
        self.assertEqual(encoder.calls, [["first"], ["second"]])

    def test_natural_sentence_segmentation_keeps_negation_and_repetition(self):
        self.assertEqual(sentences(("I don't enjoy repairs!\nI prefer helping people.", "I prefer helping people.")),
                         ["I don't enjoy repairs!", "I prefer helping people.", "I prefer helping people."])

    def test_negative_similarity_and_stable_ties(self):
        ranker = DenseRanker(self.occupations(), [Task("b", "2", "x"), Task("a", "1", "y")],
                             [[-1, 0], [-1, 0]], FixtureEncoder())
        result = ranker.score(ComponentInput(user_sentences=("first",)))
        self.assertEqual([(o.code, o.score) for o in result.occupations], [("a", -1), ("b", -1)])

    def test_nonunit_nonfinite_and_bad_owner_vectors_fail(self):
        for vectors in ([[0, 0]], [[2, 0]], [[float("nan"), 0]]):
            with self.assertRaises(ValueError):
                DenseRanker(self.occupations(), [Task("a", "1", "x")], vectors, FixtureEncoder())
        with self.assertRaisesRegex(ValueError, "eligible"):
            DenseRanker(self.occupations(), [Task("outside", "1", "x")], [[1, 0]], FixtureEncoder())

    def fixture(self, root):
        tasks = root / "tasks.csv"
        with tasks.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["code", "Task ID", "Task"])
            writer.writeheader()
            writer.writerows([{"code": code, "Task ID": task_id, "Task": text}
                             for code, task_id, text in (("a", "1", "first"), ("a", "2", "second"), ("b", "3", "third"))])
        parameters = deepcopy(BASELINE_PARAMETERS) | {"tasks_path": str(tasks), "cache_dir": str(root / "cache"), "dimension": 2}
        return parameters

    def test_cache_reload_identical_no_reembedding_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            parameters = self.fixture(Path(directory))
            encoder = FixtureEncoder()
            with patch("recommender.build_dense.BASELINE_PARAMETERS", parameters), patch("recommender.dense_ranker.BASELINE_PARAMETERS", parameters), \
                    patch("recommender.dense_ranker.model_identity", return_value="test-only-model"), \
                    patch("recommender.dense_ranker.MiniLMEncoder", return_value=encoder):
                metadata = build(self.occupations(), parameters, encoder=encoder, model_hash="test-only-model")
                self.assertGreater(metadata["artifact_bytes"], 0)
                self.assertEqual(encoder.calls, [["first", "second", "third"]])
                first = DenseRanker.from_cache(self.occupations(), parameters)
                second = DenseRanker.from_cache(self.occupations(), parameters)
                inputs = ComponentInput(user_sentences=("first",))
                self.assertEqual(first.score(inputs), second.score(inputs))
                self.assertEqual(encoder.calls[1:], [["first"], ["first"]])
                with self.assertRaisesRegex(ValueError, "already exists"):
                    build(self.occupations(), parameters, encoder=encoder, model_hash="test-only-model")

    def test_stale_and_corrupt_cache_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            parameters = self.fixture(Path(directory))
            with patch("recommender.build_dense.BASELINE_PARAMETERS", parameters), patch("recommender.dense_ranker.BASELINE_PARAMETERS", parameters), \
                    patch("recommender.dense_ranker.model_identity", return_value="test-only-model"):
                build(self.occupations(), parameters, encoder=FixtureEncoder(), model_hash="test-only-model")
                with self.assertRaisesRegex(ValueError, "stale"):
                    DenseRanker.from_cache(self.occupations()[:1], parameters)
                owners = Path(parameters["cache_dir"]) / "owners.json"
                owners.write_text(owners.read_text() + " ")
                with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                    DenseRanker.from_cache(self.occupations(), parameters)

    def test_duplicate_tasks_and_out_of_pool_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            parameters = self.fixture(Path(directory))
            self.assertEqual({t.code for t in load_tasks(parameters["tasks_path"], {"a"})}, {"a"})
            path = Path(parameters["tasks_path"])
            path.write_text(path.read_text() + "a,1,first\n")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_tasks(path, {"a", "b"})


@unittest.skipUnless(AVAILABLE and Path(BASELINE_PARAMETERS["cache_dir"]).is_dir(), "build the real M4 cache for integration fixtures")
class RealMiniLMTests(unittest.TestCase):
    def test_real_cached_pipeline_provisional_evidence_and_manifest(self):
        from recommender.ranker import RankingPipeline
        from recommender.schemas import OracleInput
        from recommender.data_loader import read_profiles
        pipeline = RankingPipeline(config=Path("configs/ranking/m4_three_way.json"))
        profile = read_profiles(Path("data/dev_profiles.jsonl"))[0]
        response = pipeline.rank_oracle(OracleInput.from_profile(profile))
        self.assertEqual(len(response.occupations), 378)
        self.assertTrue(all(o.rrf_score is not None for o in response.occupations))
        for occupation in response.occupations:
            self.assertIn("semantic", occupation.signals)
            self.assertLessEqual(len(occupation.evidence), 3)
            self.assertTrue(all(e.occupation_code == occupation.code and e.status == "provisional" for e in occupation.evidence))
        metadata = pipeline.component_metadata["m4"]
        self.assertEqual(metadata["task_count"], 8243)
        self.assertEqual(metadata["identity"]["parameters"]["model_revision"], BASELINE_PARAMETERS["model_revision"])
        dense = RankingPipeline(config=Path("configs/ranking/m4_dense.json"))
        self.assertEqual(dense.rank(dict.fromkeys(TYPES, 0), set(), []).status, "no_signal")

    def test_real_paraphrase_retrieves_meaningful_task(self):
        encoder = MiniLMEncoder(BASELINE_PARAMETERS)
        tasks = [Task("repair", "1", "Diagnose mechanical failures and repair industrial machinery."),
                 Task("nursing", "2", "Provide nursing care and administer medication to patients."),
                 Task("teaching", "3", "Prepare classroom lessons and teach mathematics to children.")]
        occupations = [Occupation(t.code, t.code, dict.fromkeys(TYPES, 0)) for t in tasks]
        ranker = DenseRanker(occupations, tasks, encoder.encode([t.text for t in tasks]), encoder)
        result = ranker.score(ComponentInput(user_sentences=("I enjoy figuring out why factory equipment stops working and getting it running again.",)))
        self.assertEqual(result.occupations[0].code, "repair")
        match = next(e for e in result.evidence if e.occupation_code == "repair")
        self.assertEqual(match.task_id, "1")
        self.assertEqual(match.status, "provisional")


if __name__ == "__main__":
    unittest.main()
