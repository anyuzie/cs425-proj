"""Explanation map support (SEMANTIC_LEDGER_PLAN section 10): rank snapshots, hexagon, t-SNE artifact."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import fakes  # noqa: F401 - registers the explicitly named test upstream components
from m5_fixture import SCRIPTS, make_system
from pipeline.interfaces import State
from recommender.build_occupation_map import (
    HEX_CORNERS, build, codes_digest, hexagon_layout, hexagon_point, map_payload, semantic_status)
from recommender.data_loader import load_eligible_occupations
from recommender.models.ranker import RankingPipeline

CATALOG = Path("data/occupations.csv")


class HexagonTests(unittest.TestCase):
    def test_formula(self):
        self.assertEqual(hexagon_point({"R": 2.0, "I": -1.0}), list(HEX_CORNERS["R"]))
        self.assertIsNone(hexagon_point({t: -0.5 for t in "RIASEC"}))
        mid = hexagon_point({"R": 1.0, "S": 1.0})   # opposite types cancel
        self.assertAlmostEqual(mid[0], 0.0, places=5)
        self.assertAlmostEqual(mid[1], 0.0, places=5)
        # Only the given (observed) types pull.
        self.assertEqual(hexagon_point({"R": 1.0, "A": 1.0}, types=("A",)), list(HEX_CORNERS["A"]))

    def test_layout_is_in_code_order(self):
        occupations = load_eligible_occupations(CATALOG)
        layout = hexagon_layout(list(reversed(occupations)))
        ordered = sorted(occupations, key=lambda o: o.code)
        self.assertEqual(len(layout["coords"]), len(ordered))
        self.assertEqual(layout["coords"][0], hexagon_point(ordered[0].z_scores) or [0.0, 0.0])


class RankSnapshotTests(unittest.TestCase):
    def test_snapshot_covers_pool_and_matches_ranking(self):
        pipeline = RankingPipeline(CATALOG, config=Path("configs/ranking/m1_cosine_bm25.json"))
        args = ({"R": .8, "I": .5, "A": 0., "S": -.4, "E": 0., "C": 0.}, {"R", "I", "S"},
                ["I love fixing machines and analysing data."], 100)
        response, snapshot = pipeline.rank_with_snapshot(*args)
        self.assertEqual([asdict(o) for o in pipeline.rank(*args).occupations], [asdict(o) for o in response.occupations])
        self.assertEqual(set(snapshot["fused"]), pipeline.eligible_codes)
        self.assertEqual(sorted(snapshot["fused"].values()), list(range(1, pipeline.eligible_count + 1)))
        for o in response.occupations:
            self.assertEqual(snapshot["fused"][o.code], o.rank)
            for name, signal in o.signals.items():
                self.assertEqual(snapshot[name][o.code], signal.rank)
        self.assertEqual(set(snapshot), {"fused", "structured", "lexical"})

    def test_no_signal_has_no_snapshot(self):
        pipeline = RankingPipeline(CATALOG, config=Path("configs/ranking/m1_cosine.json"))
        response, snapshot = pipeline.rank_with_snapshot({t: 0. for t in "RIASEC"}, set(), [], 100)
        self.assertEqual(response.status, "no_signal")
        self.assertIsNone(snapshot)

    def test_trace_rows_carry_arrays_in_map_order(self):
        system = make_system(Path("configs/ranking/m1_cosine_bm25.json"))
        state = State()
        system.start(state)
        for text in SCRIPTS[0]:
            system.step(state, text)
        codes = sorted(system.ranker.eligible_codes)
        for row in system.trace(state)["ranking_trace"]:
            snap = row["rank_snapshot"]
            self.assertEqual(snap["codes_sha256"], codes_digest(codes))
            self.assertEqual(len(snap["fused"]), len(codes))
            for o in row["occupations"]:
                i = codes.index(o["code"])
                self.assertEqual(snap["fused"][i], o["rank"])
                self.assertEqual(snap["structured"][i], o["signals"]["structured"]["rank"])
            self.assertEqual(set(row["slot_confidence"]), set(row["observed_types"]))
        # Replay export keeps its schema.
        record = system.tracker_record(state, profile_id="p", revision="r")
        self.assertEqual(set(record["turns"][0]), {"text", "interests", "observed_types"})


class SemanticArtifactTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        occupations = load_eligible_occupations(CATALOG)
        self.codes = sorted(o.code for o in occupations)
        self.occupations = occupations
        index = self.tmp / "index"
        index.mkdir()
        vectors = np.random.default_rng(0).normal(size=(len(self.codes), 16)).astype("float32")
        np.save(index / "vectors.npy", vectors / np.linalg.norm(vectors, axis=1, keepdims=True))
        (index / "codes.json").write_text(json.dumps(self.codes))
        (index / "manifest.json").write_text(json.dumps({"checkpoint_sha256": "c" * 64, "catalog_sha256": "d" * 64}))
        self.index = index

    def test_build_is_deterministic_refuses_overwrite_and_detects_staleness(self):
        first = build(self.tmp / "a", self.index, CATALOG)
        second = build(self.tmp / "b", self.index, CATALOG)
        self.assertEqual((first / "coords.json").read_bytes(), (second / "coords.json").read_bytes())
        with self.assertRaises(FileExistsError):
            build(first, self.index, CATALOG)
        coords, reason = semantic_status(first, self.index, self.codes)
        self.assertIsNone(reason)
        self.assertEqual(len(coords), len(self.codes))
        self.assertTrue(all(-0.5 <= v <= 0.5 for point in coords for v in point))
        manifest = json.loads((self.index / "manifest.json").read_text())
        (self.index / "manifest.json").write_text(json.dumps({**manifest, "checkpoint_sha256": "e" * 64}))
        self.assertIn("checkpoint", semantic_status(first, self.index, self.codes)[1])

    def test_missing_artifact_hides_only_semantic(self):
        payload = map_payload(self.occupations, self.tmp / "missing", self.index)
        self.assertIsNone(payload["semantic"])
        self.assertIn("missing", payload["semantic_unavailable"])
        self.assertEqual(len(payload["hexagon"]["coords"]), len(self.codes))
        self.assertEqual(payload["codes_sha256"], codes_digest(self.codes))


class EndpointTests(unittest.TestCase):
    def test_map_endpoint_matches_snapshot_order(self):
        from fastapi.testclient import TestClient
        from app.server import app
        config = json.loads(Path("tests/test_config.json").read_text())
        config["components"]["ranker"] = "configured"
        config["settings"].update(input_mode="mock_tracker", max_turns=2)
        config["paths"]["ranking_config"] = str(Path("configs/ranking/m1_cosine.json").resolve())
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "system.json"
            source.write_text(json.dumps(config), encoding="utf-8")
            with patch.dict(os.environ, {"CM_CONFIG": str(source), "CM_LOG_DIR": directory, "CM_STUDY_MODE": "0"}), \
                    TestClient(app) as client:
                body = client.get("/api/occupation_map").json()
                self.assertEqual(len(body["codes"]), len(body["hexagon"]["coords"]))
                session = client.post("/api/session", json={"debug": True}).json()
                turn = client.post("/api/chat", json={"session_id": session["session_id"],
                                                      "text": "I love working with machines and tools."}).json()
                snap = turn["trace"]["ranking_trace"][-1]["rank_snapshot"]
                self.assertEqual(snap["codes_sha256"], body["codes_sha256"])
                self.assertEqual(len(snap["fused"]), len(body["codes"]))


if __name__ == "__main__":
    unittest.main()
