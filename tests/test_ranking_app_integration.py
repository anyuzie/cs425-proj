"""Real ranking through HTTP; only unfinished upstream components use explicit fakes."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import fakes  # Registers explicitly named test upstream components.
from app.server import app
from pipeline.system import System
from recommender.models.ranker import RankingPipeline


class RankingAppIntegrationTests(unittest.TestCase):
    def test_configured_rankings_reach_policy_api_and_logs(self):
        for selection in ("m1_cosine.json", "m1_cosine_bm25.json"):
            with self.subTest(selection=selection), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config = json.loads(Path("tests/test_config.json").read_text())
                config["components"]["ranker"] = "configured"
                config["settings"].update(input_mode="mock_tracker", max_turns=2)
                config["paths"]["ranking_config"] = str(
                    Path("configs/ranking", selection).resolve())
                source = root / "system.json"
                source.write_text(json.dumps(config), encoding="utf-8")
                with patch.dict(os.environ, {"CM_CONFIG": str(source), "CM_LOG_DIR": str(root / "logs"),
                                             "CM_STUDY_MODE": "0"}), TestClient(app) as client:
                    system = app.state.system
                    ranker = system.ranker
                    instances = dict(ranker._components)
                    self.assertEqual(ranker.component_identifiers["m1"], "cosine_similarity@baseline_v1")
                    self.assertEqual(client.get("/").status_code, 200)
                    session = client.post("/api/session", json={"debug": True}).json()
                    history = []
                    for text in ("I love working with machines and tools.", "I love analysing data."):
                        history.append(text)
                        response = client.post("/api/chat", json={"session_id": session["session_id"], "text": text})
                        self.assertEqual(response.status_code, 200, response.text)
                        body = response.json()
                        row = body["trace"]["ranking_trace"][-1]
                        self.assertEqual(row["user_sentences"], history)
                        expected = ranker.rank(row["interests"], set(row["observed_types"]), history, 100)
                        self.assertEqual(row["occupations"], [asdict(o) for o in expected.occupations])
                        state = app.state.sessions[session["session_id"]].state
                        self.assertEqual(state.ranking, expected.occupations)
                    self.assertEqual(body["action"], "RECOMMEND")
                    self.assertEqual([r["code"] for r in body["recommendations"]],
                                     [o.code for o in expected.occupations[:5]])
                    self.assertTrue(all(ranker._components[k] is v for k, v in instances.items()))
                    participant = client.post("/api/session", json={"debug": False}).json()
                    hidden = client.post("/api/chat", json={"session_id": participant["session_id"],
                                                           "text": history[0]}).json()
                    self.assertIsNone(hidden["trace"])
                    rows = [json.loads(line) for path in (root / "logs").rglob("*.jsonl")
                            for line in path.read_text(encoding="utf-8").splitlines()]
                    logged = next(r for r in rows if r["session_id"] == participant["session_id"]
                                  and r["user_text"] is not None)
                    self.assertEqual(logged["input_mode"], "mock_tracker")
                    self.assertEqual(logged["resolved_ranking"], ranker.resolved_config)
                    self.assertEqual(logged["ranking_input"]["user_sentences"], [history[0]])

    def test_live_system_rejects_replay_only_evidence_filter_at_startup(self):
        config = json.loads(Path("configs/ranking/m1_cosine_bm25.json").read_text())
        config["evidence_filter"] = "active_only"
        ranker = RankingPipeline(config=config)
        with self.assertRaisesRegex(ValueError, "upstream lifecycle evidence"):
            System(None, None, None, None, ranker, None, None, {})
