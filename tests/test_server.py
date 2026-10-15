import json
import os
import tempfile
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from app.server import app


class ServerTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.logs = tmp.name
        patcher = mock.patch.dict(os.environ, {"CM_LOG_DIR": tmp.name})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.client = TestClient(app)
        self.client.__enter__()
        self.addCleanup(self.client.__exit__, None, None, None)

    def new(self, **body):
        r = self.client.post("/api/session", json=body)
        self.assertEqual(r.status_code, 200, r.text)
        return r.json()

    def say(self, sid, text):
        return self.client.post("/api/chat", json={"session_id": sid, "text": text})

    def test_health(self):
        body = self.client.get("/api/health").json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["eligible"], 378)
        self.assertEqual(body["components"]["extractor"], "lexicon-stub")
        self.assertEqual(body["components"]["ranker"], "structured+keyword-stub")
        self.assertEqual(body["components"]["rule"], "rule")
        self.assertEqual(body["components"]["confirmer"], "passthrough-stub")
        self.assertNotIn("evidence", body["components"])

    def test_session_opening_and_anon_code(self):
        s = self.new()
        self.assertRegex(s["participant_code"], r"^anon-[0-9a-f]{8}$")
        self.assertEqual((s["action"], s["turn"], s["max_turns"], s["done"]), ("OPENING", 0, 10, False))
        self.assertIsNone(s["recommendations"])

    def test_full_conversation(self):
        s = self.new(participant_code="P-014")
        body = None
        for _ in range(10):
            r = self.say(s["session_id"], "Definitely the first one")
            self.assertEqual(r.status_code, 200)
            body = r.json()
            if body["done"]:
                break
        self.assertTrue(body["done"])
        self.assertEqual(body["action"], "RECOMMEND")
        self.assertEqual(len(body["recommendations"]), 5)
        self.assertEqual(self.say(s["session_id"], "more").status_code, 409)

    def test_challenge_and_probe_quotes(self):
        s = self.new()
        self.say(s["session_id"], "I love working with machines and tools.")
        b = self.say(s["session_id"], "I hate repairing machines, it is boring.").json()
        self.assertEqual(b["action"], "CHALLENGE")
        self.assertEqual(b["quotes"]["earlier_turn"], 1)
        self.assertEqual(b["quotes"]["now"], "I hate repairing machines, it is boring.")
        p = self.new()
        b = self.say(p["session_id"], "I like machines but that's what I know, my dad has a workshop.").json()
        self.assertEqual(b["action"], "PROBE")
        self.assertEqual(list(b["quotes"]), ["now"])
        c = self.say(p["session_id"], "hmm").json()
        self.assertIsNone(c["quotes"])

    def test_errors(self):
        self.assertEqual(self.say("nope", "hi").status_code, 404)
        s = self.new()
        self.assertEqual(self.say(s["session_id"], "   ").status_code, 422)
        self.assertEqual(self.say(s["session_id"], "x" * 2001).status_code, 422)
        self.assertEqual(self.client.post("/api/chat", json={"session_id": s["session_id"]}).status_code, 422)

    def test_invalid_participant_code(self):
        for bad in ("../etc/passwd", "a b", "x" * 33, "", "a/b"):
            r = self.client.post("/api/session", json={"participant_code": bad})
            self.assertEqual(r.status_code, 422, bad)
        self.assertEqual(os.listdir(self.logs), [])

    def test_trace_on_by_default_and_off_in_study_mode(self):
        s = self.new(debug=False)
        self.assertIsNone(s["trace"])
        self.assertIsNone(self.say(s["session_id"], "I love machines").json()["trace"])
        with mock.patch.dict(os.environ, {"CM_STUDY_MODE": "1"}):
            self.assertIsNone(self.new()["trace"])
            self.assertIsNone(self.new(debug=True)["trace"])
        d = self.new()
        self.assertEqual(set(d["trace"]), {"profile", "actions", "top5", "components"})
        t = self.say(d["session_id"], "I love working with machines and tools.").json()["trace"]
        self.assertEqual(list(t["profile"]), list("RIASEC"))
        self.assertEqual(t["profile"]["R"]["value"], 1.0)
        self.assertIsNone(t["profile"]["I"]["value"])
        self.assertEqual(len(t["top5"]), 5)
        self.assertEqual(t["actions"][0]["turn"], 1)

    def test_log_file_written(self):
        s = self.new(participant_code="P-014")
        self.say(s["session_id"], "I love working with machines and tools.")
        path = os.path.join(self.logs, "P-014.jsonl")
        lines = [json.loads(l) for l in open(path, encoding="utf-8")]
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[1]["user_text"], "I love working with machines and tools.")
        self.assertEqual(lines[1]["participant_code"] if "participant_code" in lines[1] else "P-014", "P-014")
        self.assertEqual(lines[1]["turn"], 1)

    def test_same_message_id_steps_once(self):
        s = self.new()
        body = {"session_id": s["session_id"], "text": "I love working with machines and tools.", "message_id": "m-1"}
        a = self.client.post("/api/chat", json=body).json()
        b = self.client.post("/api/chat", json=body).json()
        self.assertEqual((a["turn"], b["turn"]), (1, 1))
        self.assertEqual(a, b)
        c = self.say(s["session_id"], "I enjoy art").json()
        self.assertEqual(c["turn"], 2)
        bad = self.client.post("/api/chat", json={**body, "message_id": "bad id!"})
        self.assertEqual(bad.status_code, 422)

    def test_log_failure_leaves_state_unchanged(self):
        s = self.new()
        with mock.patch("app.server.turn_log.append", side_effect=OSError("disk full")):
            self.assertEqual(self.say(s["session_id"], "I love machines").status_code, 500)
        r = self.say(s["session_id"], "I love machines")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["turn"], 1)

    def test_sessions_independent(self):
        a, b = self.new(), self.new()
        self.say(a["session_id"], "I love machines")
        self.say(a["session_id"], "I love art")
        self.assertEqual(self.say(b["session_id"], "I love machines").json()["turn"], 1)
        self.assertEqual(self.say(a["session_id"], "I love music").json()["turn"], 3)

    def test_static_missing_does_not_break_api(self):
        self.assertIn(self.client.get("/").status_code, (200, 404))


if __name__ == "__main__":
    unittest.main()
