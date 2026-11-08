"""Standalone annotation API/storage/export regressions, no ranking models."""
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from fastapi.testclient import TestClient
from annotation_tool.server import create_app
from annotation_tool.store import Store
from annotation_tool.export import export_rows


class AnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "collected"
        self.source.mkdir()
        self.text = "😀 I repair machines outdoors."
        self.path = self.source / "session.jsonl"
        rows = [{"event": "session", "session_id": "session", "metadata": {}},
                {"event": "submitted", "session_id": "session", "entry_id": "entry",
                 "text": self.text, "raw_text": self.text}]
        self.path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        self.original = self.path.read_bytes()
        self.store = Store(self.source, self.root / "annotations")
        self.identifier = next(iter(self.store.entries()))
        self.client = TestClient(create_app(self.source, self.root / "annotations"))
        self.url = "/api/entries/" + self.identifier

    def payload(self, status="reviewed"):
        return {"annotator": "reviewer-1", "group_id": "contributor-1", "status": status,
                "source_sha256": sha256(self.text.encode()).hexdigest(), "expected_revision": 0,
                "clauses": [{"start": 0, "end": len(self.text), "activity": True, "job_nature": True,
                             "activity_evidence": [{"start": 4, "end": 19}],
                             "job_nature_evidence": [{"start": 20, "end": 28}], "note": ""}], "note": ""}

    def test_roundtrip_export_unicode_and_no_source_mutation(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(len(self.client.get("/api/entries").json()["entries"]), 1)
        response = self.client.post(self.url, json=self.payload())
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["revision"], 1)
        dataset = self.client.get("/api/export").json()
        row = dataset["rows"][0]
        self.assertEqual(row["context"], self.text)
        self.assertEqual(row["evidence"]["activity"][0]["text"], "repair machines")
        self.assertEqual(row["labels"], {"activity": True, "job_nature": True})
        self.assertEqual(row["group_id"], "contributor-1")
        self.assertNotIn("predictions", row)
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_validation_rejects_missing_labels_evidence_and_word_gaps(self):
        payload = self.payload()
        for mutation in [lambda c:c.update(activity=None), lambda c:c.update(activity_evidence=[]),
                         lambda c:c.update(start=4), lambda c:c.update(end=999),
                         lambda c:c.update(job_nature_evidence=[{"start": 0,"end": 999}])]:
            c = json.loads(json.dumps(payload))
            mutation(c["clauses"][0])
            self.assertEqual(self.client.post(self.url, json=c).status_code, 422)

    def test_draft_neither_and_revision_conflicts(self):
        payload = self.payload("draft")
        payload["clauses"][0].update(activity=None, job_nature=None,activity_evidence=[],job_nature_evidence=[])
        self.assertEqual(self.client.post(self.url,json=payload).status_code, 200)
        self.assertEqual(self.client.get("/api/export").json()["rows"], [])
        self.assertEqual(self.client.post(self.url,json=payload).status_code, 409)
        payload.update(expected_revision=1,status="reviewed")
        payload["clauses"][0].update(activity=False,job_nature=False)
        self.assertEqual(self.client.post(self.url,json=payload).status_code, 200)
        self.assertEqual(self.client.get("/api/export").json()["rows"][0]["outcome"], "neither")
        payload.update(expected_revision=2,status="needs_review")
        self.client.post(self.url,json=payload)
        self.assertEqual(self.client.get("/api/export").json()["rows"], [])
        self.assertEqual(len(self.store.history(self.identifier)),3)

    def test_stale_source_rejected_and_pending_input_available(self):
        self.assertEqual(self.client.get(self.url).json()["processing_status"], "pending")
        self.client.post(self.url,json=self.payload())
        rows=[json.loads(s) for s in self.path.read_text().splitlines()]
        rows[1]["text"] += " changed"
        self.path.write_text("".join(json.dumps(r)+"\n" for r in rows))
        self.assertEqual(self.client.get("/api/export").status_code,422)
        self.assertEqual(self.client.post(self.url,json={**self.payload(),"expected_revision":1}).status_code,409)

    def test_malformed_collection_line_and_path_separation(self):
        with self.path.open("a") as h:h.write('{incomplete')
        result=self.client.get("/api/entries").json()
        self.assertEqual(len(result["entries"]),1)
        self.assertTrue(result["warnings"])
        with self.assertRaises(ValueError):Store(self.source,self.source / "annotations")

    def test_simple_editable_labels_save_directly_without_coverage(self):
        payload = self.payload()
        payload.update(workflow="clause_labels")
        payload["clauses"] = [{"start": 4, "end": 19, "edited_text": "Repairing machines",
             "activity": True, "job_nature": True, "uncertain": False}]
        response = self.client.post(self.url,json=payload)
        self.assertEqual(response.status_code,200,response.text)
        dataset = Path(response.json()["training_path"])
        row=json.loads(dataset.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(row["text"], "Repairing machines")
        self.assertEqual(row["source_text"], "repair machines")
        self.assertEqual(row["outcome"], "mixed")
        self.assertTrue(row["text_edited"])
        payload["expected_revision"]=1
        payload["clauses"][0].update(uncertain=True,activity=None,job_nature=None)
        self.assertEqual(self.client.post(self.url,json=payload).status_code,200)
        rows=[json.loads(s) for s in dataset.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]["outcome"],"uncertain")
        self.assertFalse(rows[0]["training_eligible"])
        self.assertEqual(rows[0]["labels"],{"activity":None,"job_nature":None})
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_simple_labels_allow_overlapping_out_of_order_selections(self):
        payload=self.payload()
        payload.update(workflow="clause_labels")
        payload["clauses"]=[{"start":20,"end":28,"activity":False,"job_nature":True},
                            {"start":4,"end":28,"activity":True,"job_nature":True}]
        self.assertEqual(self.client.post(self.url,json=payload).status_code,200)
        payload["expected_revision"]=1
        payload["clauses"][0]["edited_text"]="  "
        self.assertEqual(self.client.post(self.url,json=payload).status_code,422)

    def test_queue_refresh_and_reviewed_moves_to_bottom(self):
        new={"event":"submitted","session_id":"session","entry_id":"new",
             "text":"New input","timestamp":"2099-01-01T00:00:00+00:00"}
        with self.path.open("a") as h:h.write(json.dumps(new)+"\n")
        queue=self.client.get("/api/entries").json()["entries"]
        self.assertEqual(queue[0]["text"],"New input")
        key=queue[0]["id"]
        data=self.client.get("/api/entries/"+key).json()
        payload={"workflow":"clause_labels","annotator":"local-reviewer","status":"reviewed",
                 "source_sha256":data["source_sha256"],"expected_revision":0,"group_id":"test",
                 "clauses":[{"start":0,"end":9,"activity":None,"job_nature":None,"uncertain":True}]}
        self.assertEqual(self.client.post("/api/entries/"+key,json=payload).status_code,200)
        queue=self.client.get("/api/entries").json()["entries"]
        self.assertEqual(queue[-1]["id"],key)
        self.assertEqual(queue[-1]["status"],"reviewed")
