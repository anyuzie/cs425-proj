"""Append-only terminal observations; predictions are never training labels."""
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from uuid import uuid4


class UtteranceCollection:
    def __init__(self, directory, metadata):
        self.session_id = uuid4().hex
        self.path = Path(directory) / (self.session_id + ".jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.append("session", None, metadata=metadata)

    def append(self, event, entry_id, **payload):
        record = {"schema_version": 1, "event": event,
                  "session_id": self.session_id, "entry_id": entry_id,
                  "timestamp": datetime.now(timezone.utc).isoformat(), **payload}
        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def submitted(self, raw_text, text):
        entry_id = uuid4().hex
        self.append("submitted", entry_id, raw_text=raw_text, text=text)
        return entry_id

    def completed(self, entry_id, processed, reading, interests, observed, response):
        clauses = [{**asdict(c), "activity_detected": c.activity_detected,
                    "job_nature_detected": c.job_nature_detected, "outcome": c.outcome}
                   for c in processed.clauses]
        self.append("completed", entry_id, predictions={
            "clauses": clauses, "readings": asdict(reading),
            "interests": dict(interests), "observed_types": sorted(observed),
            "ranking": asdict(response)}, annotation_status="unlabelled")

    def failed(self, entry_id, error, processed=None):
        self.append("failed", entry_id, error={"type": type(error).__name__, "message": str(error)},
                    partial_processing=asdict(processed) if processed is not None else None)
