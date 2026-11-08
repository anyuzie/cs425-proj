"""Read-only collection input and append-only human annotation storage."""
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
from threading import RLock
from .contracts import Annotation, validate_annotation


class Store:
    def __init__(self, collection_dir, annotation_dir):
        self.collection_dir = Path(collection_dir).resolve()
        self.annotation_dir = Path(annotation_dir).resolve()
        if self.annotation_dir == self.collection_dir or self.collection_dir in self.annotation_dir.parents:
            raise ValueError("Annotation output must be separate from collection input")
        self.lock = RLock()
        self.warnings = []

    def entries(self):
        entries = {}
        self.warnings = []
        for path in sorted(self.collection_dir.glob("*.jsonl")):
            session = {}
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                try:
                    row = json.loads(line)
                    event = row["event"]
                    if event == "session":
                        session = row.get("metadata", {})
                        continue
                    key = (row["session_id"], row["entry_id"])
                    if not all(isinstance(k, str) and k for k in key):
                        raise ValueError("missing entry identity")
                    identifier = sha256((key[0] + ":" + key[1]).encode()).hexdigest()
                    if event == "submitted":
                        text = row["text"]
                        if not isinstance(text, str):
                            raise ValueError("invalid text")
                        fingerprint = sha256(text.encode("utf-8")).hexdigest()
                        entries[identifier] = {"id": identifier, "entry_id": key[1], "session_id": key[0],
                            "created_at": row.get("timestamp", datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()),
                            "source_file": path.name, "text": text, "raw_text": row.get("raw_text", text),
                            "source_sha256": fingerprint, "metadata": session,
                            "prediction": None, "processing_status": "pending"}
                    elif event in {"completed", "failed"} and identifier in entries:
                        entries[identifier]["prediction"] = row.get("predictions", row.get("partial_processing"))
                        entries[identifier]["processing_status"] = event
                except (ValueError, KeyError, TypeError) as error:
                    self.warnings.append(f"{path.name}:{line_no}: {error}")
        return entries

    def history(self, identifier):
        if len(identifier) != 64 or any(c not in "0123456789abcdef" for c in identifier):
            raise ValueError("Invalid entry identifier")
        path = self.annotation_dir / (identifier + ".jsonl")
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []

    def get(self, identifier):
        entry = self.entries().get(identifier)
        if entry is None:
            raise KeyError(identifier)
        history = self.history(identifier)
        return {**entry, "annotation": history[-1] if history else None}

    def save(self, identifier, annotation):
        with self.lock:
            entry = self.get(identifier)
            revision = entry["annotation"]["revision"] if entry["annotation"] else 0
            if annotation.expected_revision != revision:
                raise FileExistsError("Annotation changed; reload before saving")
            if annotation.source_sha256 != entry["source_sha256"]:
                raise FileExistsError("Source text changed; reload before saving")
            validate_annotation(annotation, entry["text"])
            record = {"schema_version": 2 if annotation.workflow == "clause_labels" else 1, "revision": revision + 1,
                "saved_at": datetime.now(timezone.utc).isoformat(), "id": identifier,
                "entry_id": entry["entry_id"], "session_id": entry["session_id"],
                **annotation.model_dump(exclude={"expected_revision"})}
            self.annotation_dir.mkdir(parents=True, exist_ok=True)
            with (self.annotation_dir / (identifier + ".jsonl")).open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            return record
