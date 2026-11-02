"""Export latest reviewed annotations, never predictions or automatic splits."""
from hashlib import sha256
from .contracts import Annotation, validate_annotation, outcome


def export_rows(store):
    rows, counts = [], {"reviewed_entries": 0, "skipped_entries": 0}
    for identifier, entry in store.entries().items():
        history = store.history(identifier)
        latest = history[-1] if history else None
        if not latest or latest["status"] != "reviewed":
            counts["skipped_entries"] += 1
            continue
        if latest["source_sha256"] != entry["source_sha256"]:
            raise ValueError("Reviewed annotation has stale source: " + identifier)
        fields = {k: latest[k] for k in Annotation.model_fields if k != "expected_revision" and k in latest}
        annotation = Annotation.model_validate({**fields, "expected_revision": latest["revision"]})
        validate_annotation(annotation, entry["text"])
        counts["reviewed_entries"] += 1
        for index, clause in enumerate(annotation.clauses):
            rows.append({"schema_version": 2, "example_id": f"{identifier}:r{latest['revision']}:c{index}",
                "entry_id": entry["entry_id"], "session_id": entry["session_id"],
                "group_id": latest["group_id"], "context": entry["text"],
                "context_sha256": entry["source_sha256"], "annotation_revision": latest["revision"],
                "annotator": latest["annotator"], "clause_index": index,
                "start": clause.start, "end": clause.end,
                "source_text": entry["text"][clause.start:clause.end],
                "text": clause.edited_text if clause.edited_text is not None else entry["text"][clause.start:clause.end],
                "text_edited": clause.edited_text is not None and clause.edited_text != entry["text"][clause.start:clause.end],
                "label_status": "uncertain" if clause.uncertain else "labelled",
                "training_eligible": not clause.uncertain,
                "offsets_apply_to": "source_text in context, not edited text",
                
                "labels": {"activity": clause.activity, "job_nature": clause.job_nature},
                "outcome": "uncertain" if clause.uncertain else outcome(clause.activity, clause.job_nature),
                "evidence": {name: [{"start": s.start, "end": s.end,
                    "text": entry["text"][s.start:s.end]} for s in spans]
                    for name, spans in [("activity", clause.activity_evidence), ("job_nature", clause.job_nature_evidence)]}})
    return rows, {**counts, "clauses": len(rows), "label_order": ["activity", "job_nature"],
                  "uncertain_policy": "Retained with null labels and training_eligible=false; exclude from supervised binary-label training",
                  "offset_unit": "Unicode code points", "offset_reference": "context",
                  "split_policy": "No automatic split. Group by contributor/group_id and deduplicate context_sha256 before splitting."}


def write_current_dataset(store, directory):
    """Materialize latest reviews without duplicating prior revisions in training data."""
    import json
    import os
    from pathlib import Path
    from uuid import uuid4
    rows, manifest = export_rows(store)
    directory = Path(directory).resolve()
    if directory == store.collection_dir or store.collection_dir in directory.parents:
        raise ValueError("Training output must be separate from collection input")
    if directory == store.annotation_dir or store.annotation_dir in directory.parents:
        raise ValueError("Training output must be separate from annotation history")
    directory.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    manifest["dataset_sha256"] = sha256(payload.encode("utf-8")).hexdigest()
    for name, content in [("current.jsonl", payload),
                          ("current.manifest.json", json.dumps(manifest, indent=2))]:
        target = directory / name
        temporary = directory / ("." + name + "." + uuid4().hex + ".tmp")
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    return directory / "current.jsonl"
