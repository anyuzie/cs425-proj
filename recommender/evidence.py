"""M6 query selection only. Lifecycle decisions remain owned by upstream components."""

from copy import deepcopy


def validate_evidence(records, history, previous=None):
    """One stable evidence unit per user-history entry; cumulative status snapshots.

    Text is immutable, disputed evidence may be resolved to active or superseded,
    and superseded evidence cannot be resurrected under the same ID.
    """
    if not isinstance(records, list) or len(records) != len(history):
        raise ValueError("source_evidence must cover every user-history entry exactly once")
    ids = set()
    for turn, (record, text) in enumerate(zip(records, history), 1):
        if not isinstance(record, dict) or set(record) != {"id", "turn", "text", "status"}:
            raise ValueError("source evidence requires exactly id, turn, text, status")
        if not isinstance(record["id"], str) or not record["id"].strip() or record["id"] in ids:
            raise ValueError("source evidence IDs must be nonempty and unique")
        if type(record["turn"]) is not int or record["turn"] != turn or record["text"] != text:
            raise ValueError("source evidence must preserve chronological original user text")
        if record["status"] not in ("active", "disputed", "superseded"):
            raise ValueError("source evidence status must be active, disputed or superseded")
        ids.add(record["id"])
    if previous is not None:
        if len(records) < len(previous):
            raise ValueError("source evidence history cannot shrink")
        for old, new in zip(previous, records):
            if any(old[k] != new[k] for k in ("id", "turn", "text")):
                raise ValueError("source evidence IDs and original text must remain stable across turns")
            if old["status"] == "superseded" and new["status"] != "superseded":
                raise ValueError("superseded evidence cannot be reactivated")
    return deepcopy(records)


def select_evidence(history, records, mode):
    records = validate_evidence(records, history)
    selected = [r for r in records if mode == "raw" or r["status"] == "active"]
    diagnostics = {"evidence_filter": mode, "source_evidence": records,
        "selected_source_ids": [r["id"] for r in selected],
        "excluded_source_ids": [r["id"] for r in records if r not in selected],
        "query_sentences": [r["text"] for r in selected],
        "provenance_scope": "query sources, not attribution of individual retrieved tasks"}
    return diagnostics["query_sentences"], diagnostics
