import json
import re
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

_lock = Lock()
_SAFE = re.compile(r"^[A-Za-z0-9_-]{1,32}$")


def append(log_dir, participant_code: str, record: dict) -> Path:
    """Append one JSON line to <log_dir>/<participant_code>.jsonl."""
    if not _SAFE.fullmatch(participant_code):
        raise ValueError("unsafe participant code")
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{participant_code}.jsonl"
    line = json.dumps({"ts": datetime.now(timezone.utc).isoformat(), **record}, ensure_ascii=False)
    with _lock, path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    return path
