"""Deterministic ranking data preprocessing; no model dependencies."""
from dataclasses import dataclass
from pathlib import Path
import re
from recommender.data.data_loader import read_csv


@dataclass(frozen=True)
class Task:
    code: str
    task_id: str
    text: str


def load_tasks(path, eligible_codes):
    tasks, seen = [], set()
    for row in read_csv(Path(path), {"code", "Task ID", "Task"}):
        if row["code"] not in eligible_codes:
            continue
        key = (row["code"], row["Task ID"])
        if not row["Task ID"].strip() or not row["Task"].strip() or key in seen:
            raise ValueError(f"invalid/duplicate eligible task: {key}")
        seen.add(key)
        tasks.append(Task(row["code"], row["Task ID"], row["Task"]))
    if not tasks:
        raise ValueError("no eligible tasks for dense retrieval")
    return sorted(tasks, key=lambda task: (task.code, task.task_id))


def sentences(utterances):
    # Natural text, preserving negation and repetition; no BM25 tokenization.
    return [match.group().strip() for text in utterances for match in re.finditer(r"[^.!?\n]+[.!?]*", text)
            if any(char.isalnum() for char in match.group())]

