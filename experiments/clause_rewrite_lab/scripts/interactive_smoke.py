"""Run the real interactive tool with two clauses, retaining the full output."""
from pathlib import Path
import json
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
before = set((root / "reports").glob("session_*.jsonl"))
result = subprocess.run([sys.executable, str(root / "run.py"), "--top-k", "3"],
    input="I dislike selling products and persuading customers.\nI like figuring out why my code keeps breaking.\n/quit\n",
    encoding="utf-8", capture_output=True, cwd=root, timeout=600)
(root / "reports/interactive_smoke.txt").write_text(result.stdout + "\nSTARTUP/PROGRESS:\n" + result.stderr, encoding="utf-8")
if result.returncode:
    raise RuntimeError(result.stdout + result.stderr)
created = set((root / "reports").glob("session_*.jsonl")) - before
assert len(created) == 1
session = next(iter(created))
records = [json.loads(line) for line in session.read_text(encoding="utf-8").splitlines()]
results = [record for record in records if record.get("event") == "result"]
assert len(results) == 2
assert results[0]["preference"] == "negative"
for record in results:
    expected = record["diffuseq_rewrite"] if record["rewrite_accepted"] else record["original_clause"]
    assert record["matching_text"] == expected
    assert len(record["task_matches"]) == 3
    assert len(record["candidates"]) == 3
    assert all("reconstruction" in candidate for candidate in record["candidates"])
summary = {"real_models": True, "clauses": len(results), "session_file": session.name,
           "gpu": records[0]["gpu"], "matcher": records[0]["matcher"],
           "results": [{key: record[key] for key in ["original_clause", "diffuseq_rewrite", "rewrite_is_copy",
               "reconstruction", "rewrite_accepted", "matching_text", "matching_source", "task_matches", "seconds"]} for record in results]}
(root / "reports/real_model_smoke.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=True, indent=2))
