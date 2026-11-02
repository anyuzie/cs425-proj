"""Record real resources, dependency versions, tests and original-file preservation."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
project = Path(json.loads((root / "runtime.local.json").read_text())["project_root"])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


before = json.loads((root / "reports/original_repository_sha256.json").read_text())
changed = [name for name, expected in before.items() if sha(project / name) != expected]
if changed:
    raise RuntimeError("Original repository files changed: " + ", ".join(changed))
manifest = {p.relative_to(root).as_posix(): sha(p) for directory in [root / "resources", root / "vendor"]
            for p in sorted(directory.rglob("*")) if p.is_file() and ".cache" not in p.parts
            and "__pycache__" not in p.parts and p != root / "resources/manifest.json"}
(root / "resources/manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
lock = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
(root / "reports/dependencies.lock.txt").write_text(lock, encoding="utf-8")
test = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                      cwd=root, text=True, capture_output=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
(root / "reports/unit_tests.txt").write_text(test.stdout + test.stderr, encoding="utf-8")
if test.returncode:
    raise RuntimeError(test.stdout + test.stderr)
print(test.stderr.strip())
summary = {"original_files_verified_unchanged": len(before), "changed_original_files": [],
           "scope": "standalone lab only; no integration or existing report changes",
           "resource_manifest_sha256": sha(root / "resources/manifest.json"),
           "resource_files": len(manifest), "unit_tests": "passed; see unit_tests.txt",
           "gpu": subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], text=True).strip()}
(root / "reports/protection_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
