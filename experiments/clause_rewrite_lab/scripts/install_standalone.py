"""Install only a new standalone folder, verifying every protected original."""
from pathlib import Path
import hashlib
import json
import shutil

root = Path(__file__).resolve().parents[1]
config = json.loads((root / "runtime.local.json").read_text())
project = Path(config["project_root"])
destination = project / "experiments/clause_rewrite_lab"
if destination.exists():
    raise RuntimeError("Standalone destination already exists; refusing to overwrite")
originals = json.loads((root / "reports/original_repository_sha256.json").read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


for name, expected in originals.items():
    if sha(project / name) != expected:
        raise RuntimeError("Protected original changed before install: " + name)
files = [root / name for name in ["run.py", "runtime.local.json", "README.md", "requirements.txt",
         "Start Clause Rewrite Lab.cmd", "launch.ps1", ".gitignore"]]
for folder in ["src", "scripts", "tests"]:
    files.extend(p for p in (root / folder).rglob("*.py") if "__pycache__" not in p.parts)
files.extend(root / "reports" / name for name in ["real_model_smoke.json", "interactive_smoke.txt",
    "unit_tests.txt", "protection_audit.json", "dependencies.lock.txt", "original_repository_sha256.json",
    "original_regression_tests.json"])
for path in files:
    if not path.is_file():
        raise RuntimeError("Incomplete lab package: " + str(path))
destination.mkdir(parents=True)
for path in files:
    target = destination / path.relative_to(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    assert sha(target) == sha(path)
for name, expected in originals.items():
    assert sha(project / name) == expected, name
audit = {"installed_folder": str(destination), "new_files": len(files),
         "protected_originals_unchanged": len(originals), "existing_files_modified": 0,
         "files_sha256": {path.relative_to(root).as_posix(): sha(path) for path in files}}
(destination / "reports/install_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
(root / "reports/install_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
print(json.dumps({key: value for key, value in audit.items() if key != "files_sha256"}, indent=2))
