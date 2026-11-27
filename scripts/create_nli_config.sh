#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! ${1-} =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
    printf 'Usage: bash %s <run-name>\n' "$0" >&2
    exit 2
fi

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
config_path="configs/nli/${1}.args"

"${PYTHON:-python}" - "$1" "$config_path" <<'PY'
import json
import sys
from pathlib import Path

run_name, config_path = sys.argv[1:]
output = Path("models") / run_name
if output.exists():
    sys.exit(f"Output already exists: {output}. Choose a new run name.")

try:
    manifest = json.loads(Path("configs/nli/probing_v2.json").read_text(encoding="utf-8"))
    training = manifest["training"]["arguments"]
    arguments = []
    for key in ("data", "base", "epochs", "batch_size", "learning_rate", "seed"):
        arguments.extend(("--" + key.replace("_", "-"), str(training[key])))
    arguments.extend(("--output", output.as_posix()))
    if training["allow_unreviewed"]:
        arguments.append("--allow-unreviewed")
    with Path(config_path).open("x", encoding="utf-8") as config:
        config.write("\n".join(arguments) + "\n")
except FileExistsError:
    sys.exit(f"Config already exists: {config_path}")
except (OSError, ValueError, KeyError) as error:
    sys.exit(f"Could not create {config_path}: {error}")
PY

printf 'Created %s\n' "$config_path"
printf 'Run from the repository root: python scripts/finetune_nli.py train @%s\n' "$config_path"
