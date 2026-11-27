#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! ${1-} =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
    printf 'Usage: bash %s <run-name> (letters, digits, underscores, or hyphens; start with a letter or digit)\n' "$0" >&2
    exit 2
fi

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
run_name=$1
config_path="configs/${run_name}.json"

"${PYTHON:-python}" - "$run_name" "$config_path" <<'PY'
import json
import sys
from pathlib import Path

run_name, config_path = sys.argv[1:]
template = Path("configs/occupation_semantic_v1.json")

try:
    config = json.loads(template.read_text(encoding="utf-8"))
    config["artifacts"] = f"models/{run_name}"
    config["report"] = f"reports/ranking/{run_name}"
    contents = json.dumps(config, indent=2) + "\n"
    with Path(config_path).open("x", encoding="utf-8") as destination:
        destination.write(contents)
except FileExistsError:
    sys.exit(f"Config already exists: {config_path}")
except (OSError, ValueError) as error:
    sys.exit(f"Could not create {config_path}: {error}")
PY

printf 'Created %s\n\n' "$config_path"
printf 'Run these commands from the repository root after downloading the required files:\n'
for stage in prepare train index evaluate; do
    printf 'python -m training.occupation_semantic %s --config %s\n' "$stage" "$config_path"
done
