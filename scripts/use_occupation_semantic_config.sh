#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! ${1-} =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
    printf 'Usage: bash %s <run-name> (letters, digits, underscores, or hyphens; start with a letter or digit)\n' "$0" >&2
    exit 2
fi

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

"${PYTHON:-python}" - "$1" <<'PY'
import json
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

parameters = Path(
    "configs/ranking/parameters/occupation_dual_encoder_signed_finetuned_v2_parameters.json"
)
index = f"models/{sys.argv[1]}/finetuned_index"
temporary_path = None

try:
    config = json.loads(parameters.read_text(encoding="utf-8"))
    previous_index = config["index"]
    if previous_index == index:
        print(f"Already using {index} in {parameters}")
        sys.exit(0)

    config["index"] = index
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=parameters.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        json.dump(config, temporary, indent=2, ensure_ascii=False)
        temporary.write("\n")
    temporary_path.replace(parameters)
except (OSError, ValueError, KeyError) as error:
    sys.exit(f"Could not update {parameters}: {error}")
finally:
    if temporary_path is not None:
        temporary_path.unlink(missing_ok=True)

print(f"Updated {parameters}")
print(f"Index: {previous_index} -> {index}")
PY
