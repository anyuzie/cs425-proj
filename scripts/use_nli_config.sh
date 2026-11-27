#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! ${1-} =~ ^[a-zA-Z0-9][a-zA-Z0-9_-]*$ ]]; then
    printf 'Usage: bash %s <run-name>\n' "$0" >&2
    exit 2
fi

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."

"${PYTHON:-python}" - "$1" <<'PY'
import json
import re
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile

system_config = Path("configs/system.json")
checkpoint = Path("models") / sys.argv[1] / "checkpoint"
temporary_path = None

try:
    for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"):
        if not (checkpoint / name).is_file():
            sys.exit(f"Missing checkpoint file: {checkpoint / name}. Finish training first.")

    source = system_config.read_bytes().decode("utf-8")
    previous = json.loads(source)["paths"]["nli_model"]
    target = checkpoint.as_posix()
    if previous == target:
        print(f"Already using {target} in {system_config}")
        sys.exit(0)

    updated, count = re.subn(
        r'("nli_model"\s*:\s*)"(?:[^"\\]|\\.)*"',
        lambda match: match[1] + json.dumps(target),
        source,
    )
    if count != 1 or json.loads(updated)["paths"]["nli_model"] != target:
        raise ValueError("Expected exactly one paths.nli_model entry")

    with NamedTemporaryFile(dir=system_config.parent, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        temporary.write(updated.encode("utf-8"))
    temporary_path.replace(system_config)
except (OSError, ValueError, KeyError) as error:
    sys.exit(f"Could not update {system_config}: {error}")
finally:
    if temporary_path is not None:
        temporary_path.unlink(missing_ok=True)

print(f"Updated paths.nli_model in {system_config}: {previous} -> {target}")
print("Restart the web app to load the checkpoint.")
PY
