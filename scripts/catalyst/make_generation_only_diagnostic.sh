#!/bin/bash

# Copy one canonical GPU task into a no-GPU, verifier-disabled generation
# diagnostic directory. The canonical source is never edited. Harbor flags
# still need to disable verification when the resulting directory is run.
set -euo pipefail

if (( $# != 2 )); then
  printf 'Usage: %s CANONICAL_TASK_DIR DIAGNOSTIC_DIR\n' "$0" >&2
  exit 2
fi

source_dir=$1
destination_dir=$2

if [[ ! -f $source_dir/task.toml ]]; then
  printf 'Missing canonical task.toml: %s\n' "$source_dir/task.toml" >&2
  exit 2
fi
if [[ -e $destination_dir ]]; then
  printf 'Diagnostic destination already exists: %s\n' "$destination_dir" >&2
  exit 2
fi

mkdir -p "$(dirname "$destination_dir")"
cp -a "$source_dir" "$destination_dir"

python3 - "$destination_dir/task.toml" <<'PY'
from pathlib import Path
import sys


path = Path(sys.argv[1])
text = path.read_text(encoding="utf-8")

replacements = {
    'required_gpus = 1': 'required_gpus = 0',
    'release_ready = false': (
        'release_ready = false\n'
        'diagnostic_mode = "generation-only-no-gpu"'
    ),
}
for old, new in replacements.items():
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"expected exactly one {old!r}, found {count}")
    text = text.replace(old, new, 1)

healthcheck_lines = [
    line for line in text.splitlines()
    if line.startswith('command = ') and 'direct-ptx-verify-hardware' in line
]
if len(healthcheck_lines) != 1:
    raise SystemExit(
        "expected exactly one hardware healthcheck command, found "
        f"{len(healthcheck_lines)}"
    )
text = text.replace(
    healthcheck_lines[0],
    'command = "/usr/local/bin/direct-ptx-policy-check"',
    1,
)
path.write_text(text, encoding="utf-8")
PY

python3 - "$source_dir/task.toml" "$destination_dir/task.toml" <<'PY'
from pathlib import Path
import sys


canonical = Path(sys.argv[1]).read_text(encoding="utf-8")
diagnostic = Path(sys.argv[2]).read_text(encoding="utf-8")
required = (
    'required_gpus = 0',
    'diagnostic_mode = "generation-only-no-gpu"',
    'command = "/usr/local/bin/direct-ptx-policy-check"',
)
missing = [item for item in required if item not in diagnostic]
if missing:
    raise SystemExit(f"diagnostic invariants missing: {missing}")
if 'direct-ptx-verify-hardware' in diagnostic:
    raise SystemExit("hardware verifier remains in diagnostic task.toml")
if canonical == diagnostic:
    raise SystemExit("diagnostic task.toml unexpectedly matches canonical")
PY

printf 'Created generation-only diagnostic: %s\n' "$destination_dir"
