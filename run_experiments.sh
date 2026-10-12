#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -n "${PYTHON:-}" ]]; then
    python_bin="$PYTHON"
elif command -v python >/dev/null 2>&1; then
    python_bin=python
else
    python_bin=python3
fi
export PYTHONUNBUFFERED=1
exec "$python_bin" -X utf8 experiments/launch.py "$@"
