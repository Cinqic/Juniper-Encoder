#!/usr/bin/env bash
set -euo pipefail

lock=""
while (($#)); do
  case "$1" in
    --lock) lock="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "$lock" || ! -f "$lock" ]]; then
  echo '{"error":"BLOCKED_ENVIRONMENT","message":"a repository lock file is required"}' >&2
  exit 20
fi

python3 -m venv .venv
source .venv/bin/activate
export JE_LOCK_PATH="$lock"
python -m pip install --upgrade --require-hashes -r "$lock"
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
python - <<'PY'
import hashlib, json, os, sys
from pathlib import Path
lock = Path(os.environ.get("JE_LOCK_PATH", "requirements/flowbox.lock"))
payload = {"python": sys.version, "executable": sys.executable, "lock": str(lock), "lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest()}
try:
    import torch
    payload["torch"] = torch.__version__
except ImportError:
    payload["torch"] = None
destination = Path("reports/preflight/resolved-environment.json")
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
PY
