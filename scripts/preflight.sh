#!/usr/bin/env bash
set -euo pipefail

output="reports/preflight/flowbox.json"
while (($#)); do
  case "$1" in
    --output) output="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

OUTPUT_PATH="$output" python3 - <<'PY'
import json, os, subprocess
from pathlib import Path
commands = [
    ["uname", "-a"], ["cat", "/etc/os-release"], ["lscpu"], ["free", "-b"],
    ["lsblk", "-b", "-o", "NAME,SIZE,TYPE,FSTYPE,MOUNTPOINTS"], ["df", "-B1", "."],
    ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,compute_cap,temperature.gpu,power.limit", "--format=csv"],
    ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"],
    ["ps", "-eo", "pid,comm,rss", "--sort=-rss"],
    ["python3", "--version"], ["git", "--version"], ["gh", "--version"],
]
results = []
for command in commands:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
        results.append({"command": command, "exit_code": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr})
    except Exception as exc:
        results.append({"command": command, "exit_code": None, "stdout": "", "stderr": repr(exc)})
payload = {"status": "CAPTURED", "commands": results}
destination = Path(os.environ["OUTPUT_PATH"])
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
PY
