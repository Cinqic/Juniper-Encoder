#!/usr/bin/env bash
set -euo pipefail

candidate=""
reviewer=""
while (($#)); do
  case "$1" in
    --candidate) candidate="$2"; shift 2 ;;
    --reviewer) reviewer="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "$candidate" || ( "$reviewer" != "sol" && "$reviewer" != "astra" ) ]]; then
  echo '{"error":"INVALID_INPUT","message":"candidate and reviewer sol|astra are required"}' >&2
  exit 2
fi
python -m juniper_encoder review --candidate "$candidate" --reviewer "$reviewer"
