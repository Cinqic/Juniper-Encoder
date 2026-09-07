#!/usr/bin/env bash
set -euo pipefail

candidate=""
while (($#)); do
  case "$1" in
    --candidate) candidate="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "$candidate" ]]; then
  echo '{"error":"INVALID_INPUT","message":"--candidate is required"}' >&2
  exit 2
fi
python -m juniper_encoder package verify --directory "$(dirname "$candidate")" --candidate "$candidate"
