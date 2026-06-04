#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/common.sh"

provider_load_env

found=0
while IFS= read -r runtime; do
  [ -n "$runtime" ] || continue
  if [ "$found" -eq 1 ]; then
    echo ""
  fi
  provider_status_runtime "$runtime"
  found=1
done < <(provider_known_runtimes)

if [ "$found" -eq 0 ]; then
  echo "No local provider runtime is known."
fi

if [ "${1:-}" = "--tail" ]; then
  while IFS= read -r runtime; do
    [ -n "$runtime" ] || continue
    provider_runtime_config "$runtime"
    echo ""
    echo "==> $PROVIDER_RUNTIME_LOG_FILE <=="
    tail -n "${PROVIDER_STATUS_TAIL_LINES:-40}" "$PROVIDER_RUNTIME_LOG_FILE" || true
  done < <(provider_known_runtimes)
fi
