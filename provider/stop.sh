#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/common.sh"

provider_load_env

found=0
while IFS= read -r runtime; do
  [ -n "$runtime" ] || continue
  provider_stop_runtime "$runtime"
  found=1
done < <(provider_known_runtimes)

if [ "$found" -eq 0 ]; then
  echo "No local provider runtime is known."
fi
