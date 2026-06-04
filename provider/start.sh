#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/common.sh"

provider_load_env
provider_sync_enabled_extras

started=0
while IFS= read -r runtime; do
  [ -n "$runtime" ] || continue
  provider_start_runtime "$runtime"
  started=1
done < <(provider_enabled_runtimes)

if [ "$started" -eq 0 ]; then
  echo "No local provider runtime selected."
  echo "STT_PROVIDER=${STT_PROVIDER:-}"
  echo "VOICE_AUTH_ENABLED=${VOICE_AUTH_ENABLED:-false}"
  echo "VOICE_AUTH_PROVIDER=${VOICE_AUTH_PROVIDER:-}"
fi
