#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/common.sh"

voice_auth_runtime_load_env

pid="$(voice_auth_runtime_pid)"
if voice_auth_runtime_launchctl_loaded; then
  echo "launchctl: loaded"
  echo "label: $VOICE_AUTH_RUNTIME_LABEL"
else
  echo "launchctl: unloaded"
fi

if voice_auth_runtime_is_running; then
  echo "process: running"
  echo "pid: $pid"
else
  echo "process: stopped"
  if [ -n "$pid" ]; then
    echo "stale pid file: $VOICE_AUTH_RUNTIME_PID_FILE"
  fi
fi

if health="$(voice_auth_runtime_health 2>/dev/null)"; then
  echo "health: ok"
  echo "$health"
else
  echo "health: unavailable"
  echo "health_url: $VOICE_AUTH_RUNTIME_HEALTH_URL"
fi

echo "log: $VOICE_AUTH_RUNTIME_LOG_FILE"

if [ "${1:-}" = "--tail" ]; then
  tail -n "${VOICE_AUTH_RUNTIME_STATUS_TAIL_LINES:-40}" "$VOICE_AUTH_RUNTIME_LOG_FILE"
fi
