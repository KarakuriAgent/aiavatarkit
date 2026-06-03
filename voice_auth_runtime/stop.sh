#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/common.sh"

voice_auth_runtime_load_env

pid="$(voice_auth_runtime_pid)"
if voice_auth_runtime_launchctl_loaded; then
  echo "stopping voice-auth-runtime: label=$VOICE_AUTH_RUNTIME_LABEL pid=${pid:-unknown}"
  launchctl bootout "$(voice_auth_runtime_launchctl_target)"
else
  if [ -z "$pid" ]; then
    echo "voice-auth-runtime is not running"
    rm -f "$VOICE_AUTH_RUNTIME_PID_FILE"
    exit 0
  fi

  if ! voice_auth_runtime_is_running; then
    echo "voice-auth-runtime is not running: removing stale pid file"
    rm -f "$VOICE_AUTH_RUNTIME_PID_FILE"
    exit 0
  fi

  echo "stopping voice-auth-runtime: pid=$pid"
  kill "$pid"
fi

deadline=$((SECONDS + VOICE_AUTH_RUNTIME_STOP_TIMEOUT))
while [ "$SECONDS" -lt "$deadline" ]; do
  if ! voice_auth_runtime_is_running; then
    rm -f "$VOICE_AUTH_RUNTIME_PID_FILE"
    echo "voice-auth-runtime stopped"
    exit 0
  fi
  sleep 1
done

echo "voice-auth-runtime did not stop within ${VOICE_AUTH_RUNTIME_STOP_TIMEOUT}s"
echo "pid file: $VOICE_AUTH_RUNTIME_PID_FILE"
exit 1
