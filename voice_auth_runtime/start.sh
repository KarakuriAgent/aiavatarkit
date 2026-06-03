#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/common.sh"

voice_auth_runtime_load_env

mkdir -p "$VOICE_AUTH_RUNTIME_STATE_DIR" "$VOICE_AUTH_RUNTIME_LOG_DIR"

: "${VOICE_AUTH_MODEL_PATH:=models/wespeaker-voxceleb-resnet34-LM-mlx}"
: "${VOICE_AUTH_MODEL_REPO:=Landon41/wespeaker-voxceleb-resnet34-LM-mlx}"
export VOICE_AUTH_MODEL_PATH
export VOICE_AUTH_MODEL_REPO

if voice_auth_runtime_is_running; then
  echo "voice-auth-runtime is already running: pid=$(voice_auth_runtime_pid)"
  echo "log: $VOICE_AUTH_RUNTIME_LOG_FILE"
  exit 0
fi

if [ -f "$VOICE_AUTH_RUNTIME_PID_FILE" ]; then
  rm -f "$VOICE_AUTH_RUNTIME_PID_FILE"
fi

if [ ! -x "$VOICE_AUTH_RUNTIME_COMMAND" ]; then
  cd "$VOICE_AUTH_RUNTIME_REPO_ROOT"
  uv sync --extra voice-auth
fi

{
  echo ""
  echo "===== $(date '+%Y-%m-%d %H:%M:%S') starting voice-auth-runtime ====="
  echo "repo: $VOICE_AUTH_RUNTIME_REPO_ROOT"
  echo "health: $VOICE_AUTH_RUNTIME_HEALTH_URL"
  echo "command: $VOICE_AUTH_RUNTIME_COMMAND"
  echo "model: ${VOICE_AUTH_MODEL_PATH:-}"
  echo "model_repo: ${VOICE_AUTH_MODEL_REPO:-}"
  echo "profiles: ${VOICE_AUTH_PROFILE_DIR:-}"
  echo "launchctl_label: $VOICE_AUTH_RUNTIME_LABEL"
} >> "$VOICE_AUTH_RUNTIME_LOG_FILE"

if command -v launchctl >/dev/null 2>&1; then
  voice_auth_runtime_write_plist
  if voice_auth_runtime_launchctl_loaded; then
    launchctl bootout "$(voice_auth_runtime_launchctl_target)" >/dev/null 2>&1 || true
  fi
  launchctl bootstrap "$VOICE_AUTH_RUNTIME_LAUNCHCTL_DOMAIN" "$VOICE_AUTH_RUNTIME_PLIST_FILE"
  launchctl kickstart -k "$(voice_auth_runtime_launchctl_target)"
else
  cd "$VOICE_AUTH_RUNTIME_REPO_ROOT"
  VOICE_AUTH_ENABLED=true PYTHONUNBUFFERED=1 nohup "$VOICE_AUTH_RUNTIME_COMMAND" >> "$VOICE_AUTH_RUNTIME_LOG_FILE" 2>&1 &
  echo "$!" > "$VOICE_AUTH_RUNTIME_PID_FILE"
fi

sleep 1
if ! voice_auth_runtime_is_running; then
  echo "voice-auth-runtime failed to start"
  echo "log: $VOICE_AUTH_RUNTIME_LOG_FILE"
  tail -n 40 "$VOICE_AUTH_RUNTIME_LOG_FILE" || true
  rm -f "$VOICE_AUTH_RUNTIME_PID_FILE"
  exit 1
fi

pid="$(voice_auth_runtime_pid)"
if [ -n "$pid" ]; then
  echo "$pid" > "$VOICE_AUTH_RUNTIME_PID_FILE"
fi

deadline=$((SECONDS + VOICE_AUTH_RUNTIME_START_TIMEOUT))
while [ "$SECONDS" -lt "$deadline" ]; do
  if health="$(voice_auth_runtime_health 2>/dev/null)"; then
    echo "voice-auth-runtime started: pid=$(voice_auth_runtime_pid)"
    echo "health: $health"
    echo "log: $VOICE_AUTH_RUNTIME_LOG_FILE"
    exit 0
  fi

  if ! voice_auth_runtime_is_running; then
    echo "voice-auth-runtime exited before health became ready"
    echo "log: $VOICE_AUTH_RUNTIME_LOG_FILE"
    tail -n 40 "$VOICE_AUTH_RUNTIME_LOG_FILE" || true
    rm -f "$VOICE_AUTH_RUNTIME_PID_FILE"
    exit 1
  fi

  sleep 1
done

echo "voice-auth-runtime started: pid=$pid"
echo "health is not ready yet: $VOICE_AUTH_RUNTIME_HEALTH_URL"
echo "log: $VOICE_AUTH_RUNTIME_LOG_FILE"
