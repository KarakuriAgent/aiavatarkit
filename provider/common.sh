#!/usr/bin/env bash

PROVIDER_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROVIDER_REPO_ROOT="$(cd "$PROVIDER_SCRIPT_DIR/.." && pwd)"

provider_trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

provider_load_env_file() {
  local file="$1"
  local line key value

  if [ ! -f "$file" ]; then
    return 0
  fi

  while IFS= read -r line || [ -n "$line" ]; do
    line="$(provider_trim "$line")"
    case "$line" in
      ""|\#*)
        continue
        ;;
    esac
    if [[ "$line" != *=* ]]; then
      continue
    fi

    key="$(provider_trim "${line%%=*}")"
    value="$(provider_trim "${line#*=}")"
    if [[ "$value" == \"*\" && "$value" == *\" ]] || [[ "$value" == \'*\' && "$value" == *\' ]]; then
      value="${value:1:${#value}-2}"
    fi

    if [ -n "$key" ] && [ -z "${!key+x}" ]; then
      export "$key=$value"
    fi
  done < "$file"
}

provider_bool_enabled() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

provider_load_env() {
  provider_load_env_file "$PROVIDER_REPO_ROOT/.env"
  provider_load_env_file "$PROVIDER_REPO_ROOT/server/.env"

  : "${PROVIDER_STATE_DIR:=$PROVIDER_REPO_ROOT/.provider}"
  : "${PROVIDER_LOG_DIR:=$PROVIDER_STATE_DIR/logs}"
  : "${PROVIDER_START_TIMEOUT:=90}"
  : "${PROVIDER_STOP_TIMEOUT:=15}"
  : "${STT_RUNTIME_PORT:=8766}"
  : "${VOICE_AUTH_RUNTIME_PORT:=8765}"

  export PROVIDER_STATE_DIR
  export PROVIDER_LOG_DIR
  export PROVIDER_START_TIMEOUT
  export PROVIDER_STOP_TIMEOUT
  export STT_RUNTIME_PORT
  export VOICE_AUTH_RUNTIME_PORT
}

provider_enabled_runtimes() {
  if [ "${STT_PROVIDER:-}" = "qwen3_asr_mlx" ]; then
    echo "stt/qwen3_asr_mlx"
  fi

  if provider_bool_enabled "${VOICE_AUTH_ENABLED:-false}" && [ "${VOICE_AUTH_PROVIDER:-}" = "wespeaker_mlx" ]; then
    echo "voice_auth/wespeaker_mlx"
  fi
}

provider_known_runtimes() {
  echo "stt/qwen3_asr_mlx"
  echo "voice_auth/wespeaker_mlx"
}

provider_runtime_config() {
  local runtime="$1"

  case "$runtime" in
    stt/qwen3_asr_mlx)
      PROVIDER_RUNTIME_NAME="qwen3_asr_mlx"
      PROVIDER_RUNTIME_TITLE="Qwen3-ASR MLX STT"
      PROVIDER_RUNTIME_COMMAND="${QWEN3_ASR_MLX_RUNTIME_COMMAND:-$PROVIDER_REPO_ROOT/.venv/bin/qwen3-asr-mlx-runtime}"
      PROVIDER_RUNTIME_EXTRA="qwen-stt"
      PROVIDER_RUNTIME_PID_FILE="$PROVIDER_STATE_DIR/qwen3-asr-mlx.pid"
      PROVIDER_RUNTIME_LOG_FILE="$PROVIDER_LOG_DIR/qwen3-asr-mlx.log"
      PROVIDER_RUNTIME_HEALTH_URL="${STT_RUNTIME_HEALTH_URL:-http://127.0.0.1:$STT_RUNTIME_PORT/health}"
      PROVIDER_RUNTIME_API_KEY="${STT_API_KEY:-${AIAVATAR_API_KEY:-}}"
      PROVIDER_RUNTIME_PROCESS_MATCH="qwen3-asr-mlx-runtime"
      PROVIDER_RUNTIME_PRESTART="qwen3_asr_mlx"
      ;;
    voice_auth/wespeaker_mlx)
      PROVIDER_RUNTIME_NAME="wespeaker_mlx"
      PROVIDER_RUNTIME_TITLE="WeSpeaker MLX Voice Auth"
      PROVIDER_RUNTIME_COMMAND="${WESPEAKER_MLX_RUNTIME_COMMAND:-$PROVIDER_REPO_ROOT/.venv/bin/wespeaker-mlx-runtime}"
      PROVIDER_RUNTIME_EXTRA="voice-auth"
      PROVIDER_RUNTIME_PID_FILE="$PROVIDER_STATE_DIR/wespeaker-mlx.pid"
      PROVIDER_RUNTIME_LOG_FILE="$PROVIDER_LOG_DIR/wespeaker-mlx.log"
      PROVIDER_RUNTIME_HEALTH_URL="${VOICE_AUTH_RUNTIME_HEALTH_URL:-http://127.0.0.1:$VOICE_AUTH_RUNTIME_PORT/health}"
      PROVIDER_RUNTIME_API_KEY="${VOICE_AUTH_API_KEY:-${AIAVATAR_API_KEY:-}}"
      PROVIDER_RUNTIME_PROCESS_MATCH="wespeaker-mlx-runtime"
      PROVIDER_RUNTIME_PRESTART=""
      ;;
    *)
      echo "Unsupported provider runtime: $runtime" >&2
      return 1
      ;;
  esac

  export PROVIDER_RUNTIME_NAME
  export PROVIDER_RUNTIME_TITLE
  export PROVIDER_RUNTIME_COMMAND
  export PROVIDER_RUNTIME_EXTRA
  export PROVIDER_RUNTIME_PID_FILE
  export PROVIDER_RUNTIME_LOG_FILE
  export PROVIDER_RUNTIME_HEALTH_URL
  export PROVIDER_RUNTIME_API_KEY
  export PROVIDER_RUNTIME_PROCESS_MATCH
  export PROVIDER_RUNTIME_PRESTART
}

provider_runtime_pid() {
  if [ -f "$PROVIDER_RUNTIME_PID_FILE" ]; then
    sed -n '1p' "$PROVIDER_RUNTIME_PID_FILE" | tr -d '[:space:]'
  fi
}

provider_runtime_process_matches() {
  local pid="$1"
  local command
  command="$(ps -p "$pid" -o command= 2>/dev/null || true)"
  case "$command" in
    *"$PROVIDER_RUNTIME_PROCESS_MATCH"*|*"provider.${PROVIDER_RUNTIME_NAME}"*|*"provider/"*"$PROVIDER_RUNTIME_NAME"*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

provider_runtime_is_running() {
  local pid
  pid="$(provider_runtime_pid)"
  if [ -z "$pid" ]; then
    return 1
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    return 1
  fi
  provider_runtime_process_matches "$pid"
}

provider_runtime_health() {
  if ! command -v curl >/dev/null 2>&1; then
    return 127
  fi

  if [ -n "$PROVIDER_RUNTIME_API_KEY" ]; then
    curl -fsS -H "Authorization: Bearer $PROVIDER_RUNTIME_API_KEY" "$PROVIDER_RUNTIME_HEALTH_URL"
  else
    curl -fsS "$PROVIDER_RUNTIME_HEALTH_URL"
  fi
}

provider_xml_escape() {
  local value="$1"
  value="${value//&/&amp;}"
  value="${value//</&lt;}"
  value="${value//>/&gt;}"
  value="${value//\"/&quot;}"
  value="${value//\'/&apos;}"
  printf '%s' "$value"
}

provider_shell_quote() {
  printf '%q' "$1"
}

provider_launchd_enabled() {
  [ "${PROVIDER_LAUNCH_METHOD:-auto}" != "nohup" ] || return 1
  [ "$(uname -s)" = "Darwin" ] || return 1
  command -v launchctl >/dev/null 2>&1
}

provider_launchd_domain() {
  printf 'gui/%s' "$(id -u)"
}

provider_launchd_label() {
  printf 'com.aiavatarkit.provider.%s' "$PROVIDER_RUNTIME_NAME"
}

provider_launchd_service() {
  printf '%s/%s' "$(provider_launchd_domain)" "$(provider_launchd_label)"
}

provider_launchd_launcher_file() {
  printf '%s/launchers/%s.sh' "$PROVIDER_STATE_DIR" "$PROVIDER_RUNTIME_NAME"
}

provider_launchd_plist_file() {
  printf '%s/launchers/%s.plist' "$PROVIDER_STATE_DIR" "$PROVIDER_RUNTIME_NAME"
}

provider_launchd_write_files() {
  local runtime="$1"
  local launcher plist label
  launcher="$(provider_launchd_launcher_file)"
  plist="$(provider_launchd_plist_file)"
  label="$(provider_launchd_label)"

  mkdir -p "$(dirname "$launcher")"

  cat > "$launcher" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd $(provider_shell_quote "$PROVIDER_REPO_ROOT")
. $(provider_shell_quote "$PROVIDER_SCRIPT_DIR/common.sh")
provider_load_env
provider_runtime_config $(provider_shell_quote "$runtime")
exec "\$PROVIDER_RUNTIME_COMMAND"
EOF
  chmod +x "$launcher"

  cat > "$plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$(provider_xml_escape "$label")</string>
  <key>ProgramArguments</key>
  <array>
    <string>$(provider_xml_escape "$launcher")</string>
  </array>
  <key>WorkingDirectory</key>
  <string>$(provider_xml_escape "$PROVIDER_REPO_ROOT")</string>
  <key>StandardOutPath</key>
  <string>$(provider_xml_escape "$PROVIDER_RUNTIME_LOG_FILE")</string>
  <key>StandardErrorPath</key>
  <string>$(provider_xml_escape "$PROVIDER_RUNTIME_LOG_FILE")</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PYTHONUNBUFFERED</key>
    <string>1</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
</dict>
</plist>
EOF
}

provider_launchd_is_loaded() {
  provider_launchd_enabled || return 1
  launchctl print "$(provider_launchd_service)" >/dev/null 2>&1
}

provider_launchd_pid() {
  provider_launchd_enabled || return 1
  launchctl print "$(provider_launchd_service)" 2>/dev/null \
    | awk -F'= ' '/^[[:space:]]*pid = / { print $2; exit }'
}

provider_launchd_refresh_pid_file() {
  local pid
  provider_launchd_enabled || return 1
  pid="$(provider_launchd_pid)"
  if [ -n "$pid" ]; then
    echo "$pid" > "$PROVIDER_RUNTIME_PID_FILE"
    return 0
  fi
  return 1
}

provider_launchd_start_runtime() {
  local runtime="$1"
  local plist
  provider_launchd_write_files "$runtime"
  plist="$(provider_launchd_plist_file)"

  launchctl bootout "$(provider_launchd_service)" >/dev/null 2>&1 || true
  launchctl bootstrap "$(provider_launchd_domain)" "$plist"
}

provider_launchd_stop_runtime() {
  provider_launchd_enabled || return 1
  launchctl bootout "$(provider_launchd_service)" >/dev/null 2>&1
}

provider_sync_enabled_extras() {
  local extras="" runtime extra
  local -a args=()
  while IFS= read -r runtime; do
    [ -n "$runtime" ] || continue
    provider_runtime_config "$runtime"
    extra="$PROVIDER_RUNTIME_EXTRA"
    case " $extras " in
      *" $extra "*)
        ;;
      *)
        extras="$extras $extra"
        ;;
    esac
  done < <(provider_enabled_runtimes)

  if [ -z "$(provider_trim "$extras")" ]; then
    return 0
  fi

  for extra in $extras; do
    args+=(--extra "$extra")
  done

  cd "$PROVIDER_REPO_ROOT"
  uv sync --frozen "${args[@]}"
}

provider_runtime_prestart() {
  case "$PROVIDER_RUNTIME_PRESTART" in
    qwen3_asr_mlx)
      cd "$PROVIDER_REPO_ROOT"
      "$PROVIDER_REPO_ROOT/.venv/bin/python" -m provider.stt.qwen3_asr_mlx.model
      ;;
  esac
}

provider_start_runtime() {
  local runtime="$1"
  provider_runtime_config "$runtime"

  mkdir -p "$PROVIDER_STATE_DIR" "$PROVIDER_LOG_DIR"
  provider_launchd_refresh_pid_file >/dev/null 2>&1 || true

  if provider_runtime_is_running; then
    echo "$PROVIDER_RUNTIME_NAME is already running: pid=$(provider_runtime_pid)"
    echo "health: $PROVIDER_RUNTIME_HEALTH_URL"
    echo "log: $PROVIDER_RUNTIME_LOG_FILE"
    return 0
  fi

  rm -f "$PROVIDER_RUNTIME_PID_FILE"

  if [ ! -x "$PROVIDER_RUNTIME_COMMAND" ]; then
    echo "Runtime command is missing after dependency sync: $PROVIDER_RUNTIME_COMMAND" >&2
    return 1
  fi
  provider_runtime_prestart

  {
    echo ""
    echo "===== $(date '+%Y-%m-%d %H:%M:%S') starting $PROVIDER_RUNTIME_NAME ====="
    echo "title: $PROVIDER_RUNTIME_TITLE"
    echo "repo: $PROVIDER_REPO_ROOT"
    echo "health: $PROVIDER_RUNTIME_HEALTH_URL"
    echo "command: $PROVIDER_RUNTIME_COMMAND"
    echo "extra: $PROVIDER_RUNTIME_EXTRA"
  } >> "$PROVIDER_RUNTIME_LOG_FILE"

  cd "$PROVIDER_REPO_ROOT"
  if provider_launchd_enabled; then
    provider_launchd_start_runtime "$runtime"
  else
    PYTHONUNBUFFERED=1 nohup "$PROVIDER_RUNTIME_COMMAND" >> "$PROVIDER_RUNTIME_LOG_FILE" 2>&1 </dev/null &
    echo "$!" > "$PROVIDER_RUNTIME_PID_FILE"
  fi

  sleep 1
  provider_launchd_refresh_pid_file >/dev/null 2>&1 || true
  if ! provider_runtime_is_running; then
    echo "$PROVIDER_RUNTIME_NAME failed to start"
    echo "log: $PROVIDER_RUNTIME_LOG_FILE"
    tail -n 40 "$PROVIDER_RUNTIME_LOG_FILE" || true
    rm -f "$PROVIDER_RUNTIME_PID_FILE"
    return 1
  fi

  local deadline health
  deadline=$((SECONDS + PROVIDER_START_TIMEOUT))
  while [ "$SECONDS" -lt "$deadline" ]; do
    provider_launchd_refresh_pid_file >/dev/null 2>&1 || true
    if health="$(provider_runtime_health 2>/dev/null)"; then
      echo "$PROVIDER_RUNTIME_NAME started: pid=$(provider_runtime_pid)"
      echo "health: $health"
      echo "log: $PROVIDER_RUNTIME_LOG_FILE"
      return 0
    fi

    if ! provider_runtime_is_running; then
      echo "$PROVIDER_RUNTIME_NAME exited before health became ready"
      echo "log: $PROVIDER_RUNTIME_LOG_FILE"
      tail -n 40 "$PROVIDER_RUNTIME_LOG_FILE" || true
      rm -f "$PROVIDER_RUNTIME_PID_FILE"
      return 1
    fi

    sleep 1
  done

  echo "$PROVIDER_RUNTIME_NAME started: pid=$(provider_runtime_pid)"
  echo "health is not ready yet: $PROVIDER_RUNTIME_HEALTH_URL"
  echo "log: $PROVIDER_RUNTIME_LOG_FILE"
}

provider_status_runtime() {
  local runtime="$1"
  provider_runtime_config "$runtime"
  provider_launchd_refresh_pid_file >/dev/null 2>&1 || true

  echo "runtime: $PROVIDER_RUNTIME_NAME"
  if provider_runtime_is_running; then
    echo "process: running"
    echo "pid: $(provider_runtime_pid)"
  else
    echo "process: stopped"
    if [ -f "$PROVIDER_RUNTIME_PID_FILE" ]; then
      echo "stale pid file: $PROVIDER_RUNTIME_PID_FILE"
    fi
  fi

  if health="$(provider_runtime_health 2>/dev/null)"; then
    echo "health: ok"
    echo "$health"
  else
    echo "health: unavailable"
    echo "health_url: $PROVIDER_RUNTIME_HEALTH_URL"
  fi
  echo "log: $PROVIDER_RUNTIME_LOG_FILE"
}

provider_stop_runtime() {
  local runtime="$1"
  provider_runtime_config "$runtime"

  local pid
  pid="$(provider_runtime_pid)"

  if provider_launchd_is_loaded; then
    echo "stopping $PROVIDER_RUNTIME_NAME: $(provider_launchd_service)"
    provider_launchd_stop_runtime
    rm -f "$PROVIDER_RUNTIME_PID_FILE"
    echo "$PROVIDER_RUNTIME_NAME stopped"
    return 0
  fi

  if [ -z "$pid" ]; then
    echo "$PROVIDER_RUNTIME_NAME is not running"
    rm -f "$PROVIDER_RUNTIME_PID_FILE"
    return 0
  fi

  if ! provider_runtime_is_running; then
    echo "$PROVIDER_RUNTIME_NAME is not running: removing stale pid file"
    rm -f "$PROVIDER_RUNTIME_PID_FILE"
    return 0
  fi

  echo "stopping $PROVIDER_RUNTIME_NAME: pid=$pid"
  kill "$pid"

  local deadline
  deadline=$((SECONDS + PROVIDER_STOP_TIMEOUT))
  while [ "$SECONDS" -lt "$deadline" ]; do
    if ! provider_runtime_is_running; then
      rm -f "$PROVIDER_RUNTIME_PID_FILE"
      echo "$PROVIDER_RUNTIME_NAME stopped"
      return 0
    fi
    sleep 1
  done

  echo "$PROVIDER_RUNTIME_NAME did not stop within ${PROVIDER_STOP_TIMEOUT}s"
  echo "pid file: $PROVIDER_RUNTIME_PID_FILE"
  return 1
}
