#!/usr/bin/env bash

VOICE_AUTH_RUNTIME_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOICE_AUTH_RUNTIME_REPO_ROOT="$(cd "$VOICE_AUTH_RUNTIME_SCRIPT_DIR/.." && pwd)"

voice_auth_runtime_trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "$value"
}

voice_auth_runtime_xml_escape() {
  printf '%s' "$1" | sed \
    -e 's/&/\&amp;/g' \
    -e 's/</\&lt;/g' \
    -e 's/>/\&gt;/g' \
    -e 's/"/\&quot;/g' \
    -e "s/'/\&apos;/g"
}

voice_auth_runtime_load_env_file() {
  local file="$1"
  local line key value

  if [ ! -f "$file" ]; then
    return 0
  fi

  while IFS= read -r line || [ -n "$line" ]; do
    line="$(voice_auth_runtime_trim "$line")"
    case "$line" in
      ""|\#*)
        continue
        ;;
    esac
    if [[ "$line" != *=* ]]; then
      continue
    fi

    key="$(voice_auth_runtime_trim "${line%%=*}")"
    value="$(voice_auth_runtime_trim "${line#*=}")"
    if [[ "$value" == \"*\" && "$value" == *\" ]] || [[ "$value" == \'*\' && "$value" == *\' ]]; then
      value="${value:1:${#value}-2}"
    fi

    if [ -n "$key" ] && [ -z "${!key+x}" ]; then
      export "$key=$value"
    fi
  done < "$file"
}

voice_auth_runtime_load_env() {
  voice_auth_runtime_load_env_file "$VOICE_AUTH_RUNTIME_REPO_ROOT/.env"
  voice_auth_runtime_load_env_file "$VOICE_AUTH_RUNTIME_REPO_ROOT/server/.env"

  : "${VOICE_AUTH_RUNTIME_PORT:=8765}"
  : "${VOICE_AUTH_RUNTIME_STATE_DIR:=$VOICE_AUTH_RUNTIME_REPO_ROOT/.voice_auth_runtime}"
  : "${VOICE_AUTH_RUNTIME_LOG_DIR:=$VOICE_AUTH_RUNTIME_STATE_DIR/logs}"
  : "${VOICE_AUTH_RUNTIME_PID_FILE:=$VOICE_AUTH_RUNTIME_STATE_DIR/voice-auth-runtime.pid}"
  : "${VOICE_AUTH_RUNTIME_LOG_FILE:=$VOICE_AUTH_RUNTIME_LOG_DIR/voice-auth-runtime.log}"
  : "${VOICE_AUTH_RUNTIME_HEALTH_URL:=http://127.0.0.1:$VOICE_AUTH_RUNTIME_PORT/health}"
  : "${VOICE_AUTH_RUNTIME_START_TIMEOUT:=60}"
  : "${VOICE_AUTH_RUNTIME_STOP_TIMEOUT:=15}"
  : "${VOICE_AUTH_RUNTIME_COMMAND:=$VOICE_AUTH_RUNTIME_REPO_ROOT/.venv/bin/voice-auth-runtime}"
  : "${VOICE_AUTH_RUNTIME_LABEL:=dev.aiavatarkit.voice-auth-runtime}"
  : "${VOICE_AUTH_RUNTIME_PLIST_FILE:=$VOICE_AUTH_RUNTIME_STATE_DIR/$VOICE_AUTH_RUNTIME_LABEL.plist}"
  : "${VOICE_AUTH_RUNTIME_LAUNCHCTL_DOMAIN:=gui/$(id -u)}"

  export VOICE_AUTH_RUNTIME_PORT
  export VOICE_AUTH_RUNTIME_STATE_DIR
  export VOICE_AUTH_RUNTIME_LOG_DIR
  export VOICE_AUTH_RUNTIME_PID_FILE
  export VOICE_AUTH_RUNTIME_LOG_FILE
  export VOICE_AUTH_RUNTIME_HEALTH_URL
  export VOICE_AUTH_RUNTIME_START_TIMEOUT
  export VOICE_AUTH_RUNTIME_STOP_TIMEOUT
  export VOICE_AUTH_RUNTIME_COMMAND
  export VOICE_AUTH_RUNTIME_LABEL
  export VOICE_AUTH_RUNTIME_PLIST_FILE
  export VOICE_AUTH_RUNTIME_LAUNCHCTL_DOMAIN
}

voice_auth_runtime_pid() {
  local launch_pid
  launch_pid="$(voice_auth_runtime_launchctl_pid)"
  if [ -n "$launch_pid" ]; then
    printf '%s\n' "$launch_pid"
    return 0
  fi
  if [ -f "$VOICE_AUTH_RUNTIME_PID_FILE" ]; then
    sed -n '1p' "$VOICE_AUTH_RUNTIME_PID_FILE" | tr -d '[:space:]'
  fi
  return 0
}

voice_auth_runtime_process_matches() {
  local pid="$1"
  local command
  command="$(ps -p "$pid" -o command= 2>/dev/null || true)"
  case "$command" in
    *voice-auth-runtime*|*voice_auth_runtime*|*uvicorn*voice_auth_runtime*)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

voice_auth_runtime_is_running() {
  local pid
  pid="$(voice_auth_runtime_pid)"
  if [ -z "$pid" ]; then
    return 1
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    return 1
  fi
  voice_auth_runtime_process_matches "$pid"
}

voice_auth_runtime_health() {
  if ! command -v curl >/dev/null 2>&1; then
    return 127
  fi

  local api_key="${VOICE_AUTH_API_KEY:-${AIAVATAR_API_KEY:-}}"
  if [ -n "$api_key" ]; then
    curl -fsS -H "Authorization: Bearer $api_key" "$VOICE_AUTH_RUNTIME_HEALTH_URL"
  else
    curl -fsS "$VOICE_AUTH_RUNTIME_HEALTH_URL"
  fi
}

voice_auth_runtime_launchctl_target() {
  printf '%s/%s' "$VOICE_AUTH_RUNTIME_LAUNCHCTL_DOMAIN" "$VOICE_AUTH_RUNTIME_LABEL"
}

voice_auth_runtime_launchctl_pid() {
  if ! command -v launchctl >/dev/null 2>&1; then
    return 0
  fi
  launchctl print "$(voice_auth_runtime_launchctl_target)" 2>/dev/null \
    | awk -F '= ' '/^[[:space:]]*pid = / {print $2; exit}'
}

voice_auth_runtime_launchctl_loaded() {
  if ! command -v launchctl >/dev/null 2>&1; then
    return 1
  fi
  launchctl print "$(voice_auth_runtime_launchctl_target)" >/dev/null 2>&1
}

voice_auth_runtime_write_plist() {
  local label repo command log_file
  label="$(voice_auth_runtime_xml_escape "$VOICE_AUTH_RUNTIME_LABEL")"
  repo="$(voice_auth_runtime_xml_escape "$VOICE_AUTH_RUNTIME_REPO_ROOT")"
  command="$(voice_auth_runtime_xml_escape "$VOICE_AUTH_RUNTIME_COMMAND")"
  log_file="$(voice_auth_runtime_xml_escape "$VOICE_AUTH_RUNTIME_LOG_FILE")"

  cat > "$VOICE_AUTH_RUNTIME_PLIST_FILE" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>$label</string>
  <key>WorkingDirectory</key>
  <string>$repo</string>
  <key>ProgramArguments</key>
  <array>
    <string>$command</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PYTHONUNBUFFERED</key>
    <string>1</string>
    <key>VOICE_AUTH_ENABLED</key>
    <string>true</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <false/>
  <key>StandardOutPath</key>
  <string>$log_file</string>
  <key>StandardErrorPath</key>
  <string>$log_file</string>
</dict>
</plist>
PLIST
}
