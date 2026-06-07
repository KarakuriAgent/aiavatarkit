#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

if [ -f "$REPO_ROOT/provider/common.sh" ]; then
  # Allows the launcher to be run directly as well as through provider/start.sh.
  # provider_load_env only fills variables that are not already exported.
  . "$REPO_ROOT/provider/common.sh"
  provider_load_env
fi

: "${STT_WHISPERKIT_CLI:=whisperkit-cli}"
: "${STT_WHISPERKIT_HOST:=0.0.0.0}"
: "${STT_WHISPERKIT_PORT:=5003}"
: "${STT_WHISPERKIT_MODEL_PREFIX:=openai}"

if ! command -v "$STT_WHISPERKIT_CLI" >/dev/null 2>&1; then
  echo "whisperkit-cli is not available: STT_WHISPERKIT_CLI=$STT_WHISPERKIT_CLI" >&2
  echo "Install WhisperKit CLI first, or set STT_WHISPERKIT_CLI to the executable path." >&2
  exit 1
fi

args=(
  serve
  --host "$STT_WHISPERKIT_HOST"
  --port "$STT_WHISPERKIT_PORT"
)

if [ -n "${STT_WHISPERKIT_MODEL_PATH:-}" ]; then
  if [ ! -d "$STT_WHISPERKIT_MODEL_PATH" ]; then
    echo "STT_WHISPERKIT_MODEL_PATH does not exist or is not a directory: $STT_WHISPERKIT_MODEL_PATH" >&2
    exit 1
  fi
  args+=(--model-path "$STT_WHISPERKIT_MODEL_PATH")
else
  if [ -z "${STT_WHISPERKIT_MODEL:-}" ]; then
    echo "Set STT_WHISPERKIT_MODEL_PATH for an existing CoreML model directory, or STT_WHISPERKIT_MODEL to download one." >&2
    exit 1
  fi
  args+=(--model "$STT_WHISPERKIT_MODEL" --model-prefix "$STT_WHISPERKIT_MODEL_PREFIX")
fi

if [ -n "${STT_WHISPERKIT_DOWNLOAD_MODEL_PATH:-}" ]; then
  mkdir -p "$STT_WHISPERKIT_DOWNLOAD_MODEL_PATH"
  args+=(--download-model-path "$STT_WHISPERKIT_DOWNLOAD_MODEL_PATH")
fi

if [ -n "${STT_WHISPERKIT_DOWNLOAD_TOKENIZER_PATH:-}" ]; then
  mkdir -p "$STT_WHISPERKIT_DOWNLOAD_TOKENIZER_PATH"
  args+=(--download-tokenizer-path "$STT_WHISPERKIT_DOWNLOAD_TOKENIZER_PATH")
fi

if [ -n "${STT_WHISPERKIT_ENDPOINT:-}" ]; then
  args+=(--endpoint "$STT_WHISPERKIT_ENDPOINT")
fi

language="${STT_WHISPERKIT_LANGUAGE:-${STT_LANGUAGE:-}}"
if [ -n "$language" ]; then
  args+=(--language "$language")
fi

if [ -n "${STT_WHISPERKIT_TEMPERATURE:-}" ]; then
  args+=(--temperature "$STT_WHISPERKIT_TEMPERATURE")
fi

if [ -n "${STT_WHISPERKIT_AUDIO_ENCODER_COMPUTE_UNITS:-}" ]; then
  args+=(--audio-encoder-compute-units "$STT_WHISPERKIT_AUDIO_ENCODER_COMPUTE_UNITS")
fi

if [ -n "${STT_WHISPERKIT_TEXT_DECODER_COMPUTE_UNITS:-}" ]; then
  args+=(--text-decoder-compute-units "$STT_WHISPERKIT_TEXT_DECODER_COMPUTE_UNITS")
fi

if [ -n "${STT_WHISPERKIT_CONCURRENT_WORKER_COUNT:-}" ]; then
  args+=(--concurrent-worker-count "$STT_WHISPERKIT_CONCURRENT_WORKER_COUNT")
fi

if [ -n "${STT_WHISPERKIT_CHUNKING_STRATEGY:-}" ]; then
  args+=(--chunking-strategy "$STT_WHISPERKIT_CHUNKING_STRATEGY")
fi

case "${STT_WHISPERKIT_VERBOSE:-false}" in
  1|true|TRUE|yes|YES|on|ON)
    args+=(--verbose)
    ;;
esac

echo "Starting WhisperKit STT runtime: host=$STT_WHISPERKIT_HOST port=$STT_WHISPERKIT_PORT" >&2
exec "$STT_WHISPERKIT_CLI" "${args[@]}"
