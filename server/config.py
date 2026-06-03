import os
from dataclasses import dataclass
from pathlib import Path
from typing import List


def load_env_file(path: Path):
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in (chr(34), chr(39)):
            value = value[1:-1]
        os.environ.setdefault(key, value)


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def optional_env(name: str):
    value = os.environ.get(name)
    return value if value else None


def bool_env(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.lower() in ("1", "true", "yes", "on")


def list_env(name: str, default: List[str] = None) -> List[str]:
    value = os.environ.get(name)
    if value is None or value == "":
        return list(default or [])
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    hermes_api_key: str
    hermes_base_url: str
    hermes_model: str
    hermes_request_prefix: str
    hermes_conversation_id_source: str
    hermes_store: bool
    hermes_reasoning_effort: str | None

    stt_provider: str
    stt_base_url: str | None
    stt_api_key: str | None
    stt_model: str
    stt_language: str
    stt_min_data_length: int
    stt_sample_rate: int
    stt_timeout: float

    vad_provider: str
    vad_segment_silence_threshold: float
    vad_use_iterator: bool

    llm_provider: str

    tts_provider: str
    tts_speaker: str
    aivis_api_key: str | None
    aivis_model_uuid: str
    aivis_tts_url: str
    aivis_tts_cache_dir: str | None
    aivis_tts_timeout: float

    aiavatar_admin_user: str
    aiavatar_api_key: str | None
    aiavatar_db_path: str
    aiavatar_voice_recorder_dir: str
    host: str
    port: int
    ssl_cert_path: str | None
    ssl_key_path: str | None

    merge_request_threshold: float
    use_invoke_queue: bool
    skip_tts_channels: List[str]
    response_audio_chunk_size: int | None
    debug: bool

    discord_sync_enabled: bool
    discord_bot_token: str | None
    discord_channel_id: str | None
    discord_webhook_url: str | None
    discord_guild_id: str | None
    discord_user_id: str | None
    discord_bot_id: str | None
    discord_sync_user_id: str
    discord_gateway_session_id: str
    discord_identity_cache_ttl: float
    discord_voice_message_prefix: str
    discord_api_message_prefix: str
    discord_typing_indicator_enabled: bool
    discord_typing_indicator_interval: float


def load_settings(env_path: Path | None = None) -> Settings:
    if env_path:
        load_env_file(env_path)
    else:
        server_dir = Path(__file__).resolve().parent
        project_root = server_dir.parent
        load_env_file(project_root / ".env")
        load_env_file(server_dir / ".env")

    discord_sync_user_id = os.environ.get("DISCORD_SYNC_USER_ID", "robo-kanon-stack-chan")
    discord_voice_message_prefix = os.environ.get("DISCORD_VOICE_MESSAGE_PREFIX", "🎙️ ")
    return Settings(
        openai_api_key=optional_env("OPENAI_API_KEY"),
        hermes_api_key=required_env("HERMES_API_KEY"),
        hermes_base_url=os.environ.get("HERMES_BASE_URL", "http://127.0.0.1:8642/v1"),
        hermes_model=os.environ.get("HERMES_MODEL", "hermes-agent"),
        hermes_request_prefix=os.environ.get("HERMES_REQUEST_PREFIX", "[channel:voice]"),
        hermes_conversation_id_source=os.environ.get("HERMES_CONVERSATION_ID_SOURCE", "user_id"),
        hermes_store=bool_env("HERMES_STORE", True),
        hermes_reasoning_effort=optional_env("HERMES_REASONING_EFFORT"),
        stt_provider=os.environ.get("STT_PROVIDER", "whisper_compatible"),
        stt_base_url=optional_env("STT_BASE_URL"),
        stt_api_key=optional_env("STT_API_KEY"),
        stt_model=os.environ.get("STT_MODEL", "whisperkit"),
        stt_language=os.environ.get("STT_LANGUAGE", "ja"),
        stt_min_data_length=int(os.environ.get("STT_MIN_DATA_LENGTH", "4096")),
        stt_sample_rate=int(os.environ.get("STT_SAMPLE_RATE", "16000")),
        stt_timeout=float(os.environ.get("STT_TIMEOUT", "30")),
        vad_provider=os.environ.get("VAD_PROVIDER", "silero_stream"),
        vad_segment_silence_threshold=float(os.environ.get("VAD_SEGMENT_SILENCE_THRESHOLD", "0.05")),
        vad_use_iterator=bool_env("VAD_USE_ITERATOR", True),
        llm_provider=os.environ.get("LLM_PROVIDER", "hermes"),
        tts_provider=os.environ.get("TTS_PROVIDER", "aivis"),
        tts_speaker=os.environ.get("TTS_SPEAKER", "coral"),
        aivis_api_key=optional_env("AIVIS_API_KEY"),
        aivis_model_uuid=os.environ.get("AIVIS_MODEL_UUID", "261d7c95-11d4-4f0a-9053-4d28d3dd87ee"),
        aivis_tts_url=os.environ.get("AIVIS_TTS_URL", "https://api.aivis-project.com/v1/tts/synthesize"),
        aivis_tts_cache_dir=optional_env("AIVIS_TTS_CACHE_DIR") or "aivis_tts_cache",
        aivis_tts_timeout=float(os.environ.get("AIVIS_TTS_TIMEOUT", "30")),
        aiavatar_admin_user=os.environ.get("AIAVATAR_ADMIN_USER", "admin"),
        aiavatar_api_key=optional_env("AIAVATAR_API_KEY"),
        aiavatar_db_path=os.environ.get("AIAVATAR_DB_PATH", "aiavatar.db"),
        aiavatar_voice_recorder_dir=os.environ.get("AIAVATAR_VOICE_RECORDER_DIR", "recorded_voices"),
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        ssl_cert_path=optional_env("SSL_CERT_PATH"),
        ssl_key_path=optional_env("SSL_KEY_PATH"),
        merge_request_threshold=float(os.environ.get("MERGE_REQUEST_THRESHOLD", "3.0")),
        use_invoke_queue=bool_env("USE_INVOKE_QUEUE", True),
        skip_tts_channels=list_env("SKIP_TTS_CHANNELS", ["discord"]),
        response_audio_chunk_size=int(os.environ.get("RESPONSE_AUDIO_CHUNK_SIZE", "8192")),
        debug=bool_env("DEBUG", True),
        discord_sync_enabled=bool_env("DISCORD_SYNC_ENABLED", False),
        discord_bot_token=optional_env("DISCORD_BOT_TOKEN"),
        discord_channel_id=optional_env("DISCORD_CHANNEL_ID"),
        discord_webhook_url=optional_env("DISCORD_WEBHOOK_URL"),
        discord_guild_id=optional_env("DISCORD_GUILD_ID"),
        discord_user_id=optional_env("DISCORD_USER_ID"),
        discord_bot_id=optional_env("DISCORD_BOT_ID"),
        discord_sync_user_id=discord_sync_user_id,
        discord_gateway_session_id=os.environ.get("DISCORD_GATEWAY_SESSION_ID", f"discord:{discord_sync_user_id}"),
        discord_identity_cache_ttl=float(os.environ.get("DISCORD_IDENTITY_CACHE_TTL", "300")),
        discord_voice_message_prefix=discord_voice_message_prefix,
        discord_api_message_prefix=os.environ.get("DISCORD_API_MESSAGE_PREFIX", discord_voice_message_prefix),
        discord_typing_indicator_enabled=bool_env("DISCORD_TYPING_INDICATOR_ENABLED", True),
        discord_typing_indicator_interval=float(os.environ.get("DISCORD_TYPING_INDICATOR_INTERVAL", "8")),
    )
