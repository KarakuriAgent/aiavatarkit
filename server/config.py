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


def optional_int_env(name: str) -> int | None:
    value = os.environ.get(name)
    return int(value) if value else None


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
    stt_context: str | None
    stt_max_new_tokens: int | None

    vad_provider: str
    vad_segment_silence_threshold: float
    vad_use_iterator: bool

    audio_enhancement_enabled: bool
    audio_enhancement_provider: str
    audio_enhancement_command: str
    audio_enhancement_model: str | None
    audio_enhancement_timeout: float
    audio_enhancement_fail_open: bool
    audio_enhancement_record_raw: bool
    audio_enhancement_record_enhanced: bool

    voice_auth_enabled: bool
    voice_auth_provider: str
    voice_auth_base_url: str | None
    voice_auth_api_key: str | None
    voice_auth_model_path: str | None
    voice_auth_model_repo: str
    voice_auth_auto_download_model: bool
    voice_auth_profile_dir: str
    voice_auth_enrollment_dir: str
    voice_auth_threshold: float
    voice_auth_min_duration: float
    voice_auth_sample_rate: int
    voice_auth_require_user_id: bool
    voice_auth_allow_identification: bool
    voice_auth_allowed_users: List[str]
    voice_auth_fail_open: bool
    voice_auth_apply_cmn: bool

    addressing_enabled: bool
    addressing_provider: str
    addressing_base_url: str | None
    addressing_api_key: str | None
    addressing_model: str | None
    addressing_target_names: List[str]
    addressing_primary_name: str | None
    addressing_history_limit: int
    addressing_timeout: float
    addressing_fail_open: bool
    addressing_min_confidence: float

    llm_provider: str

    tts_provider: str
    tts_api_key: str | None
    tts_base_url: str
    tts_speaker: str
    tts_model: str
    tts_instructions: str | None
    tts_response_format: str
    tts_cache_dir: str | None
    tts_timeout: float
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
    stt_provider = os.environ.get("STT_PROVIDER", "whisper_compatible")
    stt_default_model = "Qwen/Qwen3-ASR-0.6B" if stt_provider == "qwen3_asr_mlx" else "whisperkit"
    stt_model = os.environ.get("STT_MODEL", stt_default_model)
    if stt_provider == "qwen3_asr_mlx" and stt_model == "whisperkit":
        stt_model = stt_default_model
    return Settings(
        openai_api_key=optional_env("OPENAI_API_KEY"),
        hermes_api_key=required_env("HERMES_API_KEY"),
        hermes_base_url=os.environ.get("HERMES_BASE_URL", "http://127.0.0.1:8642/v1"),
        hermes_model=os.environ.get("HERMES_MODEL", "hermes-agent"),
        hermes_request_prefix=os.environ.get("HERMES_REQUEST_PREFIX", "[channel:voice]"),
        hermes_conversation_id_source=os.environ.get("HERMES_CONVERSATION_ID_SOURCE", "user_id"),
        hermes_store=bool_env("HERMES_STORE", True),
        hermes_reasoning_effort=optional_env("HERMES_REASONING_EFFORT"),
        stt_provider=stt_provider,
        stt_base_url=optional_env("STT_BASE_URL"),
        stt_api_key=optional_env("STT_API_KEY"),
        stt_model=stt_model,
        stt_language=os.environ.get("STT_LANGUAGE", "ja"),
        stt_min_data_length=int(os.environ.get("STT_MIN_DATA_LENGTH", "4096")),
        stt_sample_rate=int(os.environ.get("STT_SAMPLE_RATE", "16000")),
        stt_timeout=float(os.environ.get("STT_TIMEOUT", "30")),
        stt_context=optional_env("STT_CONTEXT"),
        stt_max_new_tokens=optional_int_env("STT_MAX_NEW_TOKENS"),
        vad_provider=os.environ.get("VAD_PROVIDER", "silero_stream"),
        vad_segment_silence_threshold=float(os.environ.get("VAD_SEGMENT_SILENCE_THRESHOLD", "0.05")),
        vad_use_iterator=bool_env("VAD_USE_ITERATOR", True),
        audio_enhancement_enabled=bool_env("AUDIO_ENHANCEMENT_ENABLED", False),
        audio_enhancement_provider=os.environ.get("AUDIO_ENHANCEMENT_PROVIDER", "deepfilternet"),
        audio_enhancement_command="/usr/local/bin/deep-filter",
        audio_enhancement_model=optional_env("AUDIO_ENHANCEMENT_MODEL"),
        audio_enhancement_timeout=float(os.environ.get("AUDIO_ENHANCEMENT_TIMEOUT", "30")),
        audio_enhancement_fail_open=bool_env("AUDIO_ENHANCEMENT_FAIL_OPEN", True),
        audio_enhancement_record_raw=bool_env("AUDIO_ENHANCEMENT_RECORD_RAW", True),
        audio_enhancement_record_enhanced=bool_env("AUDIO_ENHANCEMENT_RECORD_ENHANCED", True),
        voice_auth_enabled=bool_env("VOICE_AUTH_ENABLED", False),
        voice_auth_provider=os.environ.get("VOICE_AUTH_PROVIDER", "wespeaker_mlx"),
        voice_auth_base_url=optional_env("VOICE_AUTH_BASE_URL"),
        voice_auth_api_key=optional_env("VOICE_AUTH_API_KEY"),
        voice_auth_model_path=optional_env("VOICE_AUTH_MODEL_PATH") or "models/wespeaker-voxceleb-resnet34-LM-mlx",
        voice_auth_model_repo=os.environ.get("VOICE_AUTH_MODEL_REPO", "Landon41/wespeaker-voxceleb-resnet34-LM-mlx"),
        voice_auth_auto_download_model=bool_env("VOICE_AUTH_AUTO_DOWNLOAD_MODEL", True),
        voice_auth_profile_dir=os.environ.get("VOICE_AUTH_PROFILE_DIR", "data/voice_profiles"),
        voice_auth_enrollment_dir=os.environ.get("VOICE_AUTH_ENROLLMENT_DIR", "data/voice_auth_enrollment"),
        voice_auth_threshold=float(os.environ.get("VOICE_AUTH_THRESHOLD", "0.65")),
        voice_auth_min_duration=float(os.environ.get("VOICE_AUTH_MIN_DURATION", "1.2")),
        voice_auth_sample_rate=int(os.environ.get("VOICE_AUTH_SAMPLE_RATE", "16000")),
        voice_auth_require_user_id=bool_env("VOICE_AUTH_REQUIRE_USER_ID", True),
        voice_auth_allow_identification=bool_env("VOICE_AUTH_ALLOW_IDENTIFICATION", False),
        voice_auth_allowed_users=list_env("VOICE_AUTH_ALLOWED_USERS"),
        voice_auth_fail_open=bool_env("VOICE_AUTH_FAIL_OPEN", False),
        voice_auth_apply_cmn=bool_env("VOICE_AUTH_APPLY_CMN", True),
        addressing_enabled=bool_env("ADDRESSING_ENABLED", False),
        addressing_provider=os.environ.get("ADDRESSING_PROVIDER", "openai_compatible"),
        addressing_base_url=optional_env("ADDRESSING_BASE_URL"),
        addressing_api_key=optional_env("ADDRESSING_API_KEY"),
        addressing_model=optional_env("ADDRESSING_MODEL"),
        addressing_target_names=list_env("ADDRESSING_TARGET_NAMES"),
        addressing_primary_name=optional_env("ADDRESSING_PRIMARY_NAME"),
        addressing_history_limit=int(os.environ.get("ADDRESSING_HISTORY_LIMIT", "12")),
        addressing_timeout=float(os.environ.get("ADDRESSING_TIMEOUT", "10")),
        addressing_fail_open=bool_env("ADDRESSING_FAIL_OPEN", False),
        addressing_min_confidence=float(os.environ.get("ADDRESSING_MIN_CONFIDENCE", "0")),
        llm_provider=os.environ.get("LLM_PROVIDER", "hermes"),
        tts_provider=os.environ.get("TTS_PROVIDER", "aivis"),
        tts_api_key=optional_env("TTS_API_KEY") or optional_env("IRODORI_API_KEY"),
        tts_base_url=os.environ.get("TTS_BASE_URL", "https://api.openai.com/v1"),
        tts_speaker=os.environ.get("TTS_SPEAKER", "coral"),
        tts_model=os.environ.get("TTS_MODEL", "tts-1"),
        tts_instructions=optional_env("TTS_INSTRUCTIONS"),
        tts_response_format=os.environ.get("TTS_RESPONSE_FORMAT", "wav"),
        tts_cache_dir=optional_env("TTS_CACHE_DIR"),
        tts_timeout=float(os.environ.get("TTS_TIMEOUT", "30")),
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
