import os
from dataclasses import dataclass
from pathlib import Path


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


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    openclaw_token: str
    openclaw_base_url: str
    openclaw_model: str
    openclaw_request_prefix: str

    stt_provider: str
    stt_language: str

    vad_provider: str
    vad_segment_silence_threshold: float
    vad_use_iterator: bool

    llm_provider: str

    tts_provider: str
    tts_speaker: str

    aiavatar_admin_user: str
    aiavatar_api_key: str | None
    host: str
    port: int
    ssl_cert_path: str | None
    ssl_key_path: str | None

    merge_request_threshold: float
    use_invoke_queue: bool
    debug: bool


def load_settings(env_path: Path | None = None) -> Settings:
    load_env_file(env_path or Path(__file__).with_name(".env"))

    return Settings(
        openai_api_key=required_env("OPENAI_API_KEY"),
        openclaw_token=required_env("OPENCLAW_TOKEN"),
        openclaw_base_url=os.environ.get("OPENCLAW_BASE_URL", "http://127.0.0.1:18789/v1"),
        openclaw_model=os.environ.get("OPENCLAW_MODEL", "openclaw"),
        openclaw_request_prefix=os.environ.get("OPENCLAW_REQUEST_PREFIX", "[channel:voice]"),
        stt_provider=os.environ.get("STT_PROVIDER", "openai"),
        stt_language=os.environ.get("STT_LANGUAGE", "ja"),
        vad_provider=os.environ.get("VAD_PROVIDER", "silero_stream"),
        vad_segment_silence_threshold=float(os.environ.get("VAD_SEGMENT_SILENCE_THRESHOLD", "0.05")),
        vad_use_iterator=bool_env("VAD_USE_ITERATOR", True),
        llm_provider=os.environ.get("LLM_PROVIDER", "openclaw"),
        tts_provider=os.environ.get("TTS_PROVIDER", "openai"),
        tts_speaker=os.environ.get("TTS_SPEAKER", "coral"),
        aiavatar_admin_user=os.environ.get("AIAVATAR_ADMIN_USER", "admin"),
        aiavatar_api_key=optional_env("AIAVATAR_API_KEY"),
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
        ssl_cert_path=optional_env("SSL_CERT_PATH"),
        ssl_key_path=optional_env("SSL_KEY_PATH"),
        merge_request_threshold=float(os.environ.get("MERGE_REQUEST_THRESHOLD", "3.0")),
        use_invoke_queue=bool_env("USE_INVOKE_QUEUE", True),
        debug=bool_env("DEBUG", True),
    )
