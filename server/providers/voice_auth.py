from logging import getLogger

from ..config import Settings

logger = getLogger(__name__)


def create_voice_auth(settings: Settings):
    if not settings.voice_auth_enabled:
        return None

    if settings.voice_auth_provider in ("http", "wespeaker_mlx"):
        if not settings.voice_auth_base_url:
            raise ValueError(f"VOICE_AUTH_BASE_URL is required when VOICE_AUTH_PROVIDER={settings.voice_auth_provider}")
        from aiavatar.sts.voice_auth.http import HttpVoiceAuthenticator

        return HttpVoiceAuthenticator(
            base_url=settings.voice_auth_base_url,
            api_key=settings.voice_auth_api_key or settings.aiavatar_api_key,
            timeout=settings.stt_timeout,
            debug=settings.debug,
        )

    raise ValueError(f"Unsupported VOICE_AUTH_PROVIDER: {settings.voice_auth_provider}")
