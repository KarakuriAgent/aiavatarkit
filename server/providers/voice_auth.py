from logging import getLogger

from ..config import Settings

logger = getLogger(__name__)


def create_voice_auth(settings: Settings):
    if not settings.voice_auth_enabled:
        return None

    if settings.voice_auth_provider == "http":
        if not settings.voice_auth_base_url:
            raise ValueError("VOICE_AUTH_BASE_URL is required when VOICE_AUTH_PROVIDER=http")
        from aiavatar.sts.voice_auth.http import HttpVoiceAuthenticator

        return HttpVoiceAuthenticator(
            base_url=settings.voice_auth_base_url,
            api_key=settings.voice_auth_api_key or settings.aiavatar_api_key,
            timeout=settings.stt_timeout,
            debug=settings.debug,
        )

    if settings.voice_auth_provider == "wespeaker_mlx":
        if not settings.voice_auth_model_path:
            raise ValueError("VOICE_AUTH_MODEL_PATH is required when VOICE_AUTH_PROVIDER=wespeaker_mlx")
        from aiavatar.sts.voice_auth.wespeaker_mlx import WespeakerMlxVoiceAuthenticator

        return WespeakerMlxVoiceAuthenticator(
            model_path=settings.voice_auth_model_path,
            profile_dir=settings.voice_auth_profile_dir,
            threshold=settings.voice_auth_threshold,
            min_duration=settings.voice_auth_min_duration,
            target_sample_rate=settings.voice_auth_sample_rate,
            require_user_id=settings.voice_auth_require_user_id,
            allow_identification=settings.voice_auth_allow_identification,
            allowed_users=settings.voice_auth_allowed_users,
            fail_open=settings.voice_auth_fail_open,
            apply_cmn=settings.voice_auth_apply_cmn,
            debug=settings.debug,
        )

    raise ValueError(f"Unsupported VOICE_AUTH_PROVIDER: {settings.voice_auth_provider}")
