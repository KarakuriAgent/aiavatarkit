from aiavatar.sts.tts.openai import OpenAISpeechSynthesizer
from aiavatar.sts.tts import AudioConverter, create_instant_synthesizer

from ..config import Settings


def create_tts(settings: Settings):
    if settings.tts_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when TTS_PROVIDER=openai")
        return OpenAISpeechSynthesizer(
            openai_api_key=settings.openai_api_key,
            speaker=settings.tts_speaker,
        )
    if settings.tts_provider == "aivis":
        if not settings.aivis_api_key:
            raise ValueError("AIVIS_API_KEY is required when TTS_PROVIDER=aivis")
        return create_instant_synthesizer(
            method="POST",
            url=settings.aivis_tts_url,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {settings.aivis_api_key}",
            },
            json={
                "model_uuid": settings.aivis_model_uuid,
                "text": "{text}",
            },
            response_parser=AudioConverter(debug=settings.debug).convert,
            timeout=settings.aivis_tts_timeout,
            cache_dir=settings.aivis_tts_cache_dir,
        )
    raise ValueError(f"Unsupported TTS_PROVIDER: {settings.tts_provider}")
