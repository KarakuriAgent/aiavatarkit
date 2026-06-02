from aiavatar.sts.tts.openai import OpenAISpeechSynthesizer

from ..config import Settings


def create_tts(settings: Settings):
    if settings.tts_provider == "openai":
        return OpenAISpeechSynthesizer(
            openai_api_key=settings.openai_api_key,
            speaker=settings.tts_speaker,
        )
    raise ValueError(f"Unsupported TTS_PROVIDER: {settings.tts_provider}")
