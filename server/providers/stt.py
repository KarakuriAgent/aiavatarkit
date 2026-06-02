from aiavatar.sts.stt.openai import OpenAISpeechRecognizer

from ..config import Settings


def create_stt(settings: Settings):
    if settings.stt_provider == "openai":
        return OpenAISpeechRecognizer(
            openai_api_key=settings.openai_api_key,
            language=settings.stt_language,
        )
    raise ValueError(f"Unsupported STT_PROVIDER: {settings.stt_provider}")
