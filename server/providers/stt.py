from logging import getLogger

from aiavatar.sts.stt.openai import OpenAISpeechRecognizer
from aiavatar.sts.stt import SpeechRecognizer

from ..config import Settings

logger = getLogger(__name__)


class WhisperCompatibleSpeechRecognizer(SpeechRecognizer):
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = None,
        model: str = "whisperkit",
        min_data_length: int = 4096,
        sample_rate: int = 16000,
        language: str = "ja",
        timeout: float = 30.0,
        debug: bool = False,
    ):
        super().__init__(language=language, timeout=timeout, debug=debug)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.min_data_length = min_data_length
        self.sample_rate = sample_rate

    async def transcribe(self, data: bytes) -> str:
        if len(data) < self.min_data_length:
            if self.debug:
                logger.info("Data to transcribe is too short: %s", len(data))
            return None

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        resp = await self.http_request_with_retry(
            method="POST",
            url=f"{self.base_url}/audio/transcriptions",
            headers=headers,
            data={
                "model": self.model,
                "language": self.language,
            },
            files={
                "file": ("voice.wav", self.to_wave_file(data, self.sample_rate), "audio/wav"),
            },
        )
        if not resp:
            return None

        try:
            recognized_text = resp.json()["text"]
            if self.debug:
                logger.info("Recognized: %s", recognized_text)
            return recognized_text
        except Exception:
            logger.exception("Failed to parse transcription response: %s", resp.text)
            return None


def create_stt(settings: Settings):
    if settings.stt_provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when STT_PROVIDER=openai")
        return OpenAISpeechRecognizer(
            openai_api_key=settings.openai_api_key,
            language=settings.stt_language,
        )
    if settings.stt_provider == "whisper_compatible":
        if not settings.stt_base_url:
            raise ValueError("STT_BASE_URL is required when STT_PROVIDER=whisper_compatible")
        return WhisperCompatibleSpeechRecognizer(
            base_url=settings.stt_base_url,
            api_key=settings.stt_api_key,
            model=settings.stt_model,
            min_data_length=settings.stt_min_data_length,
            sample_rate=settings.stt_sample_rate,
            language=settings.stt_language,
            timeout=settings.stt_timeout,
            debug=settings.debug,
        )
    raise ValueError(f"Unsupported STT_PROVIDER: {settings.stt_provider}")
