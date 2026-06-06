from aiavatar.sts.vad.stream import SileroStreamSpeechDetector
from aiavatar.sts.vad.tenvad import TenVadStreamSpeechDetector, make_http_tenvad_session_factory

from ..config import Settings


def create_vad(settings: Settings, stt):
    if settings.vad_provider in ("silero", "silero_stream"):
        return SileroStreamSpeechDetector(
            speech_recognizer=stt,
            segment_silence_threshold=settings.vad_segment_silence_threshold,
            use_vad_iterator=settings.vad_use_iterator,
        )
    if settings.vad_provider == "tenvad":
        if not settings.vad_base_url:
            raise ValueError("VAD_BASE_URL is required when VAD_PROVIDER=tenvad")
        ten_vad_factory = make_http_tenvad_session_factory(
            base_url=settings.vad_base_url,
            api_key=settings.vad_api_key or settings.aiavatar_api_key,
            timeout=settings.vad_timeout,
            neg_threshold=settings.vad_neg_threshold,
        )
        return TenVadStreamSpeechDetector(
            speech_recognizer=stt,
            segment_silence_threshold=settings.vad_segment_silence_threshold,
            silence_duration_threshold=settings.vad_min_silence_ms / 1000,
            min_duration=settings.vad_min_speech_ms / 1000,
            speech_probability_threshold=settings.vad_threshold,
            negative_speech_probability_threshold=settings.vad_neg_threshold,
            hop_size=settings.vad_hop_size,
            speech_pad_ms=settings.vad_speech_pad_ms,
            ten_vad_class=ten_vad_factory,
        )
    raise ValueError(f"Unsupported VAD_PROVIDER: {settings.vad_provider}")
