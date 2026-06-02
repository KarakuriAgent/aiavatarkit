from aiavatar.sts.vad.stream import SileroStreamSpeechDetector

from ..config import Settings


def create_vad(settings: Settings, stt):
    if settings.vad_provider == "silero_stream":
        return SileroStreamSpeechDetector(
            speech_recognizer=stt,
            segment_silence_threshold=settings.vad_segment_silence_threshold,
            use_vad_iterator=settings.vad_use_iterator,
        )
    raise ValueError(f"Unsupported VAD_PROVIDER: {settings.vad_provider}")
