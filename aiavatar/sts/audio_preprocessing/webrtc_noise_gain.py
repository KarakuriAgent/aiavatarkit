import logging
from dataclasses import dataclass, field
from typing import Optional

from .base import StreamingAudioProcessor

logger = logging.getLogger(__name__)


def _load_audio_processor_class():
    try:
        from webrtc_noise_gain import AudioProcessor
    except ImportError as ex:
        raise RuntimeError(
            "webrtc-noise-gain is required when PRE_VAD_NOISE_SUPPRESSION_PROVIDER=webrtc_noise_gain. "
            "Install dependencies with `pip install webrtc-noise-gain` or your project package manager."
        ) from ex
    return AudioProcessor


@dataclass
class _WebRtcNoiseGainSession:
    processor: object
    buffer: bytearray = field(default_factory=bytearray)


class WebRtcNoiseGainAudioProcessor(StreamingAudioProcessor):
    """WebRTC noise suppression for 16 kHz mono PCM before VAD."""

    sample_rate = 16000
    frame_samples = 160
    sample_width = 2
    frame_bytes = frame_samples * sample_width

    def __init__(
        self,
        *,
        noise_suppression_level: int = 2,
        auto_gain_dbfs: int = 0,
        fail_open: bool = True,
        audio_processor_class=None,
        debug: bool = False,
    ):
        if not 0 <= noise_suppression_level <= 4:
            raise ValueError("noise_suppression_level must be between 0 and 4")
        if not 0 <= auto_gain_dbfs <= 31:
            raise ValueError("auto_gain_dbfs must be between 0 and 31")

        self.noise_suppression_level = int(noise_suppression_level)
        self.auto_gain_dbfs = int(auto_gain_dbfs)
        self.fail_open = fail_open
        self.debug = debug
        self._audio_processor_class = audio_processor_class or _load_audio_processor_class()
        self._sessions: dict[str, _WebRtcNoiseGainSession] = {}

    def process(
        self,
        audio_bytes: bytes,
        *,
        sample_rate: int,
        session_id: Optional[str] = None,
    ) -> bytes:
        if not audio_bytes:
            return b""
        if sample_rate != self.sample_rate:
            raise ValueError("WebRtcNoiseGainAudioProcessor requires 16000Hz linear16 mono input")

        key = session_id or "__default__"
        try:
            session = self._get_session(key)
            session.buffer.extend(audio_bytes)

            output = bytearray()
            while len(session.buffer) >= self.frame_bytes:
                frame = bytes(session.buffer[:self.frame_bytes])
                del session.buffer[:self.frame_bytes]
                result = session.processor.Process10ms(frame)
                output.extend(self._extract_audio(result))

            if self.debug and output:
                logger.debug(
                    "Pre-VAD WebRTC noise suppression processed: session=%s, input_bytes=%s, output_bytes=%s, buffered_bytes=%s",
                    key,
                    len(audio_bytes),
                    len(output),
                    len(session.buffer),
                )
            return bytes(output)

        except Exception:
            self.reset_session(key)
            if not self.fail_open:
                raise
            logger.warning("Pre-VAD WebRTC noise suppression failed; passing raw audio through", exc_info=True)
            return audio_bytes

    def reset_session(self, session_id: str):
        self._sessions.pop(session_id, None)

    def get_config(self) -> dict:
        return {
            "provider": "webrtc_noise_gain",
            "sample_rate": self.sample_rate,
            "frame_ms": 10,
            "noise_suppression_level": self.noise_suppression_level,
            "auto_gain_dbfs": self.auto_gain_dbfs,
            "fail_open": self.fail_open,
            "debug": self.debug,
        }

    def _get_session(self, key: str) -> _WebRtcNoiseGainSession:
        session = self._sessions.get(key)
        if session is None:
            session = _WebRtcNoiseGainSession(
                processor=self._audio_processor_class(
                    self.auto_gain_dbfs,
                    self.noise_suppression_level,
                )
            )
            self._sessions[key] = session
        return session

    def _extract_audio(self, result) -> bytes:
        audio = getattr(result, "audio", result)
        if isinstance(audio, bytes):
            return audio
        if isinstance(audio, bytearray):
            return bytes(audio)
        if isinstance(audio, memoryview):
            return audio.tobytes()
        if hasattr(audio, "tobytes"):
            return audio.tobytes()
        return bytes(audio)
