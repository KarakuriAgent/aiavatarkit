import logging
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Optional

import numpy as np

from .base import StreamingWakewordDetector, WakewordDetection

logger = logging.getLogger(__name__)


def _load_livekit_wakeword_model_class():
    try:
        from livekit.wakeword import WakeWordModel
    except ImportError as ex:
        raise RuntimeError(
            "livekit-wakeword is required when AUDIO_WAKEWORD_PROVIDER=livekit_wakeword. "
            "Install dependencies with `pip install livekit-wakeword` or your project package manager."
        ) from ex
    return WakeWordModel


@dataclass
class _WakewordSession:
    last_detection_at: float = 0.0


class LiveKitWakewordDetector(StreamingWakewordDetector):
    sample_rate = 16000
    sample_width = 2

    def __init__(
        self,
        *,
        model_paths: list[str],
        threshold: float = 0.5,
        frame_ms: int = 250,
        activation_window: float = 4.0,
        cooldown: float = 2.0,
        model_class=None,
        debug: bool = False,
    ):
        if not model_paths:
            raise ValueError("AUDIO_WAKEWORD_MODEL_PATHS is required when audio wakeword detection is enabled")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if activation_window <= 0:
            raise ValueError("activation_window must be greater than 0")
        if cooldown < 0:
            raise ValueError("cooldown must be greater than or equal to 0")
        if frame_ms <= 0:
            raise ValueError("frame_ms must be greater than 0")

        self.model_paths = [str(Path(path)) for path in model_paths]
        self.threshold = float(threshold)
        self.frame_ms = int(frame_ms)
        self.activation_window = float(activation_window)
        self.cooldown = float(cooldown)
        self.debug = debug
        self.window_samples = self.sample_rate * 2
        self.stride_samples = max(1, int(self.sample_rate * self.frame_ms / 1000))
        self._sessions: dict[str, _WakewordSession] = {}

        model_cls = model_class or _load_livekit_wakeword_model_class()
        self.model = model_cls(models=self.model_paths)

    def process(
        self,
        audio_bytes: bytes,
        *,
        sample_rate: int,
        session_id: Optional[str] = None,
    ) -> Optional[WakewordDetection]:
        if not audio_bytes:
            return None
        if sample_rate != self.sample_rate:
            raise ValueError("LiveKitWakewordDetector requires 16000Hz linear16 mono input")

        if len(audio_bytes) % self.sample_width:
            audio_bytes = audio_bytes[: -(len(audio_bytes) % self.sample_width)]
        if not audio_bytes:
            return None

        key = session_id or "__default__"
        session = self._sessions.setdefault(key, _WakewordSession())
        prediction, window_count = self._predict_across_windows(np.frombuffer(audio_bytes, dtype=np.int16))
        detection = self._prediction_to_detection(prediction, session)
        if detection:
            detection.metadata["window_count"] = window_count
        if detection and self.debug:
            logger.info(
                "Audio wakeword detected: session=%s matched=%s score=%s threshold=%s",
                key,
                detection.matched_wakeword,
                detection.score,
                detection.threshold,
            )
        return detection

    def reset_session(self, session_id: str):
        self._sessions.pop(session_id, None)

    def get_config(self) -> dict:
        return {
            "provider": "livekit_wakeword",
            "model_paths": self.model_paths,
            "threshold": self.threshold,
            "frame_ms": self.frame_ms,
            "activation_window": self.activation_window,
            "cooldown": self.cooldown,
            "debug": self.debug,
        }

    def _predict_across_windows(self, audio: np.ndarray) -> tuple[dict, int]:
        max_prediction: dict[str, float] = {}
        window_count = 0
        for window in self._iter_windows(audio):
            window_count += 1
            prediction = self.model.predict(window)
            for name, score in prediction.items():
                score = float(score)
                if score > max_prediction.get(name, 0.0):
                    max_prediction[name] = score
            if max_prediction and max(max_prediction.values()) >= self.threshold:
                return max_prediction, window_count
        return max_prediction, window_count

    def _iter_windows(self, audio: np.ndarray):
        if len(audio) < self.window_samples:
            yield np.pad(audio, (0, self.window_samples - len(audio)), mode="constant")
            return

        final_start = len(audio) - self.window_samples
        start = 0
        while start <= final_start:
            yield audio[start : start + self.window_samples]
            start += self.stride_samples

        last_start = start - self.stride_samples
        if last_start != final_start:
            yield audio[final_start:]

    def _prediction_to_detection(
        self,
        prediction: dict,
        session: _WakewordSession,
    ) -> Optional[WakewordDetection]:
        if not prediction:
            return None

        best_name, best_score = max(
            ((str(name), float(score)) for name, score in prediction.items()),
            key=lambda item: item[1],
        )
        now = monotonic()
        if best_score < self.threshold:
            return None
        if self.cooldown and now - session.last_detection_at < self.cooldown:
            return None

        session.last_detection_at = now
        return WakewordDetection(
            accepted=True,
            reason="matched",
            source="audio",
            matched_wakeword=self._display_name(best_name),
            score=best_score,
            threshold=self.threshold,
            detected_at=now,
            expires_at=now + self.activation_window,
            metadata={
                "provider": "livekit_wakeword",
                "raw_model_name": best_name,
            },
        )

    def _display_name(self, name: str) -> str:
        path = Path(name)
        return path.stem if path.suffix else name
