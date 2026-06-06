import numpy as np

from aiavatar.sts.wakeword import LiveKitWakewordDetector


class FakeLiveKitWakeWordModel:
    def __init__(self, models):
        self.models = models
        self.calls = 0

    def predict(self, audio):
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.int16
        self.calls += 1
        return {"robo-kanon": 0.8}


def test_livekit_wakeword_detector_detects_threshold_match():
    detector = LiveKitWakewordDetector(
        model_paths=["models/robo-kanon.onnx"],
        threshold=0.5,
        frame_ms=1000,
        activation_window=3,
        cooldown=0,
        model_class=FakeLiveKitWakeWordModel,
    )
    audio = bytes(48000 * 2)
    detection = detector.process(audio, sample_rate=16000, session_id="s1")

    assert detection.accepted is True
    assert detection.source == "audio"
    assert detection.reason == "matched"
    assert detection.matched_wakeword == "robo-kanon"
    assert detection.score == 0.8
    assert detection.threshold == 0.5
    assert detection.expires_at > detection.detected_at
    assert detection.metadata["window_count"] == 1
    assert detector.model.calls == 1


def test_livekit_wakeword_detector_pads_short_audio_to_required_window():
    class PaddingAwareModel(FakeLiveKitWakeWordModel):
        def predict(self, audio):
            assert len(audio) == 32000
            assert audio.dtype == np.int16
            self.calls += 1
            return {"robo-kanon": 0.9}

    detector = LiveKitWakewordDetector(
        model_paths=["models/robo-kanon.onnx"],
        threshold=0.5,
        cooldown=0,
        model_class=PaddingAwareModel,
    )
    audio = bytes(12000 * 2)
    detection = detector.process(audio, sample_rate=16000, session_id="s1")

    assert detection is not None
    assert detection.score == 0.9
    assert detection.metadata["window_count"] == 1
    assert detector.model.calls == 1


def test_livekit_wakeword_detector_uses_best_window_score():
    class WindowScoredModel(FakeLiveKitWakeWordModel):
        def predict(self, audio):
            self.calls += 1
            return {"robo-kanon": 0.2 if self.calls == 1 else 0.92}

    detector = LiveKitWakewordDetector(
        model_paths=["models/robo-kanon.onnx"],
        threshold=0.8,
        frame_ms=1000,
        cooldown=0,
        model_class=WindowScoredModel,
    )
    audio = bytes(48000 * 2)
    detection = detector.process(audio, sample_rate=16000, session_id="s1")

    assert detection is not None
    assert detection.score == 0.92
    assert detector.model.calls == 2


def test_livekit_wakeword_detector_stops_scanning_after_threshold_match():
    class EarlyMatchModel(FakeLiveKitWakeWordModel):
        def predict(self, audio):
            self.calls += 1
            return {"robo-kanon": 0.91}

    detector = LiveKitWakewordDetector(
        model_paths=["models/robo-kanon.onnx"],
        threshold=0.89,
        frame_ms=250,
        cooldown=0,
        model_class=EarlyMatchModel,
    )
    audio = bytes(64000 * 2)
    detection = detector.process(audio, sample_rate=16000, session_id="s1")

    assert detection is not None
    assert detection.score == 0.91
    assert detection.metadata["window_count"] == 1
    assert detector.model.calls == 1


def test_livekit_wakeword_detector_respects_cooldown():
    class AlwaysDetectingModel(FakeLiveKitWakeWordModel):
        def predict(self, audio):
            return {"wake": 0.9}

    detector = LiveKitWakewordDetector(
        model_paths=["wake.onnx"],
        threshold=0.5,
        cooldown=10,
        model_class=AlwaysDetectingModel,
    )
    audio = bytes(32000 * 2)

    first = detector.process(audio, sample_rate=16000, session_id="s1")
    second = detector.process(audio, sample_rate=16000, session_id="s1")

    assert first is not None
    assert second is None
