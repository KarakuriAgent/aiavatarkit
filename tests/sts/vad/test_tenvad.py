import numpy as np

from aiavatar.sts.vad.tenvad import TenVadDecisionState, TenVadStreamSpeechDetector


class FakeTenVad:
    flags = []
    instances = []

    def __init__(self, hop_size: int, threshold: float):
        self.hop_size = hop_size
        self.threshold = threshold
        self.processed_frames = []
        FakeTenVad.instances.append(self)

    def process(self, audio_data):
        self.processed_frames.append(audio_data.copy())
        flag = FakeTenVad.flags.pop(0) if FakeTenVad.flags else 0
        return (0.9 if flag else 0.1), flag


def test_detect_speech_tenvad_processes_complete_frames_and_keeps_remainder():
    FakeTenVad.flags = [0, 1]
    FakeTenVad.instances = []
    detector = TenVadStreamSpeechDetector(
        speech_recognizer=object(),
        ten_vad_class=FakeTenVad,
        hop_size=4,
    )
    session = detector.get_session("session-1")
    session.vad_buffer.extend(np.arange(10, dtype=np.int16).tobytes())

    speech_detected = detector._detect_speech_tenvad(session)

    assert speech_detected is True
    assert len(FakeTenVad.instances) == 1
    assert [frame.tolist() for frame in FakeTenVad.instances[0].processed_frames] == [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ]
    assert np.frombuffer(session.vad_buffer, dtype=np.int16).tolist() == [8, 9]


def test_tenvad_decision_state_uses_negative_threshold_hysteresis():
    state = TenVadDecisionState(threshold=0.28, neg_threshold=0.14)

    assert state.update(0.27, False) is False
    assert state.update(0.30, True) is True
    assert state.update(0.20, False) is True
    assert state.update(0.13, False) is False


def test_detect_speech_tenvad_uses_remote_batch_client():
    class FakeRemoteTenVad:
        def __init__(self, hop_size: int, threshold: float, *, session_id=None, neg_threshold=None):
            self.hop_size = hop_size
            self.threshold = threshold
            self.session_id = session_id
            self.neg_threshold = neg_threshold
            self.frames = []

        def process_frames(self, frames: bytes):
            self.frames.append(frames)
            return {"speech_detected": True, "frames": len(frames) // (self.hop_size * 2)}

        def close(self):
            pass

    detector = TenVadStreamSpeechDetector(
        speech_recognizer=object(),
        ten_vad_class=FakeRemoteTenVad,
        hop_size=4,
        negative_speech_probability_threshold=0.14,
    )
    session = detector.get_session("session-remote")
    session.vad_buffer.extend(np.arange(10, dtype=np.int16).tobytes())

    speech_detected = detector._detect_speech_tenvad(session)

    assert speech_detected is True
    assert len(session.ten_vad.frames) == 1
    assert np.frombuffer(session.ten_vad.frames[0], dtype=np.int16).tolist() == list(range(8))
    assert np.frombuffer(session.vad_buffer, dtype=np.int16).tolist() == [8, 9]
