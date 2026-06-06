import pytest

from aiavatar.sts.audio_preprocessing import WebRtcNoiseGainAudioProcessor


class _Result:
    def __init__(self, audio: bytes):
        self.audio = audio


class _RecordingAudioProcessor:
    instances = []

    def __init__(self, auto_gain_dbfs: int, noise_suppression_level: int):
        self.auto_gain_dbfs = auto_gain_dbfs
        self.noise_suppression_level = noise_suppression_level
        self.calls = []
        _RecordingAudioProcessor.instances.append(self)

    def Process10ms(self, frame: bytes):
        self.calls.append(frame)
        return _Result(frame[::-1])


def test_webrtc_noise_gain_processes_10ms_frames_and_buffers_remainder():
    _RecordingAudioProcessor.instances.clear()
    processor = WebRtcNoiseGainAudioProcessor(
        noise_suppression_level=3,
        auto_gain_dbfs=0,
        audio_processor_class=_RecordingAudioProcessor,
    )
    frame_1 = bytes([1]) * 320
    frame_2 = bytes([2]) * 320

    first_output = processor.process(frame_1 + frame_2[:100], sample_rate=16000, session_id="s1")
    second_output = processor.process(frame_2[100:], sample_rate=16000, session_id="s1")

    assert first_output == frame_1[::-1]
    assert second_output == frame_2[::-1]
    assert len(_RecordingAudioProcessor.instances) == 1
    assert _RecordingAudioProcessor.instances[0].noise_suppression_level == 3
    assert _RecordingAudioProcessor.instances[0].calls == [frame_1, frame_2]


def test_webrtc_noise_gain_fail_open_passes_raw_audio():
    class FailingAudioProcessor:
        def __init__(self, *_args):
            pass

        def Process10ms(self, _frame: bytes):
            raise RuntimeError("processing failed")

    processor = WebRtcNoiseGainAudioProcessor(
        audio_processor_class=FailingAudioProcessor,
        fail_open=True,
    )
    raw_audio = bytes([3]) * 320

    assert processor.process(raw_audio, sample_rate=16000, session_id="s1") == raw_audio


def test_webrtc_noise_gain_fail_closed_raises():
    class FailingAudioProcessor:
        def __init__(self, *_args):
            pass

        def Process10ms(self, _frame: bytes):
            raise RuntimeError("processing failed")

    processor = WebRtcNoiseGainAudioProcessor(
        audio_processor_class=FailingAudioProcessor,
        fail_open=False,
    )

    with pytest.raises(RuntimeError, match="processing failed"):
        processor.process(bytes([4]) * 320, sample_rate=16000, session_id="s1")
