import asyncio

import pytest

from aiavatar.sts.stt.base import SpeechRecognitionResult
from aiavatar.sts.vad.stream import RecordingSession, SileroStreamSpeechDetector


class FakeSpeechRecognizer:
    def __init__(self, text=None, error=None):
        self.text = text
        self.error = error
        self.calls = []

    async def recognize(self, session_id: str, data: bytes):
        self.calls.append((session_id, data))
        if self.error:
            raise self.error
        return SpeechRecognitionResult(text=self.text)


def make_detector(recognizer: FakeSpeechRecognizer):
    detector = SileroStreamSpeechDetector.__new__(SileroStreamSpeechDetector)
    detector.speech_recognizer = recognizer
    detector.debug = True
    detector._on_speech_detected = []
    detector._on_speech_recognition_error = []
    detector._validate_recognized_text = None
    return detector


@pytest.mark.asyncio
async def test_final_recognition_uses_full_audio_even_when_partial_exists():
    recognizer = FakeSpeechRecognizer(text="どんな服装がいいかな。")
    detector = make_detector(recognizer)
    detected = []

    @detector.on_speech_detected
    async def on_speech_detected(recorded_data, text, metadata, recorded_duration, session_id):
        detected.append({
            "recorded_data": recorded_data,
            "text": text,
            "metadata": metadata,
            "recorded_duration": recorded_duration,
            "session_id": session_id,
        })

    session = RecordingSession("session-1")
    session.buffer.extend(b"full-audio")
    session.last_recognized_text = "の服装がいいかな。"

    emitted = await detector._emit_final_speech_detected(session, 2.5)
    await asyncio.sleep(0)

    assert emitted is True
    assert recognizer.calls == [("session-1", b"full-audio")]
    assert len(detected) == 1
    assert detected[0]["text"] == "どんな服装がいいかな。"
    assert detected[0]["recorded_data"] == b"full-audio"
    assert detected[0]["recorded_duration"] == 2.5


@pytest.mark.asyncio
async def test_final_recognition_failure_does_not_fallback_to_partial():
    recognizer = FakeSpeechRecognizer(text="")
    detector = make_detector(recognizer)
    detected = []

    @detector.on_speech_detected
    async def on_speech_detected(recorded_data, text, metadata, recorded_duration, session_id):
        detected.append(text)

    session = RecordingSession("session-1")
    session.buffer.extend(b"full-audio")
    session.last_recognized_text = "partial text"

    emitted = await detector._emit_final_speech_detected(session, 2.5)
    await asyncio.sleep(0)

    assert emitted is False
    assert recognizer.calls == [("session-1", b"full-audio")]
    assert detected == []


@pytest.mark.asyncio
async def test_final_recognition_error_does_not_fallback_to_partial():
    recognizer = FakeSpeechRecognizer(error=RuntimeError("stt failed"))
    detector = make_detector(recognizer)
    errors = []
    detected = []

    @detector.on_speech_recognition_error
    async def on_speech_recognition_error(error, session_id):
        errors.append((str(error), session_id))

    @detector.on_speech_detected
    async def on_speech_detected(recorded_data, text, metadata, recorded_duration, session_id):
        detected.append(text)

    session = RecordingSession("session-1")
    session.buffer.extend(b"full-audio")
    session.last_recognized_text = "partial text"

    emitted = await detector._emit_final_speech_detected(session, 2.5)
    await asyncio.sleep(0)

    assert emitted is False
    assert recognizer.calls == [("session-1", b"full-audio")]
    assert errors == [("stt failed", "session-1")]
    assert detected == []
