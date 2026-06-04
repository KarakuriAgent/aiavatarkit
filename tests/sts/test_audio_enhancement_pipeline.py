import pytest

from aiavatar.sts import STSPipeline
from aiavatar.sts.audio_enhancement import AudioEnhancer, AudioEnhancementResult
from aiavatar.sts.llm import LLMServiceDummy
from aiavatar.sts.models import STSRequest
from aiavatar.sts.stt import SpeechRecognizerDummy
from aiavatar.sts.tts import SpeechSynthesizerDummy
from aiavatar.sts.vad import SpeechDetectorDummy
from aiavatar.sts.voice_auth import VoiceAuthResult, VoiceAuthenticator


class FakeAudioEnhancer(AudioEnhancer):
    def __init__(self, audio_bytes: bytes = b"enhanced", error: Exception = None):
        self.audio_bytes = audio_bytes
        self.error = error
        self.calls = []

    async def enhance(self, *, audio_bytes, sample_rate, session_id=None):
        self.calls.append({
            "audio_bytes": audio_bytes,
            "sample_rate": sample_rate,
            "session_id": session_id,
        })
        if self.error:
            raise self.error
        return AudioEnhancementResult(
            audio_bytes=self.audio_bytes,
            provider="fake",
            metadata={"changed": True},
        )

    def enhance_sync(self, *, audio_bytes, sample_rate, session_id=None):
        raise NotImplementedError

    def get_config(self):
        return {"provider": "fake"}


class RecordingVoiceAuthenticator(VoiceAuthenticator):
    def __init__(self):
        self.calls = []

    def verify_sync(self, *, user_id, audio_bytes, sample_rate, audio_duration=None):
        self.calls.append({
            "user_id": user_id,
            "audio_bytes": audio_bytes,
            "sample_rate": sample_rate,
            "audio_duration": audio_duration,
        })
        return VoiceAuthResult(
            accepted=True,
            user_id=user_id,
            matched_user_id=user_id,
            similarity=0.99,
            threshold=0.5,
            reason="matched",
        )


class RecordingSpeechRecognizer(SpeechRecognizerDummy):
    def __init__(self):
        super().__init__(recognized_text="カノン、こんにちは")
        self.calls = []
        self.sample_rate = 16000

    async def transcribe(self, data: bytes) -> str:
        self.calls.append(data)
        return await super().transcribe(data)


@pytest.mark.asyncio
async def test_audio_enhancement_runs_before_voice_auth_and_stt(tmp_path):
    db_path = str(tmp_path / "enhancement_success.db")
    enhancer = FakeAudioEnhancer(audio_bytes=b"enhanced-audio")
    voice_auth = RecordingVoiceAuthenticator()
    stt = RecordingSpeechRecognizer()
    sts = STSPipeline(
        vad=SpeechDetectorDummy(),
        stt=stt,
        llm=LLMServiceDummy(response_text="ok", db_connection_str=db_path),
        tts=SpeechSynthesizerDummy(),
        audio_enhancer=enhancer,
        voice_auth=voice_auth,
        voice_recorder_enabled=False,
        db_connection_str=db_path,
    )

    responses = [
        response
        async for response in sts.invoke(STSRequest(
            session_id="enhancement-success",
            user_id="user01",
            audio_data=b"raw-audio",
            audio_duration=1.5,
        ))
    ]

    assert enhancer.calls[0]["audio_bytes"] == b"raw-audio"
    assert voice_auth.calls[0]["audio_bytes"] == b"enhanced-audio"
    assert stt.calls[0] == b"enhanced-audio"
    start = next(response for response in responses if response.type == "start")
    assert start.metadata["audio_enhancement"]["applied"] is True
    assert start.metadata["audio_enhancement"]["provider"] == "fake"
    assert any(response.type == "final" for response in responses)

    await sts.shutdown()


@pytest.mark.asyncio
async def test_audio_enhancement_fail_open_continues_with_raw_audio(tmp_path):
    db_path = str(tmp_path / "enhancement_fail_open.db")
    enhancer = FakeAudioEnhancer(error=RuntimeError("enhancer unavailable"))
    voice_auth = RecordingVoiceAuthenticator()
    stt = RecordingSpeechRecognizer()
    sts = STSPipeline(
        vad=SpeechDetectorDummy(),
        stt=stt,
        llm=LLMServiceDummy(response_text="ok", db_connection_str=db_path),
        tts=SpeechSynthesizerDummy(),
        audio_enhancer=enhancer,
        audio_enhancement_fail_open=True,
        voice_auth=voice_auth,
        voice_recorder_enabled=False,
        db_connection_str=db_path,
    )

    responses = [
        response
        async for response in sts.invoke(STSRequest(
            session_id="enhancement-fail-open",
            user_id="user01",
            audio_data=b"raw-audio",
            audio_duration=1.5,
        ))
    ]

    assert voice_auth.calls[0]["audio_bytes"] == b"raw-audio"
    assert stt.calls[0] == b"raw-audio"
    start = next(response for response in responses if response.type == "start")
    assert start.metadata["audio_enhancement"]["applied"] is False
    assert start.metadata["audio_enhancement"]["fail_open"] is True
    assert any(response.type == "final" for response in responses)

    await sts.shutdown()


@pytest.mark.asyncio
async def test_audio_enhancement_fail_closed_cancels_before_voice_auth_and_stt(tmp_path):
    db_path = str(tmp_path / "enhancement_fail_closed.db")
    enhancer = FakeAudioEnhancer(error=RuntimeError("enhancer unavailable"))
    voice_auth = RecordingVoiceAuthenticator()
    stt = RecordingSpeechRecognizer()
    sts = STSPipeline(
        vad=SpeechDetectorDummy(),
        stt=stt,
        llm=LLMServiceDummy(response_text="ok", db_connection_str=db_path),
        tts=SpeechSynthesizerDummy(),
        audio_enhancer=enhancer,
        audio_enhancement_fail_open=False,
        voice_auth=voice_auth,
        voice_recorder_enabled=False,
        db_connection_str=db_path,
    )

    responses = [
        response
        async for response in sts.invoke(STSRequest(
            session_id="enhancement-fail-closed",
            user_id="user01",
            audio_data=b"raw-audio",
            audio_duration=1.5,
        ))
    ]

    assert voice_auth.calls == []
    assert stt.calls == []
    assert [response.type for response in responses] == ["canceled"]
    assert responses[0].metadata["reason"] == "audio_enhancement_failed"
    assert responses[0].metadata["audio_enhancement"]["fail_open"] is False

    await sts.shutdown()
