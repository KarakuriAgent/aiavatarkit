import pytest

from aiavatar.sts import STSPipeline
from aiavatar.sts.llm import LLMServiceDummy
from aiavatar.sts.models import STSRequest
from aiavatar.sts.stt import SpeechRecognizerDummy
from aiavatar.sts.tts import SpeechSynthesizerDummy
from aiavatar.sts.vad import SpeechDetectorDummy
from aiavatar.sts.voice_auth import VoiceAuthResult, VoiceAuthenticator


class FakeVoiceAuthenticator(VoiceAuthenticator):
    def __init__(self, result: VoiceAuthResult):
        self.result = result
        self.calls = 0

    def verify_sync(
        self,
        *,
        user_id,
        audio_bytes,
        sample_rate,
        audio_duration=None,
        threshold=None,
        min_duration=None,
    ):
        self.calls += 1
        return self.result


class CountingSpeechRecognizer(SpeechRecognizerDummy):
    def __init__(self):
        super().__init__(recognized_text="hello")
        self.calls = 0
        self.sample_rate = 16000

    async def transcribe(self, data: bytes) -> str:
        self.calls += 1
        return await super().transcribe(data)


@pytest.mark.asyncio
async def test_voice_auth_rejection_cancels_before_stt(tmp_path):
    db_path = str(tmp_path / "voice_auth_reject.db")
    voice_auth = FakeVoiceAuthenticator(
        VoiceAuthResult(
            accepted=False,
            user_id="user01",
            matched_user_id="user01",
            similarity=0.42,
            threshold=0.70,
            reason="below_threshold",
        )
    )
    stt = CountingSpeechRecognizer()
    sts = STSPipeline(
        vad=SpeechDetectorDummy(),
        stt=stt,
        llm=LLMServiceDummy(response_text="ok", db_connection_str=db_path),
        tts=SpeechSynthesizerDummy(),
        voice_auth=voice_auth,
        voice_recorder_enabled=False,
        db_connection_str=db_path,
    )

    responses = [
        response
        async for response in sts.invoke(STSRequest(
            session_id="voice-auth-reject",
            user_id="user01",
            audio_data=b"\x00\x00" * 16000,
            audio_duration=1.5,
        ))
    ]

    assert voice_auth.calls == 1
    assert stt.calls == 0
    assert [response.type for response in responses] == ["canceled"]
    assert responses[0].metadata["reason"] == "voice_auth_rejected"
    assert responses[0].metadata["voice_auth"]["reason"] == "below_threshold"
    assert responses[0].metadata["voice_auth"]["request_user_id"] == "user01"
    assert responses[0].metadata["voice_auth"]["matched_voice_user_id"] == "user01"

    await sts.shutdown()


@pytest.mark.asyncio
async def test_voice_auth_acceptance_continues_to_stt(tmp_path):
    db_path = str(tmp_path / "voice_auth_accept.db")
    voice_auth = FakeVoiceAuthenticator(
        VoiceAuthResult(
            accepted=True,
            user_id="user01",
            matched_user_id="user01",
            similarity=0.82,
            threshold=0.70,
            reason="matched",
        )
    )
    stt = CountingSpeechRecognizer()
    sts = STSPipeline(
        vad=SpeechDetectorDummy(),
        stt=stt,
        llm=LLMServiceDummy(response_text="ok", db_connection_str=db_path),
        tts=SpeechSynthesizerDummy(),
        voice_auth=voice_auth,
        voice_recorder_enabled=False,
        db_connection_str=db_path,
    )

    responses = [
        response
        async for response in sts.invoke(STSRequest(
            session_id="voice-auth-accept",
            user_id="user01",
            audio_data=b"\x00\x00" * 16000,
            audio_duration=1.5,
        ))
    ]

    assert voice_auth.calls == 1
    assert stt.calls == 1
    assert any(response.type == "final" for response in responses)
    assert voice_auth.result.to_dict()["request_user_id"] == "user01"
    assert voice_auth.result.to_dict()["matched_voice_user_id"] == "user01"

    await sts.shutdown()
