import pytest

from aiavatar.sts import STSPipeline
from aiavatar.sts.addressing import AddressingDecision, AddressingDetector
from aiavatar.sts.llm import LLMServiceDummy
from aiavatar.sts.models import STSRequest
from aiavatar.sts.stt import SpeechRecognizerDummy
from aiavatar.sts.tts import SpeechSynthesizerDummy
from aiavatar.sts.vad import SpeechDetectorDummy
from aiavatar.sts.voice_auth import VoiceAuthResult, VoiceAuthenticator
from aiavatar.sts.wakeword import WakewordDetection


class FakeAddressingDetector(AddressingDetector):
    def __init__(self, decision: AddressingDecision):
        self.decision = decision
        self.calls = []

    async def detect(self, *, text, recent_history=None, seconds_since_last_assistant_turn=None, recent_unaccepted_count=None):
        self.calls.append({
            "text": text,
            "recent_history": recent_history or [],
            "seconds_since_last_assistant_turn": seconds_since_last_assistant_turn,
            "recent_unaccepted_count": recent_unaccepted_count,
        })
        return self.decision

    def detect_sync(self, *, text, recent_history=None, seconds_since_last_assistant_turn=None, recent_unaccepted_count=None):
        return self.decision


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
    def __init__(self, recognized_text="hello"):
        super().__init__(recognized_text=recognized_text)
        self.calls = 0
        self.sample_rate = 16000

    async def transcribe(self, data: bytes) -> str:
        self.calls += 1
        return await super().transcribe(data)


class FakeAudioWakewordDetector:
    def __init__(self, detection=None):
        self.detection = detection
        self.calls = []

    def process(self, audio_bytes: bytes, *, sample_rate: int, session_id: str):
        self.calls.append((audio_bytes, sample_rate, session_id))
        return self.detection

    def initial_decision(self):
        return WakewordDetection(accepted=False, reason="not_detected")

    def get_config(self):
        return {"provider": "fake"}


@pytest.mark.asyncio
async def test_addressing_rejection_cancels_after_stt_before_llm(tmp_path):
    db_path = str(tmp_path / "addressing_reject.db")
    detector = FakeAddressingDetector(AddressingDecision(
        accepted=False,
        reason="monologue",
        confidence=0.91,
    ))
    stt = CountingSpeechRecognizer("ただいま")
    handled = []
    sts = STSPipeline(
        vad=SpeechDetectorDummy(),
        stt=stt,
        llm=LLMServiceDummy(response_text="ok", db_connection_str=db_path),
        tts=SpeechSynthesizerDummy(),
        addressing_detector=detector,
        voice_recorder_enabled=False,
        db_connection_str=db_path,
    )

    async def record_response(response):
        handled.append(response)

    sts.handle_response = record_response

    responses = [
        response
        async for response in sts.invoke(STSRequest(
            session_id="addressing-reject",
            user_id="user01",
            audio_data=b"\x00\x00" * 16000,
            audio_duration=1.5,
        ))
    ]

    assert stt.calls == 1
    assert len(detector.calls) == 1
    assert detector.calls[0]["text"] == "ただいま"
    assert [response.type for response in responses] == ["canceled"]
    assert responses[0].metadata["filter_reason"] == "addressing_rejected"
    assert responses[0].metadata["reason"] == "addressing_rejected"
    assert responses[0].metadata["addressing"]["reason"] == "monologue"
    assert handled == []

    histories = await sts.llm.context_manager.get_recent_histories(user_id="user01")
    assert histories == []

    await sts.shutdown()


@pytest.mark.asyncio
async def test_addressing_acceptance_continues_and_records_history_with_user_id(tmp_path):
    db_path = str(tmp_path / "addressing_accept.db")
    detector = FakeAddressingDetector(AddressingDecision(
        accepted=True,
        reason="called_by_name",
        confidence=0.99,
    ))
    stt = CountingSpeechRecognizer("カノン、天気を教えて")
    sts = STSPipeline(
        vad=SpeechDetectorDummy(),
        stt=stt,
        llm=LLMServiceDummy(response_text="晴れです。", db_connection_str=db_path),
        tts=SpeechSynthesizerDummy(),
        addressing_detector=detector,
        voice_recorder_enabled=False,
        db_connection_str=db_path,
    )

    responses = [
        response
        async for response in sts.invoke(STSRequest(
            session_id="addressing-accept",
            user_id="user01",
            audio_data=b"\x00\x00" * 16000,
            audio_duration=1.5,
        ))
    ]

    assert stt.calls == 1
    assert len(detector.calls) == 1
    assert any(response.type == "final" for response in responses)
    start = next(response for response in responses if response.type == "start")
    assert start.metadata["addressing"]["accepted"] is True

    histories = await sts.llm.context_manager.get_recent_histories(user_id="user01")
    assert [item["role"] for item in histories] == ["user", "assistant"]
    assert histories[0]["content"] == "カノン、天気を教えて"
    assert histories[1]["content"] == "晴れです。"

    await sts.shutdown()


@pytest.mark.asyncio
async def test_wakeword_acceptance_skips_addressing_detection(tmp_path):
    db_path = str(tmp_path / "wakeword_skips_addressing.db")
    detector = FakeAddressingDetector(AddressingDecision(
        accepted=False,
        reason="not_addressed",
        confidence=1.0,
    ))
    stt = CountingSpeechRecognizer("カノン、天気を教えて")
    sts = STSPipeline(
        vad=SpeechDetectorDummy(),
        stt=stt,
        llm=LLMServiceDummy(response_text="晴れです。", db_connection_str=db_path),
        tts=SpeechSynthesizerDummy(),
        addressing_detector=detector,
        wakewords=["カノン"],
        voice_recorder_enabled=False,
        db_connection_str=db_path,
    )

    responses = [
        response
        async for response in sts.invoke(STSRequest(
            session_id="wakeword-skips-addressing",
            user_id="user01",
            audio_data=b"\x00\x00" * 16000,
            audio_duration=1.5,
        ))
    ]

    assert stt.calls == 1
    assert detector.calls == []
    assert any(response.type == "final" for response in responses)
    start = next(response for response in responses if response.type == "start")
    assert start.metadata["wakeword"]["accepted"] is True
    assert start.metadata["wakeword"]["reason"] == "matched"
    assert start.metadata["addressing"] == {
        "skipped": True,
        "reason": "wakeword_accepted",
    }

    await sts.shutdown()


@pytest.mark.asyncio
async def test_audio_wakeword_acceptance_skips_addressing_detection(tmp_path):
    db_path = str(tmp_path / "audio_wakeword_skips_addressing.db")
    detector = FakeAddressingDetector(AddressingDecision(
        accepted=False,
        reason="not_addressed",
        confidence=1.0,
    ))
    audio_detector = FakeAudioWakewordDetector(WakewordDetection(
        accepted=True,
        reason="matched",
        source="audio",
        matched_wakeword="robo-kanon",
        score=0.82,
        threshold=0.5,
    ))
    stt = CountingSpeechRecognizer("天気を教えて")
    sts = STSPipeline(
        vad=SpeechDetectorDummy(),
        stt=stt,
        llm=LLMServiceDummy(response_text="晴れです。", db_connection_str=db_path),
        tts=SpeechSynthesizerDummy(),
        audio_wakeword_detector=audio_detector,
        addressing_detector=detector,
        voice_recorder_enabled=False,
        db_connection_str=db_path,
    )

    responses = [
        response
        async for response in sts.invoke(STSRequest(
            session_id="audio-wakeword-skips-addressing",
            user_id="user01",
            audio_data=b"\x00\x00" * 16000,
            audio_duration=1.5,
        ))
    ]

    assert stt.calls == 1
    assert audio_detector.calls == [(b"\x00\x00" * 16000, 16000, "audio-wakeword-skips-addressing")]
    assert detector.calls == []
    assert any(response.type == "final" for response in responses)
    start = next(response for response in responses if response.type == "start")
    assert start.metadata["wakeword"]["accepted"] is True
    assert start.metadata["wakeword"]["source"] == "audio"
    assert start.metadata["wakeword"]["matched_wakeword"] == "robo-kanon"
    assert start.metadata["addressing"] == {
        "skipped": True,
        "reason": "wakeword_accepted",
    }

    await sts.shutdown()


@pytest.mark.asyncio
async def test_wakeword_rejection_cancels_when_addressing_is_disabled(tmp_path):
    db_path = str(tmp_path / "wakeword_reject.db")
    sts = STSPipeline(
        vad=SpeechDetectorDummy(),
        stt=CountingSpeechRecognizer("ただいま"),
        llm=LLMServiceDummy(response_text="ok", db_connection_str=db_path),
        tts=SpeechSynthesizerDummy(),
        wakewords=["カノン"],
        voice_recorder_enabled=False,
        db_connection_str=db_path,
    )

    responses = [
        response
        async for response in sts.invoke(STSRequest(
            session_id="wakeword-reject",
            user_id="user01",
            audio_data=b"\x00\x00" * 16000,
            audio_duration=1.5,
        ))
    ]

    assert [response.type for response in responses] == ["canceled"]
    assert responses[0].metadata["filter_reason"] == "wakeword_rejected"
    assert responses[0].metadata["reason"] == "wakeword_rejected"
    assert responses[0].metadata["wakeword"]["reason"] == "not_detected"

    await sts.shutdown()


@pytest.mark.asyncio
async def test_audio_wakeword_rejection_cancels_when_addressing_is_disabled(tmp_path):
    db_path = str(tmp_path / "audio_wakeword_reject.db")
    audio_detector = FakeAudioWakewordDetector()
    sts = STSPipeline(
        vad=SpeechDetectorDummy(),
        stt=CountingSpeechRecognizer("ただいま"),
        llm=LLMServiceDummy(response_text="ok", db_connection_str=db_path),
        tts=SpeechSynthesizerDummy(),
        audio_wakeword_detector=audio_detector,
        voice_recorder_enabled=False,
        db_connection_str=db_path,
    )

    responses = [
        response
        async for response in sts.invoke(STSRequest(
            session_id="audio-wakeword-reject",
            user_id="user01",
            audio_data=b"\x00\x00" * 16000,
            audio_duration=1.5,
        ))
    ]

    assert audio_detector.calls == [(b"\x00\x00" * 16000, 16000, "audio-wakeword-reject")]
    assert [response.type for response in responses] == ["canceled"]
    assert responses[0].metadata["filter_reason"] == "wakeword_rejected"
    assert responses[0].metadata["wakeword"]["source"] == "audio"
    assert responses[0].metadata["wakeword"]["reason"] == "not_detected"

    await sts.shutdown()


@pytest.mark.asyncio
async def test_ptt_default_policy_runs_voice_auth_and_skips_wakeword_and_addressing(tmp_path):
    db_path = str(tmp_path / "ptt_default_policy.db")
    voice_auth = FakeVoiceAuthenticator(VoiceAuthResult(
        accepted=True,
        user_id="user01",
        matched_user_id="user01",
        similarity=0.91,
        threshold=0.70,
        reason="matched",
    ))
    audio_detector = FakeAudioWakewordDetector()
    addressing_detector = FakeAddressingDetector(AddressingDecision(
        accepted=False,
        reason="not_addressed",
        confidence=1.0,
    ))
    stt = CountingSpeechRecognizer("ただいま")
    sts = STSPipeline(
        vad=SpeechDetectorDummy(),
        stt=stt,
        llm=LLMServiceDummy(response_text="ok", db_connection_str=db_path),
        tts=SpeechSynthesizerDummy(),
        voice_auth=voice_auth,
        audio_wakeword_detector=audio_detector,
        addressing_detector=addressing_detector,
        wakewords=["カノン"],
        voice_recorder_enabled=False,
        db_connection_str=db_path,
    )

    responses = [
        response
        async for response in sts.invoke(STSRequest(
            session_id="ptt-default-policy",
            user_id="user01",
            audio_data=b"\x00\x00" * 16000,
            audio_duration=1.5,
            metadata={"input_mode": "ptt"},
        ))
    ]

    assert voice_auth.calls == 1
    assert stt.calls == 1
    assert audio_detector.calls == []
    assert addressing_detector.calls == []
    assert any(response.type == "final" for response in responses)
    start = next(response for response in responses if response.type == "start")
    assert start.metadata["input_mode"] == "ptt"
    assert start.metadata["ptt_gate_policy"] == {
        "voice_auth": True,
        "wakeword": False,
        "addressing": False,
    }
    assert "wakeword" not in start.metadata
    assert "addressing" not in start.metadata

    await sts.shutdown()


@pytest.mark.asyncio
async def test_ptt_policy_can_disable_voice_auth(tmp_path):
    db_path = str(tmp_path / "ptt_voice_auth_disabled.db")
    voice_auth = FakeVoiceAuthenticator(VoiceAuthResult(
        accepted=False,
        user_id="user01",
        matched_user_id=None,
        similarity=0.10,
        threshold=0.70,
        reason="below_threshold",
    ))
    stt = CountingSpeechRecognizer("用件です")
    sts = STSPipeline(
        vad=SpeechDetectorDummy(),
        stt=stt,
        llm=LLMServiceDummy(response_text="ok", db_connection_str=db_path),
        tts=SpeechSynthesizerDummy(),
        voice_auth=voice_auth,
        ptt_voice_auth_enabled=False,
        voice_recorder_enabled=False,
        db_connection_str=db_path,
    )

    responses = [
        response
        async for response in sts.invoke(STSRequest(
            session_id="ptt-voice-auth-disabled",
            user_id="user01",
            audio_data=b"\x00\x00" * 16000,
            audio_duration=1.5,
            metadata={"input_mode": "ptt"},
        ))
    ]

    assert voice_auth.calls == 0
    assert stt.calls == 1
    assert any(response.type == "final" for response in responses)
    start = next(response for response in responses if response.type == "start")
    assert start.metadata["ptt_gate_policy"]["voice_auth"] is False

    await sts.shutdown()


@pytest.mark.asyncio
async def test_ptt_policy_can_enable_addressing_detection(tmp_path):
    db_path = str(tmp_path / "ptt_addressing_enabled.db")
    detector = FakeAddressingDetector(AddressingDecision(
        accepted=False,
        reason="not_addressed",
        confidence=0.95,
    ))
    stt = CountingSpeechRecognizer("ただいま")
    sts = STSPipeline(
        vad=SpeechDetectorDummy(),
        stt=stt,
        llm=LLMServiceDummy(response_text="ok", db_connection_str=db_path),
        tts=SpeechSynthesizerDummy(),
        addressing_detector=detector,
        ptt_addressing_enabled=True,
        voice_recorder_enabled=False,
        db_connection_str=db_path,
    )

    responses = [
        response
        async for response in sts.invoke(STSRequest(
            session_id="ptt-addressing-enabled",
            user_id="user01",
            audio_data=b"\x00\x00" * 16000,
            audio_duration=1.5,
            metadata={"input_mode": "ptt"},
        ))
    ]

    assert stt.calls == 1
    assert len(detector.calls) == 1
    assert detector.calls[0]["text"] == "ただいま"
    assert [response.type for response in responses] == ["canceled"]
    assert responses[0].metadata["reason"] == "addressing_rejected"
    assert responses[0].metadata["ptt_gate_policy"]["addressing"] is True

    await sts.shutdown()


@pytest.mark.asyncio
async def test_addressing_detector_is_not_called_for_text_only_request(tmp_path):
    db_path = str(tmp_path / "addressing_text_bypass.db")
    detector = FakeAddressingDetector(AddressingDecision(
        accepted=False,
        reason="not_addressed",
        confidence=1.0,
    ))
    sts = STSPipeline(
        vad=SpeechDetectorDummy(),
        stt=CountingSpeechRecognizer("unused"),
        llm=LLMServiceDummy(response_text="ok", db_connection_str=db_path),
        tts=SpeechSynthesizerDummy(),
        addressing_detector=detector,
        voice_recorder_enabled=False,
        db_connection_str=db_path,
    )

    responses = [
        response
        async for response in sts.invoke(STSRequest(
            session_id="addressing-text",
            user_id="user01",
            text="APIからの明示入力",
        ))
    ]

    assert detector.calls == []
    assert any(response.type == "final" for response in responses)

    await sts.shutdown()


@pytest.mark.asyncio
async def test_addressing_rejection_count_is_passed_and_cleared_on_accept(tmp_path):
    db_path = str(tmp_path / "addressing_reject_count.db")
    detector = FakeAddressingDetector(AddressingDecision(
        accepted=False,
        reason="not_addressed",
        confidence=0.9,
    ))
    sts = STSPipeline(
        vad=SpeechDetectorDummy(),
        stt=CountingSpeechRecognizer("ただいま"),
        llm=LLMServiceDummy(response_text="ok", db_connection_str=db_path),
        tts=SpeechSynthesizerDummy(),
        addressing_detector=detector,
        voice_recorder_enabled=False,
        db_connection_str=db_path,
    )

    async def invoke_once():
        return [
            response
            async for response in sts.invoke(STSRequest(
                session_id="addressing-reject-count",
                user_id="user01",
                audio_data=b"\x00\x00" * 16000,
                audio_duration=1.5,
            ))
        ]

    await invoke_once()
    await invoke_once()
    assert detector.calls[0]["recent_unaccepted_count"] == 0
    assert detector.calls[1]["recent_unaccepted_count"] == 1

    detector.decision = AddressingDecision(
        accepted=True,
        reason="contextual_reply",
        confidence=0.95,
    )
    await invoke_once()
    assert detector.calls[2]["recent_unaccepted_count"] == 2

    detector.decision = AddressingDecision(
        accepted=False,
        reason="not_addressed",
        confidence=0.9,
    )
    await invoke_once()
    assert detector.calls[3]["recent_unaccepted_count"] == 0

    await sts.shutdown()
