import pytest

from aiavatar.sts import STSPipeline
from aiavatar.sts.addressing import AddressingDecision, AddressingDetector
from aiavatar.sts.llm import LLMServiceDummy
from aiavatar.sts.models import STSRequest
from aiavatar.sts.stt import SpeechRecognizerDummy
from aiavatar.sts.tts import SpeechSynthesizerDummy
from aiavatar.sts.vad import SpeechDetectorDummy


class FakeAddressingDetector(AddressingDetector):
    def __init__(self, decision: AddressingDecision):
        self.decision = decision
        self.calls = []

    async def detect(self, *, text, recent_history=None, seconds_since_last_assistant_turn=None):
        self.calls.append({
            "text": text,
            "recent_history": recent_history or [],
            "seconds_since_last_assistant_turn": seconds_since_last_assistant_turn,
        })
        return self.decision

    def detect_sync(self, *, text, recent_history=None, seconds_since_last_assistant_turn=None):
        return self.decision


class CountingSpeechRecognizer(SpeechRecognizerDummy):
    def __init__(self, recognized_text="hello"):
        super().__init__(recognized_text=recognized_text)
        self.calls = 0
        self.sample_rate = 16000

    async def transcribe(self, data: bytes) -> str:
        self.calls += 1
        return await super().transcribe(data)


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
