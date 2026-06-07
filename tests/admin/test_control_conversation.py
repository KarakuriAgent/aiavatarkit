from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from aiavatar.adapter.websocket.server import AIAvatarWebSocketServer
from aiavatar.admin.control import ChatRequest, SpeakRequest, process_conversation_request, process_speak_request
from aiavatar.sts.llm import LLMServiceDummy
from aiavatar.sts.tts import SpeechSynthesizerDummy
from aiavatar.sts.vad import SpeechDetectorDummy


class RecordingSpeechSynthesizer(SpeechSynthesizerDummy):
    def __init__(self):
        super().__init__(synthesized_bytes=b"voice")
        self.calls = []

    async def synthesize(self, text: str, style_info: dict = None, language: str = None) -> bytes:
        self.calls.append((text, style_info, language))
        return await super().synthesize(text, style_info, language)


def create_recording_adapter(tmp_path, *, response_text="[face:joy]hello."):
    tts = RecordingSpeechSynthesizer()
    vad = SpeechDetectorDummy()
    vad_data = {}

    def get_session_data(session_id: str, key: str):
        return vad_data.get(session_id, {}).get(key)

    def set_session_data(session_id: str, key: str, value, create_session: bool = False):
        vad_data.setdefault(session_id, {})[key] = value

    vad.get_session_data = get_session_data
    vad.set_session_data = set_session_data

    adapter = AIAvatarWebSocketServer(
        vad=vad,
        llm=LLMServiceDummy(response_text=response_text),
        tts=tts,
        db_connection_str=str(tmp_path / "aiavatar.db"),
        voice_recorder_enabled=False,
        debug=False,
    )
    return adapter, tts, vad_data


@pytest.mark.asyncio
async def test_text_delivery_returns_text_without_adapter_or_tts(tmp_path):
    adapter, tts, _vad_data = create_recording_adapter(tmp_path)
    adapter.sts.handle_response = AsyncMock()
    adapter.sts.stop_response = AsyncMock()

    response = await process_conversation_request(
        adapter,
        ChatRequest(
            text="hi",
            session_id="discord:robo-kanon-stack-chan",
            user_id="robo-kanon-stack-chan",
            channel="discord",
            delivery="text",
        ),
    )

    assert response.message == "Message processed successfully"
    assert response.text == "[face:joy]hello."
    assert response.voice_text == "hello."
    assert response.context_id
    assert tts.calls == []
    adapter.sts.handle_response.assert_not_called()
    adapter.sts.stop_response.assert_not_called()
    assert "discord" not in adapter.sts.skip_tts_channels


@pytest.mark.asyncio
async def test_speak_request_uses_active_session_and_suppresses_discord_user_log(tmp_path):
    adapter, tts, vad_data = create_recording_adapter(tmp_path, response_text="[face:joy]休憩してね。")
    adapter.sessions["stack-session"] = SimpleNamespace(id="stack-session")
    vad_data["stack-session"] = {"user_id": "robo-kanon-stack-chan"}
    adapter.handle_response = AsyncMock()
    adapter.sts.handle_response = AsyncMock()
    adapter.sts.stop_response = AsyncMock()

    response = await process_speak_request(
        adapter,
        SpeakRequest(
            text="そろそろ休憩の時間です。",
            user_id="robo-kanon-stack-chan",
            channel="cron",
        ),
    )

    assert response.message == "Message processed successfully"
    assert response.text == "[face:joy]休憩してね。"
    assert response.voice_text == "休憩してね。"
    assert response.context_id
    assert ("休憩してね。", {"styled_text": "[face:joy]休憩してね。", "info": {}}, None) in tts.calls

    start_response = next(call.args[0] for call in adapter.handle_response.call_args_list if call.args[0].type == "start")
    assert start_response.session_id == "stack-session"
    assert start_response.metadata["source"] == "avatar_speak"
    assert start_response.metadata["suppress_discord_user_log"] is True
    assert "通知内容:" in start_response.metadata["recognized_text"]
    assert "そろそろ休憩の時間です。" in start_response.metadata["recognized_text"]

    final_response = next(call.args[0] for call in adapter.handle_response.call_args_list if call.args[0].type == "final")
    assert final_response.metadata["source"] == "avatar_speak"


@pytest.mark.asyncio
async def test_speak_request_voice_false_returns_text_without_adapter_or_tts(tmp_path):
    adapter, tts, _vad_data = create_recording_adapter(tmp_path, response_text="[face:joy]完了しました。")
    adapter.handle_response = AsyncMock()
    adapter.sts.handle_response = AsyncMock()
    adapter.sts.stop_response = AsyncMock()

    response = await process_speak_request(
        adapter,
        SpeakRequest(
            text="Hermesの作業が完了しました。",
            user_id="robo-kanon-stack-chan",
            channel="hermes",
            voice=False,
        ),
    )

    assert response.message == "Message processed successfully"
    assert response.text == "[face:joy]完了しました。"
    assert response.voice_text == "完了しました。"
    assert response.context_id
    assert tts.calls == []
    adapter.handle_response.assert_not_called()
    adapter.sts.handle_response.assert_not_called()
    adapter.sts.stop_response.assert_not_called()
    assert "hermes" not in adapter.sts.skip_tts_channels


@pytest.mark.asyncio
async def test_speak_request_voice_false_can_notify_adapter_response_hooks(tmp_path):
    adapter, tts, vad_data = create_recording_adapter(tmp_path, response_text="[face:joy]完了しました。")
    vad_data["discord:robo-kanon-stack-chan"] = {"context_id": "discord-context"}
    adapter.handle_response = AsyncMock()
    adapter.sts.handle_response = AsyncMock()
    adapter.sts.stop_response = AsyncMock()

    response = await process_speak_request(
        adapter,
        SpeakRequest(
            text="Hermesの作業が完了しました。",
            session_id="discord:robo-kanon-stack-chan",
            user_id="robo-kanon-stack-chan",
            channel="hermes",
            voice=False,
            metadata={"notify_adapter_response": True},
        ),
    )

    assert response.message == "Message processed successfully"
    assert response.text == "[face:joy]完了しました。"
    assert response.voice_text == "完了しました。"
    assert response.context_id
    assert tts.calls == []
    adapter.handle_response.assert_called_once()
    final_response = adapter.handle_response.call_args.args[0]
    assert final_response.type == "final"
    assert final_response.session_id == "discord:robo-kanon-stack-chan"
    assert final_response.context_id == response.context_id
    assert final_response.metadata["notify_adapter_response"] is True
    assert final_response.metadata["speak_text"] == "Hermesの作業が完了しました。"
    assert final_response.metadata["source"] == "hermes"
    adapter.sts.handle_response.assert_not_called()
    adapter.sts.stop_response.assert_not_called()


@pytest.mark.asyncio
async def test_speak_request_requires_active_session(tmp_path):
    adapter, _tts, _vad_data = create_recording_adapter(tmp_path)

    with pytest.raises(HTTPException) as exc:
        await process_speak_request(
            adapter,
            SpeakRequest(text="そろそろ休憩の時間です。", user_id="robo-kanon-stack-chan"),
        )

    assert exc.value.status_code == 400
