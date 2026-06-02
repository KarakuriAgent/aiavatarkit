from unittest.mock import AsyncMock

import pytest

from aiavatar.adapter.websocket.server import AIAvatarWebSocketServer
from aiavatar.admin.control import ChatRequest, process_conversation_request
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


@pytest.mark.asyncio
async def test_text_delivery_returns_text_without_adapter_or_tts(tmp_path):
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
        llm=LLMServiceDummy(response_text="[face:joy]hello."),
        tts=tts,
        db_connection_str=str(tmp_path / "aiavatar.db"),
        voice_recorder_enabled=False,
        debug=False,
    )
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
    assert "discord" in adapter.sts.skip_tts_channels
