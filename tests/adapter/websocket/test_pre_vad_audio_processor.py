import base64
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from aiavatar.adapter.models import AIAvatarRequest
from aiavatar.adapter.websocket.server import AIAvatarWebSocketServer, WebSocketSessionData


class _ReceiveOnlyWebSocket:
    def __init__(self, message: str):
        self.message = message

    async def receive_text(self):
        return self.message


class _RecordingVad:
    sample_rate = 16000

    def __init__(self):
        self.calls = []

    async def process_samples(self, audio_data: bytes, session_id: str):
        self.calls.append((audio_data, session_id))


class _ReplacingPreVadProcessor:
    def __init__(self, output: bytes):
        self.output = output
        self.calls = []

    def process(self, audio_bytes: bytes, *, sample_rate: int, session_id: str):
        self.calls.append((audio_bytes, sample_rate, session_id))
        return self.output

    def reset_session(self, _session_id: str):
        pass


def _create_minimal_server(vad, pre_vad_audio_processor):
    with patch.object(AIAvatarWebSocketServer, "__init__", lambda self, **_kw: None):
        server = AIAvatarWebSocketServer()
    server._on_request_handlers = []
    server.sts = SimpleNamespace(vad=vad)
    server.pre_vad_audio_processor = pre_vad_audio_processor
    return server


@pytest.mark.asyncio
async def test_data_request_runs_pre_vad_processor_before_vad():
    vad = _RecordingVad()
    processor = _ReplacingPreVadProcessor(b"clean-audio")
    server = _create_minimal_server(vad, processor)
    raw_audio = b"raw-audio"
    request = AIAvatarRequest(
        type="data",
        session_id="session-1",
        audio_data=base64.b64encode(raw_audio).decode("ascii"),
    )

    await server.process_websocket(
        _ReceiveOnlyWebSocket(request.model_dump_json()),
        WebSocketSessionData(),
    )

    assert processor.calls == [(raw_audio, 16000, "session-1")]
    assert vad.calls == [(b"clean-audio", "session-1")]


@pytest.mark.asyncio
async def test_data_request_skips_vad_when_pre_vad_processor_buffers_all_audio():
    vad = _RecordingVad()
    processor = _ReplacingPreVadProcessor(b"")
    server = _create_minimal_server(vad, processor)
    request = AIAvatarRequest(
        type="data",
        session_id="session-1",
        audio_data=base64.b64encode(b"short").decode("ascii"),
    )

    await server.process_websocket(
        _ReceiveOnlyWebSocket(request.model_dump_json()),
        WebSocketSessionData(),
    )

    assert vad.calls == []
