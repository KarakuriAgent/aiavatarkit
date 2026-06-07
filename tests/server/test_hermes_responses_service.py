import pytest

from server.config import load_settings
from server.providers import llm as llm_module
from server.providers.llm import HermesResponsesService


class FakeStreamResponse:
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def raise_for_status(self):
        return None

    async def aiter_lines(self):
        yield "event: response.output_text.delta"
        yield 'data: {"type": "response.output_text.delta", "delta": "ok"}'
        yield "data: [DONE]"


async def collect_request_body(
    monkeypatch,
    reasoning_effort=None,
    channel=None,
    session_id=None,
    user_id="user",
):
    captured = {}

    class FakeClient:
        def __init__(self, timeout=None):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def stream(self, method, url, headers, json):
            captured["method"] = method
            captured["url"] = url
            captured["headers"] = headers
            captured["json"] = json
            return FakeStreamResponse()

    monkeypatch.setattr(llm_module.httpx, "AsyncClient", FakeClient)

    service = HermesResponsesService(
        api_key="test-key",
        base_url="http://hermes:8642/v1",
        model="hermes-agent",
        reasoning_effort=reasoning_effort,
    )
    messages = await service.compose_messages("ctx", "user", "hello")
    responses = [
        response
        async for response in service.get_llm_stream_response(
            "ctx",
            user_id,
            messages,
            session_id=session_id,
            channel=channel,
        )
    ]

    assert "".join(response.text for response in responses) == "ok"
    return captured["json"]


def request_input_text(request_body):
    return request_body["input"][0]["content"][0]["text"]


def test_load_settings_reads_hermes_reasoning_effort(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_API_KEY", raising=False)
    monkeypatch.delenv("HERMES_REASONING_EFFORT", raising=False)

    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "HERMES_API_KEY=test-key",
                "HERMES_REASONING_EFFORT=none",
            ]
        )
    )

    settings = load_settings(env_path)

    assert settings.hermes_reasoning_effort == "none"


def test_load_settings_reads_discord_and_skip_tts_settings(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_API_KEY", raising=False)
    monkeypatch.delenv("DISCORD_USER_ID", raising=False)
    monkeypatch.delenv("DISCORD_BOT_ID", raising=False)
    monkeypatch.delenv("SKIP_TTS_CHANNELS", raising=False)

    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "HERMES_API_KEY=test-key",
                "DISCORD_SYNC_ENABLED=true",
                "DISCORD_USER_ID=111",
                "DISCORD_BOT_ID=222",
                "DISCORD_SYNC_USER_ID=robo-kanon-stack-chan",
                'DISCORD_VOICE_MESSAGE_PREFIX="🎙️ "',
                'DISCORD_API_MESSAGE_PREFIX="📢 "',
                "DISCORD_TYPING_INDICATOR_ENABLED=true",
                "DISCORD_TYPING_INDICATOR_INTERVAL=8",
                "SKIP_TTS_CHANNELS=discord,linebot",
            ]
        )
    )

    settings = load_settings(env_path)

    assert settings.discord_sync_enabled is True
    assert settings.discord_user_id == "111"
    assert settings.discord_bot_id == "222"
    assert settings.discord_sync_user_id == "robo-kanon-stack-chan"
    assert settings.discord_gateway_session_id == "discord:robo-kanon-stack-chan"
    assert settings.discord_voice_message_prefix == "🎙️ "
    assert settings.discord_api_message_prefix == "📢 "
    assert settings.discord_typing_indicator_enabled is True
    assert settings.discord_typing_indicator_interval == 8
    assert settings.skip_tts_channels == ["discord", "linebot"]


@pytest.mark.asyncio
async def test_hermes_responses_service_adds_voice_prefix_when_channel_unset(monkeypatch):
    request_body = await collect_request_body(monkeypatch)

    assert request_input_text(request_body) == "[channel:voice]hello"


@pytest.mark.asyncio
async def test_hermes_responses_service_adds_voice_prefix_for_regular_channels(monkeypatch):
    request_body = await collect_request_body(monkeypatch, channel="websocket")

    assert request_input_text(request_body) == "[channel:voice]hello"


@pytest.mark.asyncio
async def test_hermes_responses_service_adds_discord_prefix_for_discord_channel(monkeypatch):
    request_body = await collect_request_body(monkeypatch, channel="discord")

    assert request_input_text(request_body) == "[channel:discord]hello"


@pytest.mark.asyncio
async def test_hermes_responses_service_sends_callback_metadata(monkeypatch):
    request_body = await collect_request_body(
        monkeypatch,
        channel="discord",
        session_id="discord:robo-kanon-stack-chan",
        user_id="robo-kanon-stack-chan",
    )

    assert request_body["callback_metadata"] == {
        "channel": "discord",
        "session_id": "discord:robo-kanon-stack-chan",
        "user_id": "robo-kanon-stack-chan",
    }


@pytest.mark.asyncio
async def test_hermes_responses_service_omits_voice_prefix_for_hermes_channel(monkeypatch):
    request_body = await collect_request_body(monkeypatch, channel="hermes")

    assert request_input_text(request_body) == "hello"


@pytest.mark.asyncio
async def test_hermes_responses_service_omits_voice_prefix_for_cron_channel(monkeypatch):
    request_body = await collect_request_body(monkeypatch, channel="cron")

    assert request_input_text(request_body) == "hello"


@pytest.mark.asyncio
async def test_hermes_responses_service_sends_reasoning_effort(monkeypatch):
    request_body = await collect_request_body(monkeypatch, reasoning_effort="none")

    assert request_body["reasoning"] == {"effort": "none"}


@pytest.mark.asyncio
async def test_hermes_responses_service_omits_reasoning_when_unset(monkeypatch):
    request_body = await collect_request_body(monkeypatch)

    assert "reasoning" not in request_body
