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


async def collect_request_body(monkeypatch, reasoning_effort=None):
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
            "user",
            messages,
        )
    ]

    assert "".join(response.text for response in responses) == "ok"
    return captured["json"]


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


@pytest.mark.asyncio
async def test_hermes_responses_service_sends_reasoning_effort(monkeypatch):
    request_body = await collect_request_body(monkeypatch, reasoning_effort="none")

    assert request_body["reasoning"] == {"effort": "none"}


@pytest.mark.asyncio
async def test_hermes_responses_service_omits_reasoning_when_unset(monkeypatch):
    request_body = await collect_request_body(monkeypatch)

    assert "reasoning" not in request_body
