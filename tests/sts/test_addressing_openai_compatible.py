import json

import pytest

from aiavatar.sts.addressing import OpenAICompatibleChatAddressingDetector


class FakeAddressingResponse:
    headers = {}

    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class FakeAddressingHttpClient:
    def __init__(self, payload):
        self.payload = payload
        self.posts = []

    async def post(self, url, *, headers, json):
        self.posts.append({
            "url": url,
            "headers": headers,
            "json": json,
        })
        return FakeAddressingResponse(self.payload)

    async def aclose(self):
        pass


@pytest.mark.asyncio
async def test_system_prompt_formats_assistant_history_as_primary_name():
    detector = OpenAICompatibleChatAddressingDetector(
        base_url="https://example.test/v1",
        api_key="test-key",
        model="test-model",
        target_names=["カノン", "スタックチャン"],
    )

    try:
        body = detector._request_body(
            text="じゃあ午後は空いてないんだね",
            recent_history=[
                {
                    "role": "user",
                    "content": "カノン、明日の予定を確認して",
                    "created_at": "2026-06-04 10:14:02.123456+00:00",
                },
                {
                    "role": "assistant",
                    "content": "明日は10時に定例、14時に歯医者があります。",
                    "created_at": "2026-06-04 10:14:08.456789+00:00",
                },
                {
                    "role": "model",
                    "content": "午後は移動時間も必要です。",
                    "created_at": "2026-06-04 10:14:12.000000+00:00",
                },
            ],
            seconds_since_last_assistant_turn=12.3,
        )
        prompt = body["messages"][0]["content"]

        assert "- 2026-06-04 10:14:02.123456+00:00 user: カノン、明日の予定を確認して" in prompt
        assert "- 2026-06-04 10:14:08.456789+00:00 カノン: 明日は10時に定例、14時に歯医者があります。" in prompt
        assert "- 2026-06-04 10:14:12.000000+00:00 カノン: 午後は移動時間も必要です。" in prompt
        assert "Current time (UTC):" in prompt
        assert prompt.startswith("Reasoning: low")
        assert "Decision procedure" in prompt
        assert "When in doubt, reject" in prompt
        assert "awaits an answer" in prompt
        assert "from about 15 to 120 seconds" in prompt
        assert "Not-addressed utterances in the last 60 seconds:" in prompt
        assert "explanation: one short sentence citing the concrete evidence" in prompt
        assert "20.0 seconds" not in prompt
        assert "assistant:" not in prompt
        assert "model:" not in prompt

        schema = body["response_format"]["json_schema"]["schema"]
        assert schema["properties"]["explanation"] == {"type": "string"}
        assert schema["properties"]["elapsed_seconds"] == {"type": ["number", "null"]}
        assert list(schema["properties"]) == ["explanation", "elapsed_seconds", "reason", "confidence", "accepted"]
        assert "explanation" in schema["required"]
        assert "elapsed_seconds" in schema["required"]
    finally:
        await detector.close()


@pytest.mark.asyncio
async def test_parse_response_includes_explanation():
    detector = OpenAICompatibleChatAddressingDetector(
        base_url="https://example.test/v1",
        api_key="test-key",
        model="test-model",
        target_names=["カノン"],
    )

    try:
        decision = detector._parse_response({
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "accepted": True,
                            "reason": "called_by_name",
                            "confidence": 0.98,
                            "explanation": "The utterance explicitly calls カノン by name.",
                        })
                    }
                }
            ]
        })

        assert decision.accepted is True
        assert decision.reason == "called_by_name"
        assert decision.confidence == 0.98
        assert decision.explanation == "The utterance explicitly calls カノン by name."
    finally:
        await detector.close()


@pytest.mark.asyncio
async def test_responses_request_body_uses_text_format_schema():
    detector = OpenAICompatibleChatAddressingDetector(
        base_url="https://example.test/v1",
        api_key="test-key",
        model="test-model",
        api_format="responses",
        target_names=["カノン"],
    )

    try:
        body = detector._request_body(
            text="カノン、聞こえる?",
            recent_history=[],
            seconds_since_last_assistant_turn=None,
        )

        assert body["model"] == "test-model"
        assert body["store"] is False
        assert body["stream"] is True
        assert "the assistant named カノン" in body["instructions"]
        assert body["input"] == [{
            "role": "user",
            "content": [{"type": "input_text", "text": "カノン、聞こえる?"}],
        }]
        assert body["text"]["format"]["type"] == "json_schema"
        assert body["text"]["format"]["name"] == "addressing_decision"
        assert body["text"]["format"]["schema"]["properties"]["reason"]["enum"]
    finally:
        await detector.close()


@pytest.mark.asyncio
async def test_chat_completions_json_object_request_body():
    detector = OpenAICompatibleChatAddressingDetector(
        base_url="https://example.test/v1",
        api_key="test-key",
        model="test-model",
        api_format="chat_completions_json_object",
        target_names=["ロボ花音", "カノン"],
        primary_name="ロボ花音",
    )

    try:
        body = detector._request_body(
            text="ロボ花音、聞こえる?",
            recent_history=[],
            seconds_since_last_assistant_turn=None,
        )
        prompt = body["messages"][0]["content"]

        assert body["model"] == "test-model"
        assert body["temperature"] == 0
        assert body["response_format"] == {"type": "json_object"}
        assert body["max_tokens"] == 1024
        assert prompt.startswith("Reasoning: low")
        assert "Return only one JSON object" in prompt
        assert "keys in this exact order: explanation, elapsed_seconds, reason, confidence, accepted" in prompt
        assert "20.0 seconds" not in prompt
        assert body["messages"][-1]["role"] == "user"
        user_content = body["messages"][-1]["content"]
        assert user_content.startswith("Actual history: none\nCurrent time (UTC): ")
        assert "\nSeconds since last assistant turn: unknown\n" in user_content
        assert "\nNot-addressed utterances in the last 60 seconds: unknown\n" in user_content
        assert user_content.endswith("Current utterance: ロボ花音、聞こえる?")
    finally:
        await detector.close()


@pytest.mark.asyncio
async def test_chat_completions_json_object_adds_history_messages():
    detector = OpenAICompatibleChatAddressingDetector(
        base_url="https://example.test/v1",
        api_key="test-key",
        model="test-model",
        api_format="chat_completions_json_object",
        target_names=["ロボ花音"],
    )

    try:
        body = detector._request_body(
            text="助かりました",
            recent_history=[
                {"role": "user", "content": "ロボ花音、確認して"},
                {"role": "assistant", "content": "確認できています。"},
            ],
            seconds_since_last_assistant_turn=3,
            recent_unaccepted_count=2,
        )

        assert body["messages"][1] == {
            "role": "user",
            "content": "ロボ花音、確認して",
        }
        assert body["messages"][2] == {
            "role": "assistant",
            "content": "確認できています。",
        }
        assert body["messages"][3]["role"] == "user"
        user_content = body["messages"][3]["content"]
        assert user_content.startswith("Actual history: present\nCurrent time (UTC): ")
        assert "\nSeconds since last assistant turn: 3.0\n" in user_content
        assert "\nNot-addressed utterances in the last 60 seconds: 2\n" in user_content
        assert user_content.endswith("Current utterance: 助かりました")
    finally:
        await detector.close()


@pytest.mark.asyncio
async def test_chat_completions_json_object_prompt_covers_context_patterns():
    detector = OpenAICompatibleChatAddressingDetector(
        base_url="https://example.test/v1",
        api_key="test-key",
        model="test-model",
        api_format="chat_completions_json_object",
        target_names=["ロボ花音", "カノン"],
        primary_name="ロボ花音",
    )

    try:
        prompt = detector._json_object_system_prompt()

        assert prompt.startswith("Reasoning: low")
        assert "When in doubt, reject" in prompt
        assert "awaits an answer" in prompt
        assert "an added instruction, a cancellation, or a request to continue or retry" in prompt
        assert "talked about to someone else" in prompt
        assert "reason=monologue" in prompt
        assert "Not-addressed utterances in the last 60 seconds" in prompt
        assert "A reply-shaped utterance alone is NOT enough" in prompt
        assert "Current time (UTC)" in prompt
        assert "20.0 seconds" not in prompt
    finally:
        await detector.close()


@pytest.mark.asyncio
async def test_local_decision_accepts_called_name_without_http():
    detector = OpenAICompatibleChatAddressingDetector(
        base_url="https://example.test/v1",
        api_key="test-key",
        model="test-model",
        target_names=["ロボ花音", "カノン"],
    )

    try:
        decision = await detector.detect(
            text="カノン、聞こえる?",
            recent_history=[],
            seconds_since_last_assistant_turn=None,
        )

        assert decision.accepted is True
        assert decision.reason == "called_by_name"
        assert decision.metadata["local_short_circuit"] is True
    finally:
        await detector.close()


@pytest.mark.asyncio
async def test_stale_unnamed_utterance_uses_llm_instead_of_local_reject():
    detector = OpenAICompatibleChatAddressingDetector(
        base_url="https://example.test/v1",
        api_key="test-key",
        model="test-model",
        api_format="chat_completions_json_object",
        target_names=["ロボ花音", "カノン"],
    )
    fake_http = FakeAddressingHttpClient({
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "accepted": False,
                        "reason": "not_addressed",
                        "confidence": 0.9,
                        "explanation": "The elapsed time makes the addressee unclear.",
                    })
                }
            }
        ]
    })
    await detector.http_client.aclose()
    detector.http_client = fake_http

    try:
        decision = await detector.detect(
            text="うん、お願い",
            recent_history=[
                {"role": "assistant", "content": "この内容で進めますか?"},
            ],
            seconds_since_last_assistant_turn=25,
        )

        assert decision.accepted is False
        assert decision.reason == "not_addressed"
        assert "local_short_circuit" not in decision.metadata
        assert len(fake_http.posts) == 1
        body = fake_http.posts[0]["json"]
        user_content = body["messages"][-1]["content"]
        assert "Current time (UTC): " in user_content
        assert "Seconds since last assistant turn: 25.0" in user_content
        assert user_content.endswith("Current utterance: うん、お願い")
    finally:
        await detector.close()


@pytest.mark.asyncio
async def test_loads_json_object_accepts_markdown_fence():
    detector = OpenAICompatibleChatAddressingDetector(
        base_url="https://example.test/v1",
        api_key="test-key",
        model="test-model",
        target_names=["ロボ花音"],
    )

    try:
        data = detector._loads_json_object(
            '```json\n{"accepted": true, "reason": "contextual_reply"}\n```'
        )

        assert data == {"accepted": True, "reason": "contextual_reply"}
    finally:
        await detector.close()


@pytest.mark.asyncio
async def test_loads_json_object_accepts_leading_text():
    detector = OpenAICompatibleChatAddressingDetector(
        base_url="https://example.test/v1",
        api_key="test-key",
        model="test-model",
        target_names=["ロボ花音"],
    )

    try:
        data = detector._loads_json_object(
            'JSON:\n{"accepted": false, "reason": "not_addressed"}'
        )

        assert data == {"accepted": False, "reason": "not_addressed"}
    finally:
        await detector.close()


@pytest.mark.asyncio
async def test_parse_responses_sse_output_text_deltas():
    detector = OpenAICompatibleChatAddressingDetector(
        base_url="https://example.test/v1",
        api_key="test-key",
        model="test-model",
        api_format="responses",
        target_names=["カノン"],
    )

    try:
        content = "\n\n".join([
            'data: {"type":"response.output_text.delta","delta":"{\\"accepted\\":true,"}',
            'data: {"type":"response.output_text.delta","delta":"\\"reason\\":\\"called_by_name\\",\\"confidence\\":1,\\"explanation\\":\\"Called by name.\\"}"}',
            "data: [DONE]",
        ])
        text = detector._extract_sse_text(content)
        decision = detector._parse_response({"output_text": text})

        assert decision.accepted is True
        assert decision.reason == "called_by_name"
        assert decision.confidence == 1
    finally:
        await detector.close()


@pytest.mark.asyncio
async def test_parse_responses_sse_does_not_duplicate_done_text():
    detector = OpenAICompatibleChatAddressingDetector(
        base_url="https://example.test/v1",
        api_key="test-key",
        model="test-model",
        api_format="responses",
        target_names=["カノン"],
    )

    try:
        expected = json.dumps({
            "accepted": True,
            "reason": "called_by_name",
            "confidence": 1,
            "explanation": "Called by name.",
        })
        content = "\n\n".join([
            f"data: {json.dumps({'type': 'response.output_text.delta', 'delta': expected})}",
            f"data: {json.dumps({'type': 'response.output_text.done', 'text': expected})}",
            "data: [DONE]",
        ])

        assert detector._extract_sse_text(content) == expected
    finally:
        await detector.close()


@pytest.mark.asyncio
async def test_parse_response_coerces_string_booleans():
    detector = OpenAICompatibleChatAddressingDetector(
        base_url="https://example.test/v1",
        api_key="test-key",
        model="test-model",
        target_names=["カノン"],
    )

    try:
        decision = detector._parse_response({
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "accepted": "false",
                            "reason": "not_addressed",
                            "confidence": 0.9,
                            "explanation": "Not addressed.",
                        })
                    }
                }
            ]
        })

        assert decision.accepted is False
        assert decision.reason == "not_addressed"
    finally:
        await detector.close()


@pytest.mark.asyncio
async def test_parse_response_normalizes_unknown_reason():
    detector = OpenAICompatibleChatAddressingDetector(
        base_url="https://example.test/v1",
        api_key="test-key",
        model="test-model",
        target_names=["カノン"],
    )

    try:
        decision = detector._parse_response({
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "accepted": True,
                            "reason": "asked a question",
                            "confidence": 0.8,
                            "explanation": "The model returned a free-form reason.",
                        })
                    }
                }
            ]
        })

        assert decision.accepted is True
        assert decision.reason == "ambiguous"
        assert decision.metadata["original_reason"] == "asked a question"
    finally:
        await detector.close()


@pytest.mark.asyncio
async def test_parse_responses_output_text():
    detector = OpenAICompatibleChatAddressingDetector(
        base_url="https://example.test/v1",
        api_key="test-key",
        model="test-model",
        api_format="responses",
        target_names=["カノン"],
    )

    try:
        decision = detector._parse_response({
            "output_text": json.dumps({
                "accepted": False,
                "reason": "not_addressed",
                "confidence": 0.93,
                "explanation": "The utterance does not address カノン.",
            })
        })

        assert decision.accepted is False
        assert decision.reason == "not_addressed"
        assert decision.confidence == 0.93
        assert decision.explanation == "The utterance does not address カノン."
    finally:
        await detector.close()
