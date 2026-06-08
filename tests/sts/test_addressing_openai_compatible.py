import json

import pytest

from aiavatar.sts.addressing import OpenAICompatibleChatAddressingDetector


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
        assert "The explanation must be one short sentence citing the concrete evidence" in prompt
        assert "assistant:" not in prompt
        assert "model:" not in prompt

        schema = body["response_format"]["json_schema"]["schema"]
        assert schema["properties"]["explanation"] == {"type": "string"}
        assert "explanation" in schema["required"]
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
        assert "You are カノン." in body["instructions"]
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
        assert body["max_tokens"] == 512
        assert "20.0以上なら" in prompt
        assert "出力JSONのトップレベルキー" in prompt
        assert body["messages"][-1] == {
            "role": "user",
            "content": "実際の履歴: なし\n最後のassistant発話からの秒数: unknown\n現在の発話: ロボ花音、聞こえる?",
        }
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
        )

        assert body["messages"][1] == {
            "role": "user",
            "content": "ロボ花音、確認して",
        }
        assert body["messages"][2] == {
            "role": "assistant",
            "content": "確認できています。",
        }
        assert body["messages"][3] == {
            "role": "user",
            "content": "実際の履歴: あり\n最後のassistant発話からの秒数: 3.0\n現在の発話: 助かりました",
        }
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

        assert "追加指示・補足条件" in prompt
        assert "選択・確定・回答" in prompt
        assert "継続・再実行・詳細化" in prompt
        assert "取り消し・変更" in prompt
        assert "省略・照応・追加・修正・回答" in prompt
        assert "20.0以上なら" in prompt
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
