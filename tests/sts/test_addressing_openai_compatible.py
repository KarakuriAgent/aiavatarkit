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
