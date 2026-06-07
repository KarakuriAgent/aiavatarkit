import sqlite3
from time import time

import pytest

from aiavatar.sts import STSPipeline
from aiavatar.sts.llm import LLMServiceDummy
from aiavatar.sts.models import STSRequest
from aiavatar.sts.stt import SpeechRecognizerDummy
from aiavatar.sts.tts import SpeechSynthesizerDummy
from aiavatar.sts.vad import SpeechDetectorDummy


class RequestStartTimingLLM(LLMServiceDummy):
    async def get_llm_stream_response(
        self,
        context_id,
        user_id,
        messages,
        system_prompt_params=None,
        tools=None,
        inline_llm_params=None,
        session_id=None,
        channel=None,
        request_start_callback=None,
    ):
        if request_start_callback:
            request_start_callback(time())
        async for chunk in super().get_llm_stream_response(
            context_id,
            user_id,
            messages,
            system_prompt_params=system_prompt_params,
            tools=tools,
            inline_llm_params=inline_llm_params,
            session_id=session_id,
            channel=channel,
            request_start_callback=request_start_callback,
        ):
            yield chunk


@pytest.mark.asyncio
async def test_llm_request_start_time_is_recorded(tmp_path):
    db_path = str(tmp_path / "llm_request_start.db")
    sts = STSPipeline(
        vad=SpeechDetectorDummy(),
        stt=SpeechRecognizerDummy(recognized_text="hello"),
        llm=RequestStartTimingLLM(response_text="ok", db_connection_str=db_path),
        tts=SpeechSynthesizerDummy(),
        voice_recorder_enabled=False,
        db_connection_str=db_path,
    )

    responses = [
        response
        async for response in sts.invoke(STSRequest(
            session_id="llm-request-start",
            user_id="user01",
            audio_data=b"\x00\x00" * 16000,
            audio_duration=1.0,
        ))
    ]

    assert any(response.type == "final" for response in responses)
    await sts.shutdown()

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT before_llm_time, llm_request_start_time FROM performance_records ORDER BY id DESC LIMIT 1"
        ).fetchone()
        assert row is not None
        assert row["llm_request_start_time"] > 0
        assert row["llm_request_start_time"] >= row["before_llm_time"]
    finally:
        conn.close()
