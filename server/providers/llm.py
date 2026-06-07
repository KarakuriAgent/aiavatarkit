import json
from logging import getLogger
from time import time
from typing import AsyncGenerator, Dict, List

import httpx

from aiavatar.sts.llm import LLMResponse, LLMService

from ..config import Settings

logger = getLogger(__name__)

CHANNEL_PREFIX_DISCORD = "[channel:discord]"
CHANNEL_PREFIX_PATTERN = "[channel:"


class HermesResponsesService(LLMService):
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        request_prefix: str = "[channel:voice]",
        conversation_id_source: str = "context_id",
        store: bool = True,
        reasoning_effort: str = None,
        system_prompt: str = None,
        model: str = "hermes-agent",
        debug: bool = False,
    ):
        super().__init__(
            system_prompt=system_prompt,
            model=model,
            split_on_control_tags=True,
            debug=debug,
        )
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.request_prefix = request_prefix
        self.conversation_id_source = conversation_id_source
        self.store = store
        self.reasoning_effort = reasoning_effort

    @property
    def dynamic_tool_name(self) -> str:
        return "execute_external_tool"

    async def compose_messages(
        self,
        context_id: str,
        user_id: str,
        text: str,
        files: List[Dict[str, str]] = None,
        system_prompt_params: Dict[str, any] = None,
    ) -> List[Dict]:
        content = []
        if text:
            content.append({"type": "input_text", "text": text})
        for f in files or []:
            if url := f.get("url"):
                content.append({"type": "input_image", "image_url": url})
        return [{"role": "user", "content": content}]

    async def update_context(self, context_id: str, user_id: str, messages: List[Dict], response_text: str):
        messages.append({"role": "assistant", "content": response_text})
        await self.context_manager.add_histories(context_id, messages, "hermes_responses", user_id=user_id)

    def _request_prefix_for_channel(self, channel: str = None) -> str | None:
        normalized_channel = channel.strip().lower() if isinstance(channel, str) else ""
        if normalized_channel == "discord":
            return CHANNEL_PREFIX_DISCORD
        if normalized_channel in {"hermes", "cron"}:
            return None
        return self.request_prefix

    @staticmethod
    def _prepend_request_prefix(text: str, prefix: str = None) -> str:
        if not text or not prefix:
            return text
        if text.lstrip().startswith(CHANNEL_PREFIX_PATTERN):
            return text
        return prefix + text

    def _apply_request_prefix(self, messages: List[Dict], channel: str = None) -> None:
        prefix = self._request_prefix_for_channel(channel)
        if not prefix:
            return

        for message in messages:
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = self._prepend_request_prefix(content, prefix)
                return
            if not isinstance(content, list):
                continue
            for index, item in enumerate(content):
                if isinstance(item, str):
                    content[index] = self._prepend_request_prefix(item, prefix)
                    return
                if not isinstance(item, dict):
                    continue
                if item.get("type") not in {"text", "input_text", "output_text"}:
                    continue
                text = item.get("text")
                if isinstance(text, str):
                    item["text"] = self._prepend_request_prefix(text, prefix)
                    return

    def _conversation_id(self, context_id: str, user_id: str, session_id: str = None) -> str:
        if self.conversation_id_source == "user_id":
            return user_id
        if self.conversation_id_source == "session_id":
            return session_id
        return context_id

    async def get_llm_stream_response(
        self,
        context_id: str,
        user_id: str,
        messages: List[dict],
        system_prompt_params: Dict[str, any] = None,
        tools: List[Dict[str, any]] = None,
        inline_llm_params: Dict[str, any] = None,
        session_id: str = None,
        channel: str = None,
        request_start_callback=None,
    ) -> AsyncGenerator[LLMResponse, None]:
        self._apply_request_prefix(messages, channel)

        request_body = {
            "model": self.model,
            "input": messages,
            "stream": True,
            "store": self.store,
        }
        callback_metadata = {
            key: value
            for key, value in {
                "channel": channel,
                "session_id": session_id,
                "user_id": user_id,
            }.items()
            if value
        }
        if callback_metadata:
            request_body["callback_metadata"] = callback_metadata

        if self.reasoning_effort is not None:
            request_body["reasoning"] = {"effort": self.reasoning_effort}

        if instructions := await self._get_system_prompt(context_id, user_id, system_prompt_params):
            request_body["instructions"] = instructions

        if conversation_id := self._conversation_id(context_id, user_id, session_id):
            request_body["conversation"] = conversation_id

        if inline_llm_params:
            request_body.update(inline_llm_params)

        if self.debug:
            logger.info("Request to Hermes Responses API: %s", request_body)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            if request_start_callback:
                request_start_callback(time())
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/responses",
                    headers=headers,
                    json=request_body,
                ) as resp:
                    resp.raise_for_status()
                    event_name = None
                    async for raw_line in resp.aiter_lines():
                        line = raw_line.strip()
                        if not line:
                            continue
                        if line.startswith("event:"):
                            event_name = line[len("event:"):].strip()
                            continue
                        if not line.startswith("data:"):
                            continue

                        data = line[len("data:"):].strip()
                        if data == "[DONE]":
                            break

                        try:
                            payload = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        event_type = payload.get("type") or event_name
                        if event_type == "response.output_text.delta":
                            if delta := payload.get("delta"):
                                yield LLMResponse(context_id=context_id, text=delta)
                        elif event_type == "response.completed" and self.debug:
                            logger.info("Hermes Responses API completed: %s", payload.get("response", {}).get("id"))

        except httpx.HTTPStatusError as ex:
            response_json = None
            try:
                response_json = ex.response.json()
            except Exception:
                pass
            logger.warning("HTTPStatusError from Hermes Responses API: %s", ex)
            yield LLMResponse(context_id=context_id, error_info={"exception": ex, "response_json": response_json})

        except Exception as ex:
            logger.warning("Error from Hermes Responses API: %s", ex)
            yield LLMResponse(context_id=context_id, error_info={"exception": ex, "response_json": None})


def create_llm(settings: Settings):
    if settings.llm_provider != "hermes":
        raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")

    llm = HermesResponsesService(
        api_key=settings.hermes_api_key,
        base_url=settings.hermes_base_url,
        model=settings.hermes_model,
        request_prefix=settings.hermes_request_prefix,
        conversation_id_source=settings.hermes_conversation_id_source,
        store=settings.hermes_store,
        reasoning_effort=settings.hermes_reasoning_effort,
        debug=settings.debug,
    )

    return llm
