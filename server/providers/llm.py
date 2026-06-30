import asyncio
import json
from logging import getLogger
from time import time
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

from aiavatar.sts.llm import LLMResponse, LLMService, Tool, ToolCall, ToolCallResult

from ..config import Settings

logger = getLogger(__name__)

CHANNEL_PREFIX_DISCORD = "[channel:discord]"
CHANNEL_PREFIX_PATTERN = "[channel:"


class HermesDelegationTracker(Tool):
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        poll_interval: float = 2.0,
        timeout: float = 1800.0,
    ):
        super().__init__(
            name="hermes_delegation",
            spec={
                "type": "function",
                "function": {
                    "name": "hermes_delegation",
                    "description": "Tracks a delegated Hermes background task.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            func=lambda: None,
        )
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.poll_interval = poll_interval
        self.timeout = timeout
        self._running_tasks: Dict[str, dict] = {}
        self._poll_tasks: Dict[str, asyncio.Task] = {}

    def track(self, delegation: Dict[str, Any], metadata: dict, request: str = None) -> Optional[str]:
        task_id = delegation.get("id")
        if not task_id or task_id in self._running_tasks:
            return task_id

        task_metadata = dict(metadata or {})
        task_metadata["task_id"] = task_id
        task_metadata["delegation"] = delegation
        self._background_metadata[task_id] = task_metadata
        self._running_tasks[task_id] = {
            "context_id": task_metadata.get("context_id"),
            "user_id": task_metadata.get("user_id"),
            "session_id": task_metadata.get("session_id"),
            "channel": task_metadata.get("channel"),
            "report_channel": None,
            "request": request or delegation.get("input_text") or "Hermes delegation",
            "progress": f"Hermes delegation queued: {task_id}\n",
            "status": delegation.get("status"),
            "delegation": delegation,
        }
        task = asyncio.create_task(self._poll_delegation(task_id))
        self._poll_tasks[task_id] = task
        task.add_done_callback(lambda _task: self._poll_tasks.pop(task_id, None))
        return task_id

    def get_running_task(self, task_id: str) -> Optional[dict]:
        task = self._running_tasks.get(task_id)
        if not task:
            return None
        return {"task_id": task_id, **task}

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _delegation_url(self, task_id: str) -> str:
        return f"{self.base_url}/delegations/{task_id}"

    async def _poll_delegation(self, task_id: str):
        deadline = time() + self.timeout
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(15.0), headers=self._headers()) as client:
                while time() < deadline:
                    await asyncio.sleep(self.poll_interval)
                    response = await client.get(self._delegation_url(task_id))
                    response.raise_for_status()
                    data = response.json()
                    status = str(data.get("status") or "")
                    running = self._running_tasks.get(task_id)
                    if running is None:
                        return
                    running["status"] = status
                    running["delegation"] = data
                    running["progress"] += f"Hermes delegation {status}\n"
                    if status in {"completed", "failed", "cancelled"}:
                        return
                running = self._running_tasks.get(task_id)
                if running:
                    running["status"] = "timeout"
                    running["progress"] += f"Hermes delegation timed out after {int(self.timeout)}s\n"
        except asyncio.CancelledError:
            return
        except Exception as exc:
            running = self._running_tasks.get(task_id)
            if running:
                running["status"] = "error"
                running["progress"] += f"Hermes delegation tracking error: {exc}\n"
            logger.warning("Hermes delegation tracking failed: task=%s error=%s", task_id, exc)
        finally:
            self._background_metadata.pop(task_id, None)
            self._running_tasks.pop(task_id, None)

    def cancel_background_tasks(
        self,
        *,
        task_id: str = None,
        session_id: str = None,
        user_id: str = None,
        context_id: str = None,
    ) -> List[str]:
        cancelled = []
        for tid, running in list(self._running_tasks.items()):
            if task_id and tid != task_id:
                continue
            if session_id and running.get("session_id") != session_id:
                continue
            if user_id and running.get("user_id") != user_id:
                continue
            if context_id and running.get("context_id") != context_id:
                continue
            poll_task = self._poll_tasks.pop(tid, None)
            if poll_task and not poll_task.done():
                poll_task.cancel()
            self._background_metadata.pop(tid, None)
            self._running_tasks.pop(tid, None)
            cancelled.append(tid)
        return cancelled


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
        delegation_poll_interval: float = 2.0,
        delegation_timeout: float = 1800.0,
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
        self.delegation_tracker = HermesDelegationTracker(
            base_url=self.base_url,
            api_key=self.api_key,
            poll_interval=delegation_poll_interval,
            timeout=delegation_timeout,
        )
        self.tools[self.delegation_tracker.name] = self.delegation_tracker

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

    @staticmethod
    def _message_text(messages: List[dict]) -> str:
        if not messages:
            return ""
        content = messages[-1].get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "\n".join(parts)
        return str(content or "")

    def _track_delegation_response(
        self,
        *,
        delegation: Dict[str, Any],
        context_id: str,
        user_id: str,
        session_id: str,
        channel: str,
        request_text: str,
        seen_delegations: set,
    ) -> Optional[LLMResponse]:
        if not isinstance(delegation, dict):
            return None
        delegation_id = delegation.get("id")
        if not delegation_id or delegation_id in seen_delegations:
            return None
        seen_delegations.add(delegation_id)
        metadata = {
            "context_id": context_id,
            "user_id": user_id,
            "session_id": session_id,
            "channel": channel,
        }
        task_id = self.delegation_tracker.track(delegation, metadata, request=request_text)
        if not task_id:
            return None
        result = ToolCallResult(
            data={
                "message": "Hermes delegation queued.",
                "task_id": task_id,
                "delegation": delegation,
            },
            task_id=task_id,
        )
        return LLMResponse(
            context_id=context_id,
            tool_call=ToolCall(
                id=task_id,
                name=self.delegation_tracker.name,
                arguments=json.dumps({"delegation": delegation}, ensure_ascii=False),
                result=result,
            ),
        )

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
        request_text = self._message_text(messages)
        seen_delegations = set()

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
                        elif event_type == "hermes.delegation.created":
                            if delegation_response := self._track_delegation_response(
                                delegation=payload,
                                context_id=context_id,
                                user_id=user_id,
                                session_id=session_id,
                                channel=channel,
                                request_text=request_text,
                                seen_delegations=seen_delegations,
                            ):
                                yield delegation_response
                        elif event_type == "response.completed" and self.debug:
                            logger.info("Hermes Responses API completed: %s", payload.get("response", {}).get("id"))
                        if event_type == "response.completed":
                            response_payload = payload.get("response") if isinstance(payload.get("response"), dict) else payload
                            for key in ("hermes_delegation",):
                                if delegation_response := self._track_delegation_response(
                                    delegation=response_payload.get(key),
                                    context_id=context_id,
                                    user_id=user_id,
                                    session_id=session_id,
                                    channel=channel,
                                    request_text=request_text,
                                    seen_delegations=seen_delegations,
                                ):
                                    yield delegation_response
                            for delegation in response_payload.get("hermes_delegations") or []:
                                if delegation_response := self._track_delegation_response(
                                    delegation=delegation,
                                    context_id=context_id,
                                    user_id=user_id,
                                    session_id=session_id,
                                    channel=channel,
                                    request_text=request_text,
                                    seen_delegations=seen_delegations,
                                ):
                                    yield delegation_response

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
        delegation_poll_interval=settings.hermes_delegation_poll_interval,
        delegation_timeout=settings.hermes_delegation_timeout,
        debug=settings.debug,
    )

    return llm
