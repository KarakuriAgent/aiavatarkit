import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx

from .base import AddressingDecision, AddressingDetector

logger = logging.getLogger(__name__)


REASONS = [
    "called_by_name",
    "contextual_reply",
    "direct_request",
    "monologue",
    "not_addressed",
    "ambiguous",
]


class OpenAICompatibleChatAddressingDetector(AddressingDetector):
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        api_format: str = "chat_completions",
        target_names: List[str],
        primary_name: str = None,
        timeout: float = 10.0,
        min_confidence: float = 0.0,
        fail_open: bool = False,
        debug: bool = False,
    ):
        if not target_names:
            raise ValueError("target_names is required")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.api_format = api_format
        self.target_names = target_names
        self.primary_name = primary_name or target_names[0]
        self.timeout = timeout
        self.min_confidence = min_confidence
        self.fail_open = fail_open
        self.debug = debug
        self.http_client = httpx.AsyncClient(timeout=httpx.Timeout(timeout))

    async def detect(
        self,
        *,
        text: str,
        recent_history: List[Dict[str, Any]] = None,
        seconds_since_last_assistant_turn: Optional[float] = None,
    ) -> AddressingDecision:
        if not text:
            return AddressingDecision(
                accepted=False,
                reason="not_addressed",
                confidence=1.0,
                explanation="The utterance is empty.",
            )

        local_decision = self._local_decision(
            text=text,
        )
        if local_decision is not None:
            return self._apply_confidence_threshold(local_decision)

        try:
            resp = await self.http_client.post(
                self._endpoint_url(),
                headers=self._headers(),
                json=self._request_body(
                    text=text,
                    recent_history=recent_history or [],
                    seconds_since_last_assistant_turn=seconds_since_last_assistant_turn,
                ),
            )
            resp.raise_for_status()
            if self.api_format == "responses" and resp.headers.get("content-type", "").startswith("text/event-stream"):
                decision = self._parse_response({"output_text": self._extract_sse_text(resp.text)})
            else:
                decision = self._parse_response(resp.json())
            return self._apply_confidence_threshold(decision)

        except Exception as ex:
            logger.warning("Addressing detection failed: %s", ex, exc_info=self.debug)
            return AddressingDecision(
                accepted=self.fail_open,
                reason="detector_error",
                confidence=0.0,
                explanation="The addressing detector failed.",
                metadata={"error": str(ex)},
            )

    def detect_sync(
        self,
        *,
        text: str,
        recent_history: List[Dict[str, Any]] = None,
        seconds_since_last_assistant_turn: Optional[float] = None,
    ) -> AddressingDecision:
        raise RuntimeError("OpenAICompatibleChatAddressingDetector.detect_sync is not supported; use detect().")

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _endpoint_url(self) -> str:
        if self.api_format == "responses":
            return f"{self.base_url}/responses"
        return f"{self.base_url}/chat/completions"

    def _request_body(
        self,
        *,
        text: str,
        recent_history: List[Dict[str, Any]],
        seconds_since_last_assistant_turn: Optional[float],
    ) -> Dict[str, Any]:
        system_prompt = self._system_prompt(
            recent_history=recent_history,
            seconds_since_last_assistant_turn=seconds_since_last_assistant_turn,
        )
        if self.api_format == "responses":
            schema = self._json_schema()
            return {
                "model": self.model,
                "instructions": system_prompt,
                "store": False,
                "stream": True,
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": text,
                            }
                        ],
                    },
                ],
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "addressing_decision",
                        "strict": True,
                        "schema": schema,
                    }
                },
            }

        if self.api_format == "chat_completions_json_object":
            system_prompt = self._json_object_system_prompt()
            response_format = {"type": "json_object"}
            max_tokens = 512
        else:
            schema = self._json_schema()
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "addressing_decision",
                    "strict": True,
                    "schema": schema,
                },
            }
            max_tokens = None

        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
        ]
        if self.api_format == "chat_completions_json_object":
            history_messages = self._json_object_history_messages(recent_history)
            messages.extend(history_messages)
            history_text = "present" if history_messages else "none"
            current_time_text = datetime.now(timezone.utc).isoformat()
            if seconds_since_last_assistant_turn is None:
                seconds_text = "unknown"
            else:
                seconds_text = f"{seconds_since_last_assistant_turn:.1f}"
            user_content = (
                f"Actual history: {history_text}\n"
                f"Current time (UTC): {current_time_text}\n"
                f"Seconds since last assistant turn: {seconds_text}\n"
                f"Current utterance: {text}"
            )
        else:
            user_content = text
        messages.append({
            "role": "user",
            "content": user_content,
        })

        body = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
            "response_format": response_format,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        return body

    def _json_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "accepted": {"type": "boolean"},
                "reason": {
                    "type": "string",
                    "enum": REASONS,
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
                "explanation": {
                    "type": "string",
                },
            },
            "required": ["accepted", "reason", "confidence", "explanation"],
        }

    def _common_prompt_rules(self) -> str:
        reasons = "|".join(REASONS)
        name = self.primary_name
        return f"""Decision rules:
- If the current utterance actually contains one of your target names or aliases, set accepted=true and reason=called_by_name.
- Use elapsed seconds as context. Decide whether the current utterance still naturally continues the recent conversation, or whether too much time has passed for the addressee to remain clear. Do not use a fixed cutoff; judge the utterance, recent history, and elapsed seconds together.
- If elapsed time makes a short acknowledgement, addition, correction, cancellation, change, choice, or continuation too disconnected from the recent conversation, and no target name is present, set accepted=false and reason=not_addressed.
- If there is actual recent conversation and the current utterance is a natural reply or follow-up to that conversation, set accepted=true and reason=contextual_reply.
- If there is actual recent conversation and the current utterance corrects or rephrases the recent conversation, set reason=contextual_reply. This includes corrections to the user's own previous request, not only corrections to your last assistant message.
- If there is actual recent conversation and the current utterance adds an instruction or condition, set reason=contextual_reply. Examples: "あと短くして", "それも保存して", "日本語で", "箇条書きにして", "もう少し詳しく".
- If there is actual recent conversation and the current utterance is a choice, confirmation, or answer, set reason=contextual_reply. Examples: "それで", "一つ目で", "後者", "明日", "3つ", "東京".
- If there is actual recent conversation and the current utterance asks to continue, retry, or elaborate, set reason=contextual_reply. Examples: "続けて", "続きを", "もう一回", "他には", "それってどういう意味？".
- If there is actual recent conversation and the current utterance cancels or changes the recent request, set reason=contextual_reply. Examples: "やっぱりやめて", "キャンセル", "戻して", "こっちに変えて".
- Accept short social utterances such as "ありがとう", "うん", "了解", "お願い", or "ごめん" only when they naturally connect to the actual recent conversation. Without actual recent conversation, set accepted=false.
- If the utterance is naturally a monologue, set accepted=false and reason=monologue.
- Reject speech to a third party, background speech, an independent new topic, an omitted instruction with no history, and utterances whose addressee is unclear by setting accepted=false.
- Even for expressions not listed above, if the current utterance naturally connects to the recent conversation as ellipsis, anaphora, addition, correction, or an answer, set reason=contextual_reply.

Examples. These examples are not conversation history, and the elapsed-second values in examples are not fixed thresholds.
No recent conversation / Current utterance: {name}、今日の予定を教えて
JSON: {{"accepted": true, "reason": "called_by_name", "confidence": 1.0, "explanation": "The current utterance contains a target name."}}
No recent conversation / Current utterance: ありがとう
JSON: {{"accepted": false, "reason": "not_addressed", "confidence": 0.85, "explanation": "It is a short social utterance with no target name and no actual history."}}
Recent conversation: user={name}、疎通確認だよ。 / assistant=はい、疎通確認できています。 / seconds: 3 / Current utterance: ありがとう
JSON: {{"accepted": true, "reason": "contextual_reply", "confidence": 0.95, "explanation": "It is a short thanks to the recent assistant response."}}
Recent conversation: assistant=この内容で進めますか？ / seconds: 2 / Current utterance: うん、お願い
JSON: {{"accepted": true, "reason": "contextual_reply", "confidence": 0.95, "explanation": "It accepts the recent assistant question."}}
Recent conversation: assistant=会議メモを要約しました。 / seconds: 4 / Current utterance: あと箇条書きにして
JSON: {{"accepted": true, "reason": "contextual_reply", "confidence": 0.95, "explanation": "It adds an instruction to the recent summary."}}
Recent conversation: assistant=A案とB案があります。どちらにしますか？ / seconds: 3 / Current utterance: 後者で
JSON: {{"accepted": true, "reason": "contextual_reply", "confidence": 0.95, "explanation": "It answers the recent choice question."}}
Recent conversation: assistant=ここまで説明しました。 / seconds: 2 / Current utterance: 続けて
JSON: {{"accepted": true, "reason": "contextual_reply", "confidence": 0.95, "explanation": "It asks to continue the recent explanation."}}
Recent conversation: user={name}、明日の予定を確認して / assistant=明日は10時に予定があります。 / seconds: 3 / Current utterance: ごめん、言い間違えた。明日じゃなくて今日
JSON: {{"accepted": true, "reason": "contextual_reply", "confidence": 0.95, "explanation": "It corrects the user's recent request."}}
Recent conversation: assistant=通知を設定します。 / seconds: 4 / Current utterance: やっぱりキャンセル
JSON: {{"accepted": true, "reason": "contextual_reply", "confidence": 0.95, "explanation": "It cancels the recent assistant action."}}
Recent conversation: user={name}、疎通確認だよ。 / assistant=はい、疎通確認できています。 / seconds: 3 / Current utterance: 太郎、これ見てくれる？
JSON: {{"accepted": false, "reason": "not_addressed", "confidence": 0.95, "explanation": "It calls a third party instead of a target name."}}
Recent conversation: assistant=今日は晴れです。 / seconds: 90 / Current utterance: うん
JSON: {{"accepted": false, "reason": "not_addressed", "confidence": 0.85, "explanation": "The elapsed time makes the short acknowledgement too disconnected from the recent conversation."}}
Recent conversation: assistant=通知を設定します。 / seconds: 90 / Current utterance: やっぱりキャンセル
JSON: {{"accepted": false, "reason": "not_addressed", "confidence": 0.85, "explanation": "The elapsed time makes the cancellation too disconnected from the recent action."}}
No recent conversation / Current utterance: あと短くして
JSON: {{"accepted": false, "reason": "not_addressed", "confidence": 0.85, "explanation": "It is an omitted instruction with no history and no clear addressee."}}

Top-level JSON keys: accepted, reason, confidence, explanation.
reason must be one of: {reasons}.
If accepted=true, reason must be one of called_by_name, contextual_reply, direct_request. Corrections, acknowledgements, added instructions, choices, continuations, and cancellations are contextual_reply.
If accepted=false, reason must be one of monologue, not_addressed, ambiguous.
The explanation must be one short sentence citing the concrete evidence for the decision."""

    def _json_object_system_prompt(self) -> str:
        names = "\n".join(f"- {name}" for name in self.target_names)
        return f"""You are {self.primary_name}.

You are a classifier that decides only whether the current utterance is addressed to {self.primary_name}.
Do not draft a reply. Return only a JSON object.
Judge only the text after "Current utterance:" in the latest user message.
Actual recent conversation is the user/assistant messages after this system message. The latest user message also contains "Actual history", "Current time (UTC)", and "Seconds since last assistant turn".
The examples in this system prompt are not conversation history.
Use the recent conversation, the current time, and the elapsed seconds when making the decision.

Your target names and aliases:
{names}

{self._common_prompt_rules()}"""

    def _matched_target_name(self, text: str) -> Optional[str]:
        if not text:
            return None
        for name in self.target_names:
            if name and name in text:
                return name
        return None

    def _local_decision(
        self,
        *,
        text: str,
    ) -> Optional[AddressingDecision]:
        matched_name = self._matched_target_name(text)
        if matched_name:
            return AddressingDecision(
                accepted=True,
                reason="called_by_name",
                confidence=1.0,
                explanation=f"The utterance explicitly contains target name: {matched_name}.",
                metadata={"local_short_circuit": True},
            )

        return None

    def _json_object_history_messages(
        self,
        recent_history: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        messages = []
        for item in recent_history:
            role = item.get("role") or "user"
            if role in ("assistant", "model"):
                role = "assistant"
            elif role != "user":
                role = "user"
            text = self._extract_text(item)
            if not text:
                continue
            messages.append({
                "role": role,
                "content": text,
            })
        return messages

    def _system_prompt(
        self,
        *,
        recent_history: List[Dict[str, Any]],
        seconds_since_last_assistant_turn: Optional[float],
    ) -> str:
        names = "\n".join(f"- {name}" for name in self.target_names)
        history = self._format_history(recent_history)
        current_time_text = datetime.now(timezone.utc).isoformat()
        if seconds_since_last_assistant_turn is None:
            seconds_text = "unknown"
        else:
            seconds_text = f"{seconds_since_last_assistant_turn:.1f}"

        return f"""You are {self.primary_name}.

You are a classifier that decides only whether the current utterance is addressed to {self.primary_name}.
Do not draft a reply. Return only the structured JSON requested by the API schema.
Judge only the latest user message as the current utterance. The examples in this prompt are not conversation history.
Use the recent conversation, the current time, and the elapsed seconds when making the decision.

Your target names and aliases:
{names}

Recent conversation:
{history}

Current time (UTC):
{current_time_text}

Seconds since your last assistant turn:
{seconds_text}

{self._common_prompt_rules()}"""

    def _format_history(self, recent_history: List[Dict[str, Any]]) -> str:
        if not recent_history:
            return "(none)"

        lines = []
        for item in recent_history:
            role = item.get("role") or "unknown"
            if role in ("assistant", "model"):
                role = self.primary_name
            text = self._extract_text(item)
            if not text:
                continue
            created_at = item.get("created_at")
            prefix = f"{created_at} {role}" if created_at else role
            lines.append(f"- {prefix}: {text}")
        return "\n".join(lines) if lines else "(none)"

    def _extract_text(self, item: Dict[str, Any]) -> str:
        if "message" in item and isinstance(item["message"], str):
            return item["message"]

        content = item.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                text = part.get("text") or part.get("input_text")
                if text:
                    parts.append(str(text))
            return " ".join(parts).strip()

        parts = item.get("parts")
        if isinstance(parts, list):
            texts = []
            for part in parts:
                if isinstance(part, dict) and part.get("text"):
                    texts.append(str(part["text"]))
            return " ".join(texts).strip()

        return ""

    def _extract_chat_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        content = payload["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict)
            )
        return self._loads_json_object(content)

    def _loads_json_object(self, content: str) -> Dict[str, Any]:
        stripped = content.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            stripped = "\n".join(lines).strip()

        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            decoder = json.JSONDecoder()
            for index, char in enumerate(stripped):
                if char != "{":
                    continue
                try:
                    data, _ = decoder.raw_decode(stripped[index:])
                    break
                except json.JSONDecodeError:
                    continue
            else:
                raise

        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object from addressing detector")
        return data

    def _parse_response(self, payload: Dict[str, Any]) -> AddressingDecision:
        if self.api_format == "responses":
            content = self._extract_responses_text(payload)
        else:
            content = payload["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict)
            )
        data = self._loads_json_object(content)
        accepted = self._coerce_bool(data["accepted"])
        reason = str(data["reason"])
        metadata = {}
        if reason not in REASONS:
            metadata["original_reason"] = reason
            reason = "ambiguous" if accepted else "not_addressed"
        return AddressingDecision(
            accepted=accepted,
            reason=reason,
            confidence=float(data["confidence"]),
            explanation=str(data.get("explanation") or ""),
            metadata=metadata,
        )

    def _coerce_bool(self, value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in ("true", "yes", "1", "accept", "accepted", "はい"):
                return True
            if normalized in ("false", "no", "0", "reject", "rejected", "いいえ"):
                return False
        return bool(value)

    def _extract_responses_text(self, payload: Dict[str, Any]) -> str:
        output_text = payload.get("output_text")
        if isinstance(output_text, str) and output_text:
            return output_text

        parts = []
        for item in payload.get("output") or []:
            if not isinstance(item, dict):
                continue
            for content in item.get("content") or []:
                if not isinstance(content, dict):
                    continue
                text = content.get("text") or content.get("output_text")
                if text:
                    parts.append(str(text))
        if parts:
            return "".join(parts)

        raise KeyError("output_text")

    def _extract_sse_text(self, content: str) -> str:
        parts = []
        done_text = None
        completed_response = None
        for line in content.splitlines():
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(str(event["error"]))
            event_type = event.get("type")
            if event_type == "response.completed" and isinstance(event.get("response"), dict):
                completed_response = event["response"]
                continue
            if event_type == "response.output_text.delta":
                delta = event.get("delta")
                if isinstance(delta, str):
                    parts.append(delta)
                continue
            if event_type == "response.output_text.done":
                text = event.get("text")
                if isinstance(text, str):
                    done_text = text

        if parts:
            return "".join(parts)
        if done_text:
            return done_text
        if completed_response:
            return self._extract_responses_text(completed_response)
        raise KeyError("output_text")

    def _apply_confidence_threshold(self, decision: AddressingDecision) -> AddressingDecision:
        if (
            decision.accepted
            and decision.confidence is not None
            and decision.confidence < self.min_confidence
        ):
            decision.metadata["original_reason"] = decision.reason
            decision.reason = "ambiguous"
            decision.accepted = False
        return decision

    def get_config(self) -> Dict[str, Any]:
        return {
            "base_url": self.base_url,
            "model": self.model,
            "api_format": self.api_format,
            "target_names": self.target_names,
            "primary_name": self.primary_name,
            "timeout": self.timeout,
            "min_confidence": self.min_confidence,
            "fail_open": self.fail_open,
            "debug": self.debug,
        }

    async def close(self):
        await self.http_client.aclose()
