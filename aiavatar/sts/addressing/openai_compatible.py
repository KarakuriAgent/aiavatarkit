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
        recent_unaccepted_count: Optional[int] = None,
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
                    recent_unaccepted_count=recent_unaccepted_count,
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
        recent_unaccepted_count: Optional[int] = None,
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
        recent_unaccepted_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        system_prompt = self._system_prompt(
            recent_history=recent_history,
            seconds_since_last_assistant_turn=seconds_since_last_assistant_turn,
            recent_unaccepted_count=recent_unaccepted_count,
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
            max_tokens = 1024
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
            unaccepted_text = self._unaccepted_count_text(recent_unaccepted_count)
            user_content = (
                f"Actual history: {history_text}\n"
                f"Current time (UTC): {current_time_text}\n"
                f"Seconds since last assistant turn: {seconds_text}\n"
                f"Not-addressed utterances in the last 60 seconds: {unaccepted_text}\n"
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
        # Property order matters: explanation first makes the model state its
        # evidence before committing to a decision, and elapsed_seconds forces
        # it to read the timing input.
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "explanation": {
                    "type": "string",
                },
                "elapsed_seconds": {
                    "type": ["number", "null"],
                },
                "reason": {
                    "type": "string",
                    "enum": REASONS,
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
                "accepted": {"type": "boolean"},
            },
            "required": ["explanation", "elapsed_seconds", "reason", "confidence", "accepted"],
        }

    def _unaccepted_count_text(self, recent_unaccepted_count: Optional[int]) -> str:
        if recent_unaccepted_count is None:
            return "unknown"
        return str(recent_unaccepted_count)

    def _common_prompt_rules(self) -> str:
        reasons = "|".join(REASONS)
        name = self.primary_name
        return f"""Environment:
- The microphone is always open in a place where people also talk to each other.
- Most utterances that do not contain a target name are NOT addressed to {name}.
- Falsely accepting an utterance interrupts a human conversation and is much worse than falsely rejecting one; the user can simply repeat with a target name.
- "Addressed to {name}" is a claim that requires positive evidence. When in doubt, reject.

Decision procedure. Follow the steps in order and stop at the first step that decides:
1. The current utterance uses a target name, or an obvious speech-recognition variant of one, as a direct address (vocative) -> accepted=true, reason=called_by_name. If the name is only being talked about to someone else (e.g. "{name}って賢いよね"), this step does not decide; continue.
2. If actual history is none: accept only a complete, self-contained request that is clearly directed at an assistant -> accepted=true, reason=direct_request. Greetings, thanks, fragments, and omitted instructions (e.g. "あと短くして") -> accepted=false, reason=not_addressed.
3. If actual history is present, set accepted=true with reason=contextual_reply ONLY when all of the following hold:
   3a. Content tie: the assistant's last turn was a question, proposal, or confirmation that awaits an answer, and the current utterance answers it (e.g. "うん、お願い", "後者で", "明日"); OR the current utterance explicitly refers to the assistant's last output or the ongoing task as a correction, an added instruction, a cancellation, or a request to continue or retry (e.g. "あと箇条書きにして", "やっぱりキャンセル", "続けて").
   3b. Timing, as a soft guide rather than hard thresholds: up to about 15 seconds a short answer still connects naturally; from about 15 to 120 seconds accept only when the content clearly refers to the recent conversation; beyond about 120 seconds accept only when the utterance explicitly refers back to the conversation.
   3c. If the not-addressed count in the last 60 seconds is 2 or more, a conversation between people is probably going on nearby: accept only a direct answer to the assistant's pending question; otherwise reject.
   A reply-shaped utterance alone is NOT enough. If the assistant's last turn was a plain statement that did not ask anything, short acknowledgements and comments (e.g. "うん", "いいんじゃない？", "それな") are usually part of a conversation between people -> accepted=false, reason=not_addressed.
4. Thinking aloud, reading aloud, or fillers -> accepted=false, reason=monologue.
5. Speech to another person or another assistant (a vocative that is not a target name, e.g. "太郎", "ねえ", "アレクサ"), background speech, or an unrelated new topic -> accepted=false, reason=not_addressed.
6. If you cannot decide -> accepted=false, reason=ambiguous, confidence 0.5 or lower.

Confidence:
- 0.9-1.0: a target name is used as a direct address, or the utterance directly answers the assistant's pending question.
- 0.7-0.9: the content explicitly refers to the assistant's last output or the ongoing task.
- below 0.7: the connection is only plausible, not evidenced; such utterances should normally be rejected.

Output exactly one JSON object with the keys in this exact order: explanation, elapsed_seconds, reason, confidence, accepted.
- explanation: one short sentence citing the concrete evidence, including the elapsed seconds.
- elapsed_seconds: copy the number from "Seconds since last assistant turn" (null if unknown).
- reason must be one of: {reasons}.
- If accepted=true, reason must be one of called_by_name, contextual_reply, direct_request. If accepted=false, reason must be one of monologue, not_addressed, ambiguous.

Examples. They are not conversation history; the seconds and counts in them are illustrations, not thresholds.

History: (none) / Seconds: unknown / Not-addressed count: 0
Utterance: {name}、今日の予定を教えて
JSON: {{"explanation": "The utterance addresses {name} directly by name.", "elapsed_seconds": null, "reason": "called_by_name", "confidence": 1.0, "accepted": true}}

History: (none) / Seconds: unknown / Not-addressed count: 0
Utterance: ありがとう
JSON: {{"explanation": "A short social phrase with no target name and no actual history.", "elapsed_seconds": null, "reason": "not_addressed", "confidence": 0.9, "accepted": false}}

History: (none) / Seconds: unknown / Not-addressed count: 0
Utterance: あと短くして
JSON: {{"explanation": "An omitted instruction with no history and no clear addressee.", "elapsed_seconds": null, "reason": "not_addressed", "confidence": 0.9, "accepted": false}}

History: assistant=この内容で進めますか？ / Seconds: 3 / Not-addressed count: 0
Utterance: うん、お願い
JSON: {{"explanation": "It directly answers the assistant's pending question 3 seconds earlier.", "elapsed_seconds": 3, "reason": "contextual_reply", "confidence": 0.95, "accepted": true}}

History: assistant=A案とB案があります。どちらにしますか？ / Seconds: 3 / Not-addressed count: 0
Utterance: 後者で
JSON: {{"explanation": "It answers the assistant's pending choice question 3 seconds earlier.", "elapsed_seconds": 3, "reason": "contextual_reply", "confidence": 0.95, "accepted": true}}

History: assistant=この内容で進めますか？ / Seconds: 4 / Not-addressed count: 0
Utterance: いいんじゃない？
JSON: {{"explanation": "It answers the assistant's pending confirmation 4 seconds earlier.", "elapsed_seconds": 4, "reason": "contextual_reply", "confidence": 0.85, "accepted": true}}

History: assistant=今日は晴れです。 / Seconds: 8 / Not-addressed count: 0
Utterance: いいんじゃない？
JSON: {{"explanation": "The assistant's last turn asked nothing, so this comment 8 seconds later is likely part of a human conversation.", "elapsed_seconds": 8, "reason": "not_addressed", "confidence": 0.85, "accepted": false}}

History: assistant=会議メモを要約しました。 / Seconds: 4 / Not-addressed count: 0
Utterance: あと箇条書きにして
JSON: {{"explanation": "It adds an instruction about the assistant's output 4 seconds earlier.", "elapsed_seconds": 4, "reason": "contextual_reply", "confidence": 0.85, "accepted": true}}

History: user={name}、明日の予定を確認して / assistant=明日は10時に予定があります。 / Seconds: 3 / Not-addressed count: 0
Utterance: ごめん、言い間違えた。明日じゃなくて今日
JSON: {{"explanation": "It corrects the user's own request in the ongoing task 3 seconds earlier.", "elapsed_seconds": 3, "reason": "contextual_reply", "confidence": 0.85, "accepted": true}}

History: assistant=通知を設定します。 / Seconds: 4 / Not-addressed count: 0
Utterance: やっぱりキャンセル
JSON: {{"explanation": "It cancels the assistant's action announced 4 seconds earlier.", "elapsed_seconds": 4, "reason": "contextual_reply", "confidence": 0.85, "accepted": true}}

History: assistant=通知を設定します。 / Seconds: 90 / Not-addressed count: 0
Utterance: やっぱりキャンセル
JSON: {{"explanation": "After 90 seconds the cancellation no longer clearly connects to the assistant's action.", "elapsed_seconds": 90, "reason": "not_addressed", "confidence": 0.85, "accepted": false}}

History: assistant=今日は晴れです。 / Seconds: 40 / Not-addressed count: 0
Utterance: うん
JSON: {{"explanation": "A generic acknowledgement 40 seconds after a plain statement that asked nothing.", "elapsed_seconds": 40, "reason": "not_addressed", "confidence": 0.85, "accepted": false}}

History: assistant=この内容で進めますか？ / Seconds: 5 / Not-addressed count: 3
Utterance: うん、進めて
JSON: {{"explanation": "Despite nearby conversation, it directly answers the assistant's pending question 5 seconds earlier.", "elapsed_seconds": 5, "reason": "contextual_reply", "confidence": 0.9, "accepted": true}}

History: assistant=今日は晴れです。 / Seconds: 6 / Not-addressed count: 4
Utterance: それでさー、昨日の話なんだけど
JSON: {{"explanation": "Several recent not-addressed utterances and a new topic indicate an ongoing human conversation 6 seconds later.", "elapsed_seconds": 6, "reason": "not_addressed", "confidence": 0.9, "accepted": false}}

History: assistant=はい、疎通確認できています。 / Seconds: 5 / Not-addressed count: 0
Utterance: 太郎、これ見てくれる？
JSON: {{"explanation": "It addresses 太郎, who is not a target name, 5 seconds later.", "elapsed_seconds": 5, "reason": "not_addressed", "confidence": 0.95, "accepted": false}}

History: assistant=はい、疎通確認できています。 / Seconds: 10 / Not-addressed count: 0
Utterance: ねえ、明日どうする？
JSON: {{"explanation": "The vocative ねえ and the unrelated new topic 10 seconds later indicate speech to another person.", "elapsed_seconds": 10, "reason": "not_addressed", "confidence": 0.9, "accepted": false}}

History: assistant=はい、疎通確認できています。 / Seconds: 10 / Not-addressed count: 0
Utterance: {name}ってけっこう賢いよね
JSON: {{"explanation": "The name is talked about to someone else, not used as a direct address, 10 seconds later.", "elapsed_seconds": 10, "reason": "not_addressed", "confidence": 0.9, "accepted": false}}

History: (none) / Seconds: unknown / Not-addressed count: 0
Utterance: えっと、どこに置いたっけな
JSON: {{"explanation": "Thinking aloud with no addressee and no history.", "elapsed_seconds": null, "reason": "monologue", "confidence": 0.85, "accepted": false}}"""

    def _json_object_system_prompt(self) -> str:
        names = "\n".join(f"- {name}" for name in self.target_names)
        return f"""Reasoning: low

You are a strict classifier for the assistant named {self.primary_name}.
Decide only whether the current utterance is addressed to {self.primary_name}.
Never draft a reply. Return only one JSON object.
Judge only the text after "Current utterance:" in the latest user message.
Actual recent conversation is the user/assistant messages after this system message. The latest user message also contains "Actual history", "Current time (UTC)", "Seconds since last assistant turn", and "Not-addressed utterances in the last 60 seconds".
The examples in this system prompt are not conversation history.

Target names and aliases:
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
        recent_unaccepted_count: Optional[int] = None,
    ) -> str:
        names = "\n".join(f"- {name}" for name in self.target_names)
        history = self._format_history(recent_history)
        current_time_text = datetime.now(timezone.utc).isoformat()
        if seconds_since_last_assistant_turn is None:
            seconds_text = "unknown"
        else:
            seconds_text = f"{seconds_since_last_assistant_turn:.1f}"
        unaccepted_text = self._unaccepted_count_text(recent_unaccepted_count)

        return f"""Reasoning: low

You are a strict classifier for the assistant named {self.primary_name}.
Decide only whether the current utterance is addressed to {self.primary_name}.
Never draft a reply. Return only the structured JSON requested by the API schema.
Judge only the latest user message as the current utterance. The examples in this prompt are not conversation history.

Target names and aliases:
{names}

Recent conversation:
{history}

Current time (UTC):
{current_time_text}

Seconds since the last assistant turn:
{seconds_text}

Not-addressed utterances in the last 60 seconds:
{unaccepted_text}

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
        if "elapsed_seconds" in data:
            metadata["elapsed_seconds"] = data["elapsed_seconds"]
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
