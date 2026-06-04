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

        try:
            resp = await self.http_client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=self._request_body(
                    text=text,
                    recent_history=recent_history or [],
                    seconds_since_last_assistant_turn=seconds_since_last_assistant_turn,
                ),
            )
            resp.raise_for_status()
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

    def _request_body(
        self,
        *,
        text: str,
        recent_history: List[Dict[str, Any]],
        seconds_since_last_assistant_turn: Optional[float],
    ) -> Dict[str, Any]:
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": self._system_prompt(
                        recent_history=recent_history,
                        seconds_since_last_assistant_turn=seconds_since_last_assistant_turn,
                    ),
                },
                {
                    "role": "user",
                    "content": text,
                },
            ],
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "addressing_decision",
                    "strict": True,
                    "schema": {
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
                    },
                },
            },
        }

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

Decide only whether the current utterance is addressed to you.
Do not draft a reply.

Your names and aliases:
{names}

Recent conversation:
{history}

Current time (UTC):
{current_time_text}

Seconds since your last assistant turn:
{seconds_text}

Decision rules:
- Accept if the utterance explicitly calls you by one of your names or aliases.
- Accept if the utterance is clearly a reply or follow-up to your recent conversation.
- Accept short utterances such as "うん", "お願い", "それで", "いいよ", or "違う" only when they are clearly replies to your recent utterance.
- Do not accept merely because there is recent conversation. There must be concrete evidence in the current utterance or the immediately preceding turns.
- If a long time has passed since your last assistant turn, lean toward rejecting short or ambiguous utterances.
- Reject monologues, background speech, speech to a third party, or utterances that are not clearly addressed to you.
- Reject when uncertain.
- The explanation must be one short sentence citing the concrete evidence for the decision."""

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

    def _parse_response(self, payload: Dict[str, Any]) -> AddressingDecision:
        content = payload["choices"][0]["message"]["content"]
        if isinstance(content, list):
            content = "".join(
                part.get("text", "")
                for part in content
                if isinstance(part, dict)
            )
        data = json.loads(content)
        return AddressingDecision(
            accepted=bool(data["accepted"]),
            reason=data["reason"],
            confidence=float(data["confidence"]),
            explanation=str(data.get("explanation") or ""),
        )

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
            "target_names": self.target_names,
            "primary_name": self.primary_name,
            "timeout": self.timeout,
            "min_confidence": self.min_confidence,
            "fail_open": self.fail_open,
            "debug": self.debug,
        }

    async def close(self):
        await self.http_client.aclose()
