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
            history_text = "あり" if history_messages else "なし"
            if seconds_since_last_assistant_turn is None:
                seconds_text = "unknown"
            else:
                seconds_text = f"{seconds_since_last_assistant_turn:.1f}"
            user_content = (
                f"実際の履歴: {history_text}\n"
                f"最後のassistant発話からの秒数: {seconds_text}\n"
                f"現在の発話: {text}"
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

    def _json_object_system_prompt(self) -> str:
        reasons = "|".join(REASONS)
        names = "、".join(self.target_names)
        name = self.primary_name
        return f"""あなたは{name}への宛先判定だけを行う分類器です。返答文は作らず、JSONだけを返してください。
最後のuser messageの「現在の発話:」以降だけを判定対象にしてください。
実際の履歴は、このsystem messageの後に追加されるuser/assistant messagesだけです。system prompt内の説明と例は会話履歴ではありません。
最後のassistant発話からの秒数と「実際の履歴: あり/なし」も判定に使ってください。
対象名一覧: {names}

判定観点:
- 最優先: 最後のassistant発話からの秒数が20.0以上なら、現在の発話に対象名が明示されている場合だけ accepted=true。対象名がなければ、相槌・追加・訂正・取り消し・変更・選択・継続に見えても必ず accepted=false。20.0以上では文脈の自然さを評価しない。
- 現在の発話に対象名一覧のいずれかが実際に含まれるなら accepted=true, reason=called_by_name。
- 現在の発話に対象名がなくても、実際の履歴があり、現在の発話が直近会話への自然な返答・続きなら accepted=true, reason=contextual_reply。
- 実際の履歴があり、現在の発話が直近会話の内容への訂正・言い直しなら contextual_reply。直前のassistant発話だけでなく、直前のuser自身の依頼内容への訂正も含む。
- 実際の履歴があり、現在の発話が追加指示・補足条件なら contextual_reply。例: 「あと短くして」「それも保存して」「日本語で」「箇条書きにして」「もう少し詳しく」。
- 実際の履歴があり、現在の発話が選択・確定・回答なら contextual_reply。例: 「それで」「一つ目で」「後者」「明日」「3つ」「東京」。
- 実際の履歴があり、現在の発話が継続・再実行・詳細化なら contextual_reply。例: 「続けて」「続きを」「もう一回」「他には」「それってどういう意味？」。
- 実際の履歴があり、現在の発話が取り消し・変更なら contextual_reply。例: 「やっぱりやめて」「キャンセル」「戻して」「こっちに変えて」。
- 短い社会的応答（ありがとう、うん、了解、お願い、ごめん等）は、実際の履歴と自然につながる場合だけ contextual_reply。履歴なしでは accepted=false。
- 独り言として自然なら accepted=false, reason=monologue。
- 第三者への呼びかけ、背景音声、独立した別話題、履歴なしの省略指示、判断不能な発話は accepted=false。
- 上記に明記されていない表現でも、現在の発話が直近会話への省略・照応・追加・修正・回答として自然につながるなら contextual_reply。

例。例は会話履歴ではありません。
実際の履歴なし / 現在の発話: {name}、今日の予定を教えて
JSON: {{"accepted": true, "reason": "called_by_name", "confidence": 1.0, "explanation": "現在の発話に対象名がある。"}}
実際の履歴なし / 現在の発話: ありがとう
JSON: {{"accepted": false, "reason": "not_addressed", "confidence": 0.85, "explanation": "名前も実際の履歴もない短い応答である。"}}
実際の履歴: user={name}、疎通確認だよ。 / assistant=はい、疎通確認できています。 / 秒数: 3 / 現在の発話: ありがとう
JSON: {{"accepted": true, "reason": "contextual_reply", "confidence": 0.95, "explanation": "直近の応答への短い謝意である。"}}
実際の履歴: assistant=この内容で進めますか？ / 秒数: 2 / 現在の発話: うん、お願い
JSON: {{"accepted": true, "reason": "contextual_reply", "confidence": 0.95, "explanation": "直近の質問への承諾である。"}}
実際の履歴: assistant=会議メモを要約しました。 / 秒数: 4 / 現在の発話: あと箇条書きにして
JSON: {{"accepted": true, "reason": "contextual_reply", "confidence": 0.95, "explanation": "直近の要約への追加指示である。"}}
実際の履歴: assistant=A案とB案があります。どちらにしますか？ / 秒数: 3 / 現在の発話: 後者で
JSON: {{"accepted": true, "reason": "contextual_reply", "confidence": 0.95, "explanation": "直近の選択肢への回答である。"}}
実際の履歴: assistant=ここまで説明しました。 / 秒数: 2 / 現在の発話: 続けて
JSON: {{"accepted": true, "reason": "contextual_reply", "confidence": 0.95, "explanation": "直近説明の継続を求めている。"}}
実際の履歴: user={name}、明日の予定を確認して / assistant=明日は10時に予定があります。 / 秒数: 3 / 現在の発話: ごめん、言い間違えた。明日じゃなくて今日
JSON: {{"accepted": true, "reason": "contextual_reply", "confidence": 0.95, "explanation": "直近の依頼内容を訂正している。"}}
実際の履歴: assistant=通知を設定します。 / 秒数: 4 / 現在の発話: やっぱりキャンセル
JSON: {{"accepted": true, "reason": "contextual_reply", "confidence": 0.95, "explanation": "直近の処理への取り消し指示である。"}}
実際の履歴: user={name}、疎通確認だよ。 / assistant=はい、疎通確認できています。 / 秒数: 3 / 現在の発話: 太郎、これ見てくれる？
JSON: {{"accepted": false, "reason": "not_addressed", "confidence": 0.95, "explanation": "対象以外の人物に呼びかけている。"}}
実際の履歴: assistant=今日は晴れです。 / 秒数: 25 / 現在の発話: うん
JSON: {{"accepted": false, "reason": "not_addressed", "confidence": 0.85, "explanation": "20秒以上後の短い相槌で、対象名もない。"}}
実際の履歴: assistant=通知を設定します。 / 秒数: 25 / 現在の発話: やっぱりキャンセル
JSON: {{"accepted": false, "reason": "not_addressed", "confidence": 0.85, "explanation": "20秒以上後の発話で、取り消しに見えても直近会話への文脈応答として扱わない。"}}
実際の履歴なし / 現在の発話: あと短くして
JSON: {{"accepted": false, "reason": "not_addressed", "confidence": 0.85, "explanation": "履歴なしの省略指示で宛先が判断できない。"}}

出力JSONのトップレベルキー: accepted, reason, confidence, explanation。
reason は {reasons} のどれか。
accepted=true の reason は called_by_name, contextual_reply, direct_request のどれか。ただし訂正・相槌・追加指示・選択・継続・取り消しは contextual_reply。
accepted=false の reason は monologue, not_addressed, ambiguous のどれか。"""

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
