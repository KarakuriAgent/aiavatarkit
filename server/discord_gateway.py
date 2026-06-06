import asyncio
import io
import json
import logging
import platform
import re
import time
import wave
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional

import httpx
import websockets
from fastapi import FastAPI

from aiavatar.adapter.base import Adapter
from aiavatar.admin.control import ChatRequest, process_conversation_request

from .config import Settings

logger = logging.getLogger(__name__)


DISCORD_API_BASE = "https://discord.com/api/v10"
DISCORD_CDN_BASE = "https://cdn.discordapp.com"
DISCORD_GATEWAY_URL = "wss://gateway.discord.gg/?v=10&encoding=json"


@dataclass
class DiscordProfile:
    username: str
    avatar_url: Optional[str] = None


@dataclass
class DiscordFile:
    filename: str
    content: bytes
    content_type: str = "application/octet-stream"


def remove_control_tags(text: Optional[str]) -> str:
    if not text:
        return ""
    clean_text = re.sub(r"\[(\w+):([^\]]+)\]", "", text)
    clean_text = re.sub(r"<\w+\s[^>]*>", "", clean_text)
    return clean_text.strip()


def _avatar_extension(avatar_hash: str) -> str:
    return "gif" if avatar_hash and avatar_hash.startswith("a_") else "webp"


def _default_avatar_url(user_id: str) -> str:
    try:
        index = (int(user_id) >> 22) % 6
    except Exception:
        index = 0
    return f"{DISCORD_CDN_BASE}/embed/avatars/{index}.png"


def _user_avatar_url(user_id: str, avatar_hash: Optional[str], *, size: int = 128) -> Optional[str]:
    if not avatar_hash:
        return None
    ext = _avatar_extension(avatar_hash)
    return f"{DISCORD_CDN_BASE}/avatars/{user_id}/{avatar_hash}.{ext}?size={size}"


def _member_avatar_url(guild_id: str, user_id: str, avatar_hash: Optional[str], *, size: int = 128) -> Optional[str]:
    if not guild_id or not avatar_hash:
        return None
    ext = _avatar_extension(avatar_hash)
    return f"{DISCORD_CDN_BASE}/guilds/{guild_id}/users/{user_id}/avatars/{avatar_hash}.{ext}?size={size}"


class DiscordIdentityResolver:
    def __init__(
        self,
        *,
        bot_token: str,
        guild_id: str = None,
        cache_ttl: float = 300,
        http_client: httpx.AsyncClient = None,
    ):
        self.bot_token = bot_token
        self.guild_id = guild_id
        self.cache_ttl = cache_ttl
        self.http_client = http_client or httpx.AsyncClient(timeout=10)
        self._owns_http_client = http_client is None
        self._cache: dict[str, tuple[float, DiscordProfile]] = {}

    async def close(self):
        if self._owns_http_client:
            await self.http_client.aclose()

    async def resolve(self, user_id: str, *, fallback_name: str = None) -> DiscordProfile:
        if not user_id:
            return DiscordProfile(username=fallback_name or "Discord")

        now = time.monotonic()
        cached = self._cache.get(user_id)
        if cached and now - cached[0] < self.cache_ttl:
            return cached[1]

        member = await self._get_member(user_id) if self.guild_id else None
        user = (member or {}).get("user") or await self._get_user(user_id) or {}

        username = (
            (member or {}).get("nick")
            or user.get("global_name")
            or user.get("username")
            or fallback_name
            or user_id
        )
        avatar_url = (
            _member_avatar_url(self.guild_id, user_id, (member or {}).get("avatar"))
            or _user_avatar_url(user_id, user.get("avatar"))
            or _default_avatar_url(user_id)
        )

        profile = DiscordProfile(username=username, avatar_url=avatar_url)
        self._cache[user_id] = (now, profile)
        return profile

    async def _get_user(self, user_id: str) -> Optional[dict]:
        return await self._get_json(f"{DISCORD_API_BASE}/users/{user_id}")

    async def _get_member(self, user_id: str) -> Optional[dict]:
        return await self._get_json(f"{DISCORD_API_BASE}/guilds/{self.guild_id}/members/{user_id}")

    async def _get_json(self, url: str) -> Optional[dict]:
        headers = {"Authorization": f"Bot {self.bot_token}"}
        try:
            resp = await self.http_client.get(url, headers=headers)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except Exception as ex:
            logger.warning("Failed to resolve Discord identity: %s", ex)
            return None


class DiscordWebhookClient:
    def __init__(self, *, webhook_url: str, http_client: httpx.AsyncClient = None):
        self.webhook_url = webhook_url
        self.http_client = http_client or httpx.AsyncClient(timeout=20)
        self._owns_http_client = http_client is None

    async def close(self):
        if self._owns_http_client:
            await self.http_client.aclose()

    async def post(self, content: str, *, profile: DiscordProfile = None, files: list[DiscordFile] = None):
        files = [f for f in (files or []) if f and f.content]
        chunks = split_discord_content(content)
        if not chunks and files:
            chunks = [""]
        for idx, chunk in enumerate(chunks):
            payload = {
                "content": chunk,
                "allowed_mentions": {"parse": []},
            }
            if profile:
                payload["username"] = profile.username
                if profile.avatar_url:
                    payload["avatar_url"] = profile.avatar_url

            try:
                if files and idx == 0:
                    resp = await self.http_client.post(
                        self.webhook_url,
                        data={"payload_json": json.dumps(payload, ensure_ascii=False)},
                        files=[
                            (f"files[{file_idx}]", (file.filename, file.content, file.content_type))
                            for file_idx, file in enumerate(files)
                        ],
                    )
                else:
                    resp = await self.http_client.post(self.webhook_url, json=payload)
                resp.raise_for_status()
            except Exception as ex:
                logger.warning("Failed to post Discord webhook message: %s", ex)


def split_discord_content(content: str, *, limit: int = 2000) -> list[str]:
    content = (content or "").strip()
    if not content:
        return []

    chunks = []
    remaining = content
    while len(remaining) > limit:
        cut = remaining.rfind("\n", 0, limit)
        if cut <= 0:
            cut = limit
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


DEBUG_FILTER_LABELS = {
    "audio_enhancement_failed": "音声補正で除外",
    "voice_auth_rejected": "話者認証で除外",
    "no_speech_recognized": "音声認識結果なし",
    "addressing_rejected": "宛先判定で除外",
    "validate_request_rejected": "リクエスト検証で除外",
    "wakeword_rejected": "ウェイクワード未検出",
}


def _canonical_filter_reason(metadata: dict) -> Optional[str]:
    reason = metadata.get("filter_reason") or metadata.get("reason")
    if reason == "No speech recognized.":
        return "no_speech_recognized"
    if reason in DEBUG_FILTER_LABELS:
        return reason
    return metadata.get("filter_reason")


def _compact_value(value):
    if value is None or value == "":
        return None
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _append_detail(lines: list[str], label: str, value):
    compact = _compact_value(value)
    if compact is not None:
        lines.append(f"{label}: {compact}")


def _is_wav(data: bytes) -> bool:
    return bool(data and data.startswith(b"RIFF") and data[8:12] == b"WAVE")


def _audio_file_extension(data: bytes, fallback: str = "wav") -> str:
    if _is_wav(data):
        return "wav"
    if data.startswith(b"OggS"):
        return "ogg"
    if data.startswith(b"ID3") or data[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "mp3"
    return fallback


def _audio_content_type(extension: str) -> str:
    return {
        "wav": "audio/wav",
        "ogg": "audio/ogg",
        "mp3": "audio/mpeg",
    }.get(extension, "application/octet-stream")


def _merge_wav_chunks(chunks: list[bytes]) -> Optional[bytes]:
    if not chunks or not all(_is_wav(chunk) for chunk in chunks):
        return None

    params = None
    frames = []
    try:
        for chunk in chunks:
            with wave.open(io.BytesIO(chunk), "rb") as wav_file:
                current_params = (
                    wav_file.getnchannels(),
                    wav_file.getsampwidth(),
                    wav_file.getframerate(),
                    wav_file.getcomptype(),
                    wav_file.getcompname(),
                )
                if params is None:
                    params = current_params
                elif params != current_params:
                    return None
                frames.append(wav_file.readframes(wav_file.getnframes()))

        out = io.BytesIO()
        with wave.open(out, "wb") as wav_file:
            channels, sample_width, frame_rate, comp_type, comp_name = params
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(frame_rate)
            wav_file.setcomptype(comp_type, comp_name)
            wav_file.writeframes(b"".join(frames))
        return out.getvalue()
    except Exception:
        return None


def _build_audio_files(prefix: str, chunks: list[bytes], *, max_files: int = 10) -> list[DiscordFile]:
    chunks = [chunk for chunk in chunks if chunk]
    if not chunks:
        return []

    merged = _merge_wav_chunks(chunks)
    if merged:
        return [DiscordFile(
            filename=f"{prefix}.wav",
            content=merged,
            content_type="audio/wav",
        )]

    files = []
    for idx, chunk in enumerate(chunks[:max_files]):
        ext = _audio_file_extension(chunk)
        files.append(DiscordFile(
            filename=f"{prefix}_{idx}.{ext}",
            content=chunk,
            content_type=_audio_content_type(ext),
        ))
    return files


class DiscordGatewayClient:
    def __init__(
        self,
        *,
        bot_token: str,
        channel_id: str,
        on_message: Callable[[str, dict], Awaitable[None]],
        intents: int = None,
        http_client: httpx.AsyncClient = None,
    ):
        self.bot_token = bot_token
        self.channel_id = channel_id
        self.on_message = on_message
        self.intents = intents if intents is not None else ((1 << 0) | (1 << 9) | (1 << 15))
        self.http_client = http_client or httpx.AsyncClient(timeout=10)
        self._owns_http_client = http_client is None
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._last_sequence = None
        self._ws = None

    def start(self):
        if self._task and not self._task.done():
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_forever())

    async def stop(self):
        self._stop_event.set()
        if self._ws:
            await self._ws.close()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self._owns_http_client:
            await self.http_client.aclose()

    async def trigger_typing(self, channel_id: str = None):
        target_channel_id = channel_id or self.channel_id
        headers = {"Authorization": f"Bot {self.bot_token}"}
        try:
            resp = await self.http_client.post(
                f"{DISCORD_API_BASE}/channels/{target_channel_id}/typing",
                headers=headers,
            )
            resp.raise_for_status()
        except Exception as ex:
            logger.warning("Failed to trigger Discord typing indicator: %s", ex)

    async def _run_forever(self):
        backoff = 1.0
        while not self._stop_event.is_set():
            try:
                await self._connect_once()
                backoff = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as ex:
                logger.warning("Discord Gateway disconnected: %s", ex)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

    async def _connect_once(self):
        heartbeat_task = None
        async with websockets.connect(DISCORD_GATEWAY_URL, ping_interval=None) as ws:
            self._ws = ws
            async for raw_message in ws:
                payload = json.loads(raw_message)
                if payload.get("s") is not None:
                    self._last_sequence = payload["s"]

                op = payload.get("op")
                if op == 10:
                    interval = payload["d"]["heartbeat_interval"] / 1000.0
                    heartbeat_task = asyncio.create_task(self._heartbeat(ws, interval))
                    await self._identify(ws)
                elif op == 7:
                    break
                elif op == 9:
                    break
                elif op == 0 and payload.get("t") == "MESSAGE_CREATE":
                    await self._handle_message(payload.get("d") or {})

                if self._stop_event.is_set():
                    break

        if heartbeat_task:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except asyncio.CancelledError:
                pass
        self._ws = None

    async def _heartbeat(self, ws, interval: float):
        while not self._stop_event.is_set():
            await asyncio.sleep(interval)
            await ws.send(json.dumps({"op": 1, "d": self._last_sequence}))

    async def _identify(self, ws):
        await ws.send(json.dumps({
            "op": 2,
            "d": {
                "token": self.bot_token,
                "intents": self.intents,
                "properties": {
                    "os": platform.system().lower(),
                    "browser": "aiavatarkit",
                    "device": "aiavatarkit",
                },
            },
        }))

    async def _handle_message(self, message: dict):
        if message.get("channel_id") != self.channel_id:
            return
        if message.get("webhook_id"):
            return

        author = message.get("author") or {}
        if author.get("bot"):
            return

        content = (message.get("content") or "").strip()
        if not content:
            return

        try:
            await self.on_message(content, message)
        except Exception:
            logger.exception("Error while handling Discord message")


class DiscordIntegration:
    def __init__(
        self,
        *,
        adapter: Adapter,
        settings: Settings,
        resolver: DiscordIdentityResolver,
        webhook: DiscordWebhookClient,
        gateway: DiscordGatewayClient,
    ):
        self.adapter = adapter
        self.settings = settings
        self.resolver = resolver
        self.webhook = webhook
        self.gateway = gateway
        self._background_tasks: set[asyncio.Task] = set()
        self._stackchan_typing_tasks: dict[str, tuple[asyncio.Event, asyncio.Task]] = {}
        self._response_audio_chunks: dict[str, list[bytes]] = {}

    def register_response_hooks(self):
        @self.adapter.on_response
        async def post_user_utterance(response, _sts_response):
            if response.user_id and response.user_id != self.settings.discord_sync_user_id:
                return
            if response.type != "start":
                return
            metadata = response.metadata or {}
            if metadata.get("source") == "discord":
                return
            if metadata.get("suppress_discord_user_log"):
                return
            await self._start_stackchan_typing(response)
            text = metadata.get("recognized_text")
            if not text:
                return
            text = f"{self.settings.discord_voice_message_prefix}{text}"
            files = []
            if self._debug_report_enabled() and self.settings.debug_report_input_audio:
                files = await self._debug_request_audio_files(response)
            self._schedule(self._post_as(self.settings.discord_user_id, text, fallback_name="User", files=files))

        @self.adapter.on_response
        async def post_ai_response(response, _sts_response):
            if response.user_id and response.user_id != self.settings.discord_sync_user_id:
                return
            metadata = response.metadata or {}
            if metadata.get("source") == "discord":
                return
            self._collect_debug_response_audio(response, _sts_response)
            if response.type in ("final", "canceled"):
                await self._stop_stackchan_typing(response)
            if response.type == "canceled":
                await self._post_debug_filter_report(response)
                self._response_audio_chunks.pop(self._response_audio_key(response, _sts_response), None)
                return
            if response.type != "final":
                return
            text = response.voice_text or remove_control_tags(response.text)
            if not text:
                return
            text = f"{self._ai_response_prefix(metadata)}{text}"
            files = []
            if self._debug_report_enabled() and self.settings.debug_report_output_audio:
                files = self._debug_response_audio_files(response, _sts_response)
            self._schedule(self._post_as(self.settings.discord_bot_id, text, fallback_name="Bot", files=files))

    def _ai_response_prefix(self, metadata: dict) -> str:
        if metadata.get("source") == "avatar_speak" or metadata.get("speak_text") is not None:
            return self.settings.discord_api_message_prefix
        return self.settings.discord_voice_message_prefix

    def _schedule(self, coro):
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _post_as(self, discord_user_id: str, text: str, *, fallback_name: str, files: list[DiscordFile] = None):
        profile = await self.resolver.resolve(discord_user_id, fallback_name=fallback_name)
        await self.webhook.post(text, profile=profile, files=files)

    def _debug_report_enabled(self) -> bool:
        return bool(getattr(self.settings, "debug_report_enabled", False))

    def _response_audio_key(self, response, sts_response=None) -> str:
        return (
            getattr(sts_response, "transaction_id", None)
            or getattr(response, "transaction_id", None)
            or getattr(response, "context_id", None)
            or getattr(response, "session_id", None)
            or self.settings.discord_sync_user_id
        )

    def _collect_debug_response_audio(self, response, sts_response=None):
        if not self._debug_report_enabled() or not self.settings.debug_report_output_audio:
            return
        if response.type != "chunk":
            return
        audio_data = getattr(sts_response, "audio_data", None)
        if not audio_data:
            audio_data = getattr(response, "audio_data", None)
        if not isinstance(audio_data, bytes) or not audio_data:
            return
        key = self._response_audio_key(response, sts_response)
        self._response_audio_chunks.setdefault(key, []).append(audio_data)

    def _debug_response_audio_files(self, response, sts_response=None) -> list[DiscordFile]:
        key = self._response_audio_key(response, sts_response)
        chunks = self._response_audio_chunks.pop(key, [])
        return _build_audio_files(f"debug_output_{key}", chunks)

    async def _debug_request_audio_files(self, response) -> list[DiscordFile]:
        metadata = response.metadata or {}
        audio_info = metadata.get("debug_request_audio") or {}
        audio_id = audio_info.get("id")
        if not audio_id:
            return []
        recorder = getattr(getattr(self.adapter, "sts", None), "voice_recorder", None)
        if not recorder:
            return []
        try:
            data = await recorder.get_voice(audio_id)
        except Exception as ex:
            logger.warning("Failed to read debug request audio: %s", ex)
            return []
        if not data:
            return []
        ext = audio_info.get("format") or _audio_file_extension(data)
        return [DiscordFile(
            filename=audio_info.get("filename") or f"{audio_id}.{ext}",
            content=data,
            content_type=_audio_content_type(ext),
        )]

    def _should_report_filter(self, metadata: dict) -> bool:
        if not self._debug_report_enabled() or not self.settings.debug_report_filter_enabled:
            return False
        reason = _canonical_filter_reason(metadata)
        if not reason:
            return False
        configured = set(self.settings.debug_report_filter_reasons or [])
        return "*" in configured or reason in configured

    def _format_filter_report(self, metadata: dict) -> str:
        reason = _canonical_filter_reason(metadata) or "unknown"
        label = DEBUG_FILTER_LABELS.get(reason, reason)
        lines = [
            "[デバッグ] 応答を生成しませんでした",
            f"種類: {label}",
        ]
        _append_detail(lines, "認識結果", metadata.get("recognized_text") or metadata.get("request_text"))
        _append_detail(lines, "入力種別", metadata.get("input_type"))
        detail_reason = metadata.get("reason")
        if detail_reason and detail_reason != reason:
            _append_detail(lines, "詳細理由", detail_reason)

        audio_enhancement = metadata.get("audio_enhancement") or {}
        _append_detail(lines, "音声補正provider", audio_enhancement.get("provider"))
        _append_detail(lines, "音声補正error", audio_enhancement.get("error"))
        _append_detail(lines, "音声補正fail_open", audio_enhancement.get("fail_open"))

        voice_auth = metadata.get("voice_auth") or {}
        _append_detail(lines, "話者認証reason", voice_auth.get("reason"))
        _append_detail(lines, "request_user_id", voice_auth.get("request_user_id"))
        _append_detail(lines, "matched_voice_user_id", voice_auth.get("matched_voice_user_id"))
        _append_detail(lines, "similarity", voice_auth.get("similarity"))
        _append_detail(lines, "threshold", voice_auth.get("threshold"))

        addressing = metadata.get("addressing") or {}
        _append_detail(lines, "宛先判定reason", addressing.get("reason"))
        _append_detail(lines, "confidence", addressing.get("confidence"))
        _append_detail(lines, "explanation", addressing.get("explanation"))

        wakeword = metadata.get("wakeword") or {}
        _append_detail(lines, "ウェイクワードreason", wakeword.get("reason"))
        _append_detail(lines, "matched_wakeword", wakeword.get("matched_wakeword"))
        return "\n".join(lines)

    async def _post_debug_filter_report(self, response):
        metadata = response.metadata or {}
        if not self._should_report_filter(metadata):
            return
        files = []
        if self.settings.debug_report_filter_audio:
            files = await self._debug_request_audio_files(response)
        await self._post_as(
            self.settings.discord_bot_id,
            self._format_filter_report(metadata),
            fallback_name="Bot",
            files=files,
        )

    async def _refresh_typing_indicator(self, stop_event: asyncio.Event):
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=self.settings.discord_typing_indicator_interval,
                )
            except asyncio.TimeoutError:
                await self.gateway.trigger_typing()

    def _stackchan_typing_key(self, response) -> str:
        return (
            getattr(response, "session_id", None)
            or getattr(response, "context_id", None)
            or self.settings.discord_sync_user_id
        )

    async def _start_stackchan_typing(self, response):
        if not self.settings.discord_typing_indicator_enabled:
            return
        key = self._stackchan_typing_key(response)
        if key in self._stackchan_typing_tasks:
            return
        stop_event = asyncio.Event()
        await self.gateway.trigger_typing()
        task = asyncio.create_task(self._refresh_typing_indicator(stop_event))
        self._stackchan_typing_tasks[key] = (stop_event, task)

    async def _stop_stackchan_typing(self, response):
        key = self._stackchan_typing_key(response)
        typing_task = self._stackchan_typing_tasks.pop(key, None)
        if not typing_task:
            return
        stop_event, task = typing_task
        stop_event.set()
        await task

    async def handle_gateway_message(self, content: str, _message: dict):
        typing_stop_event = asyncio.Event()
        typing_task = None
        if self.settings.discord_typing_indicator_enabled:
            await self.gateway.trigger_typing()
            typing_task = asyncio.create_task(self._refresh_typing_indicator(typing_stop_event))
        try:
            response = await process_conversation_request(
                self.adapter,
                ChatRequest(
                    text=content,
                    session_id=self.settings.discord_gateway_session_id,
                    user_id=self.settings.discord_sync_user_id,
                    channel="discord",
                    delivery="text",
                    wait_in_queue=True,
                ),
            )
            text = response.voice_text or remove_control_tags(response.text)
            if text:
                await self._post_as(self.settings.discord_bot_id, text, fallback_name="Bot")
        finally:
            typing_stop_event.set()
            if typing_task:
                await typing_task

    def start(self):
        self.gateway.start()

    async def stop(self):
        for key in list(self._stackchan_typing_tasks):
            stop_event, task = self._stackchan_typing_tasks.pop(key)
            stop_event.set()
            await task
        await self.gateway.stop()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        await self.webhook.close()
        await self.resolver.close()


def setup_discord_integration(app: FastAPI, *, adapter: Adapter, settings: Settings) -> Optional[DiscordIntegration]:
    if not settings.discord_sync_enabled:
        return None

    missing = [
        name for name, value in {
            "DISCORD_BOT_TOKEN": settings.discord_bot_token,
            "DISCORD_CHANNEL_ID": settings.discord_channel_id,
            "DISCORD_WEBHOOK_URL": settings.discord_webhook_url,
            "DISCORD_USER_ID": settings.discord_user_id,
            "DISCORD_BOT_ID": settings.discord_bot_id,
        }.items()
        if not value
    ]
    if missing:
        logger.warning("Discord sync disabled because required settings are missing: %s", ", ".join(missing))
        return None

    resolver = DiscordIdentityResolver(
        bot_token=settings.discord_bot_token,
        guild_id=settings.discord_guild_id,
        cache_ttl=settings.discord_identity_cache_ttl,
    )
    webhook = DiscordWebhookClient(webhook_url=settings.discord_webhook_url)

    integration_ref: dict[str, DiscordIntegration] = {}

    async def on_gateway_message(content: str, message: dict):
        await integration_ref["integration"].handle_gateway_message(content, message)

    gateway = DiscordGatewayClient(
        bot_token=settings.discord_bot_token,
        channel_id=settings.discord_channel_id,
        on_message=on_gateway_message,
    )
    integration = DiscordIntegration(
        adapter=adapter,
        settings=settings,
        resolver=resolver,
        webhook=webhook,
        gateway=gateway,
    )
    integration_ref["integration"] = integration
    integration.register_response_hooks()

    if hasattr(app, "add_event_handler"):
        app.add_event_handler("startup", integration.start)
        app.add_event_handler("shutdown", integration.stop)
    else:
        app.router.on_startup.append(integration.start)
        app.router.on_shutdown.append(integration.stop)
    logger.info("Discord sync enabled for channel_id=%s", settings.discord_channel_id)
    return integration
