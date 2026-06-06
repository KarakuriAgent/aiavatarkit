import asyncio
import json
from types import SimpleNamespace

import pytest

from server.discord_gateway import (
    DiscordGatewayClient,
    DiscordFile,
    DiscordIdentityResolver,
    DiscordIntegration,
    DiscordProfile,
    DiscordWebhookClient,
    split_discord_content,
)


class FakeResponse:
    def __init__(self, status_code=200, data=None):
        self.status_code = status_code
        self._data = data or {}

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeHTTPClient:
    def __init__(self, get_responses=None):
        self.get_responses = get_responses or {}
        self.get_urls = []
        self.posts = []
        self.closed = False

    async def get(self, url, headers=None):
        self.get_urls.append((url, headers))
        return self.get_responses[url]

    async def post(self, url, json=None, headers=None, data=None, files=None):
        self.posts.append((url, json, headers, data, files))
        return FakeResponse()

    async def aclose(self):
        self.closed = True


class FakeAdapter:
    def __init__(self):
        self.response_handlers = []

    def on_response(self, handler):
        self.response_handlers.append(handler)
        return handler


class FakeResolver:
    async def resolve(self, user_id, *, fallback_name=None):
        return DiscordProfile(username=fallback_name or user_id, avatar_url=f"https://example.com/{user_id}.png")


class RecordingWebhook:
    def __init__(self):
        self.posts = []

    async def post(self, content, *, profile=None, files=None):
        self.posts.append((content, profile, files or []))


class FakeGateway:
    def __init__(self):
        self.typing_calls = 0

    def start(self):
        return None

    async def stop(self):
        return None

    async def trigger_typing(self):
        self.typing_calls += 1


class FakeVoiceRecorder:
    def __init__(self, voices=None):
        self.voices = voices or {}

    async def get_voice(self, audio_id):
        return self.voices.get(audio_id)


def make_settings(**overrides):
    values = {
        "discord_sync_user_id": "robo-kanon-stack-chan",
        "discord_gateway_session_id": "discord:robo-kanon-stack-chan",
        "discord_user_id": "111",
        "discord_bot_id": "222",
        "discord_voice_message_prefix": "🎙️ ",
        "discord_api_message_prefix": "📢 ",
        "discord_typing_indicator_enabled": False,
        "discord_typing_indicator_interval": 8,
        "debug_report_enabled": False,
        "debug_report_input_audio": True,
        "debug_report_output_audio": True,
        "debug_report_filter_enabled": True,
        "debug_report_filter_reasons": [
            "audio_enhancement_failed",
            "no_speech_recognized",
            "addressing_rejected",
            "validate_request_rejected",
            "wakeword_rejected",
        ],
        "debug_report_filter_audio": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.asyncio
async def test_discord_identity_resolver_prefers_guild_member_profile():
    user_id = "123456789012345678"
    guild_id = "987654321098765432"
    member_url = f"https://discord.com/api/v10/guilds/{guild_id}/members/{user_id}"
    fake_client = FakeHTTPClient({
        member_url: FakeResponse(data={
            "nick": "Guild Nick",
            "avatar": "guildavatar",
            "user": {
                "id": user_id,
                "username": "user-name",
                "global_name": "Global Name",
                "avatar": "useravatar",
            },
        })
    })
    resolver = DiscordIdentityResolver(
        bot_token="token",
        guild_id=guild_id,
        cache_ttl=300,
        http_client=fake_client,
    )

    profile = await resolver.resolve(user_id)
    cached = await resolver.resolve(user_id)

    assert profile.username == "Guild Nick"
    assert profile.avatar_url == (
        f"https://cdn.discordapp.com/guilds/{guild_id}/users/{user_id}/avatars/guildavatar.webp?size=128"
    )
    assert cached == profile
    assert len(fake_client.get_urls) == 1


@pytest.mark.asyncio
async def test_discord_identity_resolver_uses_user_profile_without_guild():
    user_id = "123456789012345678"
    user_url = f"https://discord.com/api/v10/users/{user_id}"
    fake_client = FakeHTTPClient({
        user_url: FakeResponse(data={
            "id": user_id,
            "username": "user-name",
            "global_name": "Global Name",
            "avatar": "useravatar",
        })
    })
    resolver = DiscordIdentityResolver(bot_token="token", http_client=fake_client)

    profile = await resolver.resolve(user_id)

    assert profile.username == "Global Name"
    assert profile.avatar_url == f"https://cdn.discordapp.com/avatars/{user_id}/useravatar.webp?size=128"


@pytest.mark.asyncio
async def test_discord_webhook_uses_profile_and_suppresses_mentions():
    fake_client = FakeHTTPClient()
    webhook = DiscordWebhookClient(
        webhook_url="https://discord.example/webhook",
        http_client=fake_client,
    )

    await webhook.post(
        "hello",
        profile=DiscordProfile(username="Bot Name", avatar_url="https://example.com/avatar.png"),
    )

    assert fake_client.posts == [
        (
            "https://discord.example/webhook",
            {
                "content": "hello",
                "allowed_mentions": {"parse": []},
                "username": "Bot Name",
                "avatar_url": "https://example.com/avatar.png",
            },
            None,
            None,
            None,
        )
    ]


@pytest.mark.asyncio
async def test_discord_webhook_posts_file_with_payload_json():
    fake_client = FakeHTTPClient()
    webhook = DiscordWebhookClient(
        webhook_url="https://discord.example/webhook",
        http_client=fake_client,
    )

    await webhook.post(
        "hello",
        profile=DiscordProfile(username="Bot Name"),
        files=[DiscordFile(filename="voice.wav", content=b"audio", content_type="audio/wav")],
    )

    url, json_body, headers, data, files = fake_client.posts[0]
    assert url == "https://discord.example/webhook"
    assert json_body is None
    assert headers is None
    assert json.loads(data["payload_json"]) == {
        "content": "hello",
        "allowed_mentions": {"parse": []},
        "username": "Bot Name",
    }
    assert files == [("files[0]", ("voice.wav", b"audio", "audio/wav"))]


@pytest.mark.asyncio
async def test_discord_gateway_triggers_typing_indicator():
    fake_client = FakeHTTPClient()
    gateway = DiscordGatewayClient(
        bot_token="token",
        channel_id="target",
        on_message=lambda _content, _message: None,
        http_client=fake_client,
    )

    await gateway.trigger_typing()

    assert fake_client.posts == [
        (
            "https://discord.com/api/v10/channels/target/typing",
            None,
            {"Authorization": "Bot token"},
            None,
            None,
        )
    ]


@pytest.mark.asyncio
async def test_discord_integration_prefixes_stackchan_voice_user_log():
    adapter = FakeAdapter()
    webhook = RecordingWebhook()
    integration = DiscordIntegration(
        adapter=adapter,
        settings=make_settings(
            discord_sync_user_id="robo-kanon-stack-chan",
            discord_user_id="111",
            discord_bot_id="222",
            discord_voice_message_prefix="🎙️ ",
        ),
        resolver=FakeResolver(),
        webhook=webhook,
        gateway=FakeGateway(),
    )
    integration.register_response_hooks()

    await adapter.response_handlers[0](
        SimpleNamespace(
            user_id="robo-kanon-stack-chan",
            type="start",
            metadata={"recognized_text": "こんにちは", "input_type": "audio"},
        ),
        None,
    )
    await asyncio.gather(*list(integration._background_tasks))

    assert webhook.posts[0][0] == "🎙️ こんにちは"


@pytest.mark.asyncio
async def test_discord_debug_input_audio_attaches_to_user_log():
    adapter = FakeAdapter()
    adapter.sts = SimpleNamespace(voice_recorder=FakeVoiceRecorder({
        "tx_debug_request": b"RIFF....WAVEaudio",
    }))
    webhook = RecordingWebhook()
    integration = DiscordIntegration(
        adapter=adapter,
        settings=make_settings(debug_report_enabled=True),
        resolver=FakeResolver(),
        webhook=webhook,
        gateway=FakeGateway(),
    )
    integration.register_response_hooks()

    await adapter.response_handlers[0](
        SimpleNamespace(
            user_id="robo-kanon-stack-chan",
            type="start",
            metadata={
                "recognized_text": "こんにちは",
                "input_type": "audio",
                "debug_request_audio": {
                    "id": "tx_debug_request",
                    "format": "wav",
                    "filename": "input.wav",
                },
            },
        ),
        None,
    )
    await asyncio.gather(*list(integration._background_tasks))

    content, _profile, files = webhook.posts[0]
    assert content == "🎙️ こんにちは"
    assert files[0].filename == "input.wav"
    assert files[0].content == b"RIFF....WAVEaudio"


@pytest.mark.asyncio
async def test_discord_integration_prefixes_stackchan_text_user_log():
    adapter = FakeAdapter()
    webhook = RecordingWebhook()
    integration = DiscordIntegration(
        adapter=adapter,
        settings=make_settings(
            discord_sync_user_id="robo-kanon-stack-chan",
            discord_user_id="111",
            discord_bot_id="222",
            discord_voice_message_prefix="🎙️ ",
        ),
        resolver=FakeResolver(),
        webhook=webhook,
        gateway=FakeGateway(),
    )
    integration.register_response_hooks()

    await adapter.response_handlers[0](
        SimpleNamespace(
            user_id="robo-kanon-stack-chan",
            type="start",
            metadata={"recognized_text": "こんにちは", "input_type": "text"},
        ),
        None,
    )
    await asyncio.gather(*list(integration._background_tasks))

    assert webhook.posts[0][0] == "🎙️ こんにちは"


@pytest.mark.asyncio
async def test_discord_integration_does_not_log_discord_source_as_voice():
    adapter = FakeAdapter()
    webhook = RecordingWebhook()
    integration = DiscordIntegration(
        adapter=adapter,
        settings=make_settings(
            discord_sync_user_id="robo-kanon-stack-chan",
            discord_user_id="111",
            discord_bot_id="222",
            discord_voice_message_prefix="🎙️ ",
        ),
        resolver=FakeResolver(),
        webhook=webhook,
        gateway=FakeGateway(),
    )
    integration.register_response_hooks()

    await adapter.response_handlers[0](
        SimpleNamespace(
            user_id="robo-kanon-stack-chan",
            type="start",
            metadata={"source": "discord", "recognized_text": "こんにちは"},
        ),
        None,
    )

    assert webhook.posts == []


@pytest.mark.asyncio
async def test_discord_integration_suppresses_internal_speak_prompt_log():
    adapter = FakeAdapter()
    webhook = RecordingWebhook()
    integration = DiscordIntegration(
        adapter=adapter,
        settings=make_settings(),
        resolver=FakeResolver(),
        webhook=webhook,
        gateway=FakeGateway(),
    )
    integration.register_response_hooks()

    await adapter.response_handlers[0](
        SimpleNamespace(
            user_id="robo-kanon-stack-chan",
            type="start",
            metadata={
                "source": "avatar_speak",
                "suppress_discord_user_log": True,
                "recognized_text": "外部通知として以下の内容をユーザーに伝えてください。",
            },
        ),
        None,
    )

    assert webhook.posts == []


@pytest.mark.asyncio
async def test_discord_integration_prefixes_stackchan_ai_response_log():
    adapter = FakeAdapter()
    webhook = RecordingWebhook()
    integration = DiscordIntegration(
        adapter=adapter,
        settings=make_settings(
            discord_sync_user_id="robo-kanon-stack-chan",
            discord_user_id="111",
            discord_bot_id="222",
            discord_voice_message_prefix="🎙️ ",
        ),
        resolver=FakeResolver(),
        webhook=webhook,
        gateway=FakeGateway(),
    )
    integration.register_response_hooks()

    await adapter.response_handlers[1](
        SimpleNamespace(
            user_id="robo-kanon-stack-chan",
            type="final",
            metadata={},
            voice_text="どうしましたか？",
            text="[face:neutral]どうしましたか？",
        ),
        None,
    )
    await asyncio.gather(*list(integration._background_tasks))

    assert webhook.posts[0][0] == "🎙️ どうしましたか？"


@pytest.mark.asyncio
async def test_discord_debug_output_audio_attaches_to_final_response_log():
    adapter = FakeAdapter()
    webhook = RecordingWebhook()
    integration = DiscordIntegration(
        adapter=adapter,
        settings=make_settings(debug_report_enabled=True),
        resolver=FakeResolver(),
        webhook=webhook,
        gateway=FakeGateway(),
    )
    integration.register_response_hooks()

    await adapter.response_handlers[1](
        SimpleNamespace(
            user_id="robo-kanon-stack-chan",
            type="chunk",
            metadata={},
            voice_text="どうしましたか？",
            text="どうしましたか？",
        ),
        SimpleNamespace(transaction_id="tx-output", audio_data=b"audio"),
    )
    await adapter.response_handlers[1](
        SimpleNamespace(
            user_id="robo-kanon-stack-chan",
            type="final",
            metadata={},
            voice_text="どうしましたか？",
            text="どうしましたか？",
        ),
        SimpleNamespace(transaction_id="tx-output"),
    )
    await asyncio.gather(*list(integration._background_tasks))

    content, _profile, files = webhook.posts[0]
    assert content == "🎙️ どうしましたか？"
    assert files[0].filename == "debug_output_tx-output_0.wav"
    assert files[0].content == b"audio"


@pytest.mark.asyncio
async def test_discord_integration_prefixes_avatar_speak_ai_response_log_with_api_prefix():
    adapter = FakeAdapter()
    webhook = RecordingWebhook()
    integration = DiscordIntegration(
        adapter=adapter,
        settings=make_settings(
            discord_sync_user_id="robo-kanon-stack-chan",
            discord_user_id="111",
            discord_bot_id="222",
            discord_voice_message_prefix="🎙️ ",
            discord_api_message_prefix="📢 ",
        ),
        resolver=FakeResolver(),
        webhook=webhook,
        gateway=FakeGateway(),
    )
    integration.register_response_hooks()

    await adapter.response_handlers[1](
        SimpleNamespace(
            user_id="robo-kanon-stack-chan",
            type="final",
            metadata={"source": "avatar_speak", "speak_text": "そろそろ休憩の時間です。"},
            voice_text="休憩してね。",
            text="[face:joy]休憩してね。",
        ),
        None,
    )
    await asyncio.gather(*list(integration._background_tasks))

    assert webhook.posts[0][0] == "📢 休憩してね。"


@pytest.mark.asyncio
async def test_discord_integration_sends_typing_indicator_for_stackchan_response():
    adapter = FakeAdapter()
    webhook = RecordingWebhook()
    gateway = FakeGateway()
    integration = DiscordIntegration(
        adapter=adapter,
        settings=make_settings(discord_typing_indicator_enabled=True),
        resolver=FakeResolver(),
        webhook=webhook,
        gateway=gateway,
    )
    integration.register_response_hooks()

    await adapter.response_handlers[0](
        SimpleNamespace(
            session_id="stackchan-session",
            user_id="robo-kanon-stack-chan",
            type="start",
            metadata={"recognized_text": "こんにちは"},
        ),
        None,
    )
    assert gateway.typing_calls == 1
    assert "stackchan-session" in integration._stackchan_typing_tasks

    await adapter.response_handlers[1](
        SimpleNamespace(
            session_id="stackchan-session",
            user_id="robo-kanon-stack-chan",
            type="final",
            metadata={},
            voice_text="どうしましたか？",
            text="[face:neutral]どうしましたか？",
        ),
        None,
    )
    await asyncio.gather(*list(integration._background_tasks))

    assert integration._stackchan_typing_tasks == {}
    assert [post[0] for post in webhook.posts] == ["🎙️ こんにちは", "🎙️ どうしましたか？"]


@pytest.mark.asyncio
async def test_discord_integration_does_not_log_discord_source_ai_response():
    adapter = FakeAdapter()
    webhook = RecordingWebhook()
    integration = DiscordIntegration(
        adapter=adapter,
        settings=make_settings(
            discord_sync_user_id="robo-kanon-stack-chan",
            discord_user_id="111",
            discord_bot_id="222",
            discord_voice_message_prefix="🎙️ ",
        ),
        resolver=FakeResolver(),
        webhook=webhook,
        gateway=FakeGateway(),
    )
    integration.register_response_hooks()

    await adapter.response_handlers[1](
        SimpleNamespace(
            user_id="robo-kanon-stack-chan",
            type="final",
            metadata={"source": "discord"},
            voice_text="どうしましたか？",
            text="[face:neutral]どうしましたか？",
        ),
        None,
    )

    assert webhook.posts == []


@pytest.mark.asyncio
async def test_discord_debug_filter_report_uses_japanese_label_and_details():
    adapter = FakeAdapter()
    adapter.sts = SimpleNamespace(voice_recorder=FakeVoiceRecorder({
        "tx_debug_request": b"RIFF....WAVEaudio",
    }))
    webhook = RecordingWebhook()
    integration = DiscordIntegration(
        adapter=adapter,
        settings=make_settings(
            debug_report_enabled=True,
            debug_report_filter_reasons=["addressing_rejected"],
        ),
        resolver=FakeResolver(),
        webhook=webhook,
        gateway=FakeGateway(),
    )
    integration.register_response_hooks()

    await adapter.response_handlers[1](
        SimpleNamespace(
            user_id="robo-kanon-stack-chan",
            type="canceled",
            metadata={
                "filter_reason": "addressing_rejected",
                "reason": "addressing_rejected",
                "recognized_text": "ただいま",
                "input_type": "audio",
                "addressing": {
                    "reason": "monologue",
                    "confidence": 0.91,
                    "explanation": "target name was not mentioned",
                },
                "debug_request_audio": {
                    "id": "tx_debug_request",
                    "format": "wav",
                    "filename": "input.wav",
                },
            },
        ),
        None,
    )

    content, _profile, files = webhook.posts[0]
    assert "種類: 宛先判定で除外" in content
    assert "認識結果: ただいま" in content
    assert "宛先判定reason: monologue" in content
    assert "confidence: 0.910" in content
    assert "explanation: target name was not mentioned" in content
    assert files[0].filename == "input.wav"


@pytest.mark.asyncio
async def test_discord_integration_sends_typing_indicator_while_processing_gateway_message(monkeypatch):
    adapter = FakeAdapter()
    webhook = RecordingWebhook()
    gateway = FakeGateway()
    integration = DiscordIntegration(
        adapter=adapter,
        settings=make_settings(
            discord_sync_user_id="robo-kanon-stack-chan",
            discord_gateway_session_id="discord:robo-kanon-stack-chan",
            discord_bot_id="222",
            discord_typing_indicator_enabled=True,
            discord_typing_indicator_interval=8,
        ),
        resolver=FakeResolver(),
        webhook=webhook,
        gateway=gateway,
    )

    async def fake_process_conversation_request(_adapter, request):
        assert request.channel == "discord"
        assert request.delivery == "text"
        await asyncio.sleep(0)
        return SimpleNamespace(voice_text="返信です", text="[face:neutral]返信です")

    monkeypatch.setattr(
        "server.discord_gateway.process_conversation_request",
        fake_process_conversation_request,
    )

    await integration.handle_gateway_message("hello", {"id": "message-id"})

    assert gateway.typing_calls == 1
    assert webhook.posts[0][0] == "返信です"


def test_split_discord_content_respects_limit():
    chunks = split_discord_content("a" * 2001)

    assert [len(c) for c in chunks] == [2000, 1]


@pytest.mark.asyncio
async def test_gateway_message_filtering():
    received = []

    async def on_message(content, message):
        received.append((content, message["id"]))

    gateway = DiscordGatewayClient(
        bot_token="token",
        channel_id="target",
        on_message=on_message,
    )

    await gateway._handle_message({"id": "1", "channel_id": "other", "content": "ignored", "author": {}})
    await gateway._handle_message({"id": "2", "channel_id": "target", "content": "ignored", "webhook_id": "wh", "author": {}})
    await gateway._handle_message({"id": "3", "channel_id": "target", "content": "ignored", "author": {"bot": True}})
    await gateway._handle_message({"id": "4", "channel_id": "target", "content": "   ", "author": {}})
    await gateway._handle_message({"id": "5", "channel_id": "target", "content": "hello", "author": {}})

    assert received == [("hello", "5")]
