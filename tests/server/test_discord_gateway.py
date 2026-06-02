import asyncio
from types import SimpleNamespace

import pytest

from server.discord_gateway import (
    DiscordGatewayClient,
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

    async def post(self, url, json=None, headers=None):
        self.posts.append((url, json, headers))
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

    async def post(self, content, *, profile=None):
        self.posts.append((content, profile))


class FakeGateway:
    def __init__(self):
        self.typing_calls = 0

    def start(self):
        return None

    async def stop(self):
        return None

    async def trigger_typing(self):
        self.typing_calls += 1


def make_settings(**overrides):
    values = {
        "discord_sync_user_id": "robo-kanon-stack-chan",
        "discord_gateway_session_id": "discord:robo-kanon-stack-chan",
        "discord_user_id": "111",
        "discord_bot_id": "222",
        "discord_voice_message_prefix": "🎙️ ",
        "discord_typing_indicator_enabled": False,
        "discord_typing_indicator_interval": 8,
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
        )
    ]


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
