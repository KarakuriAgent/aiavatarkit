import pytest

from aiavatar.sts.voice_auth.http import HttpVoiceAuthenticator
from server.config import load_settings
from server.providers.voice_auth import create_voice_auth


@pytest.mark.asyncio
async def test_wespeaker_mlx_voice_auth_maps_to_http_client(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_API_KEY", raising=False)
    monkeypatch.delenv("VOICE_AUTH_ENABLED", raising=False)
    monkeypatch.delenv("VOICE_AUTH_PROVIDER", raising=False)
    monkeypatch.delenv("VOICE_AUTH_BASE_URL", raising=False)

    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "HERMES_API_KEY=test-key",
                "VOICE_AUTH_ENABLED=true",
                "VOICE_AUTH_PROVIDER=wespeaker_mlx",
                "VOICE_AUTH_BASE_URL=http://voice-auth-runtime:8765",
            ]
        )
    )

    authenticator = create_voice_auth(load_settings(env_path))

    assert isinstance(authenticator, HttpVoiceAuthenticator)
    assert authenticator.base_url == "http://voice-auth-runtime:8765"
    await authenticator.close()
