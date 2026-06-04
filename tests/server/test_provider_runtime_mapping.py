import pytest

from aiavatar.sts.voice_auth.http import HttpVoiceAuthenticator
from aiavatar.sts.addressing import OpenAICompatibleChatAddressingDetector
from aiavatar.sts.audio_enhancement import DeepFilterNetAudioEnhancer
from server.config import load_settings
from server.providers.audio_enhancement import create_audio_enhancer
from server.providers.addressing import create_addressing_detector
from server.providers.voice_auth import create_voice_auth


@pytest.mark.asyncio
async def test_wespeaker_mlx_voice_auth_maps_to_http_client(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_API_KEY", raising=False)
    monkeypatch.delenv("HERMES_BASE_URL", raising=False)
    monkeypatch.delenv("HERMES_MODEL", raising=False)
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


@pytest.mark.asyncio
async def test_addressing_provider_uses_hermes_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_API_KEY", raising=False)
    monkeypatch.delenv("ADDRESSING_ENABLED", raising=False)
    monkeypatch.delenv("ADDRESSING_TARGET_NAMES", raising=False)
    monkeypatch.delenv("ADDRESSING_BASE_URL", raising=False)
    monkeypatch.delenv("ADDRESSING_API_KEY", raising=False)
    monkeypatch.delenv("ADDRESSING_MODEL", raising=False)
    monkeypatch.delenv("ADDRESSING_PRIMARY_NAME", raising=False)

    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "HERMES_API_KEY=test-hermes-key",
                "HERMES_BASE_URL=http://hermes-runtime:8642/v1",
                "HERMES_MODEL=hermes-agent",
                "ADDRESSING_ENABLED=true",
                "ADDRESSING_TARGET_NAMES=カノン,スタックチャン",
            ]
        )
    )

    detector = create_addressing_detector(load_settings(env_path))

    assert isinstance(detector, OpenAICompatibleChatAddressingDetector)
    assert detector.base_url == "http://hermes-runtime:8642/v1"
    assert detector.model == "hermes-agent"
    assert detector.target_names == ["カノン", "スタックチャン"]
    assert detector.primary_name == "カノン"
    await detector.close()


@pytest.mark.asyncio
async def test_addressing_provider_override_does_not_fallback_to_hermes_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_API_KEY", raising=False)
    monkeypatch.delenv("ADDRESSING_ENABLED", raising=False)
    monkeypatch.delenv("ADDRESSING_TARGET_NAMES", raising=False)
    monkeypatch.delenv("ADDRESSING_BASE_URL", raising=False)
    monkeypatch.delenv("ADDRESSING_API_KEY", raising=False)
    monkeypatch.delenv("ADDRESSING_MODEL", raising=False)
    monkeypatch.delenv("ADDRESSING_PRIMARY_NAME", raising=False)

    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "HERMES_API_KEY=test-hermes-key",
                "HERMES_BASE_URL=http://hermes-runtime:8642/v1",
                "HERMES_MODEL=hermes-agent",
                "ADDRESSING_ENABLED=true",
                "ADDRESSING_BASE_URL=https://api.cerebras.ai/v1",
                "ADDRESSING_MODEL=gpt-oss-120b",
                "ADDRESSING_TARGET_NAMES=ロボ花音,ろぼかのん,ロボカノン,かのん,花音,カノン",
                "ADDRESSING_PRIMARY_NAME=ロボ花音",
            ]
        )
    )

    detector = create_addressing_detector(load_settings(env_path))

    assert isinstance(detector, OpenAICompatibleChatAddressingDetector)
    assert detector.base_url == "https://api.cerebras.ai/v1"
    assert detector.model == "gpt-oss-120b"
    assert detector.api_key is None
    assert detector.target_names == ["ロボ花音", "ろぼかのん", "ロボカノン", "かのん", "花音", "カノン"]
    assert detector.primary_name == "ロボ花音"
    await detector.close()


def test_audio_enhancement_provider_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_API_KEY", raising=False)
    monkeypatch.delenv("AUDIO_ENHANCEMENT_ENABLED", raising=False)

    env_path = tmp_path / ".env"
    env_path.write_text("HERMES_API_KEY=test-hermes-key")

    enhancer = create_audio_enhancer(load_settings(env_path))

    assert enhancer is None


def test_audio_enhancement_provider_maps_to_deepfilternet(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_API_KEY", raising=False)
    monkeypatch.delenv("AUDIO_ENHANCEMENT_ENABLED", raising=False)
    monkeypatch.delenv("AUDIO_ENHANCEMENT_PROVIDER", raising=False)
    monkeypatch.delenv("AUDIO_ENHANCEMENT_MODEL", raising=False)
    monkeypatch.delenv("AUDIO_ENHANCEMENT_TIMEOUT", raising=False)

    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "HERMES_API_KEY=test-hermes-key",
                "AUDIO_ENHANCEMENT_ENABLED=true",
                "AUDIO_ENHANCEMENT_PROVIDER=deepfilternet",
                "AUDIO_ENHANCEMENT_MODEL=DeepFilterNet2",
                "AUDIO_ENHANCEMENT_TIMEOUT=12",
            ]
        )
    )

    enhancer = create_audio_enhancer(load_settings(env_path))

    assert isinstance(enhancer, DeepFilterNetAudioEnhancer)
    assert enhancer.command == "/usr/local/bin/deep-filter"
    assert enhancer.model == "DeepFilterNet2"
    assert enhancer.timeout == 12
