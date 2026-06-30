import pytest

from aiavatar.sts.voice_auth.http import HttpVoiceAuthenticator
from aiavatar.sts.addressing import OpenAICompatibleChatAddressingDetector
from aiavatar.sts.audio_enhancement import DeepFilterNetAudioEnhancer
from server.config import load_settings
from server.providers.audio_enhancement import create_audio_enhancer, create_required_audio_enhancer
from server.providers.addressing import create_addressing_detector
from server.providers import audio_wakeword as audio_wakeword_provider
from server.providers import pre_vad_noise_suppression as pre_vad_provider
from server.providers import vad as vad_provider
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


def test_ptt_gate_policy_defaults(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_API_KEY", raising=False)
    monkeypatch.delenv("PTT_VOICE_AUTH_ENABLED", raising=False)
    monkeypatch.delenv("PTT_WAKEWORD_ENABLED", raising=False)
    monkeypatch.delenv("PTT_ADDRESSING_ENABLED", raising=False)

    env_path = tmp_path / ".env"
    env_path.write_text("HERMES_API_KEY=test-hermes-key")

    settings = load_settings(env_path)

    assert settings.ptt_voice_auth_enabled is True
    assert settings.ptt_wakeword_enabled is False
    assert settings.ptt_addressing_enabled is False


def test_ptt_gate_policy_env_overrides(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_API_KEY", raising=False)
    monkeypatch.delenv("PTT_VOICE_AUTH_ENABLED", raising=False)
    monkeypatch.delenv("PTT_WAKEWORD_ENABLED", raising=False)
    monkeypatch.delenv("PTT_ADDRESSING_ENABLED", raising=False)

    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "HERMES_API_KEY=test-hermes-key",
                "PTT_VOICE_AUTH_ENABLED=false",
                "PTT_WAKEWORD_ENABLED=true",
                "PTT_ADDRESSING_ENABLED=true",
            ]
        )
    )

    settings = load_settings(env_path)

    assert settings.ptt_voice_auth_enabled is False
    assert settings.ptt_wakeword_enabled is True
    assert settings.ptt_addressing_enabled is True


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


def test_pre_vad_noise_suppression_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_API_KEY", raising=False)
    monkeypatch.delenv("PRE_VAD_NOISE_SUPPRESSION_ENABLED", raising=False)

    env_path = tmp_path / ".env"
    env_path.write_text("HERMES_API_KEY=test-hermes-key")

    processor = pre_vad_provider.create_pre_vad_audio_processor(load_settings(env_path))

    assert processor is None


def test_pre_vad_noise_suppression_provider_maps_to_webrtc_noise_gain(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_API_KEY", raising=False)
    monkeypatch.delenv("PRE_VAD_NOISE_SUPPRESSION_ENABLED", raising=False)
    monkeypatch.delenv("PRE_VAD_NOISE_SUPPRESSION_PROVIDER", raising=False)
    monkeypatch.delenv("PRE_VAD_NOISE_SUPPRESSION_LEVEL", raising=False)
    monkeypatch.delenv("PRE_VAD_AUTO_GAIN_DBFS", raising=False)
    monkeypatch.delenv("PRE_VAD_NOISE_SUPPRESSION_FAIL_OPEN", raising=False)

    class DummyWebRtcNoiseGainAudioProcessor:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        pre_vad_provider,
        "WebRtcNoiseGainAudioProcessor",
        DummyWebRtcNoiseGainAudioProcessor,
    )

    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "HERMES_API_KEY=test-hermes-key",
                "PRE_VAD_NOISE_SUPPRESSION_ENABLED=true",
                "PRE_VAD_NOISE_SUPPRESSION_PROVIDER=webrtc_noise_gain",
                "PRE_VAD_NOISE_SUPPRESSION_LEVEL=4",
                "PRE_VAD_AUTO_GAIN_DBFS=0",
                "PRE_VAD_NOISE_SUPPRESSION_FAIL_OPEN=false",
            ]
        )
    )

    processor = pre_vad_provider.create_pre_vad_audio_processor(load_settings(env_path))

    assert isinstance(processor, DummyWebRtcNoiseGainAudioProcessor)
    assert processor.kwargs["noise_suppression_level"] == 4
    assert processor.kwargs["auto_gain_dbfs"] == 0
    assert processor.kwargs["fail_open"] is False


def test_audio_wakeword_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_API_KEY", raising=False)
    monkeypatch.delenv("AUDIO_WAKEWORD_ENABLED", raising=False)

    env_path = tmp_path / ".env"
    env_path.write_text("HERMES_API_KEY=test-hermes-key")

    detector = audio_wakeword_provider.create_audio_wakeword_detector(load_settings(env_path))

    assert detector is None


def test_audio_wakeword_provider_maps_to_livekit_wakeword(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_API_KEY", raising=False)
    monkeypatch.delenv("AUDIO_WAKEWORD_ENABLED", raising=False)
    monkeypatch.delenv("AUDIO_WAKEWORD_PROVIDER", raising=False)
    monkeypatch.delenv("AUDIO_WAKEWORD_MODEL_PATHS", raising=False)
    monkeypatch.delenv("AUDIO_WAKEWORD_THRESHOLD", raising=False)
    monkeypatch.delenv("AUDIO_WAKEWORD_ACTIVATION_WINDOW", raising=False)
    monkeypatch.delenv("AUDIO_WAKEWORD_COOLDOWN", raising=False)

    class DummyLiveKitWakewordDetector:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setattr(
        audio_wakeword_provider,
        "LiveKitWakewordDetector",
        DummyLiveKitWakewordDetector,
    )

    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "HERMES_API_KEY=test-hermes-key",
                "AUDIO_WAKEWORD_ENABLED=true",
                "AUDIO_WAKEWORD_PROVIDER=livekit_wakeword",
                "AUDIO_WAKEWORD_MODEL_PATHS=models/wakewords/robo.onnx,models/wakewords/kano.onnx",
                "AUDIO_WAKEWORD_THRESHOLD=0.62",
                "AUDIO_WAKEWORD_ACTIVATION_WINDOW=5",
                "AUDIO_WAKEWORD_COOLDOWN=1.5",
            ]
        )
    )

    detector = audio_wakeword_provider.create_audio_wakeword_detector(load_settings(env_path))

    assert isinstance(detector, DummyLiveKitWakewordDetector)
    assert detector.kwargs["model_paths"] == ["models/wakewords/robo.onnx", "models/wakewords/kano.onnx"]
    assert detector.kwargs["threshold"] == 0.62
    assert detector.kwargs["activation_window"] == 5
    assert detector.kwargs["cooldown"] == 1.5


def test_vad_provider_maps_tenvad_from_env(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_API_KEY", raising=False)
    monkeypatch.delenv("VAD_PROVIDER", raising=False)
    monkeypatch.delenv("VAD_BASE_URL", raising=False)
    monkeypatch.delenv("VAD_THRESHOLD", raising=False)
    monkeypatch.delenv("VAD_NEG_THRESHOLD", raising=False)
    monkeypatch.delenv("VAD_MIN_SPEECH_MS", raising=False)
    monkeypatch.delenv("VAD_MIN_SILENCE_MS", raising=False)
    monkeypatch.delenv("VAD_SPEECH_PAD_MS", raising=False)
    monkeypatch.delenv("VAD_HOP_SIZE", raising=False)
    monkeypatch.delenv("VAD_SEGMENT_SILENCE_THRESHOLD", raising=False)

    class DummyTenVadStreamSpeechDetector:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def dummy_tenvad_factory(**kwargs):
        dummy_tenvad_factory.kwargs = kwargs
        return "tenvad-session-factory"

    dummy_tenvad_factory.kwargs = {}

    monkeypatch.setattr(
        vad_provider,
        "TenVadStreamSpeechDetector",
        DummyTenVadStreamSpeechDetector,
    )
    monkeypatch.setattr(vad_provider, "make_http_tenvad_session_factory", dummy_tenvad_factory)

    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "HERMES_API_KEY=test-hermes-key",
                "VAD_PROVIDER=tenvad",
                "VAD_BASE_URL=http://vad-runtime:8767",
                "VAD_THRESHOLD=0.28",
                "VAD_NEG_THRESHOLD=0.14",
                "VAD_MIN_SPEECH_MS=500",
                "VAD_MIN_SILENCE_MS=270",
                "VAD_SPEECH_PAD_MS=120",
                "VAD_HOP_SIZE=256",
                "VAD_SEGMENT_SILENCE_THRESHOLD=0.12",
            ]
        )
    )
    stt = object()

    detector = vad_provider.create_vad(load_settings(env_path), stt)

    assert isinstance(detector, DummyTenVadStreamSpeechDetector)
    assert dummy_tenvad_factory.kwargs["base_url"] == "http://vad-runtime:8767"
    assert dummy_tenvad_factory.kwargs["neg_threshold"] == 0.14
    assert detector.kwargs["speech_recognizer"] is stt
    assert detector.kwargs["segment_silence_threshold"] == 0.12
    assert detector.kwargs["silence_duration_threshold"] == 0.27
    assert detector.kwargs["min_duration"] == 0.5
    assert detector.kwargs["speech_probability_threshold"] == 0.28
    assert detector.kwargs["negative_speech_probability_threshold"] == 0.14
    assert detector.kwargs["hop_size"] == 256
    assert detector.kwargs["speech_pad_ms"] == 120
    assert detector.kwargs["ten_vad_class"] == "tenvad-session-factory"


def test_required_audio_enhancer_ignores_enabled_flag(tmp_path, monkeypatch):
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
                "AUDIO_ENHANCEMENT_ENABLED=false",
                "AUDIO_ENHANCEMENT_PROVIDER=deepfilternet",
                "AUDIO_ENHANCEMENT_MODEL=DeepFilterNet2",
                "AUDIO_ENHANCEMENT_TIMEOUT=12",
            ]
        )
    )

    enhancer = create_required_audio_enhancer(load_settings(env_path))

    assert isinstance(enhancer, DeepFilterNetAudioEnhancer)
    assert enhancer.command == "/usr/local/bin/deep-filter"
    assert enhancer.model == "DeepFilterNet2"
    assert enhancer.timeout == 12
