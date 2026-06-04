from server.config import load_settings
from server.providers.stt import WhisperCompatibleSpeechRecognizer, create_stt


def test_load_settings_uses_qwen3_default_model(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_API_KEY", raising=False)
    monkeypatch.delenv("STT_PROVIDER", raising=False)
    monkeypatch.delenv("STT_BASE_URL", raising=False)
    monkeypatch.delenv("STT_MODEL", raising=False)

    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "HERMES_API_KEY=test-key",
                "STT_PROVIDER=qwen3_asr_mlx",
                "STT_BASE_URL=http://stt-runtime:8766/v1",
            ]
        )
    )

    settings = load_settings(env_path)

    assert settings.stt_provider == "qwen3_asr_mlx"
    assert settings.stt_model == "Qwen/Qwen3-ASR-0.6B"


def test_load_settings_ignores_whisperkit_default_for_qwen3(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_API_KEY", raising=False)
    monkeypatch.delenv("STT_PROVIDER", raising=False)
    monkeypatch.delenv("STT_BASE_URL", raising=False)
    monkeypatch.delenv("STT_MODEL", raising=False)

    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "HERMES_API_KEY=test-key",
                "STT_PROVIDER=qwen3_asr_mlx",
                "STT_BASE_URL=http://stt-runtime:8766/v1",
                "STT_MODEL=whisperkit",
            ]
        )
    )

    settings = load_settings(env_path)

    assert settings.stt_model == "Qwen/Qwen3-ASR-0.6B"


def test_create_stt_maps_qwen3_asr_mlx_to_whisper_compatible_client(tmp_path, monkeypatch):
    monkeypatch.delenv("HERMES_API_KEY", raising=False)
    monkeypatch.delenv("STT_PROVIDER", raising=False)
    monkeypatch.delenv("STT_BASE_URL", raising=False)
    monkeypatch.delenv("STT_MODEL", raising=False)

    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "HERMES_API_KEY=test-key",
                "STT_PROVIDER=qwen3_asr_mlx",
                "STT_BASE_URL=http://stt-runtime:8766/v1",
                "STT_MODEL=custom/qwen3-asr",
            ]
        )
    )

    stt = create_stt(load_settings(env_path))

    assert isinstance(stt, WhisperCompatibleSpeechRecognizer)
    assert stt.base_url == "http://stt-runtime:8766/v1"
    assert stt.model == "custom/qwen3-asr"
