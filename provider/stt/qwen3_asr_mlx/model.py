import logging
import os
from pathlib import Path

DEFAULT_QWEN3_ASR_MODEL = "Qwen/Qwen3-ASR-0.6B"

logger = logging.getLogger(__name__)


def selected_model() -> str:
    model = os.environ.get("STT_MODEL") or DEFAULT_QWEN3_ASR_MODEL
    if os.environ.get("STT_PROVIDER") == "qwen3_asr_mlx" and model == "whisperkit":
        return DEFAULT_QWEN3_ASR_MODEL
    return model


def _expanded_path(value: str) -> Path:
    return Path(os.path.expanduser(value))


def is_local_model_reference(model: str) -> bool:
    return (
        model.startswith(".")
        or model.startswith("~")
        or model.startswith("/")
        or model.startswith("models/")
    )


def ensure_model_available(model: str | None = None) -> str:
    model = model or selected_model()

    if is_local_model_reference(model):
        model_path = _expanded_path(model)
        if model_path.exists():
            logger.info("Qwen3-ASR local model is available: %s", model_path)
            return str(model_path)

        repo_id = os.environ.get("STT_MODEL_REPO") or os.environ.get("STT_RUNTIME_MODEL_REPO")
        if not repo_id:
            raise FileNotFoundError(
                f"Qwen3-ASR local model path does not exist: {model_path}. "
                "Set STT_MODEL_REPO to download a Hugging Face model into that path."
            )

        logger.info("Downloading Qwen3-ASR model repo=%s target=%s", repo_id, model_path)
        try:
            from huggingface_hub import snapshot_download
        except Exception as ex:
            raise RuntimeError(
                "huggingface-hub is required to download Qwen3-ASR models. "
                "Run with `uv sync --extra qwen-stt`."
            ) from ex

        model_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(model_path),
            local_dir_use_symlinks=False,
        )
        return str(model_path)

    logger.info("Ensuring Qwen3-ASR Hugging Face model is cached: %s", model)
    try:
        from huggingface_hub import snapshot_download
    except Exception as ex:
        raise RuntimeError(
            "huggingface-hub is required to download Qwen3-ASR models. "
            "Run with `uv sync --extra qwen-stt`."
        ) from ex

    snapshot_download(repo_id=model)
    return model


def main():
    logging.basicConfig(level=logging.INFO)
    resolved = ensure_model_available()
    print(resolved)


if __name__ == "__main__":
    main()
