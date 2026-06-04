import asyncio
import logging
import os
import wave
from io import BytesIO
from time import perf_counter
from typing import Optional

import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile

from provider.stt.qwen3_asr_mlx.model import DEFAULT_QWEN3_ASR_MODEL, ensure_model_available, selected_model
from server.config import load_settings
from server.logging_config import setup_logging

logger = logging.getLogger("aiavatar.provider.stt.qwen3_asr_mlx")

QWEN3_LANGUAGE_ALIASES = {
    "ar": "Arabic",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fa": "Persian",
    "fi": "Finnish",
    "fil": "Filipino",
    "fr": "French",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "jp": "Japanese",
    "ko": "Korean",
    "mk": "Macedonian",
    "ms": "Malay",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "yue": "Cantonese",
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-hans": "Chinese",
}


def qwen_language(language: Optional[str]) -> Optional[str]:
    if not language:
        return None
    return QWEN3_LANGUAGE_ALIASES.get(language.lower(), language)


def read_wav_upload(data: bytes) -> tuple[np.ndarray, int]:
    with wave.open(BytesIO(data), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if sample_width != 2:
        raise ValueError("Only 16-bit PCM WAV is supported")

    samples = np.frombuffer(frames, dtype="<i2")
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return samples.astype(np.float32) / 32768.0, sample_rate


def log_text(text: str) -> str:
    return (text or "").replace("\n", "\\n")


class Qwen3ASRRuntime:
    def __init__(self):
        model = selected_model()
        resolved_model = ensure_model_available(model)
        self.model = resolved_model
        self.language = os.environ.get("STT_LANGUAGE", "ja")
        self.context = os.environ.get("STT_CONTEXT") or None
        max_new_tokens = os.environ.get("STT_MAX_NEW_TOKENS")
        self.max_new_tokens = int(max_new_tokens) if max_new_tokens else None
        self._lock = asyncio.Lock()

        try:
            from mlx_qwen3_asr import Session
        except ImportError as exc:
            raise RuntimeError(
                "mlx-qwen3-asr is required for qwen3_asr_mlx runtime. "
                "Run with `uv sync --extra qwen-stt`."
            ) from exc

        logger.info("Loading Qwen3-ASR MLX runtime: model=%s language=%s", self.model, self.language)
        self.session = Session(model=self.model)
        logger.info("Qwen3-ASR MLX runtime loaded")

    def _transcribe_sync(
        self,
        *,
        audio: np.ndarray,
        sample_rate: int,
        language: Optional[str],
        context: Optional[str],
        max_new_tokens: Optional[int],
    ) -> str:
        kwargs = {}
        selected_language = qwen_language(language or self.language)
        if selected_language:
            kwargs["language"] = selected_language
        selected_context = context if context is not None else self.context
        if selected_context:
            kwargs["context"] = selected_context
        selected_max_new_tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens
        if selected_max_new_tokens is not None:
            kwargs["max_new_tokens"] = selected_max_new_tokens

        result = self.session.transcribe((audio, sample_rate), **kwargs)
        return result.text

    async def transcribe(
        self,
        *,
        audio: np.ndarray,
        sample_rate: int,
        language: Optional[str],
        context: Optional[str],
        max_new_tokens: Optional[int],
    ) -> str:
        async with self._lock:
            return self._transcribe_sync(
                audio=audio,
                sample_rate=sample_rate,
                language=language,
                context=context,
                max_new_tokens=max_new_tokens,
            )

    def health(self):
        return {
            "ok": True,
            "provider": "qwen3_asr_mlx",
            "model": self.model,
            "language": self.language,
        }


os.environ.setdefault("HERMES_API_KEY", "unused-qwen3-asr-mlx-runtime")
settings = load_settings()
setup_logging()
runtime = Qwen3ASRRuntime()

app = FastAPI()


async def require_api_key(authorization: str = Header(default=None)):
    api_key = settings.stt_api_key or settings.aiavatar_api_key
    if not api_key:
        return
    if not authorization or not authorization.lower().startswith("bearer ") or authorization[7:] != api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")


@app.get("/health", dependencies=[Depends(require_api_key)])
async def health():
    return runtime.health()


@app.post("/v1/audio/transcriptions", dependencies=[Depends(require_api_key)])
@app.post("/audio/transcriptions", dependencies=[Depends(require_api_key)])
async def audio_transcriptions(
    file: UploadFile = File(...),
    model: Optional[str] = Form(default=None),
    language: Optional[str] = Form(default=None),
    context: Optional[str] = Form(default=None),
    max_new_tokens: Optional[int] = Form(default=None),
):
    upload_data = await file.read()
    try:
        audio, sample_rate = read_wav_upload(upload_data)
    except Exception as ex:
        raise HTTPException(status_code=400, detail=str(ex)) from ex

    if model and model != runtime.model:
        logger.warning("Ignoring per-request model=%s; runtime model=%s", model, runtime.model)

    started_at = perf_counter()
    text = await runtime.transcribe(
        audio=audio,
        sample_rate=sample_rate,
        language=language,
        context=context,
        max_new_tokens=max_new_tokens,
    )
    transcribe_ms = (perf_counter() - started_at) * 1000
    audio_duration = len(audio) / sample_rate if sample_rate else 0
    logger.info(
        "Qwen3-ASR transcription: model=%s language=%s sample_rate=%d audio_duration=%.3f upload_bytes=%d transcribe_ms=%.2f text=%s",
        runtime.model,
        language or runtime.language,
        sample_rate,
        audio_duration,
        len(upload_data),
        transcribe_ms,
        log_text(text),
    )
    return {"text": text}


def main():
    host = settings.host
    port = int(os.environ.get("STT_RUNTIME_PORT", "8766"))
    uvicorn.run(
        app,
        host=host,
        port=port,
        ssl_certfile=settings.ssl_cert_path,
        ssl_keyfile=settings.ssl_key_path,
    )


if __name__ == "__main__":
    main()
