import base64
import logging
import os
import tempfile
import wave
from pathlib import Path
from typing import List, Optional

import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel

from aiavatar.sts.voice_auth.wespeaker_mlx import WespeakerMlxVoiceAuthenticator
from server.config import Settings, load_settings
from server.logging_config import setup_logging

logger = logging.getLogger("aiavatar.provider.voice_auth.wespeaker_mlx")


class VerifyRequest(BaseModel):
    user_id: Optional[str] = None
    audio_data: str
    sample_rate: int
    audio_duration: Optional[float] = None


def read_wav_pcm(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
    if sample_width != 2:
        raise ValueError(f"{path}: only 16-bit PCM WAV is supported")
    if channels == 1:
        return frames, sample_rate
    samples = np.frombuffer(frames, dtype=np.int16).reshape(-1, channels)
    mono = samples.mean(axis=1).astype(np.int16)
    return mono.tobytes(), sample_rate


def ensure_model_available(settings: Settings) -> str:
    if not settings.voice_auth_model_path:
        raise RuntimeError("VOICE_AUTH_MODEL_PATH is required for wespeaker_mlx runtime")

    model_path = Path(settings.voice_auth_model_path)
    required_files = [
        model_path / "resnet_embedding.py",
        model_path / "weights.npz",
    ]
    if all(path.exists() for path in required_files):
        return str(model_path)

    missing = [str(path) for path in required_files if not path.exists()]
    if not settings.voice_auth_auto_download_model:
        raise FileNotFoundError(
            "Missing voice auth model files and VOICE_AUTH_AUTO_DOWNLOAD_MODEL=false: "
            + ", ".join(missing)
        )

    logger.info(
        "Voice auth model files are missing; downloading model: repo=%s target=%s missing=%s",
        settings.voice_auth_model_repo,
        model_path,
        missing,
    )
    try:
        from huggingface_hub import snapshot_download
    except Exception as ex:
        raise RuntimeError(
            "huggingface-hub is required to auto-download the voice auth model. "
            "Run with `uv sync --extra voice-auth`."
        ) from ex

    model_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=settings.voice_auth_model_repo,
        local_dir=str(model_path),
        local_dir_use_symlinks=False,
    )

    still_missing = [str(path) for path in required_files if not path.exists()]
    if still_missing:
        raise FileNotFoundError(
            "Downloaded voice auth model, but required files are still missing: "
            + ", ".join(still_missing)
        )
    return str(model_path)


class WespeakerMlxRuntime:
    def __init__(self, settings: Settings):
        self.settings = settings
        model_path = ensure_model_available(settings)
        logger.info(
            "Loading wespeaker_mlx runtime: model_path=%s profile_dir=%s threshold=%.3f",
            model_path,
            settings.voice_auth_profile_dir,
            settings.voice_auth_threshold,
        )
        self.authenticator = WespeakerMlxVoiceAuthenticator(
            model_path=model_path,
            profile_dir=settings.voice_auth_profile_dir,
            threshold=settings.voice_auth_threshold,
            min_duration=settings.voice_auth_min_duration,
            target_sample_rate=settings.voice_auth_sample_rate,
            require_user_id=settings.voice_auth_require_user_id,
            allow_identification=settings.voice_auth_allow_identification,
            allowed_users=settings.voice_auth_allowed_users,
            fail_open=settings.voice_auth_fail_open,
            apply_cmn=settings.voice_auth_apply_cmn,
            debug=settings.debug,
        )
        logger.info("wespeaker_mlx runtime loaded: profiles=%d", len(self.authenticator._profiles))

    async def verify(self, request: VerifyRequest):
        audio_bytes = base64.b64decode(request.audio_data)
        result = await self.authenticator.verify(
            user_id=request.user_id,
            audio_bytes=audio_bytes,
            sample_rate=request.sample_rate,
            audio_duration=request.audio_duration,
        )
        logger.info(
            "Voice auth verify: accepted=%s reason=%s request_user_id=%s matched_voice_user_id=%s similarity=%s threshold=%s duration=%s sample_rate=%s bytes=%d",
            result.accepted,
            result.reason,
            result.request_user_id,
            result.matched_voice_user_id,
            f"{result.similarity:.4f}" if result.similarity is not None else None,
            f"{result.threshold:.4f}" if result.threshold is not None else None,
            f"{request.audio_duration:.3f}" if request.audio_duration is not None else None,
            request.sample_rate,
            len(audio_bytes),
        )
        return result.to_dict()

    async def enroll(self, *, user_id: str, files: List[UploadFile]):
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        if not files:
            raise HTTPException(status_code=400, detail="files are required")

        samples = []
        with tempfile.TemporaryDirectory(prefix="voice-auth-enroll-") as tmpdir:
            for upload in files:
                path = Path(tmpdir) / upload.filename
                path.write_bytes(await upload.read())
                samples.append(read_wav_pcm(path))
        self.authenticator.enroll_user_from_pcm(user_id, samples)
        logger.info("Voice auth profile enrolled: user_id=%s samples=%d", user_id, len(samples))
        return {
            "user_id": user_id,
            "samples": len(samples),
            "profile_dir": str(self.authenticator.profile_dir),
        }

    def profiles(self):
        return {
            "profiles": sorted(self.authenticator._profiles.keys()),
            "allowed_users": sorted(self.authenticator.allowed_users),
        }

    def health(self):
        config = self.authenticator.get_config()
        return {
            "ok": True,
            "provider": "wespeaker_mlx",
            "profiles": config.get("profiles", 0),
            "allowed_users": config.get("allowed_users", []),
        }


os.environ.setdefault("HERMES_API_KEY", "unused-wespeaker-mlx-runtime")
settings = load_settings()
setup_logging()
runtime = WespeakerMlxRuntime(settings)

app = FastAPI()


async def require_api_key(authorization: str = Header(default=None)):
    api_key = settings.voice_auth_api_key or settings.aiavatar_api_key
    if not api_key:
        return
    if not authorization or not authorization.lower().startswith("bearer ") or authorization[7:] != api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")


@app.get("/health", dependencies=[Depends(require_api_key)])
async def health():
    return runtime.health()


@app.get("/profiles", dependencies=[Depends(require_api_key)])
async def profiles():
    return runtime.profiles()


@app.post("/verify", dependencies=[Depends(require_api_key)])
async def verify(request: VerifyRequest):
    return await runtime.verify(request)


@app.post("/enroll", dependencies=[Depends(require_api_key)])
async def enroll(user_id: str = Form(...), files: List[UploadFile] = File(...)):
    return await runtime.enroll(user_id=user_id, files=files)


def main():
    host = settings.host
    port = int(os.environ.get("VOICE_AUTH_RUNTIME_PORT", "8765"))
    uvicorn.run(
        app,
        host=host,
        port=port,
        ssl_certfile=settings.ssl_cert_path,
        ssl_keyfile=settings.ssl_key_path,
    )


if __name__ == "__main__":
    main()
