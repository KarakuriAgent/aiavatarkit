import asyncio
import base64
import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from aiavatar.sts.vad.tenvad import TenVadDecisionState
from server.config import Settings, load_settings
from server.logging_config import setup_logging

logger = logging.getLogger("aiavatar.provider.vad.tenvad")


class DetectRequest(BaseModel):
    session_id: str
    audio_data: str
    sample_rate: int = 16000
    hop_size: int = 256
    threshold: float = 0.5
    neg_threshold: Optional[float] = None


class ResetRequest(BaseModel):
    session_id: Optional[str] = None


@dataclass(frozen=True)
class TenVadSessionKey:
    hop_size: int
    threshold: float
    neg_threshold: Optional[float]


class TenVadRuntimeSession:
    def __init__(self, key: TenVadSessionKey, ten_vad_class):
        self.key = key
        self.ten_vad = ten_vad_class(key.hop_size, key.threshold)
        self.decision_state = TenVadDecisionState(key.threshold, key.neg_threshold)

    def process(self, audio: np.ndarray) -> dict:
        frames = len(audio) // self.key.hop_size
        speech_detected = False
        voiced_frames = 0
        probabilities = []
        flags = []

        for index in range(frames):
            start = index * self.key.hop_size
            frame = audio[start : start + self.key.hop_size]
            probability, flag = self.ten_vad.process(frame)
            probability = float(probability)
            active = self.decision_state.update(probability, bool(flag))
            probabilities.append(probability)
            flags.append(int(flag))
            if active:
                speech_detected = True
                voiced_frames += 1

        return {
            "speech_detected": speech_detected,
            "frames": frames,
            "voiced_frames": voiced_frames,
            "last_probability": probabilities[-1] if probabilities else 0.0,
            "max_probability": max(probabilities) if probabilities else 0.0,
            "flags": flags,
        }


class TenVadRuntime:
    def __init__(self, settings: Settings):
        try:
            from ten_vad import TenVad
        except ImportError as exc:
            raise RuntimeError(
                "ten-vad is required for the TenVAD runtime. Run `uv sync` on the host."
            ) from exc

        self.settings = settings
        self.ten_vad_class = TenVad
        self.sessions: dict[str, TenVadRuntimeSession] = {}
        self._lock = asyncio.Lock()
        logger.info(
            "TenVAD runtime loaded: threshold=%s neg_threshold=%s hop_size=%s min_speech_ms=%s min_silence_ms=%s speech_pad_ms=%s",
            settings.vad_threshold,
            settings.vad_neg_threshold,
            settings.vad_hop_size,
            settings.vad_min_speech_ms,
            settings.vad_min_silence_ms,
            settings.vad_speech_pad_ms,
        )

    def _session_key(self, request: DetectRequest) -> TenVadSessionKey:
        return TenVadSessionKey(
            hop_size=request.hop_size,
            threshold=request.threshold,
            neg_threshold=request.neg_threshold,
        )

    def _get_session(self, session_id: str, key: TenVadSessionKey) -> TenVadRuntimeSession:
        session = self.sessions.get(session_id)
        if session is None or session.key != key:
            session = TenVadRuntimeSession(key, self.ten_vad_class)
            self.sessions[session_id] = session
        return session

    async def detect(self, request: DetectRequest) -> dict:
        if request.sample_rate != 16000:
            raise HTTPException(status_code=400, detail="TenVAD requires 16000Hz audio")
        if request.hop_size <= 0:
            raise HTTPException(status_code=400, detail="hop_size must be positive")

        try:
            raw_audio = base64.b64decode(request.audio_data)
        except Exception as ex:
            raise HTTPException(status_code=400, detail="Invalid base64 audio_data") from ex
        if len(raw_audio) % 2:
            raise HTTPException(status_code=400, detail="audio_data must be 16-bit PCM")

        samples = np.frombuffer(raw_audio, dtype="<i2")
        complete_samples = (len(samples) // request.hop_size) * request.hop_size
        if complete_samples <= 0:
            return {
                "speech_detected": False,
                "frames": 0,
                "voiced_frames": 0,
                "last_probability": 0.0,
                "max_probability": 0.0,
                "ignored_samples": len(samples),
            }

        async with self._lock:
            session = self._get_session(request.session_id, self._session_key(request))
            result = session.process(samples[:complete_samples])

        result["ignored_samples"] = len(samples) - complete_samples
        return result

    async def reset(self, request: ResetRequest) -> dict:
        async with self._lock:
            if request.session_id:
                removed = 1 if self.sessions.pop(request.session_id, None) is not None else 0
            else:
                removed = len(self.sessions)
                self.sessions.clear()
        return {"ok": True, "removed_sessions": removed}

    def health(self) -> dict:
        return {
            "ok": True,
            "provider": "tenvad",
            "sessions": len(self.sessions),
            "threshold": self.settings.vad_threshold,
            "neg_threshold": self.settings.vad_neg_threshold,
            "hop_size": self.settings.vad_hop_size,
            "min_speech_ms": self.settings.vad_min_speech_ms,
            "min_silence_ms": self.settings.vad_min_silence_ms,
            "speech_pad_ms": self.settings.vad_speech_pad_ms,
        }


os.environ.setdefault("HERMES_API_KEY", "unused-tenvad-runtime")
settings = load_settings()
setup_logging()
runtime = TenVadRuntime(settings)

app = FastAPI()


async def require_api_key(authorization: str = Header(default=None)):
    api_key = settings.vad_api_key or settings.aiavatar_api_key
    if not api_key:
        return
    if not authorization or not authorization.lower().startswith("bearer ") or authorization[7:] != api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")


@app.get("/health", dependencies=[Depends(require_api_key)])
async def health():
    return runtime.health()


@app.post("/detect", dependencies=[Depends(require_api_key)])
async def detect(request: DetectRequest):
    return await runtime.detect(request)


@app.post("/reset", dependencies=[Depends(require_api_key)])
async def reset(request: ResetRequest):
    return await runtime.reset(request)


def main():
    host = settings.host
    port = int(os.environ.get("VAD_RUNTIME_PORT", "8767"))
    uvicorn.run(
        app,
        host=host,
        port=port,
        ssl_certfile=settings.ssl_cert_path,
        ssl_keyfile=settings.ssl_key_path,
    )


if __name__ == "__main__":
    main()
