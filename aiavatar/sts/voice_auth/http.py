import base64
from typing import Optional

import httpx

from .base import VoiceAuthResult, VoiceAuthenticator


class HttpVoiceAuthenticator(VoiceAuthenticator):
    def __init__(
        self,
        *,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        debug: bool = False,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.debug = debug
        self.http_client = httpx.AsyncClient(timeout=httpx.Timeout(timeout))

    async def verify(
        self,
        *,
        user_id: Optional[str],
        audio_bytes: bytes,
        sample_rate: int,
        audio_duration: Optional[float] = None,
        threshold: Optional[float] = None,
        min_duration: Optional[float] = None,
    ) -> VoiceAuthResult:
        headers = self._headers()
        payload = {
            "user_id": user_id,
            "audio_data": base64.b64encode(audio_bytes).decode("ascii"),
            "sample_rate": sample_rate,
            "audio_duration": audio_duration,
        }
        if threshold is not None:
            payload["threshold"] = threshold
        if min_duration is not None:
            payload["min_duration"] = min_duration
        resp = await self.http_client.post(
            f"{self.base_url}/verify",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()
        return VoiceAuthResult(**resp.json())

    def verify_sync(
        self,
        *,
        user_id,
        audio_bytes,
        sample_rate,
        audio_duration=None,
        threshold=None,
        min_duration=None,
    ) -> VoiceAuthResult:
        raise RuntimeError("HttpVoiceAuthenticator.verify_sync is not supported; use verify().")

    async def enroll_wavs(self, *, user_id: str, wav_files: list[tuple[str, bytes]]):
        files = [
            ("files", (filename, data, "audio/wav"))
            for filename, data in wav_files
        ]
        resp = await self.http_client.post(
            f"{self.base_url}/enroll",
            headers=self._headers(),
            data={"user_id": user_id},
            files=files,
        )
        resp.raise_for_status()
        return resp.json()

    async def health(self):
        resp = await self.http_client.get(f"{self.base_url}/health", headers=self._headers())
        resp.raise_for_status()
        return resp.json()

    async def profiles(self):
        resp = await self.http_client.get(f"{self.base_url}/profiles", headers=self._headers())
        resp.raise_for_status()
        return resp.json()

    def _headers(self):
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    def get_config(self):
        return {
            "base_url": self.base_url,
            "debug": self.debug,
        }

    async def close(self):
        await self.http_client.aclose()
