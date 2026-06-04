import wave
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional
from uuid import uuid4

from server.config import Settings
from aiavatar.sts.voice_auth.http import HttpVoiceAuthenticator


@dataclass
class CandidateVoice:
    id: str
    created_at: str
    session_id: str
    user_id: Optional[str]
    duration: float
    sample_rate: int
    path: str

    def to_dict(self):
        data = asdict(self)
        data["audio_url"] = f"/api/candidates/{self.id}/audio"
        return data


class VoiceEnrollmentManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.root_dir = Path(settings.voice_auth_enrollment_dir)
        self.candidate_dir = self.root_dir / "candidates"
        self.candidate_dir.mkdir(parents=True, exist_ok=True)
        self.reading_enabled = False
        self._candidates: Dict[str, CandidateVoice] = {}
        self._http_auth: Optional[HttpVoiceAuthenticator] = None
        self._lock = Lock()
        self._load_existing_candidates()

    def _load_existing_candidates(self):
        for path in sorted(self.candidate_dir.glob("*.wav")):
            try:
                with wave.open(str(path), "rb") as wf:
                    sample_rate = wf.getframerate()
                    duration = wf.getnframes() / sample_rate if sample_rate else 0.0
                candidate_id = path.stem
                self._candidates[candidate_id] = CandidateVoice(
                    id=candidate_id,
                    created_at=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                    session_id="",
                    user_id=None,
                    duration=duration,
                    sample_rate=sample_rate,
                    path=str(path),
                )
            except Exception:
                continue

    def start_reading(self):
        self.reading_enabled = True

    def stop_reading(self):
        self.reading_enabled = False

    def state(self):
        return {
            "reading_enabled": self.reading_enabled,
            "candidate_count": len(self._candidates),
        }

    def list_candidates(self):
        return [candidate.to_dict() for candidate in sorted(self._candidates.values(), key=lambda c: c.created_at)]

    def get_candidate(self, candidate_id: str) -> CandidateVoice:
        candidate = self._candidates.get(candidate_id)
        if not candidate:
            raise KeyError(candidate_id)
        return candidate

    def clear_candidates(self):
        for candidate in list(self._candidates.values()):
            try:
                Path(candidate.path).unlink(missing_ok=True)
            except Exception:
                pass
        self._candidates.clear()

    def add_candidate(
        self,
        *,
        audio_bytes: bytes,
        sample_rate: int,
        duration: float,
        session_id: str,
        user_id: Optional[str],
    ) -> Optional[CandidateVoice]:
        if not self.reading_enabled:
            return None
        candidate_id = uuid4().hex
        path = self.candidate_dir / f"{candidate_id}.wav"
        self._write_wav(path, audio_bytes, sample_rate)
        candidate = CandidateVoice(
            id=candidate_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            session_id=session_id,
            user_id=user_id,
            duration=duration,
            sample_rate=sample_rate,
            path=str(path),
        )
        self._candidates[candidate_id] = candidate
        return candidate

    async def enroll(self, *, user_id: str, candidate_ids: List[str]):
        if not user_id:
            raise ValueError("user_id is required")
        if not candidate_ids:
            raise ValueError("candidate_ids is required")

        candidates = [self.get_candidate(candidate_id) for candidate_id in candidate_ids]
        if self.settings.voice_auth_provider in ("http", "wespeaker_mlx"):
            auth = self._get_http_auth()
            wav_files = [
                (Path(candidate.path).name, Path(candidate.path).read_bytes())
                for candidate in candidates
            ]
            return await auth.enroll_wavs(user_id=user_id, wav_files=wav_files)

        raise ValueError(f"Unsupported VOICE_AUTH_PROVIDER: {self.settings.voice_auth_provider}")

    def _get_http_auth(self) -> HttpVoiceAuthenticator:
        with self._lock:
            if self._http_auth:
                return self._http_auth
            if not self.settings.voice_auth_base_url:
                raise ValueError(f"VOICE_AUTH_BASE_URL is required when VOICE_AUTH_PROVIDER={self.settings.voice_auth_provider}")
            self._http_auth = HttpVoiceAuthenticator(
                base_url=self.settings.voice_auth_base_url,
                api_key=self.settings.voice_auth_api_key or self.settings.aiavatar_api_key,
                timeout=self.settings.stt_timeout,
                debug=self.settings.debug,
            )
            return self._http_auth

    @staticmethod
    def _write_wav(path: Path, audio_bytes: bytes, sample_rate: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_bytes)
