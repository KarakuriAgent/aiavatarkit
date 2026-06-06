import wave
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional
from uuid import uuid4

from server.config import Settings
from server.providers.audio_enhancement import create_required_audio_enhancer
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
        data["audio_url"] = f"/api/candidates/{self.id}/audio?variant=enhanced"
        data["raw_audio_url"] = f"/api/candidates/{self.id}/audio?variant=raw"
        return data


@dataclass
class VoiceAuthTestResult:
    id: str
    created_at: str
    user_id: str
    session_id: str
    duration: float
    sample_rate: int
    accepted: bool
    reason: str
    similarity: Optional[float]
    threshold: Optional[float]
    matched_user_id: Optional[str]
    metadata: dict

    def to_dict(self):
        return asdict(self)


class VoiceEnrollmentManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.root_dir = Path(settings.voice_auth_enrollment_dir)
        self.candidate_dir = self.root_dir / "candidates"
        self.enhanced_candidate_dir = self.root_dir / "candidates_enhanced"
        self.candidate_dir.mkdir(parents=True, exist_ok=True)
        self.enhanced_candidate_dir.mkdir(parents=True, exist_ok=True)
        self.audio_enhancer = create_required_audio_enhancer(settings)
        self.reading_enabled = False
        self.test_enabled = False
        self.test_user_id: Optional[str] = None
        self.test_threshold: Optional[float] = None
        self.test_min_duration: Optional[float] = None
        self._test_results: List[VoiceAuthTestResult] = []
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
        self.stop_test()

    def stop_reading(self):
        self.reading_enabled = False

    def start_test(
        self,
        user_id: str,
        threshold: Optional[float] = None,
        min_duration: Optional[float] = None,
    ):
        if not user_id:
            raise ValueError("user_id is required")
        if threshold is not None and threshold < 0:
            raise ValueError("threshold must be greater than or equal to 0")
        if min_duration is not None and min_duration < 0:
            raise ValueError("min_duration must be greater than or equal to 0")
        self.reading_enabled = False
        self.test_enabled = True
        self.test_user_id = user_id
        self.test_threshold = threshold
        self.test_min_duration = min_duration
        self._test_results.clear()

    def stop_test(self):
        self.test_enabled = False
        self.test_user_id = None
        self.test_threshold = None
        self.test_min_duration = None

    def state(self):
        return {
            "reading_enabled": self.reading_enabled,
            "candidate_count": len(self._candidates),
            "test_enabled": self.test_enabled,
            "test_user_id": self.test_user_id,
            "test_threshold": self.test_threshold,
            "test_min_duration": self.test_min_duration,
            "test_result_count": len(self._test_results),
            "last_test_result": self._test_results[-1].to_dict() if self._test_results else None,
            "default_threshold": self.settings.voice_auth_threshold,
            "default_min_duration": self.settings.voice_auth_min_duration,
        }

    def list_candidates(self):
        return [candidate.to_dict() for candidate in sorted(self._candidates.values(), key=lambda c: c.created_at)]

    async def list_profiles(self):
        if self.settings.voice_auth_provider in ("http", "wespeaker_mlx"):
            return await self._get_http_auth().profiles()
        raise ValueError(f"Unsupported VOICE_AUTH_PROVIDER: {self.settings.voice_auth_provider}")

    def list_test_results(self):
        return [result.to_dict() for result in reversed(self._test_results)]

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
            try:
                self._enhanced_candidate_path(candidate).unlink(missing_ok=True)
            except Exception:
                pass
        for path in self.enhanced_candidate_dir.glob("*.wav"):
            try:
                path.unlink(missing_ok=True)
            except Exception:
                pass
        self._candidates.clear()

    async def get_candidate_audio_path(self, candidate_id: str, variant: str = "enhanced") -> Path:
        candidate = self.get_candidate(candidate_id)
        if variant == "raw":
            return Path(candidate.path)
        if variant != "enhanced":
            raise ValueError(f"Unsupported candidate audio variant: {variant}")

        enhanced_path = self._enhanced_candidate_path(candidate)
        if enhanced_path.exists():
            return enhanced_path

        audio_bytes, sample_rate = self._read_wav(candidate.path)
        enhanced = await self.audio_enhancer.enhance(
            audio_bytes=audio_bytes,
            sample_rate=sample_rate,
            session_id=f"voice-auth-candidate-preview:{candidate.id}",
        )
        self._write_wav(enhanced_path, enhanced.audio_bytes, sample_rate)
        return enhanced_path

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

    async def add_test_result(
        self,
        *,
        audio_bytes: bytes,
        sample_rate: int,
        duration: float,
        session_id: str,
    ) -> Optional[VoiceAuthTestResult]:
        if not self.test_enabled or not self.test_user_id:
            return None

        user_id = self.test_user_id
        metadata = {}
        try:
            enhanced = await self.audio_enhancer.enhance(
                audio_bytes=audio_bytes,
                sample_rate=sample_rate,
                session_id=f"voice-auth-test:{user_id}:{session_id}",
            )
            metadata["audio_enhancement"] = enhanced.to_dict()
            result = await self._get_http_auth().verify(
                user_id=user_id,
                audio_bytes=enhanced.audio_bytes,
                sample_rate=sample_rate,
                audio_duration=duration,
                threshold=self.test_threshold,
                min_duration=self.test_min_duration,
            )
            test_result = VoiceAuthTestResult(
                id=uuid4().hex,
                created_at=datetime.now(timezone.utc).isoformat(),
                user_id=user_id,
                session_id=session_id,
                duration=duration,
                sample_rate=sample_rate,
                accepted=result.accepted,
                reason=result.reason,
                similarity=result.similarity,
                threshold=result.threshold,
                matched_user_id=result.matched_voice_user_id or result.matched_user_id,
                metadata={
                    **metadata,
                    "test_threshold": self.test_threshold,
                    "test_min_duration": self.test_min_duration,
                    **(result.metadata or {}),
                },
            )
        except Exception as ex:
            test_result = VoiceAuthTestResult(
                id=uuid4().hex,
                created_at=datetime.now(timezone.utc).isoformat(),
                user_id=user_id,
                session_id=session_id,
                duration=duration,
                sample_rate=sample_rate,
                accepted=False,
                reason="test_error",
                similarity=None,
                threshold=None,
                matched_user_id=None,
                metadata={**metadata, "error": str(ex)},
            )

        self._test_results.append(test_result)
        self._test_results = self._test_results[-50:]
        return test_result

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

    def _enhanced_candidate_path(self, candidate: CandidateVoice) -> Path:
        return self.enhanced_candidate_dir / Path(candidate.path).name

    @staticmethod
    def _read_wav(path: str | Path) -> tuple[bytes, int]:
        with wave.open(str(path), "rb") as wf:
            sample_rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
        return frames, sample_rate
