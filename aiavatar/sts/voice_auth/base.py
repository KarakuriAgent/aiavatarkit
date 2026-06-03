import asyncio
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class VoiceAuthResult:
    accepted: bool
    user_id: Optional[str] = None
    matched_user_id: Optional[str] = None
    request_user_id: Optional[str] = None
    matched_voice_user_id: Optional[str] = None
    similarity: Optional[float] = None
    threshold: Optional[float] = None
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.request_user_id is None:
            self.request_user_id = self.user_id
        if self.user_id is None:
            self.user_id = self.request_user_id
        if self.matched_voice_user_id is None:
            self.matched_voice_user_id = self.matched_user_id
        if self.matched_user_id is None:
            self.matched_user_id = self.matched_voice_user_id

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class VoiceAuthenticator(ABC):
    async def verify(
        self,
        *,
        user_id: Optional[str],
        audio_bytes: bytes,
        sample_rate: int,
        audio_duration: Optional[float] = None,
    ) -> VoiceAuthResult:
        return await asyncio.to_thread(
            self.verify_sync,
            user_id=user_id,
            audio_bytes=audio_bytes,
            sample_rate=sample_rate,
            audio_duration=audio_duration,
        )

    @abstractmethod
    def verify_sync(
        self,
        *,
        user_id: Optional[str],
        audio_bytes: bytes,
        sample_rate: int,
        audio_duration: Optional[float] = None,
    ) -> VoiceAuthResult:
        ...

    def get_config(self) -> Dict[str, Any]:
        return {}

    async def close(self):
        pass
