import asyncio
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class AudioEnhancementResult:
    audio_bytes: bytes
    provider: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.pop("audio_bytes", None)
        return data


class AudioEnhancer(ABC):
    async def enhance(
        self,
        *,
        audio_bytes: bytes,
        sample_rate: int,
        session_id: Optional[str] = None,
    ) -> AudioEnhancementResult:
        return await asyncio.to_thread(
            self.enhance_sync,
            audio_bytes=audio_bytes,
            sample_rate=sample_rate,
            session_id=session_id,
        )

    @abstractmethod
    def enhance_sync(
        self,
        *,
        audio_bytes: bytes,
        sample_rate: int,
        session_id: Optional[str] = None,
    ) -> AudioEnhancementResult:
        ...

    def get_config(self) -> Dict[str, Any]:
        return {}

    async def close(self):
        pass
