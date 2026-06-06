from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class WakewordDetection:
    accepted: bool
    reason: str
    source: str = "audio"
    matched_wakeword: Optional[str] = None
    score: Optional[float] = None
    threshold: Optional[float] = None
    detected_at: Optional[float] = None
    expires_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StreamingWakewordDetector(ABC):
    @abstractmethod
    def process(
        self,
        audio_bytes: bytes,
        *,
        sample_rate: int,
        session_id: Optional[str] = None,
    ) -> Optional[WakewordDetection]:
        ...

    def reset_session(self, session_id: str):
        pass

    def initial_decision(self) -> WakewordDetection:
        return WakewordDetection(
            accepted=False,
            reason="not_detected",
            metadata={"enabled": True},
        )

    def get_config(self) -> dict:
        return {}
