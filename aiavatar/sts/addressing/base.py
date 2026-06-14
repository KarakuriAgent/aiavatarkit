import asyncio
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class AddressingDecision:
    accepted: bool
    reason: str
    confidence: Optional[float] = None
    explanation: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AddressingDetector(ABC):
    async def detect(
        self,
        *,
        text: str,
        recent_history: List[Dict[str, Any]] = None,
        seconds_since_last_assistant_turn: Optional[float] = None,
        recent_unaccepted_count: Optional[int] = None,
    ) -> AddressingDecision:
        return await asyncio.to_thread(
            self.detect_sync,
            text=text,
            recent_history=recent_history,
            seconds_since_last_assistant_turn=seconds_since_last_assistant_turn,
            recent_unaccepted_count=recent_unaccepted_count,
        )

    @abstractmethod
    def detect_sync(
        self,
        *,
        text: str,
        recent_history: List[Dict[str, Any]] = None,
        seconds_since_last_assistant_turn: Optional[float] = None,
        recent_unaccepted_count: Optional[int] = None,
    ) -> AddressingDecision:
        ...

    def get_config(self) -> Dict[str, Any]:
        return {}

    async def close(self):
        pass
