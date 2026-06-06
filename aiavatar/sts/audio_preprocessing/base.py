from abc import ABC, abstractmethod
from typing import Optional


class StreamingAudioProcessor(ABC):
    @abstractmethod
    def process(
        self,
        audio_bytes: bytes,
        *,
        sample_rate: int,
        session_id: Optional[str] = None,
    ) -> bytes:
        ...

    def reset_session(self, session_id: str):
        pass

    def get_config(self) -> dict:
        return {}
