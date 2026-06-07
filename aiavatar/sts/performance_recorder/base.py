from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class PerformanceRecord:
    transaction_id: str
    user_id: str = None
    context_id: str = None
    stt_name: str = None
    llm_name: str = None
    tts_name: str = None
    request_text: str = None
    response_text: str = None
    request_files: str = None
    response_voice_text: str = None
    voice_length: float = 0
    pre_vad_audio_processor_time: float = 0
    vad_final_stt_time: float = 0
    vad_segment_stt_time: float = 0
    vad_silence_time: float = 0
    vad_callback_to_pipeline_time: float = 0
    debug_request_audio_save_time: float = 0
    audio_enhancement_time: float = 0
    voice_auth_time: float = 0
    request_voice_record_time: float = 0
    stt_recognition_time: float = 0
    audio_wakeword_time: float = 0
    session_context_time: float = 0
    wakeword_decision_time: float = 0
    addressing_context_time: float = 0
    addressing_detection_time: float = 0
    validate_request_time: float = 0
    merge_request_time: float = 0
    context_prepare_time: float = 0
    accepted_notify_time: float = 0
    stop_response_phase_time: float = 0
    pre_llm_handler_time: float = 0
    llm_request_start_time: float = 0
    stt_time: float = 0
    stop_response_time: float = 0
    before_llm_time: float = 0
    llm_first_chunk_time: float = 0
    llm_first_voice_chunk_time: float = 0
    llm_time: float = 0
    tts_first_chunk_time: float = 0
    tts_time: float = 0
    total_time: float = 0
    quick_response_text: str = None
    error_info: str = None
    tool_calls: str = None


class PerformanceRecorder(ABC):
    @abstractmethod
    def record(self, record: PerformanceRecord):
        pass

    @abstractmethod
    def close(self):
        pass
