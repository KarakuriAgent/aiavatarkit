import asyncio
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import json
import logging
import inspect
import re
from time import monotonic, time
import traceback
from typing import AsyncGenerator, Tuple, List, Optional
from uuid import uuid4
from ..database import PoolProvider
from .models import STSRequest, STSResponse
from .vad import SpeechDetector
from .vad.silero import SileroSpeechDetector
from .stt import SpeechRecognizer
from .stt.google import GoogleSpeechRecognizer
from .llm import LLMService, LLMResponse
from .llm.chatgpt import ChatGPTService
from .llm.context_manager import ContextManager
from .tts import SpeechSynthesizer
from .tts.voicevox import VoicevoxSpeechSynthesizer
from .performance_recorder import PerformanceRecord, PerformanceRecorder
from .performance_recorder.sqlite import SQLitePerformanceRecorder
from .voice_recorder import VoiceRecorder, RequestVoice, ResponseVoices
from .voice_recorder.file import FileVoiceRecorder
from .audio_enhancement import AudioEnhancer
from .session_state_manager import SessionStateManager, SQLiteSessionStateManager
from .voice_auth import VoiceAuthenticator
from .addressing import AddressingDetector
from .wakeword import StreamingWakewordDetector

logger = logging.getLogger(__name__)

LANGUAGE_PATTERN = re.compile(r"""\[(?:lang|language):([a-zA-Z-]+)\]|<(?:lang|language)\s[^>]*code=["']([a-zA-Z-]+)["']""")


class STSPipeline:
    def __init__(
        self,
        *,
        vad: SpeechDetector = None,
        vad_volume_db_threshold: float = -90.0,
        vad_silence_duration_threshold: float = 0.5,
        vad_sample_rate: int = 16000,
        stt: SpeechRecognizer = None,
        stt_google_api_key: str = None,
        stt_sample_rate: int = 16000,
        llm: LLMService = None,
        llm_openai_api_key: str = None,
        llm_base_url: str = None,
        llm_model: str = "gpt-4o-mini",
        llm_system_prompt: str = None,
        llm_context_manager: ContextManager = None,
        tts: SpeechSynthesizer = None,
        tts_voicevox_url: str = "http://127.0.0.1:50021",
        tts_voicevox_speaker: int = 46,
        wakewords: List[str] = None,
        audio_wakeword_detector: StreamingWakewordDetector = None,
        merge_request_threshold: float = 0.0,
        merge_request_prefix: str = "$Previous user's request and your response have been canceled. Please respond again to the following request:\n\n",
        # Japanese version
        # merge_request_prefix: str = "$直前のユーザーの要求とあなたの応答はキャンセルされました。以下の要求に対して、あらためて応答しなおしてください:\n\n"
        timestamp_interval_seconds: float = 0.0,
        timestamp_prefix: str = "$Current date and time: ",
        timestamp_timezone: str = "UTC",
        db_pool_provider: PoolProvider = None,
        db_connection_str: str = "aiavatar.db",
        session_state_manager: SessionStateManager = None,
        performance_recorder: PerformanceRecorder = None,
        voice_recorder: VoiceRecorder = None,
        voice_recorder_enabled: bool = True,
        voice_recorder_dir: str = "recorded_voices",
        audio_enhancer: AudioEnhancer = None,
        audio_enhancement_fail_open: bool = True,
        audio_enhancement_record_raw: bool = True,
        audio_enhancement_record_enhanced: bool = True,
        voice_auth: VoiceAuthenticator = None,
        addressing_detector: AddressingDetector = None,
        addressing_history_limit: int = 12,
        invoke_queue_idle_timeout: float = 10.0,
        invoke_timeout: float = 60.0,
        use_invoke_queue: bool = False,
        insert_channel_tag: bool = False,
        skip_tts_channels: List[str] = None,
        debug_report_enabled: bool = False,
        debug: bool = False
    ):
        self.debug = debug
        self.debug_report_enabled = debug_report_enabled
        self.use_invoke_queue = use_invoke_queue

        # Channel
        self.insert_channel_tag = insert_channel_tag
        self.skip_tts_channels = skip_tts_channels or []

        # Database connection pool
        if db_pool_provider:
            if db_pool_provider.db_type == "postgresql":
                self.db_pool_provider = db_pool_provider
            else:
                raise ValueError(f"Unsupported db_type: {db_pool_provider.db_type}")
        elif db_connection_str.startswith("postgresql://"):
            from ..database.postgres import PostgreSQLPoolProvider
            self.db_pool_provider = PostgreSQLPoolProvider(connection_str=db_connection_str)
        else:
            self.db_pool_provider = None

        # Session state management
        if session_state_manager:
            self.session_state_manager = session_state_manager
        elif self.db_pool_provider:
            from .session_state_manager.postgres import PostgreSQLSessionStateManager
            self.session_state_manager = PostgreSQLSessionStateManager(get_pool=self.db_pool_provider.get_pool)
        else:
            self.session_state_manager = SQLiteSessionStateManager(db_path=db_connection_str)

        # VAD
        self.vad = vad or SileroSpeechDetector(
            volume_db_threshold=vad_volume_db_threshold,
            silence_duration_threshold=vad_silence_duration_threshold,
            sample_rate=vad_sample_rate,
            debug=debug
        )

        @self.vad.on_speech_detected
        async def on_speech_detected(data: bytes, text: str, metadata: dict, recorded_duration: float, session_id: str):
            request_metadata = metadata or {}
            pre_vad_audio_processor_time = self.vad.get_session_data(
                session_id,
                "_pre_vad_audio_processor_time",
            )
            if pre_vad_audio_processor_time:
                request_metadata = {
                    **request_metadata,
                    "_pre_vad_audio_processor_time": pre_vad_audio_processor_time,
                }
                self.vad.set_session_data(
                    session_id,
                    "_pre_vad_audio_processor_time",
                    0,
                )
            audio_wakeword = self.vad.get_session_data(session_id, "audio_wakeword")
            if audio_wakeword:
                request_metadata = {
                    **request_metadata,
                    "audio_wakeword": audio_wakeword,
                }
            async for response in self.invoke(STSRequest(
                session_id=session_id,
                user_id=self.vad.get_session_data(session_id, "user_id"),
                context_id=self.vad.get_session_data(session_id, "context_id"),
                channel=self.vad.get_session_data(session_id, "channel"),
                text=text,
                audio_data=data,
                audio_duration=recorded_duration,
                system_prompt_params=self.vad.get_session_data(session_id, "system_prompt_params"),
                metadata=request_metadata,
            )):
                if response.type == "start":
                    self.vad.set_session_data(session_id, "context_id", response.context_id)
                await self.handle_response(response)

        # Speech-to-Text
        self.stt = stt or GoogleSpeechRecognizer(
            google_api_key=stt_google_api_key,
            sample_rate=stt_sample_rate,
            debug=debug
        )

        # LLM
        if llm:
            self.llm = llm
        else:
            _context_managaer = None
            if llm_context_manager:
                _context_managaer = llm_context_manager
            else:
                if self.db_pool_provider:
                    from .llm.context_manager.postgres import PostgreSQLContextManager
                    _context_managaer = PostgreSQLContextManager(get_pool=self.db_pool_provider.get_pool)

            self.llm = ChatGPTService(
                openai_api_key=llm_openai_api_key,
                base_url=llm_base_url,
                model=llm_model,
                system_prompt=llm_system_prompt,
                context_manager=_context_managaer,
                db_connection_str=db_connection_str,
                debug=debug
            )

        # Text-to-Speech
        self.tts = tts or VoicevoxSpeechSynthesizer(
            base_url=tts_voicevox_url,
            speaker=tts_voicevox_speaker,
            debug=debug
        )

        # Wakeword
        self.wakewords = wakewords
        self.audio_wakeword_detector = audio_wakeword_detector

        # Merge consecutive requests
        self.merge_request_threshold = merge_request_threshold
        self.merge_request_prefix = merge_request_prefix

        # Validate request (ex: text too short, invalid files)
        self._validate_request = None

        # Timestamp
        self.timestamp_interval_seconds = timestamp_interval_seconds
        self.timestamp_timezone = timestamp_timezone
        self.timestamp_prefix = timestamp_prefix

        # Response handler
        self.handle_response = self.handle_response_default
        self.stop_response = self.stop_response_default
        self._process_llm_chunk = self.process_llm_chunk_default

        # Performance recorder
        if performance_recorder:
            self.performance_recorder = performance_recorder
        else:
            if self.db_pool_provider:
                from .performance_recorder.postgres import PostgreSQLPerformanceRecorder
                self.performance_recorder = PostgreSQLPerformanceRecorder(connection_str=self.db_pool_provider.connection_str)
            else:
                self.performance_recorder = SQLitePerformanceRecorder(db_path=db_connection_str)

        # Voice recorder
        self.voice_recorder = voice_recorder or FileVoiceRecorder(
            record_dir=voice_recorder_dir,
            sample_rate=stt_sample_rate
        )
        self.voice_recorder_enabled = voice_recorder_enabled
        self.voice_recorder_response_audio_format = "wav"

        # Audio enhancement
        self.audio_enhancer = audio_enhancer
        self.audio_enhancement_fail_open = audio_enhancement_fail_open
        self.audio_enhancement_record_raw = audio_enhancement_record_raw
        self.audio_enhancement_record_enhanced = audio_enhancement_record_enhanced

        # Voice authentication
        self.voice_auth = voice_auth

        # Addressing detection
        self.addressing_detector = addressing_detector
        self.addressing_history_limit = addressing_history_limit
        self.addressing_rejection_window = 60.0
        self._addressing_rejections: dict[str, List[float]] = {}

        # User custom logic
        self._on_before_llm_handlers = []
        self._on_before_tts_handlers = []
        self._on_accepted_handlers = []
        self._on_finish_handlers = []

        # Queue management for invoke_queued
        self._request_queues: dict[str, asyncio.Queue] = {}
        self._invoke_workers: dict[str, asyncio.Task] = {}
        self._response_queues: dict[str, dict[str, asyncio.Queue]] = {}
        self.invoke_queue_idle_timeout = invoke_queue_idle_timeout
        self.invoke_timeout = invoke_timeout

    def get_config(self) -> dict:
        return {
            "wakewords": self.wakewords,
            "audio_wakeword_detector": self.audio_wakeword_detector.get_config()
            if self.audio_wakeword_detector
            else None,
            "merge_request_threshold": self.merge_request_threshold,
            "merge_request_prefix": self.merge_request_prefix,
            "timestamp_interval_seconds": self.timestamp_interval_seconds,
            "timestamp_prefix": self.timestamp_prefix,
            "timestamp_timezone": self.timestamp_timezone,
            "voice_recorder_enabled": self.voice_recorder_enabled,
            "audio_enhancement_fail_open": self.audio_enhancement_fail_open,
            "audio_enhancement_record_raw": self.audio_enhancement_record_raw,
            "audio_enhancement_record_enhanced": self.audio_enhancement_record_enhanced,
            "addressing_history_limit": self.addressing_history_limit,
            "invoke_queue_idle_timeout": self.invoke_queue_idle_timeout,
            "invoke_timeout": self.invoke_timeout,
            "use_invoke_queue": self.use_invoke_queue,
            "debug_report_enabled": self.debug_report_enabled,
            "debug": self.debug,
        }

    def set_config(self, config: dict) -> dict:
        allowed_keys = self.get_config().keys()
        updated = {}
        for k, v in config.items():
            if v is None:
                continue
            if k not in allowed_keys:
                continue
            try:
                setattr(self, k, v)
                updated[k] = v
            except Exception:
                pass
        return updated

    def validate_request(self, func):
        self._validate_request = func
        return func

    def on_before_llm(self, func):
        self._on_before_llm_handlers.append(func)
        return func

    def on_before_tts(self, func):
        self._on_before_tts_handlers.append(func)
        return func

    def on_accepted(self, func):
        self._on_accepted_handlers.append(func)
        return func

    def on_finish(self, func):
        self._on_finish_handlers.append(func)
        return func

    async def process_audio_samples(self, samples: bytes, context_id: str):
        await self.vad.process_samples(samples, context_id)

    def process_llm_chunk(self, func) -> dict:
        self._process_llm_chunk = func
        return func

    async def process_llm_chunk_default(self, llm_stream_chunk: LLMResponse, session_id: str, user_id: str):
        return {}

    async def handle_response_default(self, response: STSResponse):
        logger.info(f"Handle response: {response}")

    async def stop_response_default(self, session_id: str, context_id: str):
        logger.info(f"Stop response: {session_id} / {context_id}")

    def is_awake(self, request: STSRequest, last_request_at: datetime) -> bool:
        return self.get_wakeword_decision(request, last_request_at)["accepted"]

    def get_wakeword_decision(self, request: STSRequest, last_request_at: datetime) -> dict:
        audio_decision = self._get_audio_wakeword_decision(request)

        if not self.wakewords and not audio_decision:
            return {
                "enabled": False,
                "accepted": True,
                "reason": "not_configured",
            }

        if audio_decision and audio_decision.get("accepted"):
            return audio_decision

        text = request.text or ""
        for ww in self.wakewords or []:
            if ww in text:
                logger.info(f"Wake by '{ww}': {request.text}")
                return {
                    "enabled": True,
                    "accepted": True,
                    "reason": "matched",
                    "matched_wakeword": ww,
                }

        if audio_decision:
            return audio_decision

        return {
            "enabled": True,
            "accepted": False,
            "reason": "not_detected",
        }

    def _get_audio_wakeword_decision(self, request: STSRequest) -> Optional[dict]:
        metadata = request.metadata or {}
        audio_wakeword = metadata.get("audio_wakeword")
        if not isinstance(audio_wakeword, dict):
            return None

        decision = {
            **audio_wakeword,
            "enabled": True,
            "source": "audio",
        }
        if not decision.get("accepted"):
            decision["accepted"] = False
            decision.setdefault("reason", "not_detected")
            return decision

        expires_at = decision.get("expires_at")
        if expires_at is not None:
            try:
                if monotonic() > float(expires_at):
                    return {
                        **decision,
                        "accepted": False,
                        "reason": "audio_expired",
                    }
            except Exception:
                pass
        decision.setdefault("reason", "matched")
        return decision

    def _detect_audio_wakeword(self, request: STSRequest, sample_rate: int):
        if not self.audio_wakeword_detector or not request.audio_data:
            return

        detection = self.audio_wakeword_detector.process(
            request.audio_data,
            sample_rate=sample_rate,
            session_id=request.session_id,
        )
        request.metadata = request.metadata or {}
        request.metadata["audio_wakeword"] = (
            detection.to_dict()
            if detection
            else self.audio_wakeword_detector.initial_decision().to_dict()
        )

    def _metadata_with_request(self, request: STSRequest, metadata: dict = None) -> dict:
        return {
            **(request.metadata or {}),
            **(metadata or {}),
        }

    def _apply_pre_pipeline_timings(
        self,
        request: STSRequest,
        performance: PerformanceRecord,
        start_time: float,
    ):
        metadata = request.metadata or {}
        timing_fields = {
            "_pre_vad_audio_processor_time": "pre_vad_audio_processor_time",
            "_vad_final_stt_time": "vad_final_stt_time",
            "_vad_segment_stt_time": "vad_segment_stt_time",
            "_vad_silence_time": "vad_silence_time",
        }
        for metadata_key, record_attr in timing_fields.items():
            value = metadata.pop(metadata_key, None)
            if value is None:
                continue
            try:
                setattr(performance, record_attr, float(value))
            except (TypeError, ValueError):
                continue

        detected_at = metadata.pop("_vad_detected_wall_time", None)
        if detected_at is not None:
            try:
                performance.vad_callback_to_pipeline_time = max(
                    0,
                    start_time - float(detected_at),
                )
            except (TypeError, ValueError):
                pass

    def _record_performance(
        self,
        performance: Optional[PerformanceRecord],
        start_time: float,
        *,
        filter_reason: str = None,
        error_info: dict = None,
    ):
        if not performance:
            return
        if not performance.total_time:
            performance.total_time = time() - start_time
        if filter_reason or error_info:
            payload = dict(error_info or {})
            if filter_reason:
                payload["filter_reason"] = filter_reason
            performance.error_info = json.dumps(payload, ensure_ascii=False, default=str)
        self.performance_recorder.record(performance)

    async def _save_debug_request_audio(self, transaction_id: str, audio_data: bytes) -> Optional[dict]:
        if not self.debug_report_enabled or not audio_data:
            return None
        audio_id = f"{transaction_id}_debug_request"
        voice_bytes = audio_data
        if not voice_bytes.startswith(b"RIFF"):
            voice_bytes = self.voice_recorder.create_wav_header(
                data_size=len(voice_bytes),
                sample_rate=self._request_audio_sample_rate(),
                channels=getattr(self.voice_recorder, "channels", 1),
                sample_width=getattr(self.voice_recorder, "sample_width", 2),
            ) + voice_bytes
        try:
            await self.voice_recorder.save_voice(audio_id, voice_bytes, "wav")
        except Exception as ex:
            logger.warning("Failed to save debug request audio: %s", ex, exc_info=self.debug)
            return None
        return {
            "id": audio_id,
            "format": "wav",
            "filename": f"{audio_id}.wav",
        }

    def _is_addressing_detection_required(self, request: STSRequest) -> bool:
        if not self.addressing_detector:
            return False
        if not request.audio_data:
            return False
        if request.metadata and request.metadata.get("skip_addressing"):
            return False
        return True

    async def _get_addressing_context(self, user_id: str) -> Tuple[List[dict], Optional[float]]:
        recent_history = await self.llm.context_manager.get_recent_histories(
            user_id=user_id,
            limit=self.addressing_history_limit,
            include_timestamp=True,
            fallback_to_global=True,
        )
        return recent_history, self._seconds_since_last_assistant_turn(recent_history)

    def _count_recent_addressing_rejections(self, user_id: str) -> int:
        timestamps = self._addressing_rejections.get(user_id)
        if not timestamps:
            return 0
        cutoff = time() - self.addressing_rejection_window
        timestamps[:] = [ts for ts in timestamps if ts >= cutoff]
        if not timestamps:
            self._addressing_rejections.pop(user_id, None)
            return 0
        return len(timestamps)

    def _record_addressing_rejection(self, user_id: str):
        self._addressing_rejections.setdefault(user_id, []).append(time())

    def _clear_addressing_rejections(self, user_id: str):
        self._addressing_rejections.pop(user_id, None)

    def _seconds_since_last_assistant_turn(self, recent_history: List[dict]) -> Optional[float]:
        for item in reversed(recent_history or []):
            if item.get("role") not in ("assistant", "model"):
                continue
            created_at = item.get("created_at")
            if not created_at:
                return None
            try:
                if isinstance(created_at, datetime):
                    dt = created_at
                else:
                    dt = datetime.fromisoformat(str(created_at))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return max(0.0, (datetime.now(timezone.utc) - dt.astimezone(timezone.utc)).total_seconds())
            except Exception:
                return None
        return None

    def _request_audio_sample_rate(self) -> int:
        return getattr(self.stt, "sample_rate", None) or getattr(self.vad, "sample_rate", 16000)

    def _audio_enhancer_provider_name(self) -> str:
        if not self.audio_enhancer:
            return ""
        try:
            config = self.audio_enhancer.get_config()
            if config.get("provider"):
                return config["provider"]
        except Exception:
            pass
        return self.audio_enhancer.__class__.__name__

    async def is_transaction_active(self, session_id: str, transaction_id: str) -> Tuple[bool, Optional[str]]:
        state = await self.session_state_manager.get_session_state(session_id)
        return state.active_transaction_id == transaction_id, state.active_transaction_id

    async def invoke(self, request: STSRequest) -> AsyncGenerator[STSResponse, None]:
        if self.use_invoke_queue:
            async for response in self._invoke_queued(request):
                yield response
        else:
            async for response in self._invoke_direct(request):
                yield response

    async def _invoke_direct(self, request: STSRequest) -> AsyncGenerator[STSResponse, None]:
        performance = None
        start_time = time()
        transaction_id = None
        try:
            if not request.session_id:
                raise ValueError("session_id is required but not provided")

            transaction_id = str(uuid4())
            suppress_adapter_response = (request.metadata or {}).get("suppress_adapter_response") is True
            audio_sample_rate = self._request_audio_sample_rate()
            performance = PerformanceRecord(
                transaction_id=transaction_id,
                user_id=request.user_id,
                stt_name=self.stt.__class__.__name__,
                llm_name=self.llm.__class__.__name__,
                tts_name=self.tts.__class__.__name__
            )
            self._apply_pre_pipeline_timings(request, performance, start_time)

            phase_started_at = time()
            debug_request_audio = await self._save_debug_request_audio(transaction_id, request.audio_data)
            performance.debug_request_audio_save_time = time() - phase_started_at
            if debug_request_audio:
                request.metadata = request.metadata or {}
                request.metadata["debug_request_audio"] = debug_request_audio

            if self.audio_enhancer and request.audio_data:
                request.metadata = request.metadata or {}
                if self.voice_recorder_enabled and self.audio_enhancement_record_raw:
                    phase_started_at = time()
                    await self.voice_recorder.record(RequestVoice(
                        transaction_id,
                        request.audio_data,
                        suffix="request_raw",
                    ))
                    performance.request_voice_record_time += time() - phase_started_at
                try:
                    phase_started_at = time()
                    enhancement_result = await self.audio_enhancer.enhance(
                        audio_bytes=request.audio_data,
                        sample_rate=audio_sample_rate,
                        session_id=request.session_id,
                    )
                    performance.audio_enhancement_time = time() - phase_started_at
                    request.audio_data = enhancement_result.audio_bytes
                    request.metadata["audio_enhancement"] = {
                        "enabled": True,
                        "applied": True,
                        **enhancement_result.to_dict(),
                    }
                    if self.voice_recorder_enabled and self.audio_enhancement_record_enhanced:
                        phase_started_at = time()
                        await self.voice_recorder.record(RequestVoice(
                            transaction_id,
                            request.audio_data,
                            suffix="request_enhanced",
                        ))
                        performance.request_voice_record_time += time() - phase_started_at
                except Exception as ex:
                    performance.audio_enhancement_time = time() - phase_started_at
                    request.metadata["audio_enhancement"] = {
                        "enabled": True,
                        "applied": False,
                        "provider": self._audio_enhancer_provider_name(),
                        "error": str(ex),
                        "fail_open": self.audio_enhancement_fail_open,
                    }
                    logger.warning("Audio enhancement failed: %s", ex, exc_info=self.debug)
                    if not self.audio_enhancement_fail_open:
                        self._record_performance(
                            performance,
                            start_time,
                            filter_reason="audio_enhancement_failed",
                        )
                        yield STSResponse(
                            type="canceled",
                            session_id=request.session_id,
                            user_id=request.user_id,
                            context_id=request.context_id,
                            metadata=self._metadata_with_request(request, {
                                "filter_reason": "audio_enhancement_failed",
                                "reason": "audio_enhancement_failed",
                                "audio_enhancement": request.metadata["audio_enhancement"],
                            }),
                        )
                        return

            if self.voice_auth and request.audio_data:
                try:
                    phase_started_at = time()
                    voice_auth_result = await self.voice_auth.verify(
                        user_id=request.user_id,
                        audio_bytes=request.audio_data,
                        sample_rate=audio_sample_rate,
                        audio_duration=request.audio_duration,
                    )
                    performance.voice_auth_time = time() - phase_started_at
                except Exception as ex:
                    performance.voice_auth_time = time() - phase_started_at
                    logger.warning("Voice authentication failed: %s", ex, exc_info=self.debug)
                    voice_auth_result = None

                if voice_auth_result is None or not voice_auth_result.accepted:
                    metadata = {
                        "filter_reason": "voice_auth_rejected",
                        "reason": "voice_auth_rejected",
                        "voice_auth": voice_auth_result.to_dict() if voice_auth_result else {
                            "accepted": False,
                            "user_id": request.user_id,
                            "request_user_id": request.user_id,
                            "matched_user_id": None,
                            "matched_voice_user_id": None,
                            "reason": "voice_auth_error",
                        },
                    }
                    if self.debug:
                        voice_auth = metadata["voice_auth"]
                        logger.info(
                            "Voice authentication rejected request: reason=%s request_user_id=%s matched_voice_user_id=%s similarity=%s threshold=%s",
                            voice_auth.get("reason"),
                            voice_auth.get("request_user_id"),
                            voice_auth.get("matched_voice_user_id"),
                            voice_auth.get("similarity"),
                            voice_auth.get("threshold"),
                        )
                    self._record_performance(
                        performance,
                        start_time,
                        filter_reason="voice_auth_rejected",
                    )
                    yield STSResponse(
                        type="canceled",
                        session_id=request.session_id,
                        user_id=request.user_id,
                        context_id=request.context_id,
                        metadata=self._metadata_with_request(request, metadata),
                    )
                    return

                request.metadata = request.metadata or {}
                request.metadata["voice_auth"] = voice_auth_result.to_dict()

            # Record request voice
            if self.voice_recorder_enabled and request.audio_data:
                phase_started_at = time()
                await self.voice_recorder.record(RequestVoice(transaction_id, request.audio_data))
                performance.request_voice_record_time += time() - phase_started_at

            if request.text:
                # Use text if exist
                input_type = "text"
                recognized_text = request.text
                if self.debug:
                    logger.info(f"Use text in request: {recognized_text}")
            elif request.audio_data:
                # Speech-to-Text
                input_type = "audio"
                phase_started_at = time()
                recognized_text = (await self.stt.recognize(request.session_id, request.audio_data)).text
                performance.stt_recognition_time = time() - phase_started_at
                if not recognized_text:
                    if self.debug:
                        logger.info("No speech recognized.")
                    self._record_performance(
                        performance,
                        start_time,
                        filter_reason="no_speech_recognized",
                    )
                    yield STSResponse(
                        type="canceled",
                        session_id=request.session_id,
                        user_id=request.user_id,
                        context_id=request.context_id,
                        metadata=self._metadata_with_request(request, {
                            "filter_reason": "no_speech_recognized",
                            "reason": "No speech recognized.",
                            "input_type": input_type,
                        })
                    )
                    return
                if self.debug:
                    logger.info(f"Recognized text from request: {recognized_text}")
            else:
                input_type = "empty"
                recognized_text = ""    # Request without both text and audio (e.g. image only)

            # Insert channel tag
            if self.insert_channel_tag and request.channel:
                channel_tag = f"<channel name='{request.channel}' />"
                request.text = f"{channel_tag}{recognized_text}" if recognized_text else channel_tag
            else:
                request.text = recognized_text

            phase_started_at = time()
            self._detect_audio_wakeword(request, audio_sample_rate)
            performance.audio_wakeword_time = time() - phase_started_at

            # Get session state before gates that need conversation recency.
            phase_started_at = time()
            state = await self.session_state_manager.get_session_state(request.session_id)
            now = datetime.now(timezone.utc)
            last_created_at = await self.llm.context_manager.get_last_created_at(request.context_id)
            performance.session_context_time = time() - phase_started_at

            phase_started_at = time()
            wakeword_decision = self.get_wakeword_decision(request, last_created_at)
            performance.wakeword_decision_time = time() - phase_started_at
            if wakeword_decision["enabled"]:
                request.metadata = request.metadata or {}
                request.metadata["wakeword"] = wakeword_decision
                if self.debug:
                    logger.info(
                        "Wakeword decision: accepted=%s reason=%s text=%s",
                        wakeword_decision.get("accepted"),
                        wakeword_decision.get("reason"),
                        recognized_text,
                    )

            skip_addressing_by_wakeword = wakeword_decision["enabled"] and wakeword_decision["accepted"]
            addressing_required = self._is_addressing_detection_required(request)

            if not wakeword_decision["accepted"] and not addressing_required:
                self._record_performance(
                    performance,
                    start_time,
                    filter_reason="wakeword_rejected",
                )
                yield STSResponse(
                    type="canceled",
                    session_id=request.session_id,
                    user_id=request.user_id,
                    context_id=request.context_id,
                    metadata=self._metadata_with_request(request, {
                        "filter_reason": "wakeword_rejected",
                        "reason": "wakeword_rejected",
                        "wakeword": wakeword_decision,
                        "recognized_text": recognized_text,
                        "input_type": input_type,
                    }),
                )
                return

            if addressing_required and skip_addressing_by_wakeword:
                request.metadata = request.metadata or {}
                request.metadata["addressing"] = {
                    "skipped": True,
                    "reason": "wakeword_accepted",
                }
                self._clear_addressing_rejections(request.user_id)
                if self.debug:
                    logger.info("Addressing skipped by wakeword: text=%s", recognized_text)

            if addressing_required and not skip_addressing_by_wakeword:
                phase_started_at = time()
                recent_history, seconds_since_last_assistant_turn = await self._get_addressing_context(request.user_id)
                performance.addressing_context_time = time() - phase_started_at
                phase_started_at = time()
                addressing_decision = await self.addressing_detector.detect(
                    text=recognized_text,
                    recent_history=recent_history,
                    seconds_since_last_assistant_turn=seconds_since_last_assistant_turn,
                    recent_unaccepted_count=self._count_recent_addressing_rejections(request.user_id),
                )
                performance.addressing_detection_time = time() - phase_started_at
                if self.debug:
                    logger.info(
                        "Addressing decision: accepted=%s reason=%s confidence=%s explanation=%s text=%s",
                        addressing_decision.accepted,
                        addressing_decision.reason,
                        addressing_decision.confidence,
                        addressing_decision.explanation,
                        recognized_text,
                    )
                request.metadata = request.metadata or {}
                request.metadata["addressing"] = addressing_decision.to_dict()
                if addressing_decision.accepted:
                    self._clear_addressing_rejections(request.user_id)
                else:
                    self._record_addressing_rejection(request.user_id)
                    self._record_performance(
                        performance,
                        start_time,
                        filter_reason="addressing_rejected",
                    )
                    yield STSResponse(
                        type="canceled",
                        session_id=request.session_id,
                        user_id=request.user_id,
                        context_id=request.context_id,
                        metadata=self._metadata_with_request(request, {
                            "filter_reason": "addressing_rejected",
                            "reason": "addressing_rejected",
                            "addressing": addressing_decision.to_dict(),
                            "recognized_text": recognized_text,
                            "input_type": input_type,
                        }),
                    )
                    return

            if self._validate_request:
                phase_started_at = time()
                if reason := await self._validate_request(request):
                    performance.validate_request_time = time() - phase_started_at
                    if self.debug:
                        logger.info(f"Invalid request: {request.text} / reason: {reason}")
                    self._record_performance(
                        performance,
                        start_time,
                        filter_reason="validate_request_rejected",
                    )
                    yield STSResponse(
                        type="canceled",
                        session_id=request.session_id,
                        user_id=request.user_id,
                        context_id=request.context_id,
                        metadata=self._metadata_with_request(request, {
                            "filter_reason": "validate_request_rejected",
                            "reason": reason,
                            "recognized_text": recognized_text,
                            "input_type": input_type,
                        })
                    )
                    return
                performance.validate_request_time = time() - phase_started_at

            performance.request_text = request.text
            performance.request_files = json.dumps(request.files or [], ensure_ascii=False)
            performance.voice_length = request.audio_duration
            performance.stt_time = time() - start_time

            # Merge consecutive requests
            phase_started_at = time()
            if self.merge_request_threshold > 0 and request.allow_merge:
                if state.previous_request_timestamp:
                    requests_interval = (now - state.previous_request_timestamp).total_seconds()
                    if self.merge_request_threshold > requests_interval:
                        logger.info(f"Merge consecutive requests: Interval {requests_interval} < Threshold {self.merge_request_threshold}")
                        previous_request_text = (state.previous_request_text or "").replace(self.merge_request_prefix, "")
                        request.text = f"{self.merge_request_prefix}{previous_request_text}\n{request.text}"
                        request.files = request.files or state.previous_request_files
                await self.session_state_manager.update_previous_request(
                    request.session_id, now, request.text, request.files
                )
            performance.merge_request_time = time() - phase_started_at

            # Get context
            phase_started_at = time()
            if request.context_id:
                if last_created_at == datetime.min.replace(tzinfo=timezone.utc):
                    logger.info(f"Invalid context_id: {request.context_id}")
                    request.context_id = None

            if not request.context_id:
                request.context_id = str(uuid4())
                logger.info(f"Create new context_id: {request.context_id}")

            # Insert timestamp
            if self.timestamp_interval_seconds > 0 and (now - state.timestamp_inserted_at).total_seconds() > self.timestamp_interval_seconds:
                now_str = datetime.now(ZoneInfo(self.timestamp_timezone)).strftime("%Y/%m/%d %H:%M:%S")
                request.text = f"{self.timestamp_prefix}{now_str}\n\n{request.text}"
                timestamp_inserted_at = now
            else:
                timestamp_inserted_at = state.timestamp_inserted_at

            # Overwrite active transaction
            if self.debug:
                logger.info(f"Start transaction: {transaction_id} {request.text} (previous: {state.active_transaction_id})")
            await self.session_state_manager.update_transaction(request.session_id, transaction_id, timestamp_inserted_at)
            performance.context_prepare_time = time() - phase_started_at

            performance.context_id = request.context_id

            # Notify client that request is accepted only after request gates pass.
            phase_started_at = time()
            if not suppress_adapter_response:
                asyncio.create_task(self.handle_response(STSResponse(
                    type="accepted",
                    session_id=request.session_id,
                    user_id=request.user_id,
                    context_id=request.context_id,
                    transaction_id=transaction_id,
                    metadata={
                        "block_barge_in": request.block_barge_in,
                        **({"audio_enhancement": request.metadata["audio_enhancement"]} if request.metadata and "audio_enhancement" in request.metadata else {}),
                        **({"voice_auth": request.metadata["voice_auth"]} if request.metadata and "voice_auth" in request.metadata else {}),
                        **({"wakeword": request.metadata["wakeword"]} if request.metadata and "wakeword" in request.metadata else {}),
                        **({"addressing": request.metadata["addressing"]} if request.metadata and "addressing" in request.metadata else {}),
                    }
                )))
                for handler in self._on_accepted_handlers:
                    await handler(request)
            performance.accepted_notify_time = time() - phase_started_at

            # Stop on-going response before new response
            phase_started_at = time()
            if not suppress_adapter_response and (not self.use_invoke_queue or not request.wait_in_queue):
                await self.stop_response(request.session_id, request.context_id)
            performance.stop_response_phase_time = time() - phase_started_at
            performance.stop_response_time = time() - start_time

            request_metadata = request.metadata or {}
            yield STSResponse(
                type="start",
                session_id=request.session_id,
                user_id=request.user_id,
                context_id=request.context_id,
                transaction_id=transaction_id,
                metadata={
                    **request_metadata,
                    "request_text": request.text,
                    "recognized_text": recognized_text,
                    "input_type": input_type,
                }
            )

            # LLM
            phase_started_at = time()
            for handler in self._on_before_llm_handlers:
                await handler(request)
            performance.pre_llm_handler_time = time() - phase_started_at
            performance.before_llm_time = time() - start_time
            performance.quick_response_text = request.quick_response_text

            # Yield quick response if generated by on_before_llm handler
            if request.quick_response_text:
                yield STSResponse(
                    type="chunk",
                    session_id=request.session_id,
                    user_id=request.user_id,
                    context_id=request.context_id,
                    transaction_id=transaction_id,
                    text=request.quick_response_text,
                    voice_text=request.quick_response_voice_text,
                    audio_data=request.quick_response_audio,
                    metadata={"is_quick_response": True, "is_first_chunk": True}
                )

            def record_llm_request_start(started_at: float):
                if performance.llm_request_start_time:
                    return
                try:
                    performance.llm_request_start_time = max(0, float(started_at) - start_time)
                except (TypeError, ValueError):
                    pass

            llm_stream_kwargs = {
                "context_id": request.context_id,
                "user_id": request.user_id,
                "text": request.text,
                "files": request.files,
                "system_prompt_params": request.system_prompt_params,
                "session_id": request.session_id,
                "channel": request.channel,
            }
            chat_stream_params = inspect.signature(self.llm.chat_stream).parameters
            if "request_start_callback" in chat_stream_params:
                llm_stream_kwargs["request_start_callback"] = record_llm_request_start
            llm_stream = self.llm.chat_stream(**llm_stream_kwargs)

            # TTS
            async def synthesize_stream() -> AsyncGenerator[Tuple[bytes, LLMResponse], None]:
                voice_text = ""
                language = None
                async for llm_stream_chunk in llm_stream:
                    is_txn_active, active_txn = await self.is_transaction_active(request.session_id, transaction_id)
                    if not is_txn_active:
                        # Break when new transaction started in this session
                        if self.debug:
                            logger.info(f"Break llm_stream for new transaction: {active_txn} {request.text} (current: {transaction_id})")
                        break

                    # LLM error
                    if llm_stream_chunk.error_info:
                        raise Exception(f"LLM error: {llm_stream_chunk.error_info}")

                    # LLM performance
                    if performance.llm_first_chunk_time == 0:
                        performance.llm_first_chunk_time = time() - start_time

                    # ToolCall
                    if llm_stream_chunk.tool_call:
                        yield None, llm_stream_chunk, None, None
                        continue

                    # Voice
                    if llm_stream_chunk.voice_text:
                        voice_text += llm_stream_chunk.voice_text
                        performance.response_voice_text = voice_text
                        if performance.llm_first_voice_chunk_time == 0:
                            performance.llm_first_voice_chunk_time = time() - start_time
                            for handler in self._on_before_tts_handlers:
                                await handler(request)
                    performance.llm_time = time() - start_time

                    # Parse language
                    if match := LANGUAGE_PATTERN.search(llm_stream_chunk.text):
                        language = match.group(1) or match.group(2)

                    # Parse style info from LLM chunk
                    parsed_info = await self._process_llm_chunk(
                        llm_stream_chunk,
                        request.session_id,
                        request.user_id,
                    )

                    if request.channel in self.skip_tts_channels:
                        audio_chunk = None
                    else:
                        audio_chunk = await self.tts.synthesize(
                            text=llm_stream_chunk.voice_text,
                            style_info={"styled_text": llm_stream_chunk.text, "info": parsed_info or {}},
                            language=language
                        )

                    # TTS performance
                    if audio_chunk:
                        if performance.tts_first_chunk_time == 0:
                            performance.tts_first_chunk_time = time() - start_time
                        performance.tts_time = time() - start_time

                    yield audio_chunk, llm_stream_chunk, language, llm_stream_chunk.guradrail_name

            response_text = ""
            response_audios = []
            is_first_chunk = not request.quick_response_text
            tool_call_records = {}  # {tool_call_id: {name, arguments, result}}
            async for audio_chunk, llm_stream_chunk, language, guradrail_name in synthesize_stream():
                is_txn_active, active_txn = await self.is_transaction_active(request.session_id, transaction_id)
                if not is_txn_active:
                    # Break when new transaction started in this session
                    if self.debug:
                        logger.info(f"Break synthesize_stream for new transaction: {active_txn} {request.text} (current: {transaction_id})")
                    break

                if llm_stream_chunk.tool_call:
                    tc = llm_stream_chunk.tool_call
                    tc_id = tc.id or tc.name  # Use id if available, otherwise name
                    if tc_id:
                        # First yield: record name and arguments
                        # Second yield: update with result (arguments still present)
                        tool_call_records[tc_id] = {
                            "name": tc.name,
                            "arguments": tc.arguments,
                            "result": tc.result.data if tc.result and tc.result.data else None
                        }
                    yield STSResponse(
                        type="tool_call",
                        session_id=request.session_id,
                        user_id=request.user_id,
                        context_id=llm_stream_chunk.context_id,
                        transaction_id=transaction_id,
                        tool_call=llm_stream_chunk.tool_call,
                        structured_content=llm_stream_chunk.structured_content
                    )
                    continue

                response_text += llm_stream_chunk.text
                performance.response_text = response_text
                if audio_chunk:
                    response_audios.append(audio_chunk)

                yield STSResponse(
                    type="chunk",
                    session_id=request.session_id,
                    user_id=request.user_id,
                    context_id=llm_stream_chunk.context_id,
                    transaction_id=transaction_id,
                    text=llm_stream_chunk.text,
                    voice_text=llm_stream_chunk.voice_text,
                    language=language,
                    audio_data=audio_chunk,
                    metadata={
                        **request_metadata,
                        "is_first_chunk": is_first_chunk,
                        "is_guardrail_triggered": True if guradrail_name else False,
                    },
                    structured_content=llm_stream_chunk.structured_content
                )
                is_first_chunk = False

            performance.total_time = time() - start_time
            if tool_call_records:
                performance.tool_calls = json.dumps(list(tool_call_records.values()), ensure_ascii=False)
            self.performance_recorder.record(performance)

            final_response = STSResponse(
                type="final",
                session_id=request.session_id,
                user_id=request.user_id,
                context_id=request.context_id,
                transaction_id=transaction_id,
                text=(request.quick_response_text or "") + response_text,
                voice_text=(request.quick_response_voice_text or "") + (performance.response_voice_text or ""),
                metadata=request_metadata,
            )

            if self.voice_recorder_enabled:
                if request.quick_response_audio:
                    await self.voice_recorder.record(ResponseVoices(
                        transaction_id + "_qr", [request.quick_response_audio], self.voice_recorder_response_audio_format
                    ))
                await self.voice_recorder.record(ResponseVoices(
                    transaction_id, response_audios, self.voice_recorder_response_audio_format
                ))
            for handler in self._on_finish_handlers:
                await handler(request, final_response)
            yield final_response
        
        except Exception as iex:
            tb = traceback.format_exc()
            logger.error(f"Error at invoke: {iex}\n\n{tb}")

            self._record_performance(
                performance,
                start_time,
                error_info={
                    "error": str(iex),
                    "traceback": tb,
                },
            )

            yield STSResponse(
                type="error",
                session_id=request.session_id,
                user_id=request.user_id,
                context_id=request.context_id,
                transaction_id=transaction_id,
                metadata={"error": "Error in processing Speech-to-Speech pipeline"}
            )

    async def _clear_queue(self, session_id: str):
        if session_id not in self._request_queues:
            return

        queue = self._request_queues[session_id]
        pending = self._response_queues.get(session_id, {})

        while not queue.empty():
            try:
                request_id, request = queue.get_nowait()
                response_queue = pending.get(request_id)
                if response_queue:
                    await response_queue.put(STSResponse(
                        type="cancelled",
                        session_id=session_id,
                        context_id=request.context_id
                    ))
                    await response_queue.put(None)
            except asyncio.QueueEmpty:
                break

    async def _process_queue(self, session_id: str):
        queue = self._request_queues[session_id]
        if self.debug:
            logger.info(f"Queue worker started: {session_id}")

        try:
            while True:
                try:
                    request_id, request = await asyncio.wait_for(
                        queue.get(), timeout=self.invoke_queue_idle_timeout
                    )
                except asyncio.TimeoutError:
                    if queue.empty():
                        if self.debug:
                            logger.info(f"Queue worker idle timeout, cleaning up: {session_id}")
                        self._cleanup_session_queue(session_id)
                        return
                    continue

                response_queue = self._response_queues.get(session_id, {}).get(request_id)
                try:
                    async with asyncio.timeout(self.invoke_timeout):
                        async for response in self._invoke_direct(request):
                            if response_queue:
                                await response_queue.put(response)
                except asyncio.TimeoutError:
                    logger.warning(f"invoke timed out: {session_id}")
                    if response_queue:
                        await response_queue.put(STSResponse(
                            type="error",
                            session_id=session_id,
                            context_id=request.context_id,
                            metadata={"error": "invoke timed out"}
                        ))
                except Exception as ex:
                    logger.error(f"invoke error in queue worker: {session_id} - {ex}")
                    if response_queue:
                        await response_queue.put(STSResponse(
                            type="error",
                            session_id=session_id,
                            context_id=request.context_id,
                            metadata={"error": "invoke error in queue worker"}
                        ))
                finally:
                    if response_queue:
                        await response_queue.put(None)
                    if session_id in self._response_queues:
                        self._response_queues[session_id].pop(request_id, None)

        except Exception as ex:
            logger.error(f"Queue worker crashed: {session_id} - {ex}")
            self._cleanup_session_queue(session_id)

    def _cleanup_session_queue(self, session_id: str):
        self._request_queues.pop(session_id, None)
        self._invoke_workers.pop(session_id, None)
        self._response_queues.pop(session_id, None)

    async def _invoke_queued(
        self,
        request: STSRequest
    ) -> AsyncGenerator[STSResponse, None]:
        session_id = request.session_id

        if session_id not in self._request_queues:
            self._request_queues[session_id] = asyncio.Queue()
            self._response_queues[session_id] = {}
            self._invoke_workers[session_id] = asyncio.create_task(
                self._process_queue(session_id)
            )

        if not request.wait_in_queue:
            await self._clear_queue(session_id)

        request_id = str(uuid4())
        response_queue: asyncio.Queue[STSResponse] = asyncio.Queue()
        self._response_queues[session_id][request_id] = response_queue

        await self._request_queues[session_id].put((request_id, request))

        while True:
            response = await response_queue.get()
            if response is None:
                break
            yield response

    async def finalize(self, context_id: str):
        await self.vad.finalize_session(context_id)

    async def shutdown(self):
        self.performance_recorder.close()
        await self.voice_recorder.stop()
        if self.audio_enhancer:
            await self.audio_enhancer.close()
        if self.voice_auth:
            await self.voice_auth.close()
        if self.addressing_detector:
            await self.addressing_detector.close()
