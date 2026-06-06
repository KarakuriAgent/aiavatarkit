import asyncio
import base64
import logging
import math
import struct
import uuid
from collections import deque
from typing import Awaitable, Callable, Dict, List, Optional

import httpx
import numpy as np

from .base import RecordingSessionBase, SpeechDetector
from ..stt.base import SpeechRecognizer

logger = logging.getLogger(__name__)


def _load_tenvad_class():
    try:
        from ten_vad import TenVad
    except ImportError as ex:
        raise RuntimeError(
            "ten-vad is required when VAD_PROVIDER=tenvad. "
            "Install dependencies with `pip install ten-vad` or your project package manager."
        ) from ex
    return TenVad


class TenVadDecisionState:
    def __init__(self, threshold: float, neg_threshold: Optional[float] = None):
        self.threshold = threshold
        self.neg_threshold = neg_threshold
        self.active = False

    def reset(self):
        self.active = False

    def update(self, probability: float, flag: bool) -> bool:
        if self.neg_threshold is None:
            self.active = bool(flag) or probability >= self.threshold
            return self.active

        if self.active:
            if probability < self.neg_threshold:
                self.active = False
        elif probability >= self.threshold:
            self.active = True
        return self.active


class HttpTenVadRuntimeSession:
    def __init__(
        self,
        hop_size: int = 256,
        threshold: float = 0.5,
        *,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 5.0,
        session_id: Optional[str] = None,
        neg_threshold: Optional[float] = None,
        sample_rate: int = 16000,
    ):
        self.hop_size = hop_size
        self.threshold = threshold
        self.neg_threshold = neg_threshold
        self.sample_rate = sample_rate
        self.base_url = base_url.rstrip("/")
        self.session_id = session_id or f"tenvad-{uuid.uuid4()}"
        self._client = httpx.Client(timeout=timeout)
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def process_frames(self, frames: bytes) -> dict:
        response = self._client.post(
            f"{self.base_url}/detect",
            headers=self._headers,
            json={
                "session_id": self.session_id,
                "audio_data": base64.b64encode(frames).decode("ascii"),
                "sample_rate": self.sample_rate,
                "hop_size": self.hop_size,
                "threshold": self.threshold,
                "neg_threshold": self.neg_threshold,
            },
        )
        response.raise_for_status()
        return response.json()

    def reset(self):
        try:
            response = self._client.post(
                f"{self.base_url}/reset",
                headers=self._headers,
                json={"session_id": self.session_id},
            )
            response.raise_for_status()
        except Exception:
            logger.debug("Failed to reset remote TenVAD session", exc_info=True)

    def close(self):
        self.reset()
        self._client.close()


def make_http_tenvad_session_factory(
    *,
    base_url: str,
    api_key: Optional[str] = None,
    timeout: float = 5.0,
    neg_threshold: Optional[float] = None,
    sample_rate: int = 16000,
):
    def _factory(
        hop_size: int,
        threshold: float,
        *,
        session_id: Optional[str] = None,
        neg_threshold: Optional[float] = None,
    ):
        return HttpTenVadRuntimeSession(
            hop_size=hop_size,
            threshold=threshold,
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            session_id=session_id,
            neg_threshold=neg_threshold if neg_threshold is not None else _factory.neg_threshold,
            sample_rate=sample_rate,
        )

    _factory.neg_threshold = neg_threshold
    return _factory


class RecordingSession(RecordingSessionBase):
    def __init__(
        self,
        session_id: str,
        preroll_buffer_count: int = 5,
        ten_vad_class=None,
        hop_size: int = 256,
        threshold: float = 0.5,
        neg_threshold: Optional[float] = None,
    ):
        super().__init__(session_id, preroll_buffer_count)
        self.amplitude_threshold: Optional[float] = None
        self.vad_buffer: bytearray = bytearray()
        self.segment_buffer: bytearray = bytearray()
        self.segment_duration: float = 0
        self.segment_silence_duration: float = 0
        self.segment_fired: bool = False
        self.pending_recognition_task: Optional[asyncio.Task] = None
        self.recognition_sequence: int = 0
        self._ten_vad_class = ten_vad_class
        self.hop_size = hop_size
        self.threshold = threshold
        self.neg_threshold = neg_threshold
        self.vad_decision_state = TenVadDecisionState(self.threshold, self.neg_threshold)
        self.ten_vad = self._create_tenvad()

    def _create_tenvad(self):
        try:
            return self._ten_vad_class(
                self.hop_size,
                self.threshold,
                session_id=self.session_id,
                neg_threshold=self.neg_threshold,
            )
        except TypeError:
            try:
                return self._ten_vad_class(self.hop_size, self.threshold, session_id=self.session_id)
            except TypeError:
                return self._ten_vad_class(self.hop_size, self.threshold)

    def reset(self):
        super().reset()
        self.segment_buffer.clear()
        self.segment_duration = 0
        self.segment_silence_duration = 0
        self.segment_fired = False
        self.pending_recognition_task = None
        self.recognition_sequence = 0

    def reset_vad(self):
        self.vad_decision_state = TenVadDecisionState(self.threshold, self.neg_threshold)
        if hasattr(self.ten_vad, "process_frames"):
            if hasattr(self.ten_vad, "close"):
                self.ten_vad.close()
            self.ten_vad = self._create_tenvad()
        elif hasattr(self.ten_vad, "reset"):
            self.ten_vad.reset()
        else:
            self.ten_vad = self._create_tenvad()


class TenVadStreamSpeechDetector(SpeechDetector):
    def __init__(
        self,
        *,
        speech_recognizer: SpeechRecognizer,
        volume_db_threshold: Optional[float] = None,
        silence_duration_threshold: float = 0.5,
        segment_silence_threshold: float = 0.2,
        max_duration: float = 10.0,
        min_duration: float = 0.2,
        sample_rate: int = 16000,
        channels: int = 1,
        preroll_buffer_count: int = 5,
        to_linear16: Optional[Callable[[bytes], bytes]] = None,
        debug: bool = False,
        speech_probability_threshold: float = 0.5,
        negative_speech_probability_threshold: Optional[float] = None,
        hop_size: int = 256,
        speech_pad_ms: int = 0,
        on_recording_started: Optional[Callable[[str], Awaitable[None]]] = None,
        on_recording_started_min_duration: float = 1.5,
        on_recording_started_min_text_length: int = 2,
        ten_vad_class=None,
    ):
        if sample_rate != 16000:
            raise ValueError("TenVAD requires 16000Hz linear16 mono input")
        if channels != 1:
            raise ValueError("TenVAD requires mono input")
        super().__init__(
            sample_rate=sample_rate,
            on_recording_started_min_duration=on_recording_started_min_duration,
            on_recording_started_min_text_length=on_recording_started_min_text_length,
        )
        self._volume_db_threshold = volume_db_threshold
        if volume_db_threshold is not None:
            self.amplitude_threshold = 32767 * (10 ** (self.volume_db_threshold / 20.0))
        else:
            self.amplitude_threshold = None
        self.silence_duration_threshold = silence_duration_threshold
        self.segment_silence_threshold = segment_silence_threshold
        self.max_duration = max_duration
        self.min_duration = min_duration
        self.channels = channels
        self.preroll_buffer_count = preroll_buffer_count
        self.to_linear16 = to_linear16
        self.debug = debug
        self.speech_probability_threshold = speech_probability_threshold
        self.negative_speech_probability_threshold = negative_speech_probability_threshold
        self.hop_size = hop_size
        self.speech_pad_ms = speech_pad_ms
        self._ten_vad_class = ten_vad_class or _load_tenvad_class()
        self.speech_recognizer = speech_recognizer
        self.recording_sessions: Dict[str, RecordingSession] = {}
        self._on_speech_detecting: List[Callable[[str, RecordingSession], Awaitable[None]]] = []
        self._on_speech_recognition_error: List[Callable[[Exception, str], Awaitable[None]]] = []
        self._validate_recognized_text: Optional[Callable[[str], Optional[str]]] = None
        self.on_recording_started_min_text_length = on_recording_started_min_text_length
        if on_recording_started:
            self._on_recording_started.append(on_recording_started)

    def get_config(self) -> dict:
        return {
            "volume_db_threshold": self.volume_db_threshold,
            "silence_duration_threshold": self.silence_duration_threshold,
            "segment_silence_threshold": self.segment_silence_threshold,
            "max_duration": self.max_duration,
            "min_duration": self.min_duration,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "preroll_buffer_count": self.preroll_buffer_count,
            "debug": self.debug,
            "speech_probability_threshold": self.speech_probability_threshold,
            "negative_speech_probability_threshold": self.negative_speech_probability_threshold,
            "hop_size": self.hop_size,
            "speech_pad_ms": self.speech_pad_ms,
        }

    def set_config(self, config: dict) -> dict:
        allowed_keys = self.get_config().keys()
        updated = {}
        for key, value in config.items():
            if value is None or key not in allowed_keys:
                continue
            if key == "speech_probability_threshold":
                self.set_speech_probability_threshold(value)
                updated[key] = value
                continue
            if key == "negative_speech_probability_threshold":
                self.set_negative_speech_probability_threshold(value)
                updated[key] = value
                continue
            if key == "hop_size":
                self.set_hop_size(value)
                updated[key] = value
                continue
            try:
                setattr(self, key, value)
                updated[key] = value
            except Exception:
                pass
        return updated

    @property
    def volume_db_threshold(self) -> float:
        return self._volume_db_threshold

    @volume_db_threshold.setter
    def volume_db_threshold(self, value: Optional[float]):
        self._volume_db_threshold = value
        if value is not None:
            self.amplitude_threshold = 32767 * (10 ** (value / 20.0))
            logger.debug(
                "Updated volume_db_threshold to %s dB, amplitude_threshold=%s",
                value,
                self.amplitude_threshold,
            )
        else:
            self.amplitude_threshold = None
            logger.debug("Volume threshold disabled (set to None)")

    def on_speech_detecting(
        self,
        func: Callable[[str, RecordingSession], Awaitable[None]],
    ) -> Callable[[str, RecordingSession], Awaitable[None]]:
        self._on_speech_detecting.append(func)
        return func

    def validate_recognized_text(self, func: Callable[[str], Optional[str]]):
        self._validate_recognized_text = func
        return func

    def on_speech_recognition_error(
        self,
        func: Callable[[Exception, str], Awaitable[None]],
    ) -> Callable[[Exception, str], Awaitable[None]]:
        self._on_speech_recognition_error.append(func)
        return func

    async def _execute_on_speech_detecting(self, text: str, session: RecordingSession):
        for handler in self._on_speech_detecting:
            try:
                await handler(text, session)
            except Exception:
                logger.error("Error in on_speech_detecting callback", exc_info=True)

    async def _execute_on_speech_recognition_error(self, error: Exception, session_id: str):
        for handler in self._on_speech_recognition_error:
            try:
                await handler(error, session_id)
            except Exception:
                logger.error("Error in on_speech_recognition_error callback", exc_info=True)

    async def execute_on_speech_detected(
        self,
        recorded_data: bytes,
        text: str,
        metadata: dict,
        recorded_duration: float,
        session_id: str,
    ):
        await self._execute_on_speech_detected(
            recorded_data,
            text,
            metadata,
            recorded_duration,
            session_id,
        )

    def _detect_speech_tenvad(self, session: RecordingSession) -> bool:
        frame_byte_count = self.hop_size * 2
        speech_detected = False
        complete_byte_count = (len(session.vad_buffer) // frame_byte_count) * frame_byte_count

        if complete_byte_count <= 0:
            return False

        if hasattr(session.ten_vad, "process_frames"):
            frames = bytes(session.vad_buffer[:complete_byte_count])
            del session.vad_buffer[:complete_byte_count]
            try:
                result = session.ten_vad.process_frames(frames)
            except Exception:
                logger.error("Error in remote TenVAD detection", exc_info=True)
                return False

            if self.debug:
                logger.debug("Remote TenVAD result=%s", result)
            return bool(result.get("speech_detected"))

        while len(session.vad_buffer) >= frame_byte_count:
            frame = bytes(session.vad_buffer[:frame_byte_count])
            del session.vad_buffer[:frame_byte_count]
            audio_data = np.frombuffer(frame, dtype=np.int16)
            try:
                probability, flag = session.ten_vad.process(audio_data)
            except Exception:
                logger.error("Error in TenVAD detection", exc_info=True)
                continue

            if self.debug:
                logger.debug("TenVAD probability=%0.4f flag=%s", probability, flag)
            if session.vad_decision_state.update(float(probability), bool(flag)):
                speech_detected = True

        return speech_detected

    async def _recognize_final_text(self, session: RecordingSession) -> Optional[str]:
        try:
            result = await self.speech_recognizer.recognize(session.session_id, bytes(session.buffer))
            final_text = result.text if result else None
            if self.debug and session.last_recognized_text and final_text != session.last_recognized_text:
                logger.info(
                    "Final recognition superseded partial result: partial=%s final=%s",
                    session.last_recognized_text,
                    final_text,
                )
            return final_text
        except Exception as ex:
            logger.error("Error in final recognition", exc_info=True)
            await self._execute_on_speech_recognition_error(ex, session.session_id)
            return None

    async def _emit_final_speech_detected(self, session: RecordingSession, recorded_duration: float) -> bool:
        final_text = await self._recognize_final_text(session)
        if not final_text:
            if self.debug:
                logger.info("No final text recognized, skipping")
            return False

        if self._validate_recognized_text:
            if validation := self._validate_recognized_text(final_text):
                if self.debug:
                    logger.info("Invalid recognized text: %s / validation: %s", final_text, validation)
                return False

        recorded_data = bytes(session.buffer)
        asyncio.create_task(
            self.execute_on_speech_detected(
                recorded_data,
                final_text,
                None,
                recorded_duration,
                session.session_id,
            )
        )
        return True

    def _sync_preroll_padding(self, session: RecordingSession, sample_duration: float):
        if self.speech_pad_ms <= 0 or sample_duration <= 0:
            return

        target_count = max(1, math.ceil((self.speech_pad_ms / 1000.0) / sample_duration))
        if session.preroll_buffer.maxlen == target_count:
            return
        session.preroll_buffer = deque(session.preroll_buffer, maxlen=target_count)

    async def process_samples(self, samples: bytes, session_id: str) -> bool:
        if self.to_linear16:
            samples = self.to_linear16(samples)

        session = self.get_session(session_id)
        sample_duration = (len(samples) / 2) / (self.sample_rate * self.channels)
        self._sync_preroll_padding(session, sample_duration)

        if self.should_mute():
            session.reset()
            session.preroll_buffer.clear()
            session.vad_buffer.clear()
            session.reset_vad()
            logger.debug("TenVadStreamSpeechDetector is muted.")
            return False

        session.preroll_buffer.append(samples)
        session.vad_buffer.extend(samples)

        speech_detected = self._detect_speech_tenvad(session)
        if speech_detected and session.amplitude_threshold is not None and samples:
            max_amplitude = float(max(abs(sample) for sample, in struct.iter_unpack("<h", samples)))
            if max_amplitude <= session.amplitude_threshold:
                speech_detected = False

        if self.debug:
            logger.debug(
                "Speech detected: %s, duration: %.2f, session: %s",
                speech_detected,
                session.record_duration,
                session.session_id,
            )

        if speech_detected:
            await self._execute_on_voiced(session_id)

        if not session.is_recording:
            if speech_detected:
                session.reset()
                session.is_recording = True

                for frame in session.preroll_buffer:
                    session.buffer.extend(frame)

                session.buffer.extend(samples)
                session.record_duration += sample_duration

        else:
            session.buffer.extend(samples)
            session.record_duration += sample_duration
            session.segment_buffer.extend(samples)
            session.segment_duration += sample_duration

            if speech_detected:
                session.silence_duration = 0
                session.segment_silence_duration = 0
                session.segment_fired = False
            else:
                session.silence_duration += sample_duration
                session.segment_silence_duration += sample_duration

            if (
                session.segment_silence_duration >= self.segment_silence_threshold
                and len(session.segment_buffer) > 0
                and not session.segment_fired
            ):
                session.segment_fired = True
                segment_data = bytes(session.segment_buffer)
                session.recognition_sequence += 1
                current_seq = session.recognition_sequence

                async def _run_segment_recognition(data: bytes, sess: RecordingSession, seq: int):
                    try:
                        result = await self.speech_recognizer.recognize(sess.session_id, data)
                        recognized_text = result.text or ""
                        if recognized_text and seq == sess.recognition_sequence:
                            sess.last_recognized_text = recognized_text
                            await self._execute_on_speech_detecting(recognized_text, sess)
                            await self._check_and_trigger_recording_started(sess)
                    except Exception as ex:
                        logger.error("Error in segment recognition", exc_info=True)
                        await self._execute_on_speech_recognition_error(ex, sess.session_id)

                session.pending_recognition_task = asyncio.create_task(
                    _run_segment_recognition(segment_data, session, current_seq)
                )

            await self._check_and_trigger_recording_started(session)

            if session.silence_duration >= self.silence_duration_threshold:
                recorded_duration = session.record_duration - session.silence_duration
                if recorded_duration < self.min_duration:
                    if self.debug:
                        logger.info("Recording too short: %s sec", recorded_duration)
                else:
                    if self.debug:
                        logger.info("Recording finished: %s sec", recorded_duration)

                    if session.pending_recognition_task is not None:
                        try:
                            await session.pending_recognition_task
                        except Exception:
                            logger.error("Error waiting for pending recognition", exc_info=True)

                    await self._emit_final_speech_detected(session, recorded_duration)
                session.reset()

            elif session.record_duration >= self.max_duration:
                if self.debug:
                    logger.info("Recording max duration reached: %s sec", session.record_duration)

                if session.pending_recognition_task is not None:
                    try:
                        await session.pending_recognition_task
                    except Exception:
                        logger.error("Error waiting for pending recognition", exc_info=True)

                await self._emit_final_speech_detected(session, session.record_duration)
                session.reset()

        return session.is_recording

    async def process_stream(self, input_stream, session_id: str):
        logger.info("TenVadStreamSpeechDetector start processing stream.")

        async for data in input_stream:
            if not data:
                break
            await self.process_samples(data, session_id)
            await asyncio.sleep(0.0001)

        self.delete_session(session_id)

        logger.info("TenVadStreamSpeechDetector finish processing stream.")

    async def finalize_session(self, session_id: str):
        self.delete_session(session_id)

    def get_session(self, session_id: str):
        session = self.recording_sessions.get(session_id)
        if session is None:
            session = RecordingSession(
                session_id,
                self.preroll_buffer_count,
                self._ten_vad_class,
                self.hop_size,
                self.speech_probability_threshold,
                self.negative_speech_probability_threshold,
            )
            self.recording_sessions[session_id] = session
        if session.amplitude_threshold is None:
            session.amplitude_threshold = self.amplitude_threshold
        return session

    def reset_session(self, session_id: str):
        if session := self.recording_sessions.get(session_id):
            session.reset()

    def reset_session_audio_state(self, session_id: str, clear_preroll: bool = True):
        session = self.recording_sessions.get(session_id)
        if not session:
            return

        dropped_buffer_bytes = len(session.buffer)
        dropped_preroll_bytes = sum(len(frame) for frame in session.preroll_buffer)
        dropped_vad_bytes = len(session.vad_buffer)
        dropped_segment_bytes = len(session.segment_buffer)
        dropped_record_duration = session.record_duration
        dropped_silence_duration = session.silence_duration
        dropped_segment_duration = session.segment_duration
        pending_task_alive = (
            session.pending_recognition_task is not None
            and not session.pending_recognition_task.done()
        )
        was_recording = session.is_recording

        if session.pending_recognition_task is not None and not session.pending_recognition_task.done():
            session.pending_recognition_task.cancel()

        session.reset()
        session.vad_buffer.clear()
        session.reset_vad()
        if clear_preroll:
            session.preroll_buffer.clear()
        if self.debug:
            logger.info(
                "TenVAD audio state reset: session=%s, clear_preroll=%s, was_recording=%s, pending_task_alive=%s, dropped_buffer_bytes=%s, dropped_preroll_bytes=%s, dropped_vad_bytes=%s, dropped_segment_bytes=%s, dropped_record_duration=%.3f, dropped_silence_duration=%.3f, dropped_segment_duration=%.3f",
                session_id,
                clear_preroll,
                was_recording,
                pending_task_alive,
                dropped_buffer_bytes,
                dropped_preroll_bytes,
                dropped_vad_bytes,
                dropped_segment_bytes,
                dropped_record_duration,
                dropped_silence_duration,
                dropped_segment_duration,
            )

    def delete_session(self, session_id: str):
        if session_id in self.recording_sessions:
            session = self.recording_sessions[session_id]
            session.reset()
            if hasattr(session.ten_vad, "close"):
                session.ten_vad.close()
            del self.recording_sessions[session_id]

    def get_session_data(self, session_id: str, key: str):
        session = self.recording_sessions.get(session_id)
        if session:
            return session.data.get(key)

    def set_session_data(self, session_id: str, key: str, value: any, create_session: bool = False):
        if create_session:
            session = self.get_session(session_id)
        else:
            session = self.recording_sessions.get(session_id)

        if session:
            session.data[key] = value

    def set_speech_probability_threshold(self, threshold: float):
        self.speech_probability_threshold = threshold
        for session in self.recording_sessions.values():
            session.threshold = threshold
            session.vad_decision_state = TenVadDecisionState(
                threshold,
                self.negative_speech_probability_threshold,
            )
            session.reset_vad()
        logger.debug("Updated TenVAD speech probability threshold to %s", threshold)

    def set_negative_speech_probability_threshold(self, threshold: Optional[float]):
        self.negative_speech_probability_threshold = threshold
        if hasattr(self._ten_vad_class, "neg_threshold"):
            self._ten_vad_class.neg_threshold = threshold
        for session in self.recording_sessions.values():
            session.neg_threshold = threshold
            session.vad_decision_state = TenVadDecisionState(
                self.speech_probability_threshold,
                threshold,
            )
            session.reset_vad()
        logger.debug("Updated TenVAD negative speech probability threshold to %s", threshold)

    def set_hop_size(self, hop_size: int):
        self.hop_size = hop_size
        for session in self.recording_sessions.values():
            session.hop_size = hop_size
            session.vad_buffer.clear()
            session.reset_vad()
        logger.debug("Updated TenVAD hop_size to %s", hop_size)

    def reset_vad_state(self, session_id: str = None):
        if session_id:
            session = self.recording_sessions.get(session_id)
            if session:
                session.reset_vad()
                logger.debug("TenVAD state reset for session %s", session_id)
        else:
            for session in self.recording_sessions.values():
                session.reset_vad()
            logger.debug("TenVAD state reset for all sessions")

    def set_volume_db_threshold(self, session_id: str, value: float):
        session = self.get_session(session_id)
        session.amplitude_threshold = 32767 * (10 ** (value / 20.0))
