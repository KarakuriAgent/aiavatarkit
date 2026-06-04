import asyncio
import logging
import shlex
import shutil
import tempfile
import time
import wave
from pathlib import Path
from pathlib import PurePath
from typing import Optional

from .base import AudioEnhancementResult, AudioEnhancer

logger = logging.getLogger(__name__)


class DeepFilterNetAudioEnhancer(AudioEnhancer):
    def __init__(
        self,
        *,
        command: str = "deepFilter",
        model: Optional[str] = None,
        timeout: float = 30.0,
        ffmpeg_command: str = "ffmpeg",
        debug: bool = False,
    ):
        self.command = command
        self.model = model
        self.timeout = timeout
        self.ffmpeg_command = ffmpeg_command
        self.debug = debug

    async def enhance(
        self,
        *,
        audio_bytes: bytes,
        sample_rate: int,
        session_id: Optional[str] = None,
    ) -> AudioEnhancementResult:
        if not audio_bytes:
            return AudioEnhancementResult(
                audio_bytes=audio_bytes,
                provider="deepfilternet",
                metadata={"skipped": True, "reason": "empty_audio"},
            )

        self._ensure_command_available(self.ffmpeg_command)
        self._ensure_command_available(self.command)

        started_at = time.monotonic()
        with tempfile.TemporaryDirectory(prefix="aiavatar-deepfilter-") as temp_dir:
            temp_path = Path(temp_dir)
            input_wav = temp_path / "input.wav"
            input_48k_wav = temp_path / "input_48k.wav"
            output_dir = temp_path / "enhanced"
            output_wav = temp_path / "output.wav"
            output_dir.mkdir(parents=True, exist_ok=True)

            self._write_pcm_wav(input_wav, audio_bytes, sample_rate)
            await self._run_command(
                [
                    *self._split_command(self.ffmpeg_command),
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(input_wav),
                    "-ar",
                    "48000",
                    "-ac",
                    "1",
                    "-sample_fmt",
                    "s16",
                    str(input_48k_wav),
                ]
            )

            await self._run_command(self._build_deepfilter_command(input_48k_wav, output_dir))

            enhanced_path = self._find_enhanced_file(output_dir, input_48k_wav)
            await self._run_command(
                [
                    *self._split_command(self.ffmpeg_command),
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(enhanced_path),
                    "-ar",
                    str(sample_rate),
                    "-ac",
                    "1",
                    "-sample_fmt",
                    "s16",
                    str(output_wav),
                ]
            )

            enhanced_audio = self._read_pcm_wav(output_wav)

        elapsed_ms = int((time.monotonic() - started_at) * 1000)
        metadata = {
            "command": self.command,
            "model": self.model,
            "input_bytes": len(audio_bytes),
            "output_bytes": len(enhanced_audio),
            "sample_rate": sample_rate,
            "elapsed_ms": elapsed_ms,
        }
        if self.debug:
            logger.info("DeepFilterNet enhanced audio: %s", metadata)
        return AudioEnhancementResult(
            audio_bytes=enhanced_audio,
            provider="deepfilternet",
            metadata=metadata,
        )

    def enhance_sync(
        self,
        *,
        audio_bytes: bytes,
        sample_rate: int,
        session_id: Optional[str] = None,
    ) -> AudioEnhancementResult:
        raise RuntimeError("DeepFilterNetAudioEnhancer.enhance_sync is not supported")

    def get_config(self) -> dict:
        return {
            "provider": "deepfilternet",
            "command": self.command,
            "model": self.model,
            "timeout": self.timeout,
            "ffmpeg_command": self.ffmpeg_command,
            "debug": self.debug,
        }

    def _write_pcm_wav(self, path: Path, audio_bytes: bytes, sample_rate: int):
        if audio_bytes.startswith(b"RIFF"):
            path.write_bytes(audio_bytes)
            return

        with wave.open(str(path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_bytes)

    def _read_pcm_wav(self, path: Path) -> bytes:
        with wave.open(str(path), "rb") as wav_file:
            return wav_file.readframes(wav_file.getnframes())

    async def _run_command(self, command: list[str]):
        executable = command[0]
        self._ensure_executable_available(executable)

        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self.timeout)
        except asyncio.TimeoutError as ex:
            process.kill()
            await process.communicate()
            raise RuntimeError(f"Audio enhancement command timed out: {executable}") from ex

        if process.returncode != 0:
            stderr_text = stderr.decode(errors="replace").strip()
            stdout_text = stdout.decode(errors="replace").strip()
            detail = stderr_text or stdout_text or f"exit code {process.returncode}"
            raise RuntimeError(f"Audio enhancement command failed: {executable}: {detail}")

    def _split_command(self, command: str) -> list[str]:
        parts = shlex.split(command or "")
        if not parts:
            raise RuntimeError("Audio enhancement command is empty")
        return parts

    def _ensure_command_available(self, command: str):
        self._ensure_executable_available(self._split_command(command)[0])

    def _ensure_executable_available(self, executable: str):
        if shutil.which(executable) is None:
            raise RuntimeError(f"Required audio enhancement command not found: {executable}")

    def _is_rust_deep_filter_command(self) -> bool:
        command = self._split_command(self.command)
        return PurePath(command[0]).name == "deep-filter"

    def _build_deepfilter_command(self, input_wav: Path, output_dir: Path) -> list[str]:
        command = self._split_command(self.command)
        if self._is_rust_deep_filter_command():
            deepfilter_cmd = [
                *command,
                "--output-dir",
                str(output_dir),
            ]
            if self.model:
                deepfilter_cmd.extend(self._model_arguments())
            deepfilter_cmd.append(str(input_wav))
            return deepfilter_cmd

        deepfilter_cmd = [
            *command,
            str(input_wav),
            "--output-dir",
            str(output_dir),
            "--log-level",
            "error",
        ]
        if self.model:
            deepfilter_cmd.extend(self._model_arguments())
        return deepfilter_cmd

    def _model_arguments(self) -> list[str]:
        if self._is_rust_deep_filter_command():
            return ["--model", self.model]
        return ["--model-base-dir", self.model]

    def _find_enhanced_file(self, output_dir: Path, input_wav: Path) -> Path:
        candidates = sorted(output_dir.glob("*.wav"))
        if not candidates:
            raise RuntimeError("DeepFilterNet did not produce an enhanced wav file")

        preferred_names = [
            f"{input_wav.stem}_DeepFilterNet3.wav",
            f"{input_wav.stem}_DeepFilterNet2.wav",
            f"{input_wav.stem}_DeepFilterNet.wav",
            input_wav.name,
        ]
        for name in preferred_names:
            candidate = output_dir / name
            if candidate.exists():
                return candidate
        return candidates[0]
