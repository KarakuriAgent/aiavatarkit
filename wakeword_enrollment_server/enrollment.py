import asyncio
import io
import json
import os
import random
import re
import subprocess
import sys
import threading
import wave
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from uuid import uuid4

from server.config import Settings


@dataclass
class WakewordCandidate:
    id: str
    created_at: str
    wakeword: Optional[str]
    role: Optional[str]
    session_id: str
    duration: float
    sample_rate: int
    path: str

    def to_dict(self):
        data = asdict(self)
        data["audio_url"] = f"/api/candidates/{self.id}/audio"
        return data


@dataclass
class WakewordModel:
    id: str
    created_at: str
    wakeword: str
    filename: str
    path: str
    threshold: float

    def to_dict(self):
        return asdict(self)


@dataclass
class WakewordTestResult:
    id: str
    created_at: str
    model_id: str
    candidate_id: str
    wakeword: str
    detected: bool
    matched_wakeword: Optional[str]
    score: Optional[float]
    threshold: float
    duration: float

    def to_dict(self):
        return asdict(self)


@dataclass
class WakewordTrainingJob:
    id: str
    created_at: str
    wakeword: str
    model_name: str
    target_phrases: List[str]
    threshold: float
    tts_backend: str
    model_size: str
    n_samples: int
    n_samples_val: int
    steps: int
    skip_acav: bool
    run_eval: bool
    positive_source: str
    negative_source: str
    status: str
    work_dir: str
    config_path: str
    log_path: str
    onnx_path: Optional[str] = None
    model_id: Optional[str] = None
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    error: Optional[str] = None
    return_code: Optional[int] = None

    def to_dict(self):
        data = asdict(self)
        data["log_url"] = f"/api/training/jobs/{self.id}/log"
        return data


class WakewordEnrollmentManager:
    CONFIGURED_TTS_BACKEND = "configured_tts"
    CONFIG_TTS_BACKEND = "piper_vits"
    DEFAULT_NEGATIVE_PHRASES = [
        "こんにちは",
        "おはよう",
        "こんばんは",
        "ありがとう",
        "はい",
        "いいえ",
        "了解",
        "すみません",
        "ちょっと待って",
        "これはテストです",
        "ロボ",
        "花音",
        "かのん",
    ]

    def __init__(self, settings: Settings):
        self.settings = settings
        self.root_dir = Path(settings.wakeword_enrollment_dir)
        self.candidate_dir = self.root_dir / "candidates"
        self.model_dir = self.root_dir / "models"
        self.training_dir = self.root_dir / "training"
        self.training_job_dir = self.training_dir / "jobs"
        self.training_data_dir = self.training_dir / "data"
        self.candidate_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.training_job_dir.mkdir(parents=True, exist_ok=True)
        self.training_data_dir.mkdir(parents=True, exist_ok=True)
        self.reading_enabled = False
        self.reading_wakeword: Optional[str] = None
        self._candidates: Dict[str, WakewordCandidate] = {}
        self._models: Dict[str, WakewordModel] = {}
        self._test_results: List[WakewordTestResult] = []
        self._training_jobs: Dict[str, WakewordTrainingJob] = {}
        self._training_lock = threading.Lock()
        self._load_existing_candidates()
        self._load_existing_models()
        self._load_existing_training_jobs()

    def state(self):
        return {
            "reading_enabled": self.reading_enabled,
            "reading_wakeword": self.reading_wakeword,
            "candidate_count": len(self._candidates),
            "model_count": len(self._models),
            "test_result_count": len(self._test_results),
            "training_job_count": len(self._training_jobs),
            "active_training_job_count": sum(
                1 for job in self._training_jobs.values() if job.status in ("queued", "running")
            ),
            "default_threshold": self.settings.audio_wakeword_threshold,
        }

    def start_reading(self, wakeword: Optional[str] = None):
        self.reading_enabled = True
        self.reading_wakeword = wakeword or None

    def stop_reading(self):
        self.reading_enabled = False

    def add_candidate(
        self,
        *,
        audio_bytes: bytes,
        sample_rate: int,
        duration: float,
        session_id: str,
    ) -> Optional[WakewordCandidate]:
        if not self.reading_enabled:
            return None
        candidate_id = uuid4().hex
        path = self.candidate_dir / f"{candidate_id}.wav"
        self._write_wav(path, audio_bytes, sample_rate)
        candidate = WakewordCandidate(
            id=candidate_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            wakeword=self.reading_wakeword,
            role=None,
            session_id=session_id,
            duration=duration,
            sample_rate=sample_rate,
            path=str(path),
        )
        self._candidates[candidate_id] = candidate
        self._write_json(self._candidate_metadata_path(candidate_id), candidate.to_dict())
        return candidate

    def list_candidates(self):
        return [candidate.to_dict() for candidate in sorted(self._candidates.values(), key=lambda c: c.created_at)]

    def get_candidate(self, candidate_id: str) -> WakewordCandidate:
        candidate = self._candidates.get(candidate_id)
        if not candidate:
            raise KeyError(candidate_id)
        return candidate

    def get_candidate_audio_path(self, candidate_id: str) -> Path:
        return Path(self.get_candidate(candidate_id).path)

    def update_candidate_role(self, candidate_id: str, role: Optional[str]) -> dict:
        candidate = self.get_candidate(candidate_id)
        normalized_role = role.strip() if isinstance(role, str) else None
        if normalized_role == "":
            normalized_role = None
        if normalized_role not in (None, "positive", "negative", "ignore"):
            raise ValueError("role must be positive, negative, ignore, or null")
        candidate.role = normalized_role
        self._write_json(self._candidate_metadata_path(candidate.id), candidate.to_dict())
        return candidate.to_dict()

    def clear_candidates(self):
        for candidate in list(self._candidates.values()):
            Path(candidate.path).unlink(missing_ok=True)
            self._candidate_metadata_path(candidate.id).unlink(missing_ok=True)
        self._candidates.clear()

    def import_model(self, *, wakeword: str, filename: str, content: bytes, threshold: Optional[float] = None):
        if not wakeword:
            raise ValueError("wakeword is required")
        if not filename.lower().endswith(".onnx"):
            raise ValueError("only ONNX models are supported")
        if not content:
            raise ValueError("model file is empty")
        model_id = self._model_id(wakeword, filename)
        model_path = self.model_dir / f"{model_id}.onnx"
        model_path.write_bytes(content)
        model = WakewordModel(
            id=model_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            wakeword=wakeword,
            filename=filename,
            path=str(model_path),
            threshold=threshold if threshold is not None else self.settings.audio_wakeword_threshold,
        )
        self._models[model_id] = model
        self._write_json(self._model_metadata_path(model_id), model.to_dict())
        return model.to_dict()

    def list_models(self):
        return [model.to_dict() for model in sorted(self._models.values(), key=lambda m: m.created_at)]

    def get_model(self, model_id: str) -> WakewordModel:
        model = self._models.get(model_id)
        if not model:
            raise KeyError(model_id)
        return model

    def delete_model(self, model_id: str):
        model = self.get_model(model_id)
        Path(model.path).unlink(missing_ok=True)
        self._model_metadata_path(model_id).unlink(missing_ok=True)
        del self._models[model_id]

    def list_test_results(self):
        return [result.to_dict() for result in reversed(self._test_results)]

    def list_training_jobs(self):
        return [
            job.to_dict()
            for job in sorted(self._training_jobs.values(), key=lambda j: j.created_at, reverse=True)
        ]

    def get_training_job(self, job_id: str) -> WakewordTrainingJob:
        job = self._training_jobs.get(job_id)
        if not job:
            raise KeyError(job_id)
        return job

    def get_training_job_log(self, job_id: str) -> str:
        job = self.get_training_job(job_id)
        path = Path(job.log_path)
        return path.read_text(errors="replace") if path.exists() else ""

    def start_training_job(
        self,
        *,
        wakeword: str,
        target_phrases: List[str],
        model_name: Optional[str] = None,
        threshold: Optional[float] = None,
        tts_backend: str = CONFIGURED_TTS_BACKEND,
        model_size: str = "tiny",
        n_samples: int = 500,
        n_samples_val: int = 100,
        steps: int = 2000,
        skip_acav: bool = True,
        run_eval: bool = False,
        positive_source: str = "recorded_only",
        negative_source: str = "recorded_plus_tts",
        background: bool = True,
    ) -> dict:
        wakeword = (wakeword or "").strip()
        target_phrases = [phrase.strip() for phrase in target_phrases if phrase and phrase.strip()]
        if not wakeword:
            raise ValueError("wakeword is required")
        if not target_phrases:
            raise ValueError("target_phrases is required")
        if threshold is not None and not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if tts_backend not in (self.CONFIGURED_TTS_BACKEND, "piper_vits", "voxcpm"):
            raise ValueError("tts_backend must be configured_tts, piper_vits, or voxcpm")
        if model_size not in ("tiny", "small", "medium", "large"):
            raise ValueError("model_size must be tiny, small, medium, or large")
        if n_samples <= 0 or n_samples_val <= 0:
            raise ValueError("sample counts must be greater than 0")
        if steps <= 0:
            raise ValueError("steps must be greater than 0")
        if positive_source not in ("recorded_only", "recorded_plus_tts", "configured_tts_only"):
            raise ValueError("positive_source must be recorded_only, recorded_plus_tts, or configured_tts_only")
        if negative_source not in ("recorded_only", "recorded_plus_tts", "configured_tts_only"):
            raise ValueError("negative_source must be recorded_only, recorded_plus_tts, or configured_tts_only")

        with self._training_lock:
            active = [job for job in self._training_jobs.values() if job.status in ("queued", "running")]
            if active:
                raise RuntimeError(f"training job already active: {active[0].id}")

            job_id = uuid4().hex
            safe_model_name = self._training_model_name(wakeword, model_name, job_id)
            work_dir = self.training_job_dir / job_id
            output_dir = work_dir / "output"
            work_dir.mkdir(parents=True, exist_ok=True)
            config_path = work_dir / f"{safe_model_name}.yaml"
            log_path = work_dir / "training.log"
            job = WakewordTrainingJob(
                id=job_id,
                created_at=datetime.now(timezone.utc).isoformat(),
                wakeword=wakeword,
                model_name=safe_model_name,
                target_phrases=target_phrases,
                threshold=threshold if threshold is not None else self.settings.audio_wakeword_threshold,
                tts_backend=tts_backend,
                model_size=model_size,
                n_samples=int(n_samples),
                n_samples_val=int(n_samples_val),
                steps=int(steps),
                skip_acav=bool(skip_acav),
                run_eval=bool(run_eval),
                positive_source=positive_source,
                negative_source=negative_source,
                status="queued",
                work_dir=str(work_dir),
                config_path=str(config_path),
                log_path=str(log_path),
            )
            self._write_training_config(job, output_dir=output_dir)
            self._training_jobs[job.id] = job
            self._write_job_metadata(job)

        if background:
            thread = threading.Thread(target=self._run_training_job, args=(job.id,), daemon=True)
            thread.start()
        else:
            self._run_training_job(job.id)
        return job.to_dict()

    def test_model(self, *, model_id: str, candidate_id: str) -> dict:
        from aiavatar.sts.wakeword import LiveKitWakewordDetector

        model = self.get_model(model_id)
        candidate = self.get_candidate(candidate_id)
        audio_bytes, sample_rate, duration = self._read_wav(candidate.path)
        detector = LiveKitWakewordDetector(
            model_paths=[model.path],
            threshold=model.threshold,
            activation_window=max(duration, 1.0),
            cooldown=0,
            debug=self.settings.debug,
        )

        detection = detector.process(
            audio_bytes,
            sample_rate=sample_rate,
            session_id=f"test:{model_id}:{candidate_id}",
        )

        result = WakewordTestResult(
            id=uuid4().hex,
            created_at=datetime.now(timezone.utc).isoformat(),
            model_id=model_id,
            candidate_id=candidate_id,
            wakeword=model.wakeword,
            detected=bool(detection),
            matched_wakeword=detection.matched_wakeword if detection else None,
            score=detection.score if detection else None,
            threshold=model.threshold,
            duration=duration,
        )
        self._test_results.append(result)
        self._test_results = self._test_results[-50:]
        return result.to_dict()

    def _run_training_job(self, job_id: str):
        job = self.get_training_job(job_id)
        job.status = "running"
        job.started_at = datetime.now(timezone.utc).isoformat()
        self._write_job_metadata(job)

        try:
            self._append_job_log(job, f"Starting wakeword training job {job.id}")
            self._prepare_training_artifacts(job)
            if job.tts_backend == self.CONFIGURED_TTS_BACKEND:
                self._generate_configured_tts_training_clips(job)
            commands = self._training_commands(job)
            for command in commands:
                self._run_command(command, job)

            onnx_path = Path(job.work_dir) / "output" / job.model_name / f"{job.model_name}.onnx"
            if not onnx_path.exists():
                raise RuntimeError(f"ONNX output not found: {onnx_path}")

            model = self.import_model(
                wakeword=job.wakeword,
                filename=onnx_path.name,
                content=onnx_path.read_bytes(),
                threshold=job.threshold,
            )
            job.onnx_path = str(onnx_path)
            job.model_id = model["id"]
            job.status = "succeeded"
            self._append_job_log(job, f"Training completed. Registered model: {job.model_id}")
        except Exception as ex:
            job.status = "failed"
            job.error = str(ex)
            self._append_job_log(job, f"Training failed: {ex}")
        finally:
            job.finished_at = datetime.now(timezone.utc).isoformat()
            self._write_job_metadata(job)

    def _training_commands(self, job: WakewordTrainingJob) -> List[List[str]]:
        if job.tts_backend == self.CONFIGURED_TTS_BACKEND:
            commands = [
                [sys.executable, "-m", "livekit.wakeword", "augment", job.config_path],
                [sys.executable, "-m", "livekit.wakeword", "train", job.config_path],
                [sys.executable, "-m", "livekit.wakeword", "export", job.config_path],
            ]
            if job.run_eval:
                commands.append([sys.executable, "-m", "livekit.wakeword", "eval", job.config_path])
            return commands

        commands = [
            [
                sys.executable,
                "-m",
                "livekit.wakeword",
                "setup",
                "--config",
                job.config_path,
            ],
            [sys.executable, "-m", "livekit.wakeword", "generate", job.config_path],
            [sys.executable, "-m", "livekit.wakeword", "augment", job.config_path],
            [sys.executable, "-m", "livekit.wakeword", "train", job.config_path],
            [sys.executable, "-m", "livekit.wakeword", "export", job.config_path],
        ]
        if job.skip_acav:
            commands[0].append("--skip-acav")
        if job.run_eval:
            commands.append([sys.executable, "-m", "livekit.wakeword", "eval", job.config_path])
        return commands

    def _run_command(self, command: List[str], job: WakewordTrainingJob):
        self._append_job_log(job, "$ " + " ".join(command))
        env = os.environ.copy()
        env.setdefault("HF_HUB_DISABLE_XET", "1")
        with subprocess.Popen(
            command,
            cwd=job.work_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as proc:
            assert proc.stdout is not None
            for line in proc.stdout:
                self._append_job_log(job, line.rstrip("\n"))
            return_code = proc.wait()
        job.return_code = return_code
        self._write_job_metadata(job)
        if return_code != 0:
            raise RuntimeError(f"command failed with exit code {return_code}: {' '.join(command)}")

    def _prepare_training_artifacts(self, job: WakewordTrainingJob):
        if job.tts_backend != "voxcpm":
            return

        model_dir = self.training_data_dir / "voxcpm" / "VoxCPM2"
        if not model_dir.exists() or not any(model_dir.iterdir()):
            return

        has_model = (model_dir / "model.safetensors").exists() or (model_dir / "pytorch_model.bin").exists()
        has_audio_vae = (model_dir / "audiovae.safetensors").exists() or (model_dir / "audiovae.pth").exists()
        has_config = (model_dir / "config.json").exists()
        if has_model and has_audio_vae and has_config:
            return

        missing = []
        if not has_config:
            missing.append("config.json")
        if not has_model:
            missing.append("model.safetensors or pytorch_model.bin")
        if not has_audio_vae:
            missing.append("audiovae.safetensors or audiovae.pth")

        self._append_job_log(
            job,
            f"Completing incomplete VoxCPM download at {model_dir}; missing: {', '.join(missing)}",
        )
        self._download_voxcpm_snapshot(job, model_dir)

        has_model = (model_dir / "model.safetensors").exists() or (model_dir / "pytorch_model.bin").exists()
        has_audio_vae = (model_dir / "audiovae.safetensors").exists() or (model_dir / "audiovae.pth").exists()
        has_config = (model_dir / "config.json").exists()
        if not (has_model and has_audio_vae and has_config):
            raise RuntimeError(f"VoxCPM download is still incomplete: {model_dir}")

    def _download_voxcpm_snapshot(self, job: WakewordTrainingJob, model_dir: Path):
        try:
            from huggingface_hub import snapshot_download
        except ImportError as ex:
            raise RuntimeError("huggingface-hub is required for VoxCPM training") from ex

        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        model_dir.mkdir(parents=True, exist_ok=True)
        self._append_job_log(job, "Downloading missing VoxCPM files from openbmb/VoxCPM2")
        snapshot_download(repo_id="openbmb/VoxCPM2", local_dir=str(model_dir))

    def _generate_configured_tts_training_clips(self, job: WakewordTrainingJob):
        self._append_job_log(job, "Generating training clips with configured TTS provider")
        model_dir = Path(job.work_dir) / "output" / job.model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        positive_seeds = self._positive_seed_audio(job)
        if not positive_seeds:
            raise RuntimeError("no positive audio seeds were generated")

        negative_seeds = self._negative_seed_audio(job)
        if not negative_seeds:
            raise RuntimeError("no negative audio seeds were available")

        self._write_seed_split(model_dir / "positive_train", positive_seeds, job.n_samples, positive=True)
        self._write_seed_split(model_dir / "positive_test", positive_seeds, job.n_samples_val, positive=True)
        self._write_seed_split(model_dir / "negative_train", negative_seeds, job.n_samples, positive=False)
        self._write_seed_split(model_dir / "negative_test", negative_seeds, job.n_samples_val, positive=False)
        self._write_background_split(model_dir / "background_train", max(20, job.n_samples // 10))
        self._write_background_split(model_dir / "background_test", max(10, job.n_samples_val // 10))

        self._append_job_log(
            job,
            f"Generated configured-TTS clips: positive={len(positive_seeds)} seeds, "
            f"negative={len(negative_seeds)} seeds",
        )

    def _positive_seed_audio(self, job: WakewordTrainingJob):
        seeds = []
        if job.positive_source in ("recorded_only", "recorded_plus_tts"):
            for candidate in self._candidates.values():
                if self._candidate_role(candidate, job.wakeword) != "positive":
                    continue
                try:
                    audio_bytes, sample_rate, _duration = self._read_wav(candidate.path)
                    seeds.append(self._pcm16_to_float32(audio_bytes, sample_rate))
                except Exception as ex:
                    self._append_job_log(job, f"Skipping positive candidate seed {candidate.id}: {ex}")

        if job.positive_source in ("recorded_plus_tts", "configured_tts_only"):
            for phrase in job.target_phrases:
                try:
                    seeds.append(self._synthesize_configured_tts_audio(phrase))
                except Exception as ex:
                    self._append_job_log(job, f"Configured TTS failed for positive phrase '{phrase}': {ex}")

        return [seed for seed in seeds if seed.size > 0]

    def _negative_seed_audio(self, job: WakewordTrainingJob):
        seeds = []
        if job.negative_source in ("recorded_only", "recorded_plus_tts"):
            for candidate in self._candidates.values():
                if self._candidate_role(candidate, job.wakeword) != "negative":
                    continue
                try:
                    audio_bytes, sample_rate, _duration = self._read_wav(candidate.path)
                    seeds.append(self._pcm16_to_float32(audio_bytes, sample_rate))
                except Exception as ex:
                    self._append_job_log(job, f"Skipping negative candidate seed {candidate.id}: {ex}")

        if job.negative_source in ("recorded_plus_tts", "configured_tts_only"):
            for phrase in self._negative_phrases(job):
                try:
                    seeds.append(self._synthesize_configured_tts_audio(phrase))
                except Exception as ex:
                    self._append_job_log(job, f"Configured TTS failed for negative phrase '{phrase}': {ex}")

        return [seed for seed in seeds if seed.size > 0]

    @staticmethod
    def _candidate_role(candidate: WakewordCandidate, wakeword: str) -> str:
        if candidate.role in ("positive", "negative", "ignore"):
            return candidate.role
        if not candidate.wakeword:
            return "ignore"
        return "positive" if candidate.wakeword == wakeword else "negative"

    def _negative_phrases(self, job: WakewordTrainingJob) -> List[str]:
        target = {phrase.strip() for phrase in job.target_phrases}
        phrases = []
        for phrase in self.DEFAULT_NEGATIVE_PHRASES:
            if phrase and phrase not in target and phrase not in phrases:
                phrases.append(phrase)
        return phrases

    def _synthesize_configured_tts_audio(self, text: str):
        from server.providers.tts import create_tts

        async def _run():
            synthesizer = create_tts(self.settings)
            try:
                return await synthesizer.synthesize(text, language="ja")
            finally:
                close = getattr(synthesizer, "close", None)
                if close:
                    await close()

        audio_bytes = asyncio.run(_run())
        return self._decode_audio_bytes(audio_bytes)

    def _write_seed_split(self, split_dir: Path, seeds: List["np.ndarray"], count: int, *, positive: bool):
        existing = self._count_original_clips(split_dir)
        if existing >= count:
            return
        split_dir.mkdir(parents=True, exist_ok=True)
        rng = random.Random(f"{split_dir}:{count}:{positive}")
        for index in range(existing, count):
            seed = rng.choice(seeds)
            audio = self._variant_audio(seed, rng)
            path = split_dir / f"clip_{index:06d}.wav"
            self._write_float_wav(path, audio, 16000)

    def _write_background_split(self, split_dir: Path, count: int):
        existing = self._count_original_clips(split_dir)
        if existing >= count:
            return
        import numpy as np

        split_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(abs(hash(str(split_dir))) % (2**32))
        for index in range(existing, count):
            noise = rng.normal(0.0, 0.012, 32000).astype(np.float32)
            self._write_float_wav(split_dir / f"clip_{index:06d}.wav", noise, 16000)

    def _variant_audio(self, seed, rng: random.Random):
        import numpy as np

        audio = self._trim_silence(seed)
        if audio.size == 0:
            audio = seed
        speed = rng.uniform(0.90, 1.10)
        if audio.size > 1 and abs(speed - 1.0) > 0.01:
            target_len = max(1, int(audio.size / speed))
            audio = np.interp(
                np.linspace(0, audio.size - 1, target_len),
                np.arange(audio.size),
                audio,
            ).astype(np.float32)
        audio = audio * rng.uniform(0.75, 1.15)
        noise_level = rng.uniform(0.0, 0.003)
        if noise_level > 0:
            audio = audio + np.random.default_rng(rng.randrange(2**32)).normal(
                0.0,
                noise_level,
                audio.shape,
            ).astype(np.float32)
        return np.clip(audio, -1.0, 1.0).astype(np.float32)

    @staticmethod
    def _trim_silence(audio):
        import numpy as np

        if audio.size == 0:
            return audio
        mask = np.abs(audio) > 0.01
        if not np.any(mask):
            return audio
        indices = np.flatnonzero(mask)
        pad = 1600
        start = max(0, int(indices[0]) - pad)
        end = min(audio.size, int(indices[-1]) + pad)
        return audio[start:end]

    @staticmethod
    def _decode_audio_bytes(audio_bytes: bytes):
        import numpy as np

        if not audio_bytes:
            raise ValueError("configured TTS returned empty audio")
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())

        if sampwidth == 1:
            audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif sampwidth == 2:
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"unsupported WAV sample width: {sampwidth}")

        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
        if sample_rate != 16000:
            audio = WakewordEnrollmentManager._resample_linear(audio, sample_rate, 16000)
        return np.asarray(audio, dtype=np.float32)

    @staticmethod
    def _resample_linear(audio, sample_rate: int, target_sample_rate: int):
        import numpy as np

        if sample_rate == target_sample_rate or audio.size == 0:
            return audio.astype(np.float32)
        target_len = max(1, int(round(audio.size * target_sample_rate / sample_rate)))
        return np.interp(
            np.linspace(0, audio.size - 1, target_len),
            np.arange(audio.size),
            audio,
        ).astype(np.float32)

    @staticmethod
    def _pcm16_to_float32(audio_bytes: bytes, sample_rate: int):
        import numpy as np

        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if sample_rate != 16000:
            audio = WakewordEnrollmentManager._resample_linear(audio, sample_rate, 16000)
        return np.asarray(audio, dtype=np.float32)

    @staticmethod
    def _write_float_wav(path: Path, audio, sample_rate: int):
        import numpy as np

        path.parent.mkdir(parents=True, exist_ok=True)
        pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm.tobytes())

    @staticmethod
    def _count_original_clips(directory: Path) -> int:
        if not directory.is_dir():
            return 0
        return sum(1 for path in directory.iterdir() if re.match(r"^clip_\d{6}\.wav$", path.name))

    def _write_training_config(self, job: WakewordTrainingJob, *, output_dir: Path):
        data_dir = self.training_data_dir
        config = {
            "model_name": job.model_name,
            "target_phrases": job.target_phrases,
            "n_samples": job.n_samples,
            "n_samples_val": job.n_samples_val,
            "n_background_samples": max(20, job.n_samples // 10),
            "n_background_samples_val": max(10, job.n_samples_val // 10),
            "tts_batch_size": 20,
            "tts_backend": self.CONFIG_TTS_BACKEND if job.tts_backend == self.CONFIGURED_TTS_BACKEND else job.tts_backend,
            "data_dir": str(data_dir),
            "output_dir": str(output_dir),
            "augmentation": {
                "clip_duration": 2.0,
                "batch_size": 16,
                "rounds": 1,
                "background_paths": [str(data_dir / "backgrounds")],
                "rir_paths": [str(data_dir / "rirs")],
            },
            "model": {
                "model_type": "conv_attention",
                "model_size": job.model_size,
            },
            "steps": job.steps,
            "target_fp_per_hour": 0.2,
        }
        Path(job.config_path).write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")

    def _append_job_log(self, job: WakewordTrainingJob, message: str):
        path = Path(job.log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).isoformat()
        with path.open("a") as fp:
            fp.write(f"[{timestamp}] {message}\n")

    def _load_existing_candidates(self):
        for metadata_path in sorted(self.candidate_dir.glob("*.json")):
            try:
                data = json.loads(metadata_path.read_text())
                candidate = WakewordCandidate(
                    id=data["id"],
                    created_at=data["created_at"],
                    wakeword=data.get("wakeword"),
                    role=data.get("role") if data.get("role") in ("positive", "negative", "ignore") else None,
                    session_id=data.get("session_id") or "",
                    duration=float(data.get("duration") or 0),
                    sample_rate=int(data.get("sample_rate") or 16000),
                    path=data["path"],
                )
                if Path(candidate.path).exists():
                    self._candidates[candidate.id] = candidate
            except Exception:
                continue

    def _load_existing_models(self):
        for metadata_path in sorted(self.model_dir.glob("*.json")):
            try:
                data = json.loads(metadata_path.read_text())
                model = WakewordModel(
                    id=data["id"],
                    created_at=data["created_at"],
                    wakeword=data["wakeword"],
                    filename=data.get("filename") or f"{data['id']}.onnx",
                    path=data["path"],
                    threshold=float(data.get("threshold") or self.settings.audio_wakeword_threshold),
                )
                if Path(model.path).exists():
                    self._models[model.id] = model
            except Exception:
                continue

    def _load_existing_training_jobs(self):
        for metadata_path in sorted(self.training_job_dir.glob("*/job.json")):
            try:
                data = json.loads(metadata_path.read_text())
                job = WakewordTrainingJob(
                    id=data["id"],
                    created_at=data["created_at"],
                    wakeword=data["wakeword"],
                    model_name=data["model_name"],
                    target_phrases=list(data.get("target_phrases") or [data["wakeword"]]),
                    threshold=float(data.get("threshold") or self.settings.audio_wakeword_threshold),
                    tts_backend=data.get("tts_backend") or "voxcpm",
                    model_size=data.get("model_size") or "tiny",
                    n_samples=int(data.get("n_samples") or 500),
                    n_samples_val=int(data.get("n_samples_val") or 100),
                    steps=int(data.get("steps") or 2000),
                    skip_acav=bool(data.get("skip_acav", True)),
                    run_eval=bool(data.get("run_eval", False)),
                    positive_source=data.get("positive_source") or "recorded_plus_tts",
                    negative_source=data.get("negative_source") or "recorded_plus_tts",
                    status=data.get("status") or "failed",
                    work_dir=data["work_dir"],
                    config_path=data["config_path"],
                    log_path=data["log_path"],
                    onnx_path=data.get("onnx_path"),
                    model_id=data.get("model_id"),
                    started_at=data.get("started_at"),
                    finished_at=data.get("finished_at"),
                    error=data.get("error"),
                    return_code=data.get("return_code"),
                )
                if job.status in ("queued", "running"):
                    job.status = "failed"
                    job.error = "server restarted while job was active"
                    job.finished_at = datetime.now(timezone.utc).isoformat()
                    self._write_job_metadata(job)
                self._training_jobs[job.id] = job
            except Exception:
                continue

    def _candidate_metadata_path(self, candidate_id: str) -> Path:
        return self.candidate_dir / f"{candidate_id}.json"

    def _model_metadata_path(self, model_id: str) -> Path:
        return self.model_dir / f"{model_id}.json"

    def _write_job_metadata(self, job: WakewordTrainingJob):
        path = Path(job.work_dir) / "job.json"
        self._write_json(path, job.to_dict())

    def _model_id(self, wakeword: str, filename: str) -> str:
        stem = Path(filename).stem
        base = self._slug(f"{wakeword}-{stem}") or uuid4().hex
        model_id = base
        index = 2
        while model_id in self._models:
            model_id = f"{base}-{index}"
            index += 1
        return model_id

    def _training_model_name(self, wakeword: str, model_name: Optional[str], job_id: str) -> str:
        base = self._slug(model_name or wakeword) or f"wakeword-{job_id[:8]}"
        return base.replace("-", "_")

    @staticmethod
    def _slug(value: str) -> str:
        slug = re.sub(r"[^0-9A-Za-z_.-]+", "-", value.strip()).strip("-_.")
        return slug.lower()

    @staticmethod
    def _write_json(path: Path, data: dict):
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    @staticmethod
    def _write_wav(path: Path, audio_bytes: bytes, sample_rate: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_bytes)

    @staticmethod
    def _read_wav(path: str | Path) -> tuple[bytes, int, float]:
        with wave.open(str(path), "rb") as wf:
            sample_rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            duration = wf.getnframes() / sample_rate if sample_rate else 0.0
        return frames, sample_rate, duration
