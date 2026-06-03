import importlib.util
import logging
from pathlib import Path
from threading import Lock
from typing import Dict, Optional, Sequence
from urllib.parse import quote

import numpy as np

from .base import VoiceAuthResult, VoiceAuthenticator

logger = logging.getLogger(__name__)


class WespeakerMlxVoiceAuthenticator(VoiceAuthenticator):
    """WeSpeaker ResNet34-LM verifier backed by a local MLX model directory.

    The expected model directory is Landon41/wespeaker-voxceleb-resnet34-LM-mlx,
    downloaded from Hugging Face. It must include resnet_embedding.py and
    weights.npz.
    """

    def __init__(
        self,
        *,
        model_path: str,
        profile_dir: str,
        threshold: float = 0.70,
        min_duration: float = 1.2,
        target_sample_rate: int = 16000,
        require_user_id: bool = True,
        allow_identification: bool = False,
        allowed_users: Optional[Sequence[str]] = None,
        fail_open: bool = False,
        apply_cmn: bool = True,
        debug: bool = False,
    ):
        self.model_path = Path(model_path)
        self.profile_dir = Path(profile_dir)
        self.threshold = float(threshold)
        self.min_duration = float(min_duration)
        self.target_sample_rate = int(target_sample_rate)
        self.require_user_id = bool(require_user_id)
        self.allow_identification = bool(allow_identification)
        self.allowed_users = {user for user in (allowed_users or []) if user}
        self.fail_open = bool(fail_open)
        self.apply_cmn = bool(apply_cmn)
        self.debug = debug

        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self._profiles: Dict[str, np.ndarray] = {}
        self._lock = Lock()

        self._mx = self._import_required("mlx.core", "mlx")
        self.model = self._load_model()
        self.reload_profiles()

    def _import_required(self, module_name: str, package_name: str):
        try:
            return __import__(module_name, fromlist=["*"])
        except Exception as ex:
            raise RuntimeError(
                f"{package_name} is required for VOICE_AUTH_PROVIDER=wespeaker_mlx. "
                f"Install the voice-auth extra before enabling this provider."
            ) from ex

    def _load_model(self):
        model_file = self.model_path / "resnet_embedding.py"
        weights_file = self.model_path / "weights.npz"
        if not model_file.exists():
            raise FileNotFoundError(f"Missing MLX model file: {model_file}")
        if not weights_file.exists():
            raise FileNotFoundError(f"Missing MLX weights file: {weights_file}")

        spec = importlib.util.spec_from_file_location("wespeaker_mlx_resnet_embedding", model_file)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load MLX model module: {model_file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        model = module.ResNet34Embedding()
        weights = np.load(weights_file)
        for key in weights.files:
            target = model
            parts = key.split(".")
            for attr in parts[:-1]:
                if attr.isdigit():
                    target = target[int(attr)]
                elif attr == "layers":
                    target = target.layers
                else:
                    target = getattr(target, attr)
            setattr(target, parts[-1], self._mx.array(weights[key]))
        model.eval()
        return model

    def reload_profiles(self):
        profiles: Dict[str, np.ndarray] = {}
        for path in sorted(self.profile_dir.glob("*.npz")):
            try:
                data = np.load(path, allow_pickle=True)
                embedding = data["embedding"].astype(np.float32, copy=False)
                user_id = str(data["user_id"]) if "user_id" in data else path.stem
                profiles[user_id] = self._normalize(embedding)
            except Exception:
                logger.warning("Failed to load voice profile: %s", path, exc_info=True)
        self._profiles = profiles
        logger.info("Loaded %d voice auth profiles from %s", len(self._profiles), self.profile_dir)

    def save_profile(self, user_id: str, embeddings: np.ndarray):
        if not user_id:
            raise ValueError("user_id is required")
        emb = np.asarray(embeddings, dtype=np.float32)
        if emb.ndim == 2:
            emb = emb.mean(axis=0)
        emb = self._normalize(emb)
        path = self._profile_path(user_id)
        np.savez_compressed(
            path,
            user_id=user_id,
            embedding=emb,
            model="wespeaker-resnet34-lm-mlx",
            samples=int(np.asarray(embeddings).shape[0]) if np.asarray(embeddings).ndim == 2 else 1,
        )
        self._profiles[user_id] = emb

    def enroll_user_from_pcm(self, user_id: str, samples: list[tuple[bytes, int]]):
        embeddings = [
            self.embed_pcm(audio_bytes=audio_bytes, sample_rate=sample_rate)
            for audio_bytes, sample_rate in samples
        ]
        self.save_profile(user_id, np.vstack(embeddings))

    def verify_sync(
        self,
        *,
        user_id: Optional[str],
        audio_bytes: bytes,
        sample_rate: int,
        audio_duration: Optional[float] = None,
    ) -> VoiceAuthResult:
        try:
            return self._verify_sync(
                user_id=user_id,
                audio_bytes=audio_bytes,
                sample_rate=sample_rate,
                audio_duration=audio_duration,
            )
        except Exception as ex:
            logger.warning("Voice authentication failed internally: %s", ex, exc_info=self.debug)
            if self.fail_open:
                return VoiceAuthResult(
                    accepted=True,
                    user_id=user_id,
                    reason="voice_auth_error_fail_open",
                    metadata={"error": str(ex)},
                )
            return VoiceAuthResult(
                accepted=False,
                user_id=user_id,
                reason="voice_auth_error",
                metadata={"error": str(ex)},
            )

    def _verify_sync(
        self,
        *,
        user_id: Optional[str],
        audio_bytes: bytes,
        sample_rate: int,
        audio_duration: Optional[float] = None,
    ) -> VoiceAuthResult:
        if not audio_bytes:
            return VoiceAuthResult(False, user_id=user_id, threshold=self.threshold, reason="empty_audio")

        duration = audio_duration
        if duration is None and sample_rate > 0:
            duration = (len(audio_bytes) / 2) / sample_rate
        if duration is not None and duration < self.min_duration:
            return VoiceAuthResult(
                False,
                user_id=user_id,
                threshold=self.threshold,
                reason="audio_too_short",
                metadata={"duration": duration, "min_duration": self.min_duration},
            )

        if self.require_user_id and not user_id and not self.allow_identification:
            return VoiceAuthResult(False, threshold=self.threshold, reason="missing_user_id")

        emb = self.embed_pcm(audio_bytes=audio_bytes, sample_rate=sample_rate)

        if self.allow_identification:
            return self._identify(user_id=user_id, emb=emb)

        if user_id and user_id in self._profiles:
            return self._verify_claimed(user_id=user_id, emb=emb)

        if user_id and user_id not in self._profiles:
            return VoiceAuthResult(False, user_id=user_id, threshold=self.threshold, reason="profile_not_found")

        if self.require_user_id:
            return VoiceAuthResult(False, threshold=self.threshold, reason="missing_user_id")

        return self._identify(user_id=user_id, emb=emb)

    def _verify_claimed(self, *, user_id: str, emb: np.ndarray) -> VoiceAuthResult:
        target = self._profiles[user_id]
        sim = self._cosine(target, emb)
        if sim < self.threshold:
            return VoiceAuthResult(
                accepted=False,
                user_id=user_id,
                matched_user_id=user_id,
                similarity=sim,
                threshold=self.threshold,
                reason="below_threshold",
            )
        if self.allowed_users and user_id not in self.allowed_users:
            return VoiceAuthResult(
                accepted=False,
                user_id=user_id,
                matched_user_id=user_id,
                similarity=sim,
                threshold=self.threshold,
                reason="user_not_allowed",
            )
        return VoiceAuthResult(
            accepted=True,
            user_id=user_id,
            matched_user_id=user_id,
            similarity=sim,
            threshold=self.threshold,
            reason="matched",
        )

    def _identify(self, *, user_id: Optional[str], emb: np.ndarray) -> VoiceAuthResult:
        if not self._profiles:
            return VoiceAuthResult(False, user_id=user_id, threshold=self.threshold, reason="no_profiles")

        matched_user_id, sim = max(
            ((profile_user_id, self._cosine(profile, emb)) for profile_user_id, profile in self._profiles.items()),
            key=lambda item: item[1],
        )
        if sim < self.threshold:
            return VoiceAuthResult(
                accepted=False,
                user_id=user_id,
                matched_user_id=matched_user_id,
                similarity=sim,
                threshold=self.threshold,
                reason="below_threshold",
            )
        if self.allowed_users and matched_user_id not in self.allowed_users:
            return VoiceAuthResult(
                accepted=False,
                user_id=user_id,
                matched_user_id=matched_user_id,
                similarity=sim,
                threshold=self.threshold,
                reason="user_not_allowed",
            )
        return VoiceAuthResult(
            accepted=True,
            user_id=user_id,
            matched_user_id=matched_user_id,
            similarity=sim,
            threshold=self.threshold,
            reason="identified",
        )

    def embed_pcm(self, *, audio_bytes: bytes, sample_rate: int) -> np.ndarray:
        audio = self._pcm_to_float32(audio_bytes)
        if sample_rate != self.target_sample_rate:
            audio = self._resample_linear(audio, sample_rate, self.target_sample_rate)
        fbank = self._compute_fbank(audio, self.target_sample_rate)
        with self._lock:
            out = self.model(self._mx.array(fbank[np.newaxis, :, :]))
            self._mx.eval(out)
        return self._normalize(np.asarray(out, dtype=np.float32).reshape(-1))

    def _compute_fbank(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        try:
            import torch
            import torchaudio.compliance.kaldi as kaldi
        except Exception as ex:
            raise RuntimeError(
                "torch and torchaudio are required to compute WeSpeaker-compatible fbank features."
            ) from ex

        waveform = torch.from_numpy(audio.astype(np.float32, copy=False)).unsqueeze(0)
        feats = kaldi.fbank(
            waveform,
            num_mel_bins=80,
            frame_length=25,
            frame_shift=10,
            sample_frequency=sample_rate,
            dither=0.0,
            energy_floor=0.0,
        )
        if self.apply_cmn:
            feats = feats - feats.mean(dim=0, keepdim=True)
        return feats.cpu().numpy().astype(np.float32, copy=False)

    def _profile_path(self, user_id: str) -> Path:
        return self.profile_dir / f"{quote(user_id, safe='')}.npz"

    @staticmethod
    def _pcm_to_float32(audio_bytes: bytes) -> np.ndarray:
        if len(audio_bytes) % 2:
            audio_bytes = audio_bytes[:-1]
        return np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    @staticmethod
    def _resample_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
        if source_rate <= 0 or source_rate == target_rate:
            return audio
        source_duration = len(audio) / source_rate
        target_len = max(1, int(round(source_duration * target_rate)))
        source_x = np.linspace(0.0, source_duration, num=len(audio), endpoint=False)
        target_x = np.linspace(0.0, source_duration, num=target_len, endpoint=False)
        return np.interp(target_x, source_x, audio).astype(np.float32)

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float32)
        return v / (np.linalg.norm(v) + 1e-9)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(WespeakerMlxVoiceAuthenticator._normalize(a), WespeakerMlxVoiceAuthenticator._normalize(b)))

    def get_config(self):
        return {
            "model_path": str(self.model_path),
            "profile_dir": str(self.profile_dir),
            "threshold": self.threshold,
            "min_duration": self.min_duration,
            "target_sample_rate": self.target_sample_rate,
            "require_user_id": self.require_user_id,
            "allow_identification": self.allow_identification,
            "allowed_users": sorted(self.allowed_users),
            "fail_open": self.fail_open,
            "apply_cmn": self.apply_cmn,
            "profiles": len(self._profiles),
            "debug": self.debug,
        }
