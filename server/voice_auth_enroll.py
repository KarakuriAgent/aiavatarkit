import argparse
import os
import wave
from pathlib import Path

import numpy as np

from .config import load_settings
from aiavatar.sts.voice_auth.wespeaker_mlx import WespeakerMlxVoiceAuthenticator


def read_wav_pcm(path: Path) -> tuple[bytes, int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if sample_width != 2:
        raise ValueError(f"{path}: only 16-bit PCM WAV is supported")

    if channels == 1:
        return frames, sample_rate

    samples = np.frombuffer(frames, dtype=np.int16).reshape(-1, channels)
    mono = samples.mean(axis=1).astype(np.int16)
    return mono.tobytes(), sample_rate


def main():
    parser = argparse.ArgumentParser(description="Enroll a WeSpeaker MLX voice profile from WAV files.")
    parser.add_argument("user_id")
    parser.add_argument("wav_files", nargs="+", type=Path)
    parser.add_argument("--env", type=Path, default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--profile-dir", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    os.environ.setdefault("HERMES_API_KEY", "unused-voice-auth-enroll")
    settings = load_settings(args.env)
    model_path = args.model_path or settings.voice_auth_model_path
    if not model_path:
        raise RuntimeError("VOICE_AUTH_MODEL_PATH or --model-path is required")

    auth = WespeakerMlxVoiceAuthenticator(
        model_path=model_path,
        profile_dir=args.profile_dir or settings.voice_auth_profile_dir,
        threshold=args.threshold if args.threshold is not None else settings.voice_auth_threshold,
        min_duration=settings.voice_auth_min_duration,
        target_sample_rate=settings.voice_auth_sample_rate,
        require_user_id=settings.voice_auth_require_user_id,
        allow_identification=settings.voice_auth_allow_identification,
        fail_open=settings.voice_auth_fail_open,
        apply_cmn=settings.voice_auth_apply_cmn,
        debug=settings.debug,
    )
    samples = [read_wav_pcm(path) for path in args.wav_files]
    auth.enroll_user_from_pcm(args.user_id, samples)
    print(f"Saved voice profile for {args.user_id} in {auth.profile_dir}")


if __name__ == "__main__":
    main()
