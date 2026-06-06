import asyncio
import wave

from aiavatar.sts.audio_enhancement import AudioEnhancementResult
from server import voice_auth_enroll


class FakeEnrollmentAudioEnhancer:
    def __init__(self):
        self.calls = []
        self.closed = False

    async def enhance(self, *, audio_bytes, sample_rate, session_id=None):
        self.calls.append(
            {
                "audio_bytes": audio_bytes,
                "sample_rate": sample_rate,
                "session_id": session_id,
            }
        )
        return AudioEnhancementResult(
            audio_bytes=b"enhanced-pcm",
            provider="fake",
            metadata={"forced": True},
        )

    async def close(self):
        self.closed = True


def write_wav(path, frames: bytes, sample_rate: int = 16000):
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frames)


def test_voice_auth_enroll_uses_enhanced_audio(monkeypatch, tmp_path):
    fake_enhancer = FakeEnrollmentAudioEnhancer()
    monkeypatch.setattr(
        voice_auth_enroll,
        "create_required_audio_enhancer",
        lambda settings: fake_enhancer,
    )
    raw_pcm = b"\x01\x00\x02\x00"
    wav_path = tmp_path / "sample.wav"
    write_wav(wav_path, raw_pcm)

    samples = asyncio.run(
        voice_auth_enroll.read_enhanced_samples(
            "user01",
            [wav_path],
            settings=object(),
        )
    )

    assert samples == [(b"enhanced-pcm", 16000)]
    assert fake_enhancer.calls == [
        {
            "audio_bytes": raw_pcm,
            "sample_rate": 16000,
            "session_id": "voice-auth-enroll:user01:sample.wav",
        }
    ]
    assert fake_enhancer.closed is True
