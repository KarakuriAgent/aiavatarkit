import asyncio
import wave

from aiavatar.sts.audio_enhancement import AudioEnhancementResult
from aiavatar.sts.voice_auth import VoiceAuthResult
from voice_auth_server import enrollment


class FakeSettings:
    def __init__(self, root_dir):
        self.voice_auth_enrollment_dir = str(root_dir)
        self.voice_auth_provider = "http"
        self.voice_auth_base_url = "http://voice-auth-runtime:8765"
        self.voice_auth_api_key = None
        self.aiavatar_api_key = None
        self.stt_timeout = 30
        self.debug = False


class FakeAudioEnhancer:
    def __init__(self):
        self.calls = []

    async def enhance(self, *, audio_bytes, sample_rate, session_id=None):
        self.calls.append(
            {
                "audio_bytes": audio_bytes,
                "sample_rate": sample_rate,
                "session_id": session_id,
            }
        )
        return AudioEnhancementResult(
            audio_bytes=b"\x03\x00\x04\x00",
            provider="fake",
            metadata={},
        )


class FakeVoiceAuthClient:
    def __init__(self):
        self.verify_calls = []

    async def profiles(self):
        return {"profiles": ["user01"], "allowed_users": ["user01"]}

    async def verify(
        self,
        *,
        user_id,
        audio_bytes,
        sample_rate,
        audio_duration=None,
        threshold=None,
        min_duration=None,
    ):
        self.verify_calls.append(
            {
                "user_id": user_id,
                "audio_bytes": audio_bytes,
                "sample_rate": sample_rate,
                "audio_duration": audio_duration,
                "threshold": threshold,
                "min_duration": min_duration,
            }
        )
        return VoiceAuthResult(
            accepted=True,
            user_id=user_id,
            matched_user_id=user_id,
            similarity=0.91,
            threshold=0.5,
            reason="matched",
        )


def read_wav_frames(path):
    with wave.open(str(path), "rb") as wf:
        return wf.readframes(wf.getnframes()), wf.getframerate()


def test_candidate_audio_url_points_to_enhanced_preview(monkeypatch, tmp_path):
    enhancer = FakeAudioEnhancer()
    monkeypatch.setattr(enrollment, "create_required_audio_enhancer", lambda settings: enhancer)
    manager = enrollment.VoiceEnrollmentManager(FakeSettings(tmp_path))
    manager.start_reading()

    candidate = manager.add_candidate(
        audio_bytes=b"\x01\x00\x02\x00",
        sample_rate=16000,
        duration=0.1,
        session_id="session-1",
        user_id="user01",
    )

    data = candidate.to_dict()

    assert data["audio_url"].endswith(f"/api/candidates/{candidate.id}/audio?variant=enhanced")
    assert data["raw_audio_url"].endswith(f"/api/candidates/{candidate.id}/audio?variant=raw")


def test_candidate_enhanced_audio_is_generated_and_cached(monkeypatch, tmp_path):
    enhancer = FakeAudioEnhancer()
    monkeypatch.setattr(enrollment, "create_required_audio_enhancer", lambda settings: enhancer)
    manager = enrollment.VoiceEnrollmentManager(FakeSettings(tmp_path))
    manager.start_reading()
    candidate = manager.add_candidate(
        audio_bytes=b"\x01\x00\x02\x00",
        sample_rate=16000,
        duration=0.1,
        session_id="session-1",
        user_id="user01",
    )

    enhanced_path = asyncio.run(manager.get_candidate_audio_path(candidate.id))
    second_path = asyncio.run(manager.get_candidate_audio_path(candidate.id))
    raw_path = asyncio.run(manager.get_candidate_audio_path(candidate.id, variant="raw"))

    assert enhanced_path == second_path
    assert len(enhancer.calls) == 1
    assert read_wav_frames(enhanced_path) == (b"\x03\x00\x04\x00", 16000)
    assert read_wav_frames(raw_path) == (b"\x01\x00\x02\x00", 16000)


def test_clear_candidates_removes_raw_and_enhanced_audio(monkeypatch, tmp_path):
    enhancer = FakeAudioEnhancer()
    monkeypatch.setattr(enrollment, "create_required_audio_enhancer", lambda settings: enhancer)
    manager = enrollment.VoiceEnrollmentManager(FakeSettings(tmp_path))
    manager.start_reading()
    candidate = manager.add_candidate(
        audio_bytes=b"\x01\x00\x02\x00",
        sample_rate=16000,
        duration=0.1,
        session_id="session-1",
        user_id="user01",
    )
    raw_path = asyncio.run(manager.get_candidate_audio_path(candidate.id, variant="raw"))
    enhanced_path = asyncio.run(manager.get_candidate_audio_path(candidate.id))

    manager.clear_candidates()

    assert not raw_path.exists()
    assert not enhanced_path.exists()


def test_voice_auth_test_verifies_enhanced_audio(monkeypatch, tmp_path):
    enhancer = FakeAudioEnhancer()
    auth_client = FakeVoiceAuthClient()
    monkeypatch.setattr(enrollment, "create_required_audio_enhancer", lambda settings: enhancer)
    manager = enrollment.VoiceEnrollmentManager(FakeSettings(tmp_path))
    manager._get_http_auth = lambda: auth_client

    manager.start_reading()
    manager.start_test("user01")
    result = asyncio.run(
        manager.add_test_result(
            audio_bytes=b"\x01\x00\x02\x00",
            sample_rate=16000,
            duration=1.5,
            session_id="session-1",
        )
    )

    assert manager.reading_enabled is False
    assert result.accepted is True
    assert result.reason == "matched"
    assert result.similarity == 0.91
    assert auth_client.verify_calls == [
        {
            "user_id": "user01",
            "audio_bytes": b"\x03\x00\x04\x00",
            "sample_rate": 16000,
            "audio_duration": 1.5,
            "threshold": None,
            "min_duration": None,
        }
    ]
    assert manager.list_test_results()[0]["accepted"] is True


def test_voice_auth_test_passes_setting_overrides(monkeypatch, tmp_path):
    enhancer = FakeAudioEnhancer()
    auth_client = FakeVoiceAuthClient()
    monkeypatch.setattr(enrollment, "create_required_audio_enhancer", lambda settings: enhancer)
    manager = enrollment.VoiceEnrollmentManager(FakeSettings(tmp_path))
    manager._get_http_auth = lambda: auth_client

    manager.start_test("user01", threshold=0.42, min_duration=0.0)
    result = asyncio.run(
        manager.add_test_result(
            audio_bytes=b"\x01\x00\x02\x00",
            sample_rate=16000,
            duration=0.4,
            session_id="session-1",
        )
    )

    assert result.metadata["test_threshold"] == 0.42
    assert result.metadata["test_min_duration"] == 0.0
    assert auth_client.verify_calls[0]["threshold"] == 0.42
    assert auth_client.verify_calls[0]["min_duration"] == 0.0


def test_start_reading_stops_voice_auth_test(monkeypatch, tmp_path):
    monkeypatch.setattr(enrollment, "create_required_audio_enhancer", lambda settings: FakeAudioEnhancer())
    manager = enrollment.VoiceEnrollmentManager(FakeSettings(tmp_path))

    manager.start_test("user01")
    manager.start_reading()

    assert manager.reading_enabled is True
    assert manager.test_enabled is False
    assert manager.test_user_id is None
