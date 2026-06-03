import numpy as np

from aiavatar.sts.voice_auth.wespeaker_mlx import WespeakerMlxVoiceAuthenticator


def make_auth(*, allowed_users=None):
    auth = object.__new__(WespeakerMlxVoiceAuthenticator)
    auth.threshold = 0.70
    auth.allowed_users = set(allowed_users or [])
    auth._profiles = {
        "allowed": np.array([1.0, 0.0], dtype=np.float32),
        "blocked": np.array([0.0, 1.0], dtype=np.float32),
    }
    return auth


def test_identification_accepts_allowed_user():
    auth = make_auth(allowed_users=["allowed"])

    result = auth._identify(user_id=None, emb=np.array([1.0, 0.0], dtype=np.float32))

    assert result.accepted is True
    assert result.matched_user_id == "allowed"
    assert result.reason == "identified"


def test_identification_rejects_user_outside_allowlist():
    auth = make_auth(allowed_users=["allowed"])

    result = auth._identify(user_id=None, emb=np.array([0.0, 1.0], dtype=np.float32))

    assert result.accepted is False
    assert result.matched_user_id == "blocked"
    assert result.reason == "user_not_allowed"


def test_claimed_verification_rejects_user_outside_allowlist():
    auth = make_auth(allowed_users=["allowed"])

    result = auth._verify_claimed(user_id="blocked", emb=np.array([0.0, 1.0], dtype=np.float32))

    assert result.accepted is False
    assert result.matched_user_id == "blocked"
    assert result.reason == "user_not_allowed"
