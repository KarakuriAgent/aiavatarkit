from aiavatar.sts.wakeword import LiveKitWakewordDetector

from ..config import Settings


def create_audio_wakeword_detector(settings: Settings):
    if not settings.audio_wakeword_enabled:
        return None

    if settings.audio_wakeword_provider != "livekit_wakeword":
        raise ValueError(f"Unsupported AUDIO_WAKEWORD_PROVIDER: {settings.audio_wakeword_provider}")

    return LiveKitWakewordDetector(
        model_paths=settings.audio_wakeword_model_paths,
        threshold=settings.audio_wakeword_threshold,
        frame_ms=settings.audio_wakeword_frame_ms,
        activation_window=settings.audio_wakeword_activation_window,
        cooldown=settings.audio_wakeword_cooldown,
        debug=settings.debug,
    )
