from logging import getLogger

from ..config import Settings

logger = getLogger(__name__)


def create_audio_enhancer(settings: Settings):
    if not settings.audio_enhancement_enabled:
        return None

    if settings.audio_enhancement_provider != "deepfilternet":
        raise ValueError(f"Unsupported AUDIO_ENHANCEMENT_PROVIDER: {settings.audio_enhancement_provider}")

    from aiavatar.sts.audio_enhancement import DeepFilterNetAudioEnhancer

    return DeepFilterNetAudioEnhancer(
        command=settings.audio_enhancement_command,
        model=settings.audio_enhancement_model,
        timeout=settings.audio_enhancement_timeout,
        debug=settings.debug,
    )
