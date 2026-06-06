from aiavatar.sts.audio_preprocessing import WebRtcNoiseGainAudioProcessor

from ..config import Settings


def create_pre_vad_audio_processor(settings: Settings):
    if not settings.pre_vad_noise_suppression_enabled:
        return None

    if settings.pre_vad_noise_suppression_provider != "webrtc_noise_gain":
        raise ValueError(
            f"Unsupported PRE_VAD_NOISE_SUPPRESSION_PROVIDER: {settings.pre_vad_noise_suppression_provider}"
        )

    return WebRtcNoiseGainAudioProcessor(
        noise_suppression_level=settings.pre_vad_noise_suppression_level,
        auto_gain_dbfs=settings.pre_vad_auto_gain_dbfs,
        fail_open=settings.pre_vad_noise_suppression_fail_open,
        debug=settings.debug,
    )
