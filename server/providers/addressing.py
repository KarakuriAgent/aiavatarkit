from logging import getLogger

from ..config import Settings

logger = getLogger(__name__)


def create_addressing_detector(settings: Settings):
    if not settings.addressing_enabled:
        return None

    if settings.addressing_provider != "openai_compatible":
        raise ValueError(f"Unsupported ADDRESSING_PROVIDER: {settings.addressing_provider}")

    target_names = settings.addressing_target_names
    if not target_names:
        raise ValueError("ADDRESSING_TARGET_NAMES is required when ADDRESSING_ENABLED=true")

    from aiavatar.sts.addressing import OpenAICompatibleChatAddressingDetector

    uses_addressing_provider_override = bool(
        settings.addressing_base_url
        or settings.addressing_model
        or settings.addressing_api_key
    )

    return OpenAICompatibleChatAddressingDetector(
        base_url=settings.addressing_base_url or settings.hermes_base_url,
        api_key=(
            settings.addressing_api_key
            if uses_addressing_provider_override
            else settings.hermes_api_key
        ),
        model=settings.addressing_model or settings.hermes_model,
        target_names=target_names,
        primary_name=settings.addressing_primary_name,
        timeout=settings.addressing_timeout,
        min_confidence=settings.addressing_min_confidence,
        fail_open=settings.addressing_fail_open,
        debug=settings.debug,
    )
