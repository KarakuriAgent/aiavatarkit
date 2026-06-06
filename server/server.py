import uvicorn
import shlex
import shutil
from fastapi import FastAPI

from aiavatar.admin import setup_admin_panel
from .config import load_settings
from .discord_gateway import setup_discord_integration
from .logging_config import setup_logging
from .pipeline import create_aiavatar_app


settings = load_settings()
setup_logging()
aiavatar_app = create_aiavatar_app(settings)

app = FastAPI()


@app.get("/health")
async def health():
    audio_enhancement_command_available = False
    if settings.audio_enhancement_enabled and settings.audio_enhancement_command:
        command = shlex.split(settings.audio_enhancement_command)
        audio_enhancement_command_available = bool(command and shutil.which(command[0]))

    return {
        "ok": True,
        "mode": "conversation",
        "audio_enhancement_enabled": settings.audio_enhancement_enabled,
        "audio_enhancement_provider": settings.audio_enhancement_provider if settings.audio_enhancement_enabled else None,
        "audio_enhancement_command_available": audio_enhancement_command_available,
        "pre_vad_noise_suppression_enabled": settings.pre_vad_noise_suppression_enabled,
        "pre_vad_noise_suppression_provider": (
            settings.pre_vad_noise_suppression_provider
            if settings.pre_vad_noise_suppression_enabled
            else None
        ),
        "voice_auth_enabled": settings.voice_auth_enabled,
        "voice_auth_provider": settings.voice_auth_provider if settings.voice_auth_enabled else None,
    }


app.include_router(aiavatar_app.get_websocket_router())

setup_admin_panel(
    app,
    adapter=aiavatar_app,
    title="AIAvatarKit Admin Panel",
    api_key=settings.aiavatar_api_key,
    basic_auth_username=settings.aiavatar_admin_user,
    basic_auth_password=settings.aiavatar_api_key,
)

setup_discord_integration(
    app,
    adapter=aiavatar_app,
    settings=settings,
)


def main():
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        ssl_certfile=settings.ssl_cert_path,
        ssl_keyfile=settings.ssl_key_path,
    )


if __name__ == "__main__":
    main()
