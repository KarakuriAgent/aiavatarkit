import uvicorn
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
    return {
        "ok": True,
        "mode": "conversation",
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
