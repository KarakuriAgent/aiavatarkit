from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from aiavatar.admin import setup_admin_panel
from .config import load_settings
from .logging_config import setup_logging
from .pipeline import create_aiavatar_app


settings = load_settings()
setup_logging()
aiavatar_app = create_aiavatar_app(settings)

app = FastAPI()
app.include_router(aiavatar_app.get_websocket_router())
app.mount("/static", StaticFiles(directory=Path(__file__).with_name("html")), name="static")

setup_admin_panel(
    app,
    adapter=aiavatar_app,
    title="AIAvatarKit Admin Panel",
    api_key=settings.aiavatar_api_key,
    basic_auth_username=settings.aiavatar_admin_user,
    basic_auth_password=settings.aiavatar_api_key,
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

# open http://localhost:8000/static/vrm.html
