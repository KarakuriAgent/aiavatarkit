import base64
import logging
import os
from pathlib import Path
from typing import Dict, List

import uvicorn
from fastapi import APIRouter, Depends, FastAPI, HTTPException, WebSocket
from fastapi.responses import FileResponse
from pydantic import BaseModel

from aiavatar.adapter.models import AIAvatarRequest, AIAvatarResponse
from aiavatar.sts.vad.silero import SileroSpeechDetector
from server.config import Settings, load_settings
from server.logging_config import setup_logging

from .auth import authenticate_websocket, create_admin_dependency
from .enrollment import VoiceEnrollmentManager

logger = logging.getLogger("aiavatar.voice_auth_enrollment")


class WebSocketSession:
    def __init__(self):
        self.id = None
        self.data = {}


class EnrollRequest(BaseModel):
    user_id: str
    candidate_ids: List[str]


class VoiceAuthEnrollmentServer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.manager = VoiceEnrollmentManager(settings)
        self.sessions: Dict[str, WebSocketSession] = {}
        self.websockets: Dict[str, WebSocket] = {}
        self.vad = SileroSpeechDetector(
            sample_rate=settings.voice_auth_sample_rate,
            debug=settings.debug,
            use_vad_iterator=settings.vad_use_iterator,
        )

        @self.vad.on_speech_detected
        async def on_speech_detected(data: bytes, text: str, metadata: dict, recorded_duration: float, session_id: str):
            user_id = self.vad.get_session_data(session_id, "user_id")
            candidate = self.manager.add_candidate(
                audio_bytes=data,
                sample_rate=self.vad.sample_rate,
                duration=recorded_duration,
                session_id=session_id,
                user_id=user_id,
            )
            if candidate:
                logger.info(
                    "Voice auth candidate captured: id=%s, session=%s, user=%s, duration=%.3f",
                    candidate.id,
                    session_id,
                    user_id,
                    recorded_duration,
                )

    async def process_websocket(self, websocket: WebSocket, session: WebSocketSession):
        request = AIAvatarRequest.model_validate_json(await websocket.receive_text())

        if not request.session_id:
            await websocket.send_text(AIAvatarResponse(
                type="final",
                session_id=request.session_id,
                user_id=request.user_id,
                context_id=request.context_id,
                metadata={"error": "WebSocket disconnect: session_id is required."},
            ).model_dump_json())
            await websocket.close()
            return

        if request.type == "start":
            self.websockets[request.session_id] = websocket
            session.id = request.session_id
            session.data["metadata"] = request.metadata
            self.sessions[session.id] = session

            if request.user_id:
                self.vad.set_session_data(request.session_id, "user_id", request.user_id, True)
            if request.context_id:
                self.vad.set_session_data(request.session_id, "context_id", request.context_id, True)
            if request.channel:
                self.vad.set_session_data(request.session_id, "channel", request.channel, True)

            await websocket.send_text(AIAvatarResponse(
                type="connected",
                session_id=request.session_id,
                user_id=request.user_id,
                context_id=request.context_id,
                metadata={"mode": "voice_auth_enrollment"},
            ).model_dump_json())
            return

        if request.type in ("data", "invoke"):
            if request.user_id:
                self.vad.set_session_data(request.session_id, "user_id", request.user_id, True)
            if request.audio_data:
                await self.vad.process_samples(base64.b64decode(request.audio_data), request.session_id)
            return

        if request.type == "config":
            return

        if request.type == "stop":
            await websocket.close()

    def get_websocket_router(self, path: str = "/ws"):
        auth = create_admin_dependency(self.settings.aiavatar_api_key, self.settings.aiavatar_admin_user)
        router = APIRouter()

        @router.get(path + "/sessions")
        async def get_session_by_user_id(user_id: str, _auth=Depends(auth)):
            for session_id, session in reversed(self.sessions.items()):
                if self.vad.get_session_data(session_id, "user_id") == user_id:
                    return {"session_id": session.id, "data": session.data}
            return {"session_id": None}

        @router.websocket(path)
        async def websocket_endpoint(websocket: WebSocket):
            subprotocol = authenticate_websocket(websocket, self.settings.aiavatar_api_key)
            await websocket.accept(subprotocol=subprotocol)
            session = WebSocketSession()
            try:
                while True:
                    await self.process_websocket(websocket, session)
            except Exception as ex:
                message = str(ex)
                if "WebSocket is not connected" in message or "<CloseCode.NO_STATUS_RCVD: 1005>" in message:
                    logger.info("WebSocket disconnected: session_id=%s", session.id)
                else:
                    raise
            finally:
                if session.id:
                    await self.vad.finalize_session(session.id)
                    self.websockets.pop(session.id, None)
                    self.sessions.pop(session.id, None)

        return router

    def get_admin_router(self):
        auth = create_admin_dependency(self.settings.aiavatar_api_key, self.settings.aiavatar_admin_user)
        router = APIRouter(dependencies=[Depends(auth)])

        @router.get("/")
        async def index():
            return FileResponse(Path(__file__).resolve().parent / "static" / "index.html")

        @router.get("/api/state")
        async def get_state():
            return self.manager.state()

        @router.post("/api/reading/start")
        async def start_reading():
            self.manager.start_reading()
            return self.manager.state()

        @router.post("/api/reading/stop")
        async def stop_reading():
            self.manager.stop_reading()
            return self.manager.state()

        @router.get("/api/candidates")
        async def list_candidates():
            return {"candidates": self.manager.list_candidates()}

        @router.delete("/api/candidates")
        async def clear_candidates():
            self.manager.clear_candidates()
            return {"candidates": []}

        @router.get("/api/candidates/{candidate_id}/audio")
        async def get_candidate_audio(candidate_id: str):
            try:
                candidate = self.manager.get_candidate(candidate_id)
            except KeyError:
                raise HTTPException(status_code=404, detail="candidate not found")
            return FileResponse(candidate.path, media_type="audio/wav")

        @router.post("/api/enroll")
        async def enroll(request: EnrollRequest):
            try:
                return await self.manager.enroll(user_id=request.user_id, candidate_ids=request.candidate_ids)
            except KeyError as ex:
                raise HTTPException(status_code=404, detail=f"candidate not found: {ex}")
            except ValueError as ex:
                raise HTTPException(status_code=400, detail=str(ex))

        return router


# Enrollment does not use Hermes, but load_settings() is shared with the
# conversation server and requires HERMES_API_KEY there.
os.environ.setdefault("HERMES_API_KEY", "unused-voice-auth-enrollment")
settings = load_settings()
setup_logging()
enrollment_server = VoiceAuthEnrollmentServer(settings)

app = FastAPI()


@app.get("/health")
async def health():
    return {
        "ok": True,
        "mode": "voice_auth_enrollment",
        "reading_enabled": enrollment_server.manager.reading_enabled,
        "candidate_count": len(enrollment_server.manager.list_candidates()),
        "voice_auth_provider": settings.voice_auth_provider,
    }


app.include_router(enrollment_server.get_websocket_router())
app.include_router(enrollment_server.get_admin_router())


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
