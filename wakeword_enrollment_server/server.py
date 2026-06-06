import base64
import logging
import os
from pathlib import Path
from typing import Dict

import uvicorn
from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, UploadFile, WebSocket
from fastapi.responses import FileResponse
from pydantic import BaseModel

from aiavatar.adapter.models import AIAvatarRequest, AIAvatarResponse
from aiavatar.sts.vad.silero import SileroSpeechDetector
from server.config import Settings, load_settings
from server.logging_config import setup_logging
from voice_auth_server.auth import authenticate_websocket, create_admin_dependency

from .enrollment import WakewordEnrollmentManager

logger = logging.getLogger("aiavatar.wakeword_enrollment")


class WebSocketSession:
    def __init__(self):
        self.id = None
        self.data = {}


class ReadingStartRequest(BaseModel):
    wakeword: str | None = None


class ModelTestRequest(BaseModel):
    candidate_id: str


class CandidateUpdateRequest(BaseModel):
    role: str | None = None


class TrainingStartRequest(BaseModel):
    wakeword: str
    target_phrases: list[str] | None = None
    model_name: str | None = None
    threshold: float | None = None
    tts_backend: str = "configured_tts"
    model_size: str = "tiny"
    n_samples: int = 500
    n_samples_val: int = 100
    steps: int = 2000
    skip_acav: bool = True
    run_eval: bool = False
    positive_source: str = "recorded_only"
    negative_source: str = "recorded_plus_tts"


class WakewordEnrollmentServer:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.manager = WakewordEnrollmentManager(settings)
        self.sessions: Dict[str, WebSocketSession] = {}
        self.websockets: Dict[str, WebSocket] = {}
        self.vad = SileroSpeechDetector(
            sample_rate=settings.voice_auth_sample_rate,
            debug=settings.debug,
            use_vad_iterator=settings.vad_use_iterator,
        )

        @self.vad.on_speech_detected
        async def on_speech_detected(data: bytes, text: str, metadata: dict, recorded_duration: float, session_id: str):
            candidate = self.manager.add_candidate(
                audio_bytes=data,
                sample_rate=self.vad.sample_rate,
                duration=recorded_duration,
                session_id=session_id,
            )
            if candidate:
                logger.info(
                    "Wakeword candidate captured: id=%s, wakeword=%s, session=%s, duration=%.3f",
                    candidate.id,
                    candidate.wakeword,
                    session_id,
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
                metadata={"mode": "wakeword_enrollment"},
            ).model_dump_json())
            return

        if request.type in ("data", "invoke"):
            if request.user_id:
                self.vad.set_session_data(request.session_id, "user_id", request.user_id, True)
            if request.context_id:
                self.vad.set_session_data(request.session_id, "context_id", request.context_id, True)
            if request.channel:
                self.vad.set_session_data(request.session_id, "channel", request.channel, True)
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
        async def get_session_by_user_id(_auth=Depends(auth)):
            session = next(reversed(self.sessions.values()), None) if self.sessions else None
            return {"session_id": session.id if session else None, "data": session.data if session else {}}

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
                    logger.info("Wakeword WebSocket disconnected: session_id=%s", session.id)
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
        async def start_reading(request: ReadingStartRequest):
            self.manager.start_reading(request.wakeword)
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
                audio_path = self.manager.get_candidate_audio_path(candidate_id)
            except KeyError:
                raise HTTPException(status_code=404, detail="candidate not found")
            return FileResponse(audio_path, media_type="audio/wav")

        @router.patch("/api/candidates/{candidate_id}")
        async def update_candidate(candidate_id: str, request: CandidateUpdateRequest):
            try:
                return self.manager.update_candidate_role(candidate_id, request.role)
            except KeyError:
                raise HTTPException(status_code=404, detail="candidate not found")
            except ValueError as ex:
                raise HTTPException(status_code=400, detail=str(ex))

        @router.get("/api/models")
        async def list_models():
            return {"models": self.manager.list_models()}

        @router.post("/api/models")
        async def import_model(
            wakeword: str = Form(...),
            threshold: float | None = Form(default=None),
            file: UploadFile = File(...),
        ):
            try:
                return self.manager.import_model(
                    wakeword=wakeword,
                    filename=file.filename or "wakeword.onnx",
                    content=await file.read(),
                    threshold=threshold,
                )
            except ValueError as ex:
                raise HTTPException(status_code=400, detail=str(ex))

        @router.delete("/api/models/{model_id}")
        async def delete_model(model_id: str):
            try:
                self.manager.delete_model(model_id)
            except KeyError:
                raise HTTPException(status_code=404, detail="model not found")
            return {"models": self.manager.list_models()}

        @router.post("/api/models/{model_id}/test")
        async def test_model(model_id: str, request: ModelTestRequest):
            try:
                return self.manager.test_model(model_id=model_id, candidate_id=request.candidate_id)
            except KeyError as ex:
                raise HTTPException(status_code=404, detail=f"not found: {ex}")
            except ValueError as ex:
                raise HTTPException(status_code=400, detail=str(ex))

        @router.get("/api/test/results")
        async def list_test_results():
            return {"results": self.manager.list_test_results()}

        @router.get("/api/training/jobs")
        async def list_training_jobs():
            return {"jobs": self.manager.list_training_jobs()}

        @router.post("/api/training/jobs")
        async def start_training_job(request: TrainingStartRequest):
            try:
                return self.manager.start_training_job(
                    wakeword=request.wakeword,
                    target_phrases=request.target_phrases or [request.wakeword],
                    model_name=request.model_name,
                    threshold=request.threshold,
                    tts_backend=request.tts_backend,
                    model_size=request.model_size,
                    n_samples=request.n_samples,
                    n_samples_val=request.n_samples_val,
                    steps=request.steps,
                    skip_acav=request.skip_acav,
                    run_eval=request.run_eval,
                    positive_source=request.positive_source,
                    negative_source=request.negative_source,
                )
            except ValueError as ex:
                raise HTTPException(status_code=400, detail=str(ex))
            except RuntimeError as ex:
                raise HTTPException(status_code=409, detail=str(ex))

        @router.get("/api/training/jobs/{job_id}/log")
        async def get_training_job_log(job_id: str):
            try:
                return {"log": self.manager.get_training_job_log(job_id)}
            except KeyError:
                raise HTTPException(status_code=404, detail="job not found")

        return router


os.environ.setdefault("HERMES_API_KEY", "unused-wakeword-enrollment")
settings = load_settings()
setup_logging()
wakeword_enrollment_server = WakewordEnrollmentServer(settings)

app = FastAPI()


@app.get("/health")
async def health():
    return {
        "ok": True,
        "mode": "wakeword_enrollment",
        "reading_enabled": wakeword_enrollment_server.manager.reading_enabled,
        "reading_wakeword": wakeword_enrollment_server.manager.reading_wakeword,
        "candidate_count": len(wakeword_enrollment_server.manager.list_candidates()),
        "model_count": len(wakeword_enrollment_server.manager.list_models()),
        "training_job_count": len(wakeword_enrollment_server.manager.list_training_jobs()),
    }


app.include_router(wakeword_enrollment_server.get_websocket_router())
app.include_router(wakeword_enrollment_server.get_admin_router())


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
