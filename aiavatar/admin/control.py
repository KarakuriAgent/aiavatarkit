import logging
import re
from typing import Literal, Optional
from uuid import uuid4
from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from ..sts.models import STSRequest, STSResponse
from ..adapter.base import Adapter
from .auth import create_api_key_dependency

logger = logging.getLogger(__name__)


class APIResponse(BaseModel):
    """
    APIResponse is a standard response model for successful API operations.
    
    Used to return a simple success message or status information.
    """
    message: str = Field(
        ..., 
        example="Message from API", 
        description="Success message from API operation"
    )


class ChatResponse(APIResponse):
    text: Optional[str] = Field(
        default=None,
        example="こんにちは！",
        description="Response text including control tags when available"
    )
    voice_text: Optional[str] = Field(
        default=None,
        example="こんにちは！",
        description="Speech/display text with control tags removed when available"
    )
    context_id: Optional[str] = Field(
        default=None,
        example="context_001",
        description="Conversation context ID used by AIAvatarKit"
    )


class SpeechRequest(BaseModel):
    """
    SpeechRequest contains text for avatar speech synthesis and performance.

    Supports control tags for face expressions and animations embedded in text.
    """
    text: str = Field(
        ...,
        example="[face:joy]Hi, let's talk with me!",
        description="Text to synthesize with optional face and animation control tags"
    )
    session_id: Optional[str] = Field(
        default=None,
        example="local_session",
        description="Session Id to route the response to a specific client (e.g., WebSocket connection)"
    )
    user_id: Optional[str] = Field(
        default=None,
        example="user_001",
        description="User Id to resolve session. Used when session_id is not provided."
    )


class ChatRequest(BaseModel):
    """
    ChatRequest contains a text message for conversation processing.

    Used to send messages to the AIAvatar's conversation processing pipeline.
    """
    text: str = Field(
        ...,
        example="こんにちは！",
        description="Text message to send to the conversation processor"
    )
    session_id: Optional[str] = Field(
        default=None,
        example="local_session",
        description="Session Id to route the response to a specific client (e.g., WebSocket connection)"
    )
    user_id: Optional[str] = Field(
        default=None,
        example="user_001",
        description="User Id for conversation context management (e.g., per-user memory)"
    )
    channel: Optional[str] = Field(
        default=None,
        example="discord",
        description="Input channel name used by the conversation processor"
    )
    delivery: Literal["avatar", "text"] = Field(
        default="avatar",
        description="Response delivery mode. 'avatar' sends responses to the adapter; 'text' returns text only."
    )
    wait_in_queue: bool = Field(
        default=True,
        description="When invoke queue is enabled, wait behind pending requests instead of interrupting."
    )


async def process_conversation_request(
    adapter: Adapter,
    request: ChatRequest,
    *,
    default_session_id: str = None,
) -> ChatResponse:
    session_id = request.session_id or default_session_id

    if request.delivery == "text":
        user_id = request.user_id
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required for text delivery")

        session_id = session_id or f"text:{user_id}"
        channel = request.channel or "text"

        # Text delivery must never synthesize audio. Make this true even if the
        # caller forgot to configure STSPipeline.skip_tts_channels.
        if hasattr(adapter.sts, "skip_tts_channels") and channel not in adapter.sts.skip_tts_channels:
            adapter.sts.skip_tts_channels.append(channel)

        context_id = None
        if hasattr(adapter.sts.vad, "get_session_data"):
            context_id = adapter.sts.vad.get_session_data(session_id, "context_id")

        response_text = ""
        response_voice_text = ""
        latest_context_id = context_id
        async for resp in adapter.sts.invoke(STSRequest(
            session_id=session_id,
            user_id=user_id,
            context_id=context_id,
            text=request.text,
            channel=channel,
            wait_in_queue=request.wait_in_queue,
            metadata={"source": channel, "delivery": "text", "suppress_adapter_response": True}
        )):
            if resp.type == "start":
                latest_context_id = resp.context_id
                if hasattr(adapter.sts.vad, "set_session_data"):
                    adapter.sts.vad.set_session_data(session_id, "context_id", resp.context_id, create_session=True)
            elif resp.type == "chunk":
                response_text += resp.text or ""
                response_voice_text += resp.voice_text or ""
                latest_context_id = resp.context_id or latest_context_id
            elif resp.type == "final":
                response_text = resp.text or response_text
                response_voice_text = resp.voice_text or response_voice_text
                latest_context_id = resp.context_id or latest_context_id
            elif resp.type == "error":
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=(resp.metadata or {}).get("error", "Error in processing conversation")
                )

        return ChatResponse(
            message="Message processed successfully",
            text=response_text,
            voice_text=response_voice_text,
            context_id=latest_context_id,
        )

    context_id = adapter.sts.vad.get_session_data(session_id, "context_id") if session_id else None
    async for resp in adapter.sts.invoke(STSRequest(
        session_id=session_id,
        user_id=request.user_id,
        context_id=context_id,
        text=request.text,
        channel=request.channel,
        wait_in_queue=request.wait_in_queue,
    )):
        if resp.type == "start":
            if session_id:
                adapter.sts.vad.set_session_data(session_id, "context_id", resp.context_id)
        await adapter.handle_response(resp)

    return ChatResponse(message="Message processed successfully")


class ControlAPI:
    def __init__(self, adapter: Adapter, default_session_id: str = None):
        self.adapter = adapter
        self.default_session_id = default_session_id

    def remove_control_tags(self, text: str) -> str:
        clean_text = text
        clean_text = re.sub(r"\[(\w+):([^\]]+)\]", "", clean_text)
        clean_text = re.sub(r"<\w+\s[^>]*>", "", clean_text)
        clean_text = clean_text.strip()
        return clean_text

    def get_router(self) -> APIRouter:
        router = APIRouter()

        @router.post(
            "/avatar/perform", 
            tags=["Avatar Control"], 
            summary="Perform avatar speech with controls",
            description="Synthesize speech and perform avatar controls (face/animation) based on embedded tags",
            response_description="Performance completed successfully",
            responses={
                200: {"description": "Avatar performance completed successfully"},
                422: {"description": "Invalid text or control tags"},
                500: {"description": "Internal server error"}
            }
        )
        async def post_avatar_perform(request: SpeechRequest) -> APIResponse:
            """
            Perform comprehensive avatar actions including speech, face, and animation.
            
            This endpoint processes text with embedded control tags:
            - Synthesizes speech from the provided text (excluding control tags)
            - Parses control tags for face expressions and animations
            - Executes synchronized avatar performance
            
            Control tag format:
            - Face: [face:expression_name] (e.g., [face:joy])
            - Animation: [animation:animation_name] (e.g., [animation:wave_hands])
            
            Example: "[face:joy]Hello there! [animation:wave_hands]Nice to meet you!"
            """
            try:
                session_id = request.session_id or self.default_session_id
                requested_session_id = session_id
                if not session_id and request.user_id:
                    if hasattr(self.adapter, "get_session_by_user_id"):
                        session_data = self.adapter.get_session_by_user_id(request.user_id)
                        if session_data:
                            session_id = session_data.id
                if not session_id:
                    logger.warning(
                        "Avatar perform requested but no active session found: user_id=%s, requested_session_id=%s",
                        request.user_id,
                        requested_session_id,
                    )
                    raise HTTPException(status_code=400, detail="No active session found")

                voice_text = self.remove_control_tags(request.text)
                logger.info(
                    "Avatar perform request: user_id=%s, requested_session_id=%s, resolved_session_id=%s, voice_text=%s",
                    request.user_id,
                    requested_session_id,
                    session_id,
                    voice_text,
                )
                voice = await self.adapter.sts.tts.synthesize(text=voice_text)
                context_id = str(uuid4())
                transaction_id = str(uuid4())

                await self.adapter.stop_response(session_id, "_")
                await self.adapter.handle_response(STSResponse(
                    type="accepted",
                    session_id=session_id,
                    user_id=request.user_id,
                    context_id=context_id,
                    transaction_id=transaction_id,
                    metadata={"block_barge_in": True, "source": "avatar_perform"}
                ))
                await self.adapter.handle_response(STSResponse(
                    type="start",
                    session_id=session_id,
                    user_id=request.user_id,
                    context_id=context_id,
                    transaction_id=transaction_id,
                    metadata={"source": "avatar_perform"}
                ))

                await self.adapter.handle_response(STSResponse(
                    type="chunk",
                    session_id=session_id,
                    user_id=request.user_id,
                    context_id=context_id,
                    transaction_id=transaction_id,
                    text=request.text,
                    voice_text=voice_text,
                    audio_data=voice,
                    metadata={"is_first_chunk": True, "source": "avatar_perform"}
                ))
                await self.adapter.handle_response(STSResponse(
                    type="final",
                    session_id=session_id,
                    user_id=request.user_id,
                    context_id=context_id,
                    transaction_id=transaction_id,
                    text=request.text,
                    voice_text=voice_text,
                    metadata={"source": "avatar_perform"}
                ))

                return APIResponse(message="Avatar performance completed successfully")
            
            except Exception as ex:
                logger.exception(f"Error performing avatar actions")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error while performing avatar actions"
                )

        @router.post(
            "/conversation", 
            tags=["Avatar Control"], 
            summary="Send message to conversation processor",
            description="Send a text message to the AIAvatar conversation processing pipeline",
            response_description="Success message indicating conversation processing completion",
            responses={
                200: {"description": "Message processed successfully"},
                400: {"description": "AIAvatar is not listening"},
                422: {"description": "Invalid message format"},
                500: {"description": "Internal server error"}
            }
        )
        async def processor_chat(request: ChatRequest) -> ChatResponse:
            """
            Send a text message to the conversation processing pipeline.
            
            This endpoint processes a text message through the complete STS pipeline:
            - Validates that the avatar is currently listening
            - Invokes the Speech-to-Speech pipeline with the text input
            - Processes LLM response and generates appropriate avatar actions
            - Handles TTS synthesis and avatar control synchronization
            
            The message will be processed as if it were spoken input, triggering
            the full conversation flow including context management and response generation.
            """
            try:
                return await process_conversation_request(
                    self.adapter,
                    request,
                    default_session_id=self.default_session_id,
                )
            
            except HTTPException:
                raise
            except Exception as ex:
                logger.error(f"Error processing conversation message: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error while processing conversation message"
                )

        return router


def setup_control_api(
    app: FastAPI,
    *,
    adapter: Adapter,
    default_session_id: str = None,
    api_key: str = None,
):
    deps = [Depends(create_api_key_dependency(api_key))] if api_key else []
    app.include_router(
        ControlAPI(adapter=adapter, default_session_id=default_session_id).get_router(),
        dependencies=deps,
    )
