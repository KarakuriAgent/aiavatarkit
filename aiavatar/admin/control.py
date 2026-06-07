import logging
import re
from typing import Dict, Literal, Optional
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
    metadata: Optional[Dict] = Field(
        default=None,
        description="Internal metadata passed to the conversation processor"
    )


class SpeakRequest(BaseModel):
    """
    SpeakRequest contains an external notification that should be rewritten by
    the conversation model, stored in the active conversation, and spoken by the avatar.
    """
    text: str = Field(
        ...,
        example="そろそろ休憩の時間です。",
        description="External notification text to convert into a spoken avatar response"
    )
    session_id: Optional[str] = Field(
        default=None,
        example="local_session",
        description="Session Id to route the spoken response to a specific client"
    )
    user_id: Optional[str] = Field(
        default=None,
        example="user_001",
        description="User Id to resolve the active session and conversation"
    )
    channel: Optional[str] = Field(
        default="cron",
        example="cron",
        description="Input channel name used by the conversation processor"
    )
    wait_in_queue: bool = Field(
        default=True,
        description="When invoke queue is enabled, wait behind pending requests instead of interrupting."
    )
    voice: bool = Field(
        default=True,
        description="Whether to deliver the generated response to the avatar voice output. False stores/returns the conversation response without TTS."
    )
    metadata: Optional[Dict] = Field(
        default=None,
        description="Internal metadata passed to the conversation processor"
    )


def build_speak_prompt(text: str) -> str:
    return (
        "外部通知として以下の内容をユーザーに伝えてください。\n"
        "返答はそのまま音声で読み上げられます。\n"
        "通知本文だけを短く自然な日本語で返してください。\n"
        "説明、前置き、引用符、箇条書き、Markdownは使わないでください。\n\n"
        f"通知内容:\n{text}"
    )


def resolve_active_session_id(adapter: Adapter, *, session_id: str = None, user_id: str = None) -> str:
    resolved_session_id = session_id
    if not resolved_session_id and user_id and hasattr(adapter, "get_session_by_user_id"):
        session_data = adapter.get_session_by_user_id(user_id)
        if session_data:
            resolved_session_id = session_data.id
    if not resolved_session_id:
        logger.warning(
            "No active session found: user_id=%s, requested_session_id=%s",
            user_id,
            session_id,
        )
        raise HTTPException(status_code=400, detail="No active session found")
    return resolved_session_id


def resolve_user_id(adapter: Adapter, *, session_id: str = None, user_id: str = None) -> str:
    resolved_user_id = user_id
    if not resolved_user_id and session_id and hasattr(adapter.sts.vad, "get_session_data"):
        resolved_user_id = adapter.sts.vad.get_session_data(session_id, "user_id")
    if not resolved_user_id:
        raise HTTPException(status_code=400, detail="user_id is required")
    return resolved_user_id


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

        # Text delivery must never synthesize audio. Make this true for this
        # invocation even if the caller forgot to configure skip_tts_channels,
        # but do not permanently mark operational channels such as "hermes" as
        # TTS-skipped.
        added_skip_tts_channel = False
        if hasattr(adapter.sts, "skip_tts_channels") and channel not in adapter.sts.skip_tts_channels:
            adapter.sts.skip_tts_channels.append(channel)
            added_skip_tts_channel = True

        try:
            notify_adapter_response = bool((request.metadata or {}).get("notify_adapter_response"))
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
                metadata={
                    **(request.metadata or {}),
                    "source": channel,
                    "delivery": "text",
                    "suppress_adapter_response": True,
                }
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
                    if notify_adapter_response:
                        await adapter.handle_response(resp)
                elif resp.type == "error":
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=(resp.metadata or {}).get("error", "Error in processing conversation")
                    )
        finally:
            if added_skip_tts_channel:
                try:
                    adapter.sts.skip_tts_channels.remove(channel)
                except ValueError:
                    pass

        return ChatResponse(
            message="Message processed successfully",
            text=response_text,
            voice_text=response_voice_text,
            context_id=latest_context_id,
        )

    context_id = adapter.sts.vad.get_session_data(session_id, "context_id") if session_id else None
    response_text = ""
    response_voice_text = ""
    latest_context_id = context_id
    async for resp in adapter.sts.invoke(STSRequest(
        session_id=session_id,
        user_id=request.user_id,
        context_id=context_id,
        text=request.text,
        channel=request.channel,
        wait_in_queue=request.wait_in_queue,
        metadata=request.metadata,
    )):
        if resp.type == "start":
            latest_context_id = resp.context_id
            if session_id:
                adapter.sts.vad.set_session_data(session_id, "context_id", resp.context_id)
        elif resp.type == "chunk":
            response_text += resp.text or ""
            response_voice_text += resp.voice_text or ""
            latest_context_id = resp.context_id or latest_context_id
        elif resp.type == "final":
            response_text = resp.text or response_text
            response_voice_text = resp.voice_text or response_voice_text
            latest_context_id = resp.context_id or latest_context_id
        await adapter.handle_response(resp)

    return ChatResponse(
        message="Message processed successfully",
        text=response_text or None,
        voice_text=response_voice_text or None,
        context_id=latest_context_id,
    )


async def process_speak_request(
    adapter: Adapter,
    request: SpeakRequest,
    *,
    default_session_id: str = None,
) -> ChatResponse:
    requested_session_id = request.session_id or default_session_id
    if request.voice:
        session_id = resolve_active_session_id(
            adapter,
            session_id=requested_session_id,
            user_id=request.user_id,
        )
    else:
        session_id = requested_session_id
    user_id = resolve_user_id(adapter, session_id=session_id, user_id=request.user_id)
    return await process_conversation_request(
        adapter,
        ChatRequest(
            text=build_speak_prompt(request.text),
            session_id=session_id,
            user_id=user_id,
            channel=request.channel or "cron",
            delivery="avatar" if request.voice else "text",
            wait_in_queue=request.wait_in_queue,
            metadata={
                **(request.metadata or {}),
                "source": "avatar_speak",
                "suppress_discord_user_log": True,
                "speak_text": request.text,
                "voice": request.voice,
            },
        ),
    )


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
                requested_session_id = request.session_id or self.default_session_id
                session_id = resolve_active_session_id(
                    self.adapter,
                    session_id=requested_session_id,
                    user_id=request.user_id,
                )

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
            
            except HTTPException:
                raise
            except Exception as ex:
                logger.exception(f"Error performing avatar actions")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error while performing avatar actions"
                )

        @router.post(
            "/avatar/speak",
            tags=["Avatar Control"],
            summary="Speak external notification through conversation",
            description="Send an external notification to the conversation model, store the generated response, and speak it through the avatar",
            response_description="Notification speech completed successfully",
            responses={
                200: {"description": "Notification speech completed successfully"},
                400: {"description": "No active session found"},
                422: {"description": "Invalid message format"},
                500: {"description": "Internal server error"}
            }
        )
        async def post_avatar_speak(request: SpeakRequest) -> ChatResponse:
            """
            Convert an external notification into a model response and speak it.

            The generated response is delivered through the normal avatar
            conversation path, so it is stored in the same conversation and is
            synthesized by the existing TTS pipeline.
            """
            try:
                return await process_speak_request(
                    self.adapter,
                    request,
                    default_session_id=self.default_session_id,
                )

            except HTTPException:
                raise
            except Exception as ex:
                logger.error(f"Error processing avatar speak request: {ex}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Internal server error while processing avatar speak request"
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
