from aiavatar.adapter.websocket.server import AIAvatarWebSocketServer

from .config import Settings
from .providers.llm import create_llm
from .providers.stt import create_stt
from .providers.tts import create_tts
from .providers.vad import create_vad
from .providers.voice_auth import create_voice_auth


def create_aiavatar_app(settings: Settings):
    stt = create_stt(settings)
    vad = create_vad(settings, stt)
    llm = create_llm(settings)
    tts = create_tts(settings)
    voice_auth = create_voice_auth(settings)

    return AIAvatarWebSocketServer(
        vad=vad,
        stt=stt,
        llm=llm,
        tts=tts,
        voice_auth=voice_auth,
        merge_request_threshold=settings.merge_request_threshold,
        use_invoke_queue=settings.use_invoke_queue,
        api_key=settings.aiavatar_api_key,
        db_connection_str=settings.aiavatar_db_path,
        voice_recorder_dir=settings.aiavatar_voice_recorder_dir,
        response_audio_chunk_size=settings.response_audio_chunk_size,
        skip_tts_channels=settings.skip_tts_channels,
        debug=settings.debug,
    )
