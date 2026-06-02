from aiavatar.adapter.websocket.server import AIAvatarWebSocketServer

from .config import Settings
from .providers.llm import create_llm
from .providers.stt import create_stt
from .providers.tts import create_tts
from .providers.vad import create_vad


def create_aiavatar_app(settings: Settings):
    stt = create_stt(settings)
    vad = create_vad(settings, stt)
    llm = create_llm(settings)
    tts = create_tts(settings)

    return AIAvatarWebSocketServer(
        vad=vad,
        stt=stt,
        llm=llm,
        tts=tts,
        merge_request_threshold=settings.merge_request_threshold,
        use_invoke_queue=settings.use_invoke_queue,
        api_key=settings.aiavatar_api_key,
        debug=settings.debug,
    )
