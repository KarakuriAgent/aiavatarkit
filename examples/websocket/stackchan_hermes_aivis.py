# pip install aiavatar fastapi uvicorn websockets
import logging
import os
from pathlib import Path


def load_env_file(path: Path):
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in (chr(34), chr(39)):
            value = value[1:-1]
        os.environ.setdefault(key, value)


load_env_file(Path(__file__).with_name(".env.stackchan"))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn

from aiavatar.adapter.websocket.server import AIAvatarWebSocketServer
from aiavatar.admin import setup_admin_panel
from aiavatar.sts.llm.chatgpt import ChatGPTService
from aiavatar.sts.stt import SpeechRecognizer
from aiavatar.sts.tts import AudioConverter, create_instant_synthesizer
from aiavatar.sts.vad.stream import SileroStreamSpeechDetector
from aiavatar.util import download_example


AIVIS_API_KEY = os.environ["AIVIS_API_KEY"]

STT_BASE_URL = os.environ.get("STT_BASE_URL", "http://192.168.0.171:5003/v1")
STT_API_KEY = os.environ.get("STT_API_KEY")
STT_MODEL = os.environ.get("STT_MODEL", "whisperkit")
STT_LANGUAGE = os.environ.get("STT_LANGUAGE", "ja")

HERMES_BASE_URL = os.environ.get("HERMES_BASE_URL", "http://127.0.0.1:8642/v1")
HERMES_API_KEY = os.environ["HERMES_API_KEY"]
HERMES_MODEL = os.environ.get("HERMES_MODEL", "gpt-5.5")

AIVIS_MODEL_UUID = os.environ.get(
    "AIVIS_MODEL_UUID",
    "261d7c95-11d4-4f0a-9053-4d28d3dd87ee",
)

PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "https://stack-chan.yamashita.app")
AIAVATAR_API_KEY = os.environ.get("AIAVATAR_API_KEY")
AIAVATAR_ADMIN_USER = os.environ.get("AIAVATAR_ADMIN_USER", "admin")


SYSTEM_PROMPT = """
あなたは Stack-chan として、ユーザーと音声で会話します。

## 話し方
- 日本語で自然に話してください。
- 音声合成で読み上げるため、返答は原則1〜2文、30〜80文字程度にしてください。
- 絵文字、箇条書き、Markdown、長い前置きは使わないでください。
- 聞き取りミスらしい入力は、文脈から自然に補って応答してください。

## 表情
以下の表情タグを応答に含められます。
- [face:neutral]
- [face:joy]
- [face:angry]
- [face:sorrow]
- [face:fun]
- [face:surprised]

必要なときだけ、文頭または感情が変わる位置に表情タグを入れてください。

## カメラ
視覚情報が必要な場合は [vision:camera] を含めてください。
このタグを出した直後は、短い一言だけにしてください。

例:
[face:joy]いいよ、見てみるね。
[vision:camera]どれどれ。

## 出力制約
- 表情タグと vision タグ以外の制御タグは出さないでください。
- 内部思考や説明は出さず、ユーザーに聞こえる返答だけを出してください。
""".strip()


logger = logging.getLogger("aiavatar")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("[%(levelname)s] %(asctime)s : %(message)s"))
logger.addHandler(handler)


class WhisperCompatibleSpeechRecognizer(SpeechRecognizer):
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str = None,
        model: str = "whisperkit",
        min_data_length: int = 4096,
        sample_rate: int = 16000,
        language: str = "ja",
        timeout: float = 30.0,
        debug: bool = False,
    ):
        super().__init__(language=language, timeout=timeout, debug=debug)
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.min_data_length = min_data_length
        self.sample_rate = sample_rate

    async def transcribe(self, data: bytes) -> str:
        if len(data) < self.min_data_length:
            if self.debug:
                logger.info("Data to transcribe is too short: %s", len(data))
            return None

        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        form_data = {
            "model": self.model,
            "language": self.language,
        }
        files = {
            "file": ("voice.wav", self.to_wave_file(data, self.sample_rate), "audio/wav"),
        }

        resp = await self.http_request_with_retry(
            method="POST",
            url=f"{self.base_url}/audio/transcriptions",
            headers=headers,
            data=form_data,
            files=files,
        )
        if not resp:
            return None

        try:
            recognized_text = resp.json()["text"]
            if self.debug:
                logger.info("Recognized: %s", recognized_text)
            return recognized_text
        except Exception:
            logger.exception("Failed to parse transcription response: %s", resp.text)
            return None


stt = WhisperCompatibleSpeechRecognizer(
    base_url=STT_BASE_URL,
    api_key=STT_API_KEY,
    model=STT_MODEL,
    language=STT_LANGUAGE,
    debug=True,
)

vad = SileroStreamSpeechDetector(
    speech_recognizer=stt,
    segment_silence_threshold=0.05,
    use_vad_iterator=True,
)

llm = ChatGPTService(
    openai_api_key=HERMES_API_KEY,
    base_url=HERMES_BASE_URL,
    system_prompt=SYSTEM_PROMPT,
    model=HERMES_MODEL,
    split_on_control_tags=True,
    debug=True,
)

tts = create_instant_synthesizer(
    method="POST",
    url="https://api.aivis-project.com/v1/tts/synthesize",
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIVIS_API_KEY}",
    },
    json={
        "model_uuid": AIVIS_MODEL_UUID,
        "text": "{text}",
    },
    response_parser=AudioConverter(debug=True).convert,
    timeout=30,
    cache_dir="aivis_tts_cache",
)

aiavatar_app = AIAvatarWebSocketServer(
    vad=vad,
    stt=stt,
    llm=llm,
    tts=tts,
    merge_request_threshold=3.0,
    use_invoke_queue=True,
    api_key=AIAVATAR_API_KEY,
    response_audio_chunk_size=8192,
    debug=True,
)

download_example("websocket/html")

app = FastAPI()
app.include_router(aiavatar_app.get_websocket_router())
app.mount("/static", StaticFiles(directory="html"), name="static")

setup_admin_panel(
    app,
    adapter=aiavatar_app,
    title="AIAvatarKit Stack-chan Admin",
    api_key=AIAVATAR_API_KEY,
    basic_auth_username=AIAVATAR_ADMIN_USER,
    basic_auth_password=AIAVATAR_API_KEY,
)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "public_base_url": PUBLIC_BASE_URL,
        "stt_base_url": STT_BASE_URL,
        "stt_model": STT_MODEL,
        "hermes_base_url": HERMES_BASE_URL,
        "hermes_model": HERMES_MODEL,
        "aivis_model_uuid": AIVIS_MODEL_UUID,
    }


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
    )
