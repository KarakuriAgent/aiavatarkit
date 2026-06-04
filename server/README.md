# AIAvatarKit + Hermes

このドキュメントは、Hermes の OpenAI-compatible API を AIAvatarKit の LLM バックエンドとして使うための設定メモです。

このサーバーは Hermes の `/v1/responses` を使います。`/v1/chat/completions` ではなく Responses API を使うことで、Hermes 側の `conversation` による会話セッション維持を利用します。

## Hermes Setup

Hermes の API Server を有効化し、OpenAI-compatible API が使える状態にします。

Hermes 側の `~/.hermes/.env` には、少なくとも以下を設定します。

```env
API_SERVER_ENABLED=true
API_SERVER_KEY=change-me-local-dev
```

その後、Hermes gateway を起動します。

```sh
hermes gateway
```

このサーバーはデフォルトで以下に接続します。

```text
http://127.0.0.1:8642/v1
```

`.env` では `HERMES_BASE_URL` に `/v1` まで含めて指定します。サーバー側では `/responses` を付けて呼び出します。

```env
HERMES_BASE_URL=http://127.0.0.1:8642/v1
HERMES_MODEL=hermes-agent
```

Hermes API の bearer token は、Hermes 側の `API_SERVER_KEY` と同じ値を `HERMES_API_KEY` に設定します。

```env
HERMES_API_KEY=change-me-local-dev
```

`HERMES_MODEL` は Hermes API に渡す model ID です。Hermes docs では、実際に使われる LLM は Hermes 側の設定で決まり、API リクエストの `model` フィールドは主に OpenAI-compatible frontend 向けの model ID として扱われます。

Responses API の `reasoning.effort` を request body に含めたい場合は、`.env` に `HERMES_REASONING_EFFORT` を設定します。

```env
HERMES_REASONING_EFFORT=none
```

この値を設定すると、サーバーは Hermes の `/v1/responses` に `reasoning: {"effort": "<value>"}` を送ります。未設定または空の場合は `reasoning` を送信しません。

## Session

Hermes 側のセッションは Responses API の `conversation` で維持します。

このサーバーでは、デフォルトで AIAvatarKit の `user_id` を Hermes の `conversation` に渡します。

```env
HERMES_CONVERSATION_ID_SOURCE=user_id
```

指定できる値は以下です。

```text
context_id
user_id
session_id
```

通常は `user_id` のまま使います。これにより、通常会話、Hermes の conversation、`/avatar/perform` の通知先を同じ ID で揃えられます。`context_id` にすると AIAvatarKit の会話コンテキスト単位、`session_id` にすると WebSocket 接続セッション単位で分かれます。

## Voice Prefix

音声経由の入力には、デフォルトで `[channel:voice]` を先頭に付けて Hermes に送ります。

```env
HERMES_REQUEST_PREFIX=[channel:voice]
```

Hermes 側のプロンプトやエージェント設定で、この prefix が付いた入力を音声会話として扱うようにしてください。不要な場合は空にできます。

```env
HERMES_REQUEST_PREFIX=
```

## AIAvatarKit Setup

依存関係はリポジトリルートで `uv` を使ってセットアップします。

```sh
uv sync
```

`.env` はリポジトリルートに作ります。

```sh
cp server/.env.example .env
```

最低限、以下を設定します。

```env
HERMES_API_KEY=change-me-local-dev
HERMES_BASE_URL=http://127.0.0.1:8642/v1
HERMES_MODEL=hermes-agent
HERMES_REQUEST_PREFIX=[channel:voice]
HERMES_CONVERSATION_ID_SOURCE=user_id
HERMES_REASONING_EFFORT=none

STT_PROVIDER=whisper_compatible
STT_BASE_URL=http://127.0.0.1:5003/v1
STT_MODEL=whisperkit
STT_LANGUAGE=ja

TTS_PROVIDER=aivis
AIVIS_API_KEY=YOUR_AIVIS_API_KEY
AIVIS_MODEL_UUID=261d7c95-11d4-4f0a-9053-4d28d3dd87ee
```

`STT_BASE_URL` は Whisper 互換の `/audio/transcriptions` API を提供するサーバーを指定します。実際のリクエスト先は以下になります。

```text
{STT_BASE_URL}/audio/transcriptions
```

### Local Provider Runtimes

Apple Silicon の MLX provider は Docker コンテナ内ではなく、Mac ホスト上の provider runtime として起動します。会話サーバーからはHTTPで呼びます。

Qwen3-ASR MLX を使う場合、`.env` では以下のように指定します。`STT_MODEL` を省略した場合も `STT_PROVIDER=qwen3_asr_mlx` では `Qwen/Qwen3-ASR-0.6B` が使われます。

```env
STT_PROVIDER=qwen3_asr_mlx
STT_BASE_URL=http://host.docker.internal:8766/v1
STT_MODEL=Qwen/Qwen3-ASR-0.6B
STT_LANGUAGE=ja
STT_RUNTIME_PORT=8766
```

ホストruntimeを起動します。

```sh
provider/start.sh
provider/status.sh
provider/status.sh --tail
provider/stop.sh
```

精度優先にする場合は `STT_MODEL=Qwen/Qwen3-ASR-1.7B` に差し替えます。4bit/8bit に量子化したローカルモデルを使う場合も、`STT_MODEL=models/qwen3-asr-0.6b-4bit` のようにモデルディレクトリを指定できます。`provider/start.sh` は起動前にモデルを確認し、Hugging Face repo ID の場合はキャッシュへダウンロードします。ローカルパスが無い場合は `STT_MODEL_REPO` を指定すると、そのrepoを指定パスへダウンロードします。

固有名詞や専門用語に寄せたい場合は、スペース区切りで `STT_CONTEXT` を設定します。

```env
STT_CONTEXT=Hermes AIAvatarKit StackChan
```

任意の既存Whisper互換サーバーを使う場合は `STT_PROVIDER=whisper_compatible` のまま `STT_BASE_URL` をそのサーバーに向けます。この場合 `provider/start.sh` はSTT runtimeを起動しません。

## Run

Hermes、STT サーバー、Aivis Cloud API の設定を用意してから、リポジトリルートで起動します。

```sh
uv run server
```

Docker Compose で起動する場合:

```sh
docker compose up --build
```

この compose 構成ではコードを image に COPY せず、`aiavatar/` と `server/` を bind mount します。Python 依存はコンテナ内の named volume `/app/.venv` に入ります。

ホスト側の公開ポートを変える場合は `.env` に `HOST_PORT=8001` のように指定します。コンテナ内のアプリは常に `PORT=8000` で起動します。Docker の port mapping は `127.0.0.1:${HOST_PORT}:8000` なので、ホスト外には公開されません。

Docker Compose では会話ログ DB を `/app/data/aiavatar.db`、録音ファイルを `/app/recorded_voices` に保存し、それぞれ named volume で永続化します。`docker compose down` / `up` でコンテナを作り直してもログは残ります。

Hermes や STT が Mac ホスト上で動いている場合は、`.env` で以下のように指定します。

```env
HERMES_BASE_URL=http://host.docker.internal:8642/v1
STT_BASE_URL=http://host.docker.internal:5003/v1
```

Hermes が別コンテナで動いていてホストにポート公開されている場合は、公開ポートを `host.docker.internal` 経由で指定します。

```env
HERMES_BASE_URL=http://host.docker.internal:8647/v1
```

この構成では `user_id` を共通キーとして使います。StackChan 側の `config.json`、Hermes の `conversation`、`/avatar/perform` の通知先を同じ値にしてください。

```json
{
  "user_id": "user01"
}
```

StackChan 側の `config.json` の `user_id` を固定してください。

## Voice Authentication

`VOICE_AUTH_ENABLED=true` にすると、VAD で確定した音声を STT / LLM に渡す前に登録済みユーザーの声と照合します。拒否された音声は `canceled` response になり、STT には送られません。

最初の provider は `wespeaker_mlx` です。Apple Silicon では `Landon41/wespeaker-voxceleb-resnet34-LM-mlx` を使います。`provider/start.sh` は `VOICE_AUTH_MODEL_PATH` に必要ファイルが無い場合、起動時に `VOICE_AUTH_MODEL_REPO` から自動ダウンロードします。

手動で先に落とす場合は以下も使えます。

```sh
uv sync --extra voice-auth
uv run huggingface-cli download --local-dir models/wespeaker-voxceleb-resnet34-LM-mlx Landon41/wespeaker-voxceleb-resnet34-LM-mlx
```

Docker で会話サーバーや登録サーバーを動かす場合、MLX/Metal は Docker コンテナ内では使わず、Mac ホスト上でprovider runtimeを常駐させます。

```sh
provider/start.sh
```

runtime はデフォルトで `0.0.0.0:8765` に起動します。Docker 側からは `http://host.docker.internal:8765` で呼び出します。Docker Compose 単体ではホスト上のruntimeプロセスを安全に起動/停止できないため、登録サーバーや会話サーバーを起動する前に別ターミナルで起動してください。ログはデフォルトで `.provider/logs/` に追記されます。

```sh
provider/start.sh
provider/status.sh
provider/status.sh --tail
provider/stop.sh
```

runtime のヘルスチェックは `GET /health` です。`AIAVATAR_API_KEY` または `VOICE_AUTH_API_KEY` を設定している場合、`Authorization: Bearer ...` が必要です。

```sh
curl -H "Authorization: Bearer $AIAVATAR_API_KEY" http://127.0.0.1:8765/health
```

```env
VOICE_AUTH_ENABLED=true
VOICE_AUTH_PROVIDER=wespeaker_mlx
VOICE_AUTH_BASE_URL=http://host.docker.internal:8765
# Optional. Defaults to AIAVATAR_API_KEY if empty.
# VOICE_AUTH_API_KEY=
VOICE_AUTH_MODEL_PATH=models/wespeaker-voxceleb-resnet34-LM-mlx
VOICE_AUTH_MODEL_REPO=Landon41/wespeaker-voxceleb-resnet34-LM-mlx
VOICE_AUTH_AUTO_DOWNLOAD_MODEL=true
VOICE_AUTH_PROFILE_DIR=data/voice_profiles
VOICE_AUTH_ENROLLMENT_DIR=data/voice_auth_enrollment
VOICE_AUTH_THRESHOLD=0.65
VOICE_AUTH_MIN_DURATION=1.2
VOICE_AUTH_SAMPLE_RATE=16000
VOICE_AUTH_REQUIRE_USER_ID=false
VOICE_AUTH_ALLOW_IDENTIFICATION=true
VOICE_AUTH_ALLOWED_USERS=user01
VOICE_AUTH_FAIL_OPEN=false
VOICE_AUTH_APPLY_CMN=true
VOICE_AUTH_RUNTIME_PORT=8765
# Optional provider runtime state/log paths.
# PROVIDER_STATE_DIR=.provider
# PROVIDER_LOG_DIR=.provider/logs
```

登録profileは `VOICE_AUTH_PROFILE_DIR` の `{user_id}.npz` として読み込みます。中には `embedding` という 256 次元の L2 normalize 済み vector を保存してください。会話時の自動登録は行いません。

会話サーバーは `VOICE_AUTH_ALLOW_IDENTIFICATION=true` の場合、発話音声を登録済みprofile全体から識別し、`VOICE_AUTH_ALLOWED_USERS` に含まれる名前だけを許可します。

音声登録時は本番会話サーバーを停止して、同じ `.env` と同じ `HOST_PORT` で登録サーバーを起動します。StackChan 側の接続先は変えません。

```sh
VOICE_AUTH_ENABLED=true VOICE_AUTH_PROVIDER=wespeaker_mlx provider/start.sh
docker compose down
docker compose -f compose.voice-auth.yml up --build
```

登録UIは既存サーバーと同じポートの `/` で開きます。`AIAVATAR_API_KEY` が設定されている場合は、`AIAVATAR_ADMIN_USER` / `AIAVATAR_API_KEY` の Basic 認証を使います。

WAV ファイルから直接登録する場合は以下も使えます。

```sh
uv run voice-auth-enroll user01 samples/user01_*.wav
```

採用前の合格条件:

```text
MLX embedding が参照実装と一致する: 同一音声 cosine >= 0.999
手元マイクで本人/別人スコアが分離する: 本人下限 > 別人上限 + 余裕
profileなし、短すぎる音声、推論エラーは拒否される
拒否時に STT / LLM へ進まない
```

## Discord Sync

Discord チャンネルに StackChan との音声会話ログを流し、同じ Hermes conversation に Discord からもテキストで問い合わせる場合は Discord sync を有効化します。

```env
DISCORD_SYNC_ENABLED=true
DISCORD_BOT_TOKEN=your-discord-bot-token
DISCORD_CHANNEL_ID=123456789012345678
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
DISCORD_GUILD_ID=123456789012345678

DISCORD_USER_ID=123456789012345678
DISCORD_BOT_ID=234567890123456789
DISCORD_SYNC_USER_ID=robo-kanon-stack-chan
DISCORD_GATEWAY_SESSION_ID=discord:robo-kanon-stack-chan
DISCORD_VOICE_MESSAGE_PREFIX="🎙️ "
DISCORD_API_MESSAGE_PREFIX="📢 "
DISCORD_TYPING_INDICATOR_ENABLED=true
DISCORD_TYPING_INDICATOR_INTERVAL=8

SKIP_TTS_CHANNELS=discord
```

`DISCORD_USER_ID` はユーザー発話ログの webhook 表示名/avatar 解決元、`DISCORD_BOT_ID` は AI 応答ログの webhook 表示名/avatar 解決元です。サーバーは Discord API から user / guild member を取得し、webhook payload の `username` / `avatar_url` に設定します。

`DISCORD_SYNC_USER_ID` は Hermes conversation に渡す AIAvatarKit `user_id` です。StackChan 側の `config.json` の `user_id` と同じ値にしてください。Discord 経由の入力は `DISCORD_GATEWAY_SESSION_ID` を疑似 session として使い、StackChan の WebSocket session には送信しません。

`DISCORD_VOICE_MESSAGE_PREFIX` は StackChan 側から同期されたユーザー発話ログと通常の AI 応答ログに付く prefix です。`DISCORD_API_MESSAGE_PREFIX` は `/avatar/speak` など API 経由の外部通知に対する AI 応答ログに付く prefix です。未設定の場合は `DISCORD_VOICE_MESSAGE_PREFIX` と同じ値を使います。Discord からのテキスト入力と、その入力への AI 応答には付きません。

`DISCORD_TYPING_INDICATOR_ENABLED` を有効にすると、Discord からのテキスト入力を処理している間と、StackChan 側の会話を Discord に同期している AI 応答処理中に、bot が対象チャンネルに typing indicator を出します。Discord の typing indicator は約 10 秒で消えるため、`DISCORD_TYPING_INDICATOR_INTERVAL` 秒ごとに更新します。

Discord からのテキスト入力は `/conversation` の `delivery=text` 経路で処理されます。この経路は `adapter.handle_response()` を呼ばず、`channel=discord` を `skip_tts_channels` に追加するため、StackChan は喋らず AIVIS TTS も呼ばれません。

Discord bot は対象チャンネルの `MESSAGE_CREATE` を Gateway で受けるため、Discord Developer Portal 側で Message Content Intent を有効化してください。Webhook 投稿は Gateway にも見えるため、サーバー側では `webhook_id` 付き message と bot message を無視してループを防ぎます。

## Push Notification

Hermes の cron や tool から接続中の StackChan に発話させたい場合は、このサーバーの `/avatar/speak` を呼び出します。`/avatar/speak` は通知本文を Hermes に渡し、返ってきた response を通常の avatar 会話として TTS します。そのため、読み上げ内容は同じ Hermes conversation に残ります。

```text
Hermes cron/tool
  -> POST /avatar/speak user_id=user01
  -> AIAvatarKit server
  -> Hermes conversation=user01
  -> WebSocket
  -> StackChan user_id=user01
```

例:

```http
POST http://localhost:8000/avatar/speak
Authorization: Bearer {AIAVATAR_API_KEY}
Content-Type: application/json

{"user_id": "user01", "text": "そろそろ休憩の時間です。"}
```

Discord sync が有効な場合、`/avatar/speak` の内部指示文は Discord には投稿されず、Hermes から返った bot 応答だけが webhook に流れます。この bot 応答には `DISCORD_API_MESSAGE_PREFIX` が付きます。Hermes cron/tool 由来の通知として見せたい場合は `.env` に `DISCORD_API_MESSAGE_PREFIX="📢 "` を設定してください。

`/avatar/perform` は Hermes conversation を通さず、指定 text を直接 TTS する endpoint です。履歴に残したい通知では `/avatar/speak` を使ってください。詳しくは [voice_push_notification_skill.md](voice_push_notification_skill.md) を参照してください。

## Hermes Prompt Guidance

Hermes docs では、Responses API の `instructions` は Hermes agent の core system prompt の上に重ねられます。このサーバーは現在、固定の system prompt や `instructions` は追加していません。音声入力 prefix と音声出力制約を使う場合は、Hermes 側の agent 設定に以下のような内容を入れてください。

```markdown
## Communication Modes & Voice Constraints

### Channel Detection
- 音声チャンネルからのリクエストは `[channel:voice]` で始まります。
- この prefix がある場合は音声会話として応答してください。

### Voice Mode
- 返答は短く、1〜2文程度にしてください。
- 絵文字、長い前置き、Markdown、箇条書きは避けてください。
- AIAvatarKit のブラウザ UI で表情を変えたい場合は `[face:joy]` のような face tag を使えます。
- 利用可能な face tag は `neutral`, `joy`, `angry`, `sorrow`, `fun`, `surprised` です。
- AIAvatarKit のカメラ取得を要求したい場合は `[vision:camera]` を出力してください。
- 音声認識ミスらしい入力は、文脈から自然に補って応答してください。

### Text Mode
- `[channel:voice]` がない場合は通常のテキスト会話として応答してください。
- face tag や vision tag は必要な場合以外は出さないでください。

## System Logic & Agentic Behavior

- `$` で始まる文は、ユーザーに聞こえる発話ではなく、処理上の補助情報として扱ってください。
- `$` で始まる指示に対して「実行します」のようにそのまま返答せず、会話文脈に合う自然な応答をしてください。
```

## Accessing from Another Host

別の端末のブラウザからマイクやカメラを使う場合、HTTPS が必要になることがあります。

選択肢は以下です。

- ドメインを用意して正式な SSL 証明書を発行する
- ngrok などの tunnel を使う
- 自己署名証明書を使う

自己署名証明書は `cert.py` で生成できます。アクセスに使う IP アドレスまたはホスト名を指定してください。

```sh
uv run python server/cert.py 192.168.1.123
```

生成された証明書と鍵のパスを `.env` に設定します。

```env
SSL_CERT_PATH=192.168.1.123.pem
SSL_KEY_PATH=192.168.1.123-key.pem
```
