---
name: hermes-voice-message
description: "ユーザーまたは外部システムが音声送信、読み上げ、声で通知することを明示した場合だけ使う。`[channel:voice]` が付いた通常会話への返答では使わない。"
required_environment_variables:
  - AIAVATAR_BASE_URL
  - AIAVATAR_API_KEY
  - AIAVATAR_USER_ID
---

# Hermes 音声メッセージ

明示的に依頼された音声通知を AIAvatarKit 経由で送ります。

リクエストが「音声で送って」「読み上げて」「声で通知して」「喋らせて」など、音声配送を明示している場合だけ使います。`[channel:voice]` が付いた通常会話への返答には使いません。

## 使用条件

次のすべてを満たす場合だけ使います。

- `音声で送って`、`読み上げて`、`声で伝えて`、`喋らせて` など、音声配送の指示が明示されている。
- 送信するメッセージ本文がある、またはリクエストから直接作れる。
- `AIAVATAR_BASE_URL`、`AIAVATAR_API_KEY`、`AIAVATAR_USER_ID` が使える。

次の場合は使いません。

- ユーザーとの通常会話。
- ユーザーがテキストだけの出力を求めている。
- 何を返すかだけが指定され、音声配送が明示されていない。
- `[channel:voice]` が付いた通常会話リクエスト。

## エンドポイント

AIAvatarKit に次のリクエストを送ります。

```http
POST {AIAVATAR_BASE_URL}/avatar/speak
Authorization: Bearer {AIAVATAR_API_KEY}
Content-Type: application/json
User-Agent: Hermes-Agent/1.0 AIAvatar-Client
```

body:

```json
{
  "user_id": "{AIAVATAR_USER_ID}",
  "channel": "hermes",
  "text": "{message_to_deliver}"
}
```

`/avatar/speak` は `text` を対象 AIAvatarKit conversation に送り、返ってきたモデル応答を読み上げます。読み上げられた応答は同じ conversation に保存されます。この skill では `/avatar/perform` を呼びません。

## メッセージ本文のルール

- `text` には通知本文だけを入れる。
- `音声で送って`、`読み上げて`、`目的`、`実行`、`記録` などのメタ指示は入れない。
- `[channel:voice]` は入れない。AIAvatarKit 側で voice channel prefix が付く。
- ユーザーが長い正確な文面を指定していない限り、短くする。
- ユーザーが特定の文を送るよう依頼した場合は、文面を保持する。
- Markdown、表、URL、装飾記号は、ユーザーが明示していない限り避ける。

## レスポンス処理

成功時は、音声メッセージを送信したことを現在の会話に記録します。必要なら返ってきた `voice_text` または `text` を添えます。

失敗時は、HTTP status と response body を含めて、現在の会話に失敗した事実を記録します。主な失敗は以下です。

- `401`: API key がない、または不正。
- `400 No active session found`: `AIAVATAR_USER_ID` に対応する avatar が接続されていない。
- `422`: request JSON が不正、または `text` がない。
