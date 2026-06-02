---
name: voice-message
description: "`$システムメッセージです` などの明示的な外部通知依頼があり、`発話内容` が指定されている場合に限り、環境変数 `AIAVATAR_USER_ID` の avatar へ音声メッセージ、表情タグ、アニメーションタグを送る skill。通常会話では使わない。"
required_environment_variables:
  - AIAVATAR_BASE_URL
  - AIAVATAR_API_KEY
  - AIAVATAR_USER_ID
---

# 音声メッセージ送信

外部通知を受け取り、接続中の avatar に音声で発話させるための手順です。通常の音声会話への返答では使いません。

## 実行条件

次のすべてを満たす場合だけ実行します。

- 入力が `$システムメッセージです` などの system-style instruction である。
- cron、scheduler、monitoring、外部システム通知などからの依頼である。
- `発話内容` が明示されている。
- avatar に音声通知として送る指示が明示されている。
- 環境変数 `AIAVATAR_BASE_URL`、`AIAVATAR_API_KEY`、`AIAVATAR_USER_ID` が設定されている。

次の場合はこの skill を使いません。

- ユーザーとの通常会話への返答。
- `[channel:voice]` など AIAvatarKit 側から来た通常の音声会話リクエスト。
- 通常の LLM 応答を返せば AIAvatarKit 側で読み上げられるケース。
- `発話内容` がないケース。
- `AIAVATAR_USER_ID` が設定されていないケース。

## 入力形式

外部通知は必ず対象 conversation に投入され、以下の形式を含みます。

```text
$システムメッセージです。cron からの通知依頼です。
目的: ユーザーへの音声通知
実行: この skill を使って avatar に発話させる
発話内容: [face:joy]15時のミーティングがもうすぐ始まります。
記録: この通知を送ったことを会話文脈に残す
```

送信先は instruction から推測しません。常に環境変数 `AIAVATAR_USER_ID` の値を使います。

`発話内容:` の後ろだけが avatar に喋らせる本文です。`目的`、`実行`、`記録` などの制御指示は発話してはいけません。

## 実行手順

1. 現在の conversation に入った system-style instruction を通常会話として返答せず、この skill の実行対象として扱う。
2. 環境変数 `AIAVATAR_USER_ID` を送信先 `user_id` として読む。
3. instruction から `発話内容` を抽出する。
4. 必要なら `発話内容` に表情タグまたはアニメーションタグを追加・調整する。
5. AIAvatarKit の `/avatar/perform` に POST する。
6. この通知を送った事実を現在の conversation の文脈に残す。

## 環境変数

```text
AIAVATAR_BASE_URL=https://example.com
AIAVATAR_API_KEY=...
AIAVATAR_USER_ID=hermes:user:default
```

`AIAVATAR_USER_ID` は固定の送信先です。外部通知ごとに変更しません。

## API 呼び出し

```http
POST {AIAVATAR_BASE_URL}/avatar/perform
Authorization: Bearer {AIAVATAR_API_KEY}
Content-Type: application/json
User-Agent: Hermes-Agent/1.0 AIAvatar-Client
```

Cloudflare 等で `403 error code: 1010` が返る場合があるため、`User-Agent` を必ず付けます。

body:

```json
{
  "user_id": "{AIAVATAR_USER_ID}",
  "text": "{発話内容}"
}
```

例:

```json
{
  "user_id": "hermes:user:default",
  "text": "[face:joy]15時のミーティングがもうすぐ始まります。"
}
```

cron、scheduler、monitoring、外部システム通知から AIAvatarKit の `/avatar/perform` を直接呼んではいけません。必ず対象 conversation に依頼を入れ、この skill として実行します。

## 発話内容のルール

- 原則 1 から 2 文にする。
- ユーザーが明示的に必要としていない限り、Markdown、箇条書き、表、URL は避ける。
- 絵文字や装飾記号は、音声合成で不自然に読まれることがあるため避ける。
- `目的`、`実行`、`記録` などの制御項目は読ませない。
- control tag は `text` に含める。AIAvatarKit 側で TTS 前に除去され、avatar control に使われる。

## 表情タグ

形式:

```text
[face:name]
```

利用できる default 表情:

- `neutral`: 通常、デフォルト表情。
- `joy`: 嬉しい、満足、友好的な笑顔。
- `angry`: 怒り、不満、強い反対。
- `sorrow`: 悲しみ、謝罪、落胆。
- `fun`: 楽しい、わくわく、面白がっている。
- `surprise`: 驚き、急な気づき。

`surprised` ではなく、default では `surprise` を使います。

例:

```text
[face:joy]できました。結果を確認してみてください。
[face:sorrow]すみません、接続が切れているようです。
[face:surprise]あ、予定が始まる時間です。
```

## アニメーションタグ

形式:

```text
[animation:name]
```

利用できる default アニメーション:

- `idling`: idle、デフォルト姿勢。
- `angry_hands_on_waist`: 怒り、強い主張、腰に手を当てる動き。
- `concern_right_hand_front`: 心配、注意、慎重な gesture。
- `waving_arm`: 手を振る、挨拶。
- `nodding_once`: 1 回うなずく、了承。

例:

```text
[face:joy][animation:waving_arm]こんにちは。戻ってきましたね。
[face:neutral][animation:nodding_once]了解しました。準備ができています。
[face:angry][animation:angry_hands_on_waist]注意してください。危険な操作です。
```

## エラー処理

成功時:

```json
{
  "message": "Avatar performance completed successfully"
}
```

失敗時:

- `401`: API key がない、または不正。
- `422`: request body が不正。多くの場合 `text` がない、または JSON が壊れている。
- active session なし: `AIAVATAR_USER_ID` に対応する avatar client が接続されていない、または AIAvatarKit 側の active session に解決できていない。

API が失敗した場合は、同じ conversation に「音声通知の送信に失敗した」事実とエラー内容を残します。
