# Voice Push Notification Skill

Hermes の cron や tool から、特定の StackChan / ブラウザアバターセッションに音声メッセージを送信します。

この構成では `user_id` を共通キーとして使います。

```text
Hermes conversation
AIAvatarKit user_id
StackChan config.json の user_id
/avatar/perform の user_id
```

これらを同じ値にしてください。

## 使い方

`perform` API を呼び出し、通知先の `user_id` を指定して音声通知を送信します。

- **Endpoint**: `POST {base_url}/avatar/perform`
- **Headers**:
  - `Authorization: Bearer {api_key}`
  - `Content-Type: application/json`
- **Body**:

```json
{
  "text": "[face:joy]Your notification message here",
  "user_id": "user01"
}
```

### Control tags

`text` には control tag を埋め込めます。これにより、アバターの表情やアニメーションを制御できます。

- Face: `[face:expression_name]` 例: `[face:joy]`, `[face:surprise]`
- Animation: `[animation:animation_name]` 例: `[animation:wave_hands]`

### Response

```json
{
  "message": "Avatar performance completed successfully"
}
```

### Error responses

- `400`: 指定された `user_id` に対応するアクティブなセッションが見つからない
- `401`: API Key が不正、または指定されていない

## Example

`user01` に挨拶を通知する例:

```http
POST {base_url}/avatar/perform
Authorization: Bearer {api_key}
Content-Type: application/json

{"text": "[face:joy]Hello! You have a new message.", "user_id": "user01"}
```
