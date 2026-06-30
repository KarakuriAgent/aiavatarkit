# ウェイクワードモデル再学習 手順書（robo_kanon v2）

上から順に実行すれば完了するように書いてあります。所要時間の目安：データ整理30分、収録30〜60分（話者数による）、学習1〜2時間（放置可）、検証・配備30分。

## 背景（なぜ作り直すか）

- 現行モデルは普通の会話に対してスコア0.66〜0.87で誤発火する（2026-06-12の実測：通常会話24クリップ中14クリップが閾値0.65を超過）
- 原因は学習データ：正例103件のうち約35件が2秒超の「ロボ花音、〇〇して」型の文章。トレーナーは2.0秒窓で切り出すため、**名前を含まない会話部分が正例として学習されていた**
- さらに負例が約45件しかなく、実際の雑談・生活音がほぼ含まれていなかった

対策：正例は「ロボ花音」単体のみ（2秒未満）、負例を実環境音声中心に300件規模へ増強。

## 事前メモ

- すべてリポジトリルート（`~/workspace/aiavatarkit`）で実行
- メインサーバーと enrollment サーバーはポート8000を共有するため**同時には起動できない**。収録〜学習中は Stack-chan は応答しません
- 収録は必ず **Stack-chan 本体のマイク経由**で行う（推論時と同じ音響経路で録るため）。enrollment サーバー起動中はデバイスが自動的にそちらへ接続される

---

## Phase 0: enrollment サーバーへ切り替え

```bash
cd ~/workspace/aiavatarkit
docker compose stop stackchan-server
docker compose -f compose.wakeword-enrollment.yml up -d
# healthy になるまで待つ（初回は uv sync で数分かかる）
curl -fsS http://127.0.0.1:8000/health && echo OK
```

API キーを環境変数にセット（以降のコマンドで使用）：

```bash
export AIAVATAR_API_KEY=$(grep '^AIAVATAR_API_KEY=' .env | cut -d= -f2)
export AUTH="Authorization: Bearer $AIAVATAR_API_KEY"
```

---

## Phase 1: 既存正例の整理（2秒超の除外＋ホールドアウト選定）

```bash
python3 tools/wakeword/relabel_and_holdout.py
```

- 2.0秒以上の正例（約35件）が `ignore` になる
- 検証用に正例8本がランダム選定されて `ignore` になり、IDが `wakeword_holdout_ids.txt` に保存される（**このファイルは消さない**。Phase 6 で使う）
- 「学習に使われる正例: 約60件」と表示されればOK

---

## Phase 2: 誤発火クリップを負例として登録

2026-06-12 の実運用音声のうち、現行モデルが発火した14クリップ。まず試聴して、**実際に「ロボ花音」と呼んでいるものが混ざっていないか確認**する（混ざっていたらその行を削除）：

```bash
cd ~/workspace/aiavatarkit
cat > /tmp/fp_clips.txt << 'EOF'
recorded_voices/56677deb-e5fb-40aa-8b45-76216db799df_debug_request.wav
recorded_voices/598547ef-9035-4719-aa1e-0a4e4594ae16_debug_request.wav
recorded_voices/b4eb22cb-1ba7-4e0d-9540-f1c76ea62424_debug_request.wav
recorded_voices/a30b212e-5733-4b23-819d-9559de931905_debug_request.wav
recorded_voices/e76b68c1-8beb-4514-b338-c49dfddac074_debug_request.wav
recorded_voices/4cccc7de-0796-441c-878a-382035c51daf_debug_request.wav
recorded_voices/1c5907fd-7eb2-46e7-96a8-9ee067b7108f_debug_request.wav
recorded_voices/1c656187-ab35-407b-b6fa-349c7c640917_debug_request.wav
recorded_voices/1f1e326d-ec6d-4622-ae00-6055381b9d84_debug_request.wav
recorded_voices/652c44cf-d740-440f-979b-631c61c476e5_debug_request.wav
recorded_voices/d961ee82-6480-463f-826b-0a6ecfecd1c5_debug_request.wav
recorded_voices/91db2ea1-cf4f-485a-a04f-9aaa180097e2_debug_request.wav
recorded_voices/8a2206e7-1391-4096-8bd7-1750753979ce_debug_request.wav
recorded_voices/7478fbf1-cb83-46fd-b585-5d16596eba62_debug_request.wav
EOF
# 1本ずつ再生して確認（スペースで次へ）
while read f; do echo "$f"; afplay "$f"; done < /tmp/fp_clips.txt
```

問題なければ負例として登録（サーバーが自動で再起動される）：

```bash
python3 tools/wakeword/import_negatives.py 誤発火 $(cat /tmp/fp_clips.txt)
```

---

## Phase 3: 家族の雑談を負例として登録

1. iPhone のボイスメモ等で、**普段の環境（テレビ・キッチン音込み）での雑談を10〜20分**録音する。
   **注意：録音中は絶対に「ロボ花音」「かのん」等の名前を言わない**（言ってしまったらその部分を聞いて特定し、分割後にチャンクを削除）
2. AirDrop 等で Mac に転送して変換・分割・登録：

```bash
# m4a → 16kHz モノラル wav（macOS標準の afconvert を使用）
afconvert -f WAVE -d LEI16@16000 -c 1 ~/Downloads/家族雑談.m4a /tmp/chat.wav

# 3秒チャンクに分割（無音は自動で捨てられる）
python3 tools/wakeword/split_audio.py /tmp/chat.wav /tmp/chat_chunks

# 負例として登録
python3 tools/wakeword/import_negatives.py 雑談 /tmp/chat_chunks/*.wav
```

15分の録音なら150〜250チャンクになる想定。

---

## Phase 4: 読み上げ収録（Stack-chan のマイクに向かって）

### 4-1. 正例の収録（話者ごとに実施）

収録モードを開始：

```bash
curl -X POST -H "$AUTH" -H "Content-Type: application/json" \
  -d '{"wakeword": "ロボ花音"}' http://127.0.0.1:8000/api/reading/start
```

以下を**1行ずつ、行間に1秒以上の間を空けて**発話する（VADが1発話=1候補として切り出す）。話者ごとに全部読む：

```
【正例 読み上げ台本：すべて「ロボ花音」とだけ言う。言い方を変える】
 1. ロボ花音        （普通に）
 2. ロボ花音        （普通に・もう一回）
 3. ロボ花音？      （問いかけるように）
 4. ロボ花音！      （強めに呼ぶ）
 5. ねぇロボ花音    （軽い前置き付き）
 6. ロボ花音        （小さい声で）
 7. ロボ花音        （ささやき声で）
 8. ロボ花音        （大きい声で）
 9. ロボ花音        （2〜3m離れた場所から呼ぶ）
10. ロボ花音        （デバイスの横や後ろなど別方向から）
11. ロボ花音        （早口で）
12. ロボ花音        （ゆっくりはっきり）
13. ロボ花音        （笑いながら）
14. ロボ花音        （気だるい感じで）
15. ロボ花音        （語尾を上げて）
16. ロボ花音        （語尾を下げて）
--- ここからテレビや音楽をつけた状態で ---
17. ロボ花音        （普通に）
18. ロボ花音        （大きい声で）
19. ロボ花音        （小さい声で）
20. ロボ花音        （2〜3m離れて）
```

収録モードを終了：

```bash
curl -X POST -H "$AUTH" http://127.0.0.1:8000/api/reading/stop
```

話者を交代して 4-1 を繰り返す（家でデバイスを使う人全員分）。

### 4-2. 負例の収録（話者ごとに実施）

ラベルを変えて収録モードを開始（「ロボ花音」以外のラベルは自動的に負例になる）：

```bash
curl -X POST -H "$AUTH" -H "Content-Type: application/json" \
  -d '{"wakeword": "負例読み上げ"}' http://127.0.0.1:8000/api/reading/start
```

以下を1行ずつ、普段の話し声で読む：

```
【負例 読み上げ台本A：名前に音が似ているもの】
 1. ロボ
 2. ロボット
 3. ロボットみたいだね
 4. 花音
 5. かのん
 6. かの
 7. かな
 8. お母さん
 9. たかのさんから電話だよ
10. 我慢の限界
11. 簡単じゃん
12. 鞄の中見た？
13. 何の音？
14. あの音なに
15. ぼかんと音がした
16. この間のやつ

【負例 読み上げ台本B：実際に誤発火した文（音が近いのでそのまま読む）】
17. 木造の音はなんないよ
18. いらないゴム持ってない
19. 一瞬だけだとチューミング祭で

【負例 読み上げ台本C：くだけた日常会話】
20. ねえこれ見て
21. それでさー
22. やっぱいいや
23. ちょっと何それ
24. まじで？
25. うそでしょ
26. どうしようかな
27. なんでもない
28. もういいよ
29. そうなんだ
30. わかんない
31. なんないよ
32. もう買わん
33. 持ってないよ
```

```bash
curl -X POST -H "$AUTH" http://127.0.0.1:8000/api/reading/stop
```

話者を交代して 4-2 を繰り返す。

### 4-3. 収録結果の確認

```bash
curl -s -H "$AUTH" http://127.0.0.1:8000/api/candidates | python3 -c "
import sys, json
cs = json.load(sys.stdin)
def is_pos(c): return c.get('role')=='positive' or (c.get('role') is None and c.get('wakeword')=='ロボ花音')
def is_neg(c): return c.get('role')=='negative' or (c.get('role') is None and c.get('wakeword') not in (None,'ロボ花音'))
print('正例:', sum(map(is_pos, cs)), '/ 負例:', sum(map(is_neg, cs)), '/ ignore:', sum(c.get('role')=='ignore' for c in cs))
"
```

目標：**正例100〜130件 / 負例250件以上**。正例の収録ミス（咳・無言など）があれば、長さが極端なものを `PATCH /api/candidates/<id>` で `{"role": "ignore"}` にして除外。

---

## Phase 5: 学習

```bash
curl -X POST -H "$AUTH" -H "Content-Type: application/json" \
  -d '{
    "wakeword": "ロボ花音",
    "target_phrases": ["ロボ花音"],
    "model_name": "robo_kanon_v2",
    "model_size": "small",
    "n_samples": 1500,
    "n_samples_val": 200,
    "steps": 4000,
    "threshold": 0.7,
    "tts_backend": "configured_tts",
    "positive_source": "recorded_only",
    "negative_source": "recorded_plus_tts",
    "run_eval": true
  }' http://127.0.0.1:8000/api/training/jobs
```

ポイント：
- `positive_source: recorded_only` — TTS正例を使わない（合成音声の「ロボ花音」はアバター自身の声になり、自己発話で発火する原因になりうるため）
- `model_size: small` — tiny より誤発火耐性が上がる。推論負荷はまだ軽い
- `threshold: 0.7` は仮置き。運用閾値は Phase 6 で実測して決める

進捗確認（`status` が `succeeded` になるまで。1〜2時間程度）：

```bash
curl -s -H "$AUTH" http://127.0.0.1:8000/api/training/jobs | python3 -m json.tool | head -30
# ログを見る場合（job_id は上の出力から）
curl -s -H "$AUTH" http://127.0.0.1:8000/api/training/jobs/<job_id>/log | python3 -c "import sys,json; print(json.load(sys.stdin)['log'])" | tail -30
```

---

## Phase 6: 検証と閾値決定

ホールドアウト正例（学習に使っていない8本）と全負例で新モデルをスコアリング：

```bash
docker compose -f compose.wakeword-enrollment.yml exec -T \
  -e HOLDOUT_IDS="$(paste -sd, wakeword_holdout_ids.txt)" \
  wakeword-enrollment-server uv run python - robo_kanon_v2 \
  < tools/wakeword/validate_model.py
```

判定基準：
- **正例の検出失敗が0件**であること（失敗があれば正例を追加収録して Phase 5 をやり直し）
- **TP最小ピークと FP最大ピークのギャップが0.15以上**あること（足りなければ負例追加 or steps を6000に増やして再学習）
- 表示された**推奨閾値をメモする**

---

## Phase 7: 配備

```bash
# モデルをホストへコピー
docker compose -f compose.wakeword-enrollment.yml cp \
  wakeword-enrollment-server:/app/data/wakeword_enrollment/models/robo_kanon_v2.onnx \
  ./models/wakewords/robo_kanon_v2.onnx
```

`.env` を編集（2行変更。閾値は Phase 6 の推奨値に置き換える）：

```
AUDIO_WAKEWORD_MODEL_PATHS=/app/models/wakewords/robo_kanon_v2.onnx
AUDIO_WAKEWORD_THRESHOLD=<Phase 6 の推奨値>
```

サーバーを切り替え：

```bash
docker compose -f compose.wakeword-enrollment.yml down
docker compose up -d --force-recreate stackchan-server
curl -fsS http://127.0.0.1:8000/health && echo OK
```

---

## Phase 8: 運用（継続改善ループ）

発火スコアはログに残る。誤発火を見つけたら：

```bash
docker compose logs stackchan-server | grep "Audio wakeword detected"
```

1. 該当時刻の `recorded_voices/*_debug_request.wav` を特定（コンテナログはUTC表記。ファイル時刻はJSTなので+9時間）
2. `python3 tools/wakeword/import_negatives.py 誤発火 <ファイル>` で負例に追加（enrollment サーバー起動時）
3. 負例が20〜30件貯まったら Phase 5〜7 を再実行

これを2〜3周すると実環境に強いモデルになる。

## 補足

- 検出漏れ（呼んでも起きない）が増えた場合：その状況（距離・声量）の正例を追加収録して再学習、または閾値を下げる（FP最大ピークより上の範囲で）
- それでも誤発火が残る場合の次の一手：音響発火時にSTTテキストに名前がなければ宛先LLM判定へ回す pipeline 修正（設計済み・未実装）
