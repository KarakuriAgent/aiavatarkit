# 学習データ整理スクリプト:
# 1) 2.0秒以上の正例（文章入り）を ignore に変更
# 2) 残った正例から検証用ホールドアウトを8本選んで ignore に変更し、IDを保存
#
# 使い方（enrollment サーバー起動中に、リポジトリルートで実行）:
#   export AIAVATAR_API_KEY=$(grep '^AIAVATAR_API_KEY=' .env | cut -d= -f2)
#   python3 tools/wakeword/relabel_and_holdout.py
import json
import os
import random
import sys
import urllib.request

BASE = os.environ.get("ENROLLMENT_BASE_URL", "http://127.0.0.1:8000")
API_KEY = os.environ.get("AIAVATAR_API_KEY", "")
WAKEWORD = "ロボ花音"
MAX_POSITIVE_DURATION = 2.0
HOLDOUT_COUNT = 8
HOLDOUT_FILE = "wakeword_holdout_ids.txt"


def api(method: str, path: str, body=None):
    req = urllib.request.Request(
        BASE + path,
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read() or "{}")


def is_positive(c: dict) -> bool:
    if c.get("role") == "positive":
        return True
    return c.get("role") is None and c.get("wakeword") == WAKEWORD


if os.path.exists(HOLDOUT_FILE):
    sys.exit(f"{HOLDOUT_FILE} が既に存在します。再実行する場合は削除してください。")

candidates = api("GET", "/api/candidates")

long_positives = [c for c in candidates if is_positive(c) and c["duration"] >= MAX_POSITIVE_DURATION]
for c in long_positives:
    api("PATCH", f"/api/candidates/{c['id']}", {"role": "ignore"})
    print(f"ignore (長すぎ {c['duration']:.1f}s): {c['id']}")

short_positives = [c for c in candidates if is_positive(c) and c["duration"] < MAX_POSITIVE_DURATION]
random.seed(42)
holdout = random.sample(short_positives, min(HOLDOUT_COUNT, len(short_positives)))
for c in holdout:
    api("PATCH", f"/api/candidates/{c['id']}", {"role": "ignore"})
    print(f"ignore (ホールドアウト {c['duration']:.1f}s): {c['id']}")

with open(HOLDOUT_FILE, "w") as f:
    f.write("\n".join(c["id"] for c in holdout) + "\n")

remaining = len(short_positives) - len(holdout)
print(f"\n長すぎ正例の除外: {len(long_positives)}件 / ホールドアウト: {len(holdout)}件 ({HOLDOUT_FILE})")
print(f"学習に使われる正例: {remaining}件")
