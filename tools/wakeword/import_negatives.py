# wav ファイル群を負例候補として enrollment サーバーに一括登録するスクリプト。
# コンテナの candidates ディレクトリへコピーし、サーバーを再起動して読み込ませる。
#
# 使い方（リポジトリルートで実行）:
#   python3 tools/wakeword/import_negatives.py <ラベル> file1.wav [file2.wav ...]
# 例:
#   python3 tools/wakeword/import_negatives.py 誤発火 recorded_voices/xxx_debug_request.wav
#   python3 tools/wakeword/import_negatives.py 雑談 /tmp/chat_chunks/*.wav
#
# 入力は 16bit PCM モノラル wav（サンプルレートは任意。学習時に16kHzへ変換される）。
import datetime
import json
import shutil
import subprocess
import sys
import tempfile
import uuid
import wave
from pathlib import Path

COMPOSE = ["docker", "compose", "-f", "compose.wakeword-enrollment.yml"]
SERVICE = "wakeword-enrollment-server"
CONTAINER_DIR = "/app/data/wakeword_enrollment/candidates"

if len(sys.argv) < 3:
    sys.exit("usage: python3 tools/wakeword/import_negatives.py <ラベル> file1.wav [file2.wav ...]")

label = sys.argv[1]
files = sys.argv[2:]
staging = Path(tempfile.mkdtemp(prefix="ww_negatives_"))

count = 0
for file in files:
    with wave.open(file, "rb") as wf:
        sample_rate = wf.getframerate()
        duration = wf.getnframes() / sample_rate if sample_rate else 0.0
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            sys.exit(
                f"{file}: 16bit モノラル PCM が必要です。"
                "afconvert -f WAVE -d LEI16@16000 -c 1 <入力> <出力.wav> で変換してください。"
            )
    candidate_id = uuid.uuid4().hex
    shutil.copy(file, staging / f"{candidate_id}.wav")
    metadata = {
        "id": candidate_id,
        "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "wakeword": label,
        "role": "negative",
        "session_id": "import",
        "duration": duration,
        "sample_rate": sample_rate,
        "path": f"{CONTAINER_DIR}/{candidate_id}.wav",
    }
    (staging / f"{candidate_id}.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    count += 1
    print(f"prepared: {file} -> {candidate_id} ({duration:.1f}s)")

subprocess.run(COMPOSE + ["cp", f"{staging}/.", f"{SERVICE}:{CONTAINER_DIR}/"], check=True)
print(f"\n{count}件をコピーしました。サーバーを再起動して読み込みます...")
subprocess.run(COMPOSE + ["restart", SERVICE], check=True)
shutil.rmtree(staging)
print("完了。GET /api/candidates で件数を確認してください。")
