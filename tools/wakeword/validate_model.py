# 学習済みモデルをホールドアウト正例と負例でスコアリングし、運用閾値を提案する。
# コンテナ内で実行する（実際の推論ランタイムと同条件で測るため）。
#
# 使い方（リポジトリルートで実行）:
#   docker compose -f compose.wakeword-enrollment.yml exec -T \
#     -e HOLDOUT_IDS="$(paste -sd, wakeword_holdout_ids.txt)" \
#     wakeword-enrollment-server uv run python - robo_kanon_v2 \
#     < tools/wakeword/validate_model.py
#
# 引数: モデルID（/app/data/wakeword_enrollment/models/<モデルID>.onnx）
import json
import os
import sys
import wave
from pathlib import Path

from aiavatar.sts.wakeword.livekit import LiveKitWakewordDetector

CANDIDATE_DIR = Path("/app/data/wakeword_enrollment/candidates")
MODEL_DIR = Path("/app/data/wakeword_enrollment/models")
THRESHOLD_GRID = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

model_id = sys.argv[1] if len(sys.argv) > 1 else sys.exit("usage: ... python - <model_id>")
model_path = MODEL_DIR / f"{model_id}.onnx"
if not model_path.exists():
    sys.exit(f"モデルが見つかりません: {model_path}\n登録済み: {[p.stem for p in MODEL_DIR.glob('*.onnx')]}")

holdout_ids = {i for i in os.environ.get("HOLDOUT_IDS", "").split(",") if i}
if not holdout_ids:
    sys.exit("HOLDOUT_IDS 環境変数が空です")

positives, negatives = [], []
for meta_path in CANDIDATE_DIR.glob("*.json"):
    c = json.loads(meta_path.read_text())
    if not Path(c["path"]).exists():
        continue
    if c["id"] in holdout_ids:
        positives.append(c)
    elif c.get("role") == "negative" or (c.get("role") is None and c.get("wakeword") not in (None, "ロボ花音")):
        negatives.append(c)

print(f"model={model_id} / ホールドアウト正例={len(positives)} / 負例={len(negatives)}")


def read_wav(path):
    with wave.open(path, "rb") as wf:
        sr = wf.getframerate()
        return wf.readframes(wf.getnframes()), sr, wf.getnframes() / sr


def peak_threshold(clip):
    """検出が維持される最大の閾値（ピークスコアの近似）を返す。未検出なら None。"""
    audio, sr, dur = read_wav(clip["path"])
    peak = None
    for th in THRESHOLD_GRID:
        detector = LiveKitWakewordDetector(
            model_paths=[str(model_path)],
            threshold=th, activation_window=max(dur, 1.0), cooldown=0,
        )
        if detector.process(audio, sample_rate=sr, session_id=f"val:{clip['id']}:{th}"):
            peak = th
        else:
            break
    return peak


tp_peaks = []
print("\n[ホールドアウト正例]")
for c in positives:
    peak = peak_threshold(c)
    tp_peaks.append(peak)
    print(f"  {c['id'][:8]} dur={c['duration']:.1f}s peak>={peak}")

fp_peaks = []
print("\n[負例] (検出されたものだけ表示)")
for c in negatives:
    peak = peak_threshold(c)
    if peak is not None:
        fp_peaks.append(peak)
        print(f"  {c['id'][:8]} ({c.get('wakeword')}) dur={c['duration']:.1f}s peak>={peak}")

missed = [p for p in tp_peaks if p is None]
tp_min = min((p for p in tp_peaks if p is not None), default=None)
fp_max = max(fp_peaks, default=None)

print("\n==== 結果 ====")
print(f"正例の最小ピーク: {tp_min} / 検出失敗: {len(missed)}件")
print(f"負例の最大ピーク: {fp_max} (閾値0.40以上で発火した負例: {len(fp_peaks)}/{len(negatives)})")

if missed:
    print("→ 検出できない正例があります。正例の追加収録か steps 増で再学習してください。")
elif tp_min is None:
    print("→ 正例がありません。HOLDOUT_IDS を確認してください。")
elif fp_max is None:
    print(f"→ 負例はすべて 0.40 未満。閾値は {max(0.55, tp_min - 0.1):.2f} 前後を推奨。")
elif tp_min - fp_max >= 0.15:
    print(f"→ 推奨閾値: {fp_max + (tp_min - fp_max) * 0.4:.2f} (FP最大 {fp_max} と TP最小 {tp_min} の間)")
else:
    print(f"→ TP最小 {tp_min} と FP最大 {fp_max} のギャップが不足。負例追加 or steps 増で再学習を推奨。")
