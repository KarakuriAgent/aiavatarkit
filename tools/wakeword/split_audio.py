# 長い録音（家族の雑談など）を約3秒のチャンクに分割し、ほぼ無音のチャンクを捨てる。
# 出力は import_negatives.py にそのまま渡せる。
#
# 使い方:
#   python3 tools/wakeword/split_audio.py input.wav /tmp/chat_chunks
# 入力は 16bit PCM モノラル wav（afconvert -f WAVE -d LEI16@16000 -c 1 で変換しておく）。
import array
import math
import sys
import wave
from pathlib import Path

CHUNK_SECONDS = 3.0
MIN_RMS = 200  # 16bit 振幅。これ未満のチャンクは無音とみなして捨てる

if len(sys.argv) != 3:
    sys.exit("usage: python3 tools/wakeword/split_audio.py input.wav outdir")

src, outdir = sys.argv[1], Path(sys.argv[2])
outdir.mkdir(parents=True, exist_ok=True)

with wave.open(src, "rb") as wf:
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
        sys.exit("16bit モノラル PCM が必要です (afconvert -f WAVE -d LEI16@16000 -c 1 で変換)")
    sample_rate = wf.getframerate()
    frames = wf.readframes(wf.getnframes())

samples = array.array("h", frames)
chunk_size = int(sample_rate * CHUNK_SECONDS)
kept = skipped = 0
for index, start in enumerate(range(0, len(samples), chunk_size)):
    chunk = samples[start:start + chunk_size]
    if len(chunk) < sample_rate:  # 1秒未満の端切れは捨てる
        continue
    rms = math.sqrt(sum(s * s for s in chunk) / len(chunk))
    if rms < MIN_RMS:
        skipped += 1
        continue
    path = outdir / f"chunk_{index:04d}.wav"
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes(chunk.tobytes())
    kept += 1

print(f"出力 {kept}件 / 無音スキップ {skipped}件 -> {outdir}")
