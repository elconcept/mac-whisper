#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
  echo "error: native Apple Silicon macOS required" >&2
  exit 1
fi

mkdir -p build/support dist
uv run --no-sync python - <<'PY'
from pathlib import Path
from shutil import copy2
import imageio_ffmpeg
source = Path(imageio_ffmpeg.get_ffmpeg_exe())
target = Path("build/support/ffmpeg")
copy2(source, target)
target.chmod(0o755)
print("Bundling FFmpeg:", source)
PY

uv run --no-sync pyinstaller --noconfirm --clean --windowed --onedir \
  --name MacWhisper \
  --collect-all whisperx \
  --collect-all mlx_whisper \
  --collect-all mlx_audio \
  --collect-all pyannote.audio \
  --collect-all transformers \
  --collect-all mlx \
  --hidden-import=mlx_whisper \
  --hidden-import=whisperx.pipeline \
  --hidden-import=whisperx.alignment \
  --hidden-import=whisperx.diarize \
  --hidden-import=whisperx.backends.diarization_backend \
  --add-binary "build/support/ffmpeg:." \
  gui.py

test -d dist/MacWhisper.app
test -x dist/MacWhisper.app/Contents/MacOS/MacWhisper
QT_QPA_PLATFORM=offscreen dist/MacWhisper.app/Contents/MacOS/MacWhisper --self-test
codesign --force --deep --sign - dist/MacWhisper.app
codesign --verify --deep --strict dist/MacWhisper.app
ditto -c -k --keepParent dist/MacWhisper.app dist/MacWhisper-macos-arm64.zip
shasum -a 256 dist/MacWhisper-macos-arm64.zip > dist/SHA256SUMS
echo "Build OK: dist/MacWhisper-macos-arm64.zip"
