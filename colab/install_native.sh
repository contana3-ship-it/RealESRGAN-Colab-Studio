#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python - <<'PY'
import torch
print('PyTorch:', torch.__version__, '| GPU CUDA:', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('Bật GPU: Runtime > Change runtime type > T4 GPU, sau đó chạy lại.')
PY
if ! command -v ffmpeg >/dev/null || ! command -v ffprobe >/dev/null; then
  apt-get update -qq && apt-get install -y -qq ffmpeg
fi
python -m pip install -q 'opencv-python-headless>=4.8,<5' 'tqdm>=4.66,<5' 'ipywidgets>=8,<9'
echo "Da cai dat xong. San sang chon video tu Google Drive trong notebook Colab."
