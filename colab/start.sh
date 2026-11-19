#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
echo '[1/4] Kiểm tra runtime GPU'
python - <<'PY'
import torch,sys
print('Python:',sys.version.split()[0], '| torch:',torch.__version__, '| CUDA:',torch.cuda.is_available())
if not torch.cuda.is_available(): raise SystemExit('CHƯA CÓ GPU: Colab > Runtime > Change runtime type > T4 GPU')
PY
command -v ffmpeg >/dev/null || { apt-get update -qq && apt-get install -y ffmpeg; }
echo '[2/4] Cài thư viện giao diện'
python -m pip install -q 'fastapi>=0.110,<1' 'uvicorn>=0.27,<1' 'python-multipart>=0.0.9' 'opencv-python-headless>=4.8,<5'
echo '[3/4] Cài Cloudflare Tunnel'
if ! command -v cloudflared >/dev/null 2>&1; then
  ARCH=$(dpkg --print-architecture)
  curl -fLsS "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${ARCH}.deb" -o /tmp/cloudflared.deb
  dpkg -i /tmp/cloudflared.deb
fi
export STUDIO_PASSWORD="${STUDIO_PASSWORD:-$(python - <<'PY'
import secrets
print(secrets.token_urlsafe(16))
PY
)}"
export STUDIO_DATA="${STUDIO_DATA:-/content/realesrgan-studio-data}"
echo '[4/4] Khởi động giao diện'
echo "TÀI KHOẢN: admin"
echo "MẬT KHẨU: ${STUDIO_PASSWORD}"
echo 'Đang tạo Cloudflare Quick Tunnel...'
python -m uvicorn app.server:app --host 127.0.0.1 --port 7860 >/tmp/realesrgan-web.log 2>&1 &
WEBPID=$!
trap 'kill $WEBPID 2>/dev/null || true' EXIT
sleep 3
if ! kill -0 "$WEBPID" 2>/dev/null; then cat /tmp/realesrgan-web.log; exit 1; fi
cloudflared tunnel --url http://127.0.0.1:7860 --no-autoupdate 2>&1 | tee /tmp/realesrgan-tunnel.log
