"""Google Colab-native uploader and automatic RealESRGAN_x4plus video restoration.
Run after colab/install_native.sh, inside Google Colab only.
"""
from __future__ import annotations
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen

from app.model import Upscaler, WEIGHTS_URL

ROOT = Path(os.environ.get('STUDIO_DATA', '/content/realesrgan-native-data')).resolve()
ROOT.mkdir(parents=True, exist_ok=True)
VALID_EXTS = {'.mp4', '.mov', '.mkv', '.avi', '.webm'}


def video_info(path):
    result = subprocess.check_output([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,avg_frame_rate,nb_frames',
        '-of', 'json', str(path)], text=True)
    streams = json.loads(result).get('streams', [])
    if not streams:
        raise RuntimeError('Không tìm thấy luồng hình ảnh trong video.')
    stream = streams[0]
    a, b = (stream.get('avg_frame_rate') or '30/1').split('/')
    fps = float(a) / max(float(b), 1)
    if fps <= 0:
        raise RuntimeError('FPS không hợp lệ.')
    total = stream.get('nb_frames', 'N/A')
    return fps, int(stream['width']), int(stream['height']), int(total) if str(total).isdigit() else None


def install_weights():
    """Cache the official weights on Drive; stage a local copy for fast model load.

    On subsequent Colab runtimes, this avoids downloading weights from GitHub again.
    """
    weights = ROOT / 'weights' / 'RealESRGAN_x4plus.pth'
    weights.parent.mkdir(parents=True, exist_ok=True)
    if weights.exists() and weights.stat().st_size > 10_000_000:
        return weights
    drive_cache = Path('/content/drive/MyDrive/RealESRGAN-Studio/weights/RealESRGAN_x4plus.pth')
    if drive_cache.exists() and drive_cache.stat().st_size > 10_000_000:
        print('Đang sử dụng model đã lưu trên Drive...', flush=True)
        temp = weights.with_suffix('.download')
        shutil.copyfile(drive_cache, temp)
        temp.replace(weights)
        return weights
    temp = weights.with_suffix('.download')
    print('Đang tải trọng số RealESRGAN_x4plus chính thức...', flush=True)
    from tqdm.auto import tqdm
    try:
        with urlopen(WEIGHTS_URL, timeout=90) as resp, temp.open('wb') as fh:
            total = int(resp.headers.get('Content-Length', 0))
            with tqdm(total=total or None, unit='B', unit_scale=True, desc='Trọng số x4') as bar:
                while True:
                    block = resp.read(1024 * 1024)
                    if not block:
                        break
                    fh.write(block)
                    bar.update(len(block))
        if temp.stat().st_size < 10_000_000:
            raise RuntimeError('Trọng số model tải về không đủ dung lượng.')
        temp.replace(weights)
        if Path('/content/drive/MyDrive').is_dir():
            drive_cache.parent.mkdir(parents=True, exist_ok=True)
            pending = drive_cache.with_suffix('.uploading')
            shutil.copyfile(weights, pending)
            if pending.stat().st_size != weights.stat().st_size:
                raise IOError('Không thể lưu model lên Drive.')
            pending.replace(drive_cache)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    return weights


def run_restore(src: Path, tile=128, quality=95, clean_frames=True):
    """Legacy notebook entry point, now using resumable encoded segments.

    For persistence across Colab runtimes, mount Google Drive first. The Drive
    workflow in drive_app.py is the recommended interface.
    """
    from colab.checkpoint_engine import restore_video
    persistent = Path('/content/drive/MyDrive/RealESRGAN-Studio/checkpoints')
    checkpoint_root = persistent if Path('/content/drive/MyDrive').is_dir() else ROOT / 'checkpoints'
    if not persistent.parent.is_dir():
        checkpoint_root.mkdir(parents=True, exist_ok=True)
    output = ROOT / 'outputs'
    # Legacy JPEG quality=95 maps to high-quality H.264 CRF 14.
    return restore_video(Path(src), output, checkpoint_root,
                         tile=int(tile), crf=14 if quality >= 95 else 16,
                         segment_frames=60, precision='fp16', stage_source=False)


def main():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Chưa bật GPU. Vào Runtime → Change runtime type → T4 GPU.')
    from google.colab import files
    from IPython.display import display, HTML, FileLink
    from colab.chunk_upload import upload_video
    display(HTML("""<div style="padding:18px;border-radius:14px;background:#142338;color:#fff;font:16px system-ui">
      <b style="font-size:21px">RealESRGAN x4 · Colab Native Studio</b>
      <p>Chọn video. Upload chia khối hiển thị phần trăm thực và tự khôi phục sau khi upload.</p>
      <small>Không Tunnel, không server; file lưu trực tiếp trên runtime Colab.</small></div>"""))
    target = upload_video(ROOT / 'uploads')
    tile = int(os.getenv('REALESRGAN_TILE', '128'))
    quality = int(os.getenv('REALESRGAN_JPEG_QUALITY', '95'))
    auto_download_limit_mb = int(os.getenv('AUTO_DOWNLOAD_LIMIT_MB', '150'))
    print(f'Đã tải lên: {target.name} ({target.stat().st_size/1048576:.1f} MB). Bắt đầu khôi phục...', flush=True)
    output = run_restore(target, tile=tile, quality=quality)
    display(HTML('<h3 style="color:#16834a">✓ Khôi phục hoàn tất — file MP4 đã sẵn sàng</h3>'))
    display(FileLink(str(output), result_html_prefix='Tải file MP4: '))
    if output.stat().st_size <= auto_download_limit_mb * 1048576:
        files.download(str(output))
    else:
        print('File lớn: nhấn liên kết phía trên để tải xuống.')


if __name__ == '__main__':
    main()
