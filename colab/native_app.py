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
    weights = ROOT / 'weights' / 'RealESRGAN_x4plus.pth'
    weights.parent.mkdir(parents=True, exist_ok=True)
    if weights.exists() and weights.stat().st_size > 10_000_000:
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
        temp.replace(weights)
    except Exception:
        temp.unlink(missing_ok=True)
        raise
    return weights


def run_restore(src: Path, tile=128, quality=95, clean_frames=True):
    import cv2
    from tqdm.auto import tqdm
    import torch
    fps, w, h, total = video_info(src)
    if w * 4 > 8192 or h * 4 > 8192:
        raise RuntimeError('Video đầu vào quá lớn để phóng x4 (giới hạn 8192px mỗi chiều).')
    token = re.sub(r'[^A-Za-z0-9_-]+', '_', src.stem)[:36] or 'video'
    folder = ROOT / f'{token}_{src.stat().st_size}'
    frames = folder / 'frames'
    frames.mkdir(parents=True, exist_ok=True)
    output = folder / f'{token}_RealESRGAN_x4.mp4'
    if output.exists() and output.stat().st_size > 0:
        print('Đã có file hoàn chỉnh trong thư mục kết quả:', output)
        return output
    print(f'Video: {w}×{h}, {fps:.3f} FPS, số frame: {total or "đang xác định"}', flush=True)
    weights = install_weights()
    model = Upscaler(weights, tile=tile)
    print(f'GPU: {torch.cuda.get_device_name(0)} | Model: RealESRGAN_x4plus | Tile: {tile}', flush=True)
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError('OpenCV không thể đọc video đầu vào.')
    if not total:
        counted = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total = counted if counted > 0 else None
    bar = tqdm(total=total, desc='Khôi phục khung hình', unit='frame', dynamic_ncols=True)
    index = 0
    completed_this_run = 0
    started = time.monotonic()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            destination = frames / f'{index:08d}.jpg'
            if not destination.exists() or destination.stat().st_size == 0:
                restored = model.enhance(frame)
                if not cv2.imwrite(str(destination), restored, [cv2.IMWRITE_JPEG_QUALITY, quality]):
                    raise RuntimeError(f'Không lưu được frame {index}.')
                completed_this_run += 1
            index += 1
            bar.update(1)
            if index % 10 == 0:
                rate = completed_this_run / max(time.monotonic() - started, .01)
                bar.set_postfix({'GPU fps': f'{rate:.2f}', 'xử lý mới': completed_this_run})
    finally:
        bar.close()
        cap.release()
    if index == 0:
        raise RuntimeError('Video không có khung hình.')
    print('Đã xử lý hình ảnh. Đang ghép video và âm thanh gốc...', flush=True)
    temp = folder / 'restored.partial.mp4'
    cmd = [
        'ffmpeg', '-hide_banner', '-y', '-loglevel', 'error', '-stats',
        '-framerate', f'{fps:.9f}', '-i', str(frames / '%08d.jpg'),
        '-i', str(src), '-map', '0:v:0', '-map', '1:a:0?',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-b:a', '192k', '-shortest', '-movflags', '+faststart', str(temp),
    ]
    p = subprocess.run(cmd, text=True, capture_output=True)
    if p.returncode:
        raise RuntimeError('FFmpeg ghép video thất bại: ' + p.stderr[-2000:])
    temp.replace(output)
    if clean_frames:
        shutil.rmtree(frames, ignore_errors=True)
    print(f'✓ HOÀN THÀNH — {output} ({output.stat().st_size / 1048576:.1f} MB)', flush=True)
    return output


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
