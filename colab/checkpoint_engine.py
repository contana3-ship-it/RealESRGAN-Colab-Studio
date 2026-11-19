"""Resumable, loss-minimizing RealESRGAN video restoration on Colab.

Durability boundary: only validated and completely copied MP4 segments on Drive
are considered committed. A dead Colab VM loses at most the current segment.
No idle-prevention tricks, background runtime promises, or silent reprocessing.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

MODEL_NAME = 'RealESRGAN_x4plus'
ENGINE_VERSION = 2
SEG_RE = re.compile(r'^seg_(\d{10})_(\d{5})\.mp4$')


def _now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def _run(cmd: list[str], *, label: str):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(f'{label}: {p.stderr[-2500:]}')
    return p.stdout


def ffprobe_info(path: Path):
    return json.loads(_run([
        'ffprobe', '-v', 'error', '-show_entries',
        'stream=index,codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames',
        '-show_entries', 'format=duration', '-of', 'json', str(path)
    ], label=f'Không đọc được metadata: {path.name}'))


def _frames_in_segment(path: Path) -> int:
    # Our ffmpeg-produced MP4 segments carry an exact nb_frames in their header.
    # Reading this metadata is MUCH faster than decoding each segment on resume.
    result = json.loads(_run([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=nb_frames', '-of', 'json', str(path)
    ], label=f'Không kiểm tra được segment: {path.name}'))
    streams = result.get('streams') or []
    value = streams[0].get('nb_frames') if streams else None
    if str(value).isdigit():
        return int(value)
    result = json.loads(_run([
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames',
        '-show_entries', 'stream=nb_read_frames', '-of', 'json', str(path)
    ], label=f'Không kiểm tra được segment: {path.name}'))
    streams = result.get('streams') or []
    if not streams or not str(streams[0].get('nb_read_frames', '')).isdigit():
        raise RuntimeError(f'Không đếm được frame của {path.name}')
    return int(streams[0]['nb_read_frames'])


def _fps(src: Path):
    from fractions import Fraction
    info = ffprobe_info(src)
    streams = [s for s in info.get('streams', []) if s.get('codec_type') == 'video']
    if not streams:
        raise RuntimeError('File nguồn không có video stream.')
    s = streams[0]
    rate = s.get('avg_frame_rate', '0/0')
    try:
        frame_rate = Fraction(rate)
        if frame_rate <= 0:
            frame_rate = Fraction(s['r_frame_rate'])
    except (ValueError, ZeroDivisionError, KeyError):
        raise RuntimeError('Không xác định được FPS video.')
    if frame_rate <= 0:
        raise RuntimeError('FPS video không hợp lệ.')
    total = s.get('nb_frames', '')
    total = int(total) if str(total).isdigit() else None
    return frame_rate, int(s['width']), int(s['height']), total


def sample_signature(path: Path, sample_bytes: int = 1024 * 1024) -> dict:
    size = path.stat().st_size
    h = hashlib.sha256()
    h.update(str(size).encode())
    with path.open('rb') as f:
        h.update(f.read(sample_bytes))
        if size > sample_bytes:
            f.seek(max(0, size - sample_bytes))
            h.update(f.read(sample_bytes))
    return {'bytes': size, 'sample_sha256': h.hexdigest(), 'path': str(path.resolve())}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def quick_signature(path: Path, sample_bytes: int = 1024 * 1024) -> str:
    size = path.stat().st_size
    h = hashlib.sha256(str(size).encode())
    with path.open('rb') as f:
        h.update(f.read(sample_bytes))
        if size > sample_bytes:
            f.seek(max(0, size - sample_bytes))
            h.update(f.read(sample_bytes))
    return h.hexdigest()


def _json_atomic(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.writing')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass  # Drive FUSE may not support this syscall.
    tmp.replace(path)


def _format_size(n):
    return f'{n/1048576:.1f} MB'


def _copy_stream(source: Path, target: Path, *, resume: bool = False):
    """Partial copies are never promoted. Retry final upload if source prefix differs."""
    total = source.stat().st_size
    target.parent.mkdir(parents=True, exist_ok=True)
    pos = 0
    if resume and target.exists():
        pos = target.stat().st_size
        if pos > total:
            pos = 0
        elif pos:
            # Validate both ends of existing prefix before trusting offset.
            n = min(1024 * 1024, pos)
            with source.open('rb') as src, target.open('rb') as dst:
                if src.read(n) != dst.read(n):
                    pos = 0
                else:
                    src.seek(pos - n); dst.seek(pos - n)
                    if src.read(n) != dst.read(n):
                        pos = 0
    with source.open('rb') as src, target.open('ab' if pos else 'wb') as dst:
        src.seek(pos)
        while True:
            block = src.read(8 * 1024 * 1024)
            if not block:
                break
            dst.write(block)
        dst.flush()
        try:
            os.fsync(dst.fileno())
        except OSError:
            pass
    if target.stat().st_size != total:
        raise IOError('Sao chép lên Drive không hoàn tất.')
    # Verify sampled contents (fast even for large files).
    with source.open('rb') as src, target.open('rb') as dst:
        for offset in (0, max(0, total // 2 - 524288), max(0, total - 1048576)):
            src.seek(offset); dst.seek(offset)
            if src.read(1048576) != dst.read(1048576):
                raise IOError('Drive trả về dữ liệu khác file gốc (kiểm tra lại dung lượng Drive).')


@contextmanager
def _single_runtime_lock(lock_path: Path):
    """Prevent duplicate jobs in the same Colab runtime / kernel."""
    import fcntl
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a+') as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Công việc này đã đang chạy trong Colab; không mở phiên xử lý trùng.')
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _copy_retry(source: Path, partial: Path, attempts: int = 3):
    for attempt in range(attempts):
        try:
            _copy_stream(source, partial, resume=True)
            return
        except (OSError, IOError) as e:
            if attempt == attempts - 1:
                raise
            delay = 2 ** attempt
            print(f'Lỗi đọc/ghi Drive: {e}. Thử lại sau {delay}s...', flush=True)
            time.sleep(delay)


def stage_source_video(src: Path, local_root: Path) -> Path:
    size = src.stat().st_size
    free = shutil.disk_usage(local_root).free
    if free < size + 3 * 1024**3:
        print('Không đủ chỗ staging; đọc video trực tiếp từ Drive.', flush=True)
        return src
    dest = local_root / ('input' + src.suffix.lower())
    if dest.exists() and dest.stat().st_size == size:
        return dest
    print(f'Đang sao chép video {_format_size(size)} về SSD Colab để đọc nhanh hơn...', flush=True)
    partial = dest.with_suffix(dest.suffix + '.partial')
    _copy_retry(src, partial)
    partial.replace(dest)
    return dest


def _valid_segments(segments_dir: Path, chunk_frames: int, manifest: dict):
    """Reconstruct contiguous durable progress from files, including orphan commits."""
    expected = 0
    segments = []
    previously_known = {r['name']: r for r in manifest.get('segments', [])}
    # A single listing is much faster than repeated Drive directory traversals.
    by_start = {}
    for path in segments_dir.glob('seg_*.mp4'):
        m = SEG_RE.fullmatch(path.name)
        if m:
            count = int(m.group(2))
            if 0 < count <= chunk_frames:
                by_start.setdefault(int(m.group(1)), []).append((count, path))
    while True:
        candidates = by_start.get(expected, [])
        if not candidates:
            break
        # A full chunk takes priority over a dangling shorter segment.
        candidates.sort(key=lambda item: item[0], reverse=True)
        valid = None
        for count, path in candidates:
            try:
                if path.stat().st_size <= 0 or _frames_in_segment(path) != count:
                    continue
                previous = previously_known.get(path.name)
                if previous and previous.get('bytes') != path.stat().st_size:
                    continue
                if previous and previous.get('sample_sha256') and quick_signature(path) != previous['sample_sha256']:
                    continue
                valid = (count, path)
                break
            except (OSError, RuntimeError, ValueError):
                continue
        if valid is None:
            break
        count, path = valid
        segments.append({'name': path.name, 'start': expected, 'frames': count,
                         'bytes': path.stat().st_size, 'sample_sha256': quick_signature(path)})
        expected += count
        if count < chunk_frames:  # final incomplete chunk means EOF.
            break
    return segments, expected


def _ffmpeg_encoder(dst: Path, width: int, height: int, fps_text: str, crf: int, preset: str):
    log = dst.with_suffix('.ffmpeg.log')
    err = log.open('wb')
    args = [
        'ffmpeg', '-hide_banner', '-nostdin', '-y', '-loglevel', 'error',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s:v', f'{width}x{height}',
        '-framerate', fps_text, '-i', 'pipe:0', '-an',
        '-c:v', 'libx264', '-preset', preset, '-crf', str(crf),
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(dst),
    ]
    try:
        p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=err)
    except Exception:
        err.close()
        raise
    return p, err, log


def _encode_frames(cap, model, local_segment: Path, fps_text: str, size_out: tuple[int, int],
                   chunk_frames: int, crf: int, preset: str, progress: Callable[[int], None]):
    import numpy as np
    width_out, height_out = size_out
    p, error_fh, logfile = _ffmpeg_encoder(local_segment, width_out, height_out, fps_text, crf, preset)
    count = 0
    failed = False
    try:
        for _ in range(chunk_frames):
            ok, frame = cap.read()
            if not ok:
                break
            enhanced = model.enhance(frame)
            if enhanced.shape != (height_out, width_out, 3) or enhanced.dtype != np.uint8:
                raise RuntimeError('Kết quả khung hình sai kích thước hoặc định dạng.')
            p.stdin.write(memoryview(np.ascontiguousarray(enhanced)))
            count += 1
            progress(1)
        p.stdin.close()
        if p.wait() != 0:
            raise RuntimeError(f'FFmpeg encoder lỗi: {logfile.read_text(errors="replace")[-2000:]}')
    except BaseException:
        failed = True
        try:
            p.stdin.close()
        except Exception:
            pass
        if p.poll() is None:
            p.kill()
        p.wait()
        raise
    finally:
        error_fh.close()
        if not failed:
            logfile.unlink(missing_ok=True)
    return count


def _final_mux(seg_paths: list[Path], src: Path, local: Path):
    concat_txt = local.parent / 'concat_list.txt'
    with concat_txt.open('w', encoding='utf-8') as f:
        for segment in seg_paths:
            # Checkpoint directories contain only generated ASCII paths.
            f.write("file '" + str(segment).replace("'", "'\\''") + "'\n")
    # Copy the original audio bitstream where MP4 supports it; never re-encode video.
    command = [
        'ffmpeg', '-hide_banner', '-nostdin', '-y', '-loglevel', 'error',
        '-f', 'concat', '-safe', '0', '-i', str(concat_txt), '-i', str(src),
        '-map', '0:v:0', '-map', '1:a:0?', '-c:v', 'copy',
    ]
    try:
        _run(command + ['-c:a', 'copy', '-shortest', '-movflags', '+faststart',
                        str(local)], label='Ghép video và âm thanh gốc')
    except RuntimeError as original_error:
        # PCM/unsupported input codecs may not fit an MP4 container.
        print('MP4 không hỗ trợ codec âm thanh gốc; chuyển sang AAC 192k.', flush=True)
        try:
            _run(command + ['-c:a', 'aac', '-b:a', '192k', '-shortest',
                            '-movflags', '+faststart', str(local)], label='Ghép video AAC')
        except RuntimeError:
            raise original_error
    if not local.exists() or local.stat().st_size < 1000:
        raise RuntimeError('Không tạo được MP4 kết quả.')


def restore_video(
    src: Path,
    output_dir: Path,
    checkpoint_root: Path,
    *,
    tile: int = 0,
    segment_frames: int = 60,
    crf: int = 14,
    preset: str = 'veryfast',
    precision: str = 'fp16',
    stage_source: bool = True,
    weights: Optional[Path] = None,
    upscaler_factory=None,
    local_root: Optional[Path] = None,
):
    """Restore with durable, crash-resumable 60-frame video chunks.

    upscaler_factory(weights, tile, precision) is injectable for CPU integration tests.
    Input video is CFR-oriented. Variable-FPS videos may have A/V timing drift.
    """
    import cv2
    from tqdm.auto import tqdm
    src = Path(src).resolve(strict=True)
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = Path(checkpoint_root).resolve()
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    if not 10 <= segment_frames <= 1200:
        raise ValueError('segment_frames phải từ 10 đến 1200.')
    if not 0 <= crf <= 24:
        raise ValueError('CRF phải từ 0 đến 24; thấp hơn nghĩa là nét hơn và file lớn hơn.')
    if preset not in ('ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium'):
        raise ValueError('Preset encoder không hợp lệ.')
    if precision not in ('fp16', 'fp32'):
        raise ValueError('Chỉ hỗ trợ fp16 hoặc fp32.')
    if tile not in (0, 64, 96, 128, 192, 256, 384, 512):
        raise ValueError('Tile không hợp lệ.')
    fps, width, height, total = _fps(src)
    if width * 4 > 8192 or height * 4 > 8192:
        raise RuntimeError('Đầu ra x4 vượt giới hạn 8192 pixel mỗi chiều; cần giảm độ phân giải nguồn.')
    if weights is None:
        from colab.native_app import install_weights
        weights = install_weights()
    weights = Path(weights).resolve(strict=True)
    model_hash = sha256(weights)
    signature = sample_signature(src)
    settings = dict(version=ENGINE_VERSION, model=MODEL_NAME, model_sha256=model_hash,
                    upscale=4, precision=precision, tile=tile,
                    segment_frames=segment_frames, crf=crf, preset=preset)
    job_data = {'source': signature, 'settings': settings}
    job_id = hashlib.sha256(json.dumps(job_data, sort_keys=True).encode()).hexdigest()[:20]
    lock_base = Path(local_root or os.getenv('REALESRGAN_LOCAL_WORK', '/content/realesrgan-work'))
    with _single_runtime_lock(lock_base / (job_id + '.lock')):
        job_dir = checkpoint_root / job_id
        segments_dir = job_dir / 'segments'
        segments_dir.mkdir(parents=True, exist_ok=True)
        manifest_file = job_dir / 'checkpoint.json'
        manifest = json.loads(manifest_file.read_text(encoding='utf-8')) if manifest_file.exists() else {}
        if manifest and (manifest.get('source') != signature or manifest.get('settings') != settings):
            raise RuntimeError('Checkpoint và video/cấu hình không trùng khớp; cần tạo job mới.')
        manifest.update({'job_id': job_id, 'source': signature, 'settings': settings,
                         'fps': str(fps), 'dimensions': [width, height],
                         'created_at': manifest.get('created_at', _now()), 'updated_at': _now()})
        stem = re.sub(r'[^0-9a-zA-Z_-]+', '_', src.stem)[:44] or 'video'
        output_path = output_dir / f'{stem}_RealESRGAN_x4_{job_id[:8]}.mp4'
        if manifest.get('state') == 'done' and output_path.exists() and output_path.stat().st_size > 1000:
            try:
                ffprobe_info(output_path)
            except RuntimeError:
                print('MP4 cũ không đọc được; dựng lại từ checkpoint.', flush=True)
            else:
                print('Đã hoàn thành từ phiên trước:', output_path, flush=True)
                return output_path
        segments, completed = _valid_segments(segments_dir, segment_frames, manifest)
        manifest['segments'], manifest['completed_frames'] = segments, completed
        manifest['state'] = 'processing'
        _json_atomic(manifest_file, manifest)
        print(f'Job: {job_id} | nguồn {width}x{height} @ {float(fps):.4f} FPS | x4 | '
              f'CRF {crf} | {precision} | checkpoint mỗi {segment_frames} frame', flush=True)
        if completed:
            print(f'PHỤC HỒI CHECKPOINT: {len(segments)} đoạn, {completed} frame đã lưu trên Drive.', flush=True)
        if segments and segments[-1]['frames'] < segment_frames:
            finished = True
        else:
            finished = False
        local = Path(local_root or os.getenv('REALESRGAN_LOCAL_WORK', '/content/realesrgan-work')) / job_id
        local.mkdir(parents=True, exist_ok=True)
        read_source = stage_source_video(src, local) if stage_source and not finished else src
        if not finished:
            if upscaler_factory is None:
                from app.model import Upscaler
                upscaler_factory = lambda path, t, p: Upscaler(path, tile=t, precision=p)
            model = upscaler_factory(weights, tile, precision)
            print('Model đã nạp; tile thực tế:', getattr(model, 'tile', tile), flush=True)
            cap = cv2.VideoCapture(str(read_source))
            if not cap.isOpened():
                raise RuntimeError('Không thể mở video bằng OpenCV.')
            bar = tqdm(total=total, initial=min(completed, total) if total else 0,
                       desc='Khôi phục x4', unit='frame', dynamic_ncols=True)
            started = time.monotonic()
            processed_this_run = 0
            try:
                if completed:
                    print(f'Bỏ qua {completed} frame đã lưu trên Drive (chỉ giải mã CPU)...', flush=True)
                    for n in range(completed):
                        if not cap.grab():
                            raise RuntimeError('Video nguồn đã bị rút ngắn so với checkpoint.')
                while True:
                    chunk_start = completed
                    temp = local / f'seg_{chunk_start:010d}.partial.mp4'
                    temp.unlink(missing_ok=True)
                    def advance(n):
                        nonlocal processed_this_run
                        processed_this_run += n
                        bar.update(n)
                        if processed_this_run % 10 == 0:
                            rate = processed_this_run / max(0.01, time.monotonic() - started)
                            bar.set_postfix({'GPU FPS': f'{rate:.2f}', 'đã lưu Drive': completed})
                    count = _encode_frames(cap, model, temp, str(fps),
                                           (width * 4, height * 4),
                                           segment_frames, crf, preset, advance)
                    if count == 0:
                        temp.unlink(missing_ok=True)
                        finished = True
                        break
                    if _frames_in_segment(temp) != count:
                        raise RuntimeError('Segment encode bị thiếu frame, không thể checkpoint.')
                    name = f'seg_{chunk_start:010d}_{count:05d}.mp4'
                    destination = segments_dir / name
                    partial = segments_dir / (name + '.uploading')
                    _copy_retry(temp, partial)
                    if _frames_in_segment(partial) != count:
                        raise RuntimeError('Segment trên Drive bị lỗi, không đánh dấu đã xong.')
                    partial.replace(destination)
                    segments.append({'name': name, 'start': chunk_start, 'frames': count,
                                     'bytes': destination.stat().st_size, 'sample_sha256': quick_signature(destination)})
                    completed += count
                    manifest['segments'], manifest['completed_frames'] = segments, completed
                    manifest['updated_at'], manifest['state'] = _now(), 'processing'
                    _json_atomic(manifest_file, manifest)
                    temp.unlink(missing_ok=True)
                    print(f'✓ Checkpoint {len(segments)}: {completed} frame, '
                          f'đã lưu {name} trên Drive.', flush=True)
                    if count < segment_frames:
                        finished = True
                        break
            finally:
                bar.close()
                cap.release()
        if not finished or completed == 0:
            raise RuntimeError('Chưa có dữ liệu hoàn chỉnh để ghép video.')
        # A lost runtime during finalization simply rebuilds from durable video chunks.
        manifest['state'], manifest['updated_at'] = 'muxing', _now()
        _json_atomic(manifest_file, manifest)
        print(f'Đang ghép {len(segments)} segment (không mã hóa lại video), giữ âm thanh gốc...', flush=True)
        joined = local / 'final.mp4'
        joined.unlink(missing_ok=True)
        _final_mux([segments_dir / s['name'] for s in segments], read_source, joined)
        partial_out = output_path.with_name(output_path.name + '.uploading')
        manifest['state'], manifest['updated_at'] = 'copying_output', _now()
        _json_atomic(manifest_file, manifest)
        print(f'Đang sao chép MP4 {_format_size(joined.stat().st_size)} sang Google Drive...', flush=True)
        _copy_retry(joined, partial_out)
        partial_out.replace(output_path)
        # Optional sanity checks on the completed file.
        ffprobe_info(output_path)
        manifest['state'], manifest['updated_at'] = 'done', _now()
        manifest['output'], manifest['output_bytes'] = str(output_path), output_path.stat().st_size
        _json_atomic(manifest_file, manifest)
        print(f'✓ HOÀN TẤT — {output_path} ({_format_size(output_path.stat().st_size)})', flush=True)
        return output_path
