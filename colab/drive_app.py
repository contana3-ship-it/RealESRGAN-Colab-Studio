"""Colab notebook UI: Google Drive video -> durable x4 restoration -> Google Drive.

Use ordinary Colab mount authorization. No tunnel and no browser upload.
"""
from __future__ import annotations

import html
import json
import os
import traceback
from pathlib import Path

from colab.checkpoint_engine import restore_video

VIDEO_TYPES = {'.mp4', '.mkv', '.mov', '.avi', '.webm', '.m4v'}
DRIVE_ROOT = Path('/content/drive/MyDrive')
STUDIO_ROOT = DRIVE_ROOT / 'RealESRGAN-Studio'
OUTPUT_DEFAULT = STUDIO_ROOT / 'outputs'
CHECKPOINT_DEFAULT = STUDIO_ROOT / 'checkpoints'


def list_videos(folder: Path, max_files: int = 2000):
    if not folder.is_dir():
        return []
    result = []
    for base, dirs, files in os.walk(folder, followlinks=False):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != 'checkpoints']
        for filename in files:
            path = Path(base) / filename
            if path.suffix.lower() in VIDEO_TYPES and path.is_file():
                result.append(path)
                if len(result) >= max_files:
                    return sorted(result, key=lambda p: str(p).lower())
    return sorted(result, key=lambda p: str(p).lower())


def inside_drive(raw: str) -> Path:
    p = Path(raw.strip()).expanduser()
    if not p.is_absolute():
        # Accept either "Movies/video.mp4" or "MyDrive/Movies/video.mp4".
        if p.parts and p.parts[0].lower() == 'mydrive':
            p = Path(*p.parts[1:])
        p = DRIVE_ROOT / p
    p = p.resolve()
    p.relative_to(DRIVE_ROOT.resolve())
    return p


def safe_drive_file(raw: str) -> Path:
    p = inside_drive(raw)
    if not p.is_file() or p.suffix.lower() not in VIDEO_TYPES:
        raise ValueError('Hãy chọn một file video hợp lệ trên Google Drive.')
    return p


def checkpoint_summary(folder: Path):
    if not folder.is_dir():
        return []
    items = []
    for p in folder.glob('*/checkpoint.json'):
        try:
            x = json.loads(p.read_text(encoding='utf-8'))
            items.append(x)
        except (OSError, json.JSONDecodeError):
            pass
    items.sort(key=lambda x: x.get('updated_at', ''), reverse=True)
    return items


def restore_from_drive(src: Path, output_folder: Path, tile: int = 0,
                       *, segment_frames: int = 60, crf: int = 14,
                       precision: str = 'fp16', stage_source: bool = True,
                       preset: str = 'veryfast'):
    """Public API; source/output/checkpoints stay on mounted Drive."""
    src = safe_drive_file(str(src))
    output_folder = inside_drive(str(output_folder))
    checkpoint_root = inside_drive(os.getenv('DRIVE_CHECKPOINT_DIR', str(CHECKPOINT_DEFAULT)))
    output_folder.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    print('Nguồn:', src, flush=True)
    print('Checkpoint bền vững:', checkpoint_root, flush=True)
    print('MP4 đích:', output_folder, flush=True)
    return restore_video(src, output_folder, checkpoint_root,
                         tile=tile, segment_frames=segment_frames, crf=crf,
                         precision=precision, preset=preset, stage_source=stage_source)


def main():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Chưa có CUDA GPU. Trong Colab chọn Runtime → Change runtime type → T4 GPU.')
    from google.colab import drive
    from IPython.display import display, HTML, clear_output, FileLink
    import ipywidgets as widgets

    print('Mount Google Drive...', flush=True)
    drive.mount('/content/drive', force_remount=False)
    if not DRIVE_ROOT.is_dir():
        raise RuntimeError('Không tìm thấy Google Drive MyDrive.')

    watch_folder = inside_drive(os.getenv('DRIVE_VIDEO_DIR', str(DRIVE_ROOT)))
    output_folder = inside_drive(os.getenv('DRIVE_OUTPUT_DIR', str(OUTPUT_DEFAULT)))
    checkpoints = inside_drive(os.getenv('DRIVE_CHECKPOINT_DIR', str(CHECKPOINT_DEFAULT)))
    videos = list_videos(watch_folder)
    gpu_name = html.escape(torch.cuda.get_device_name(0))

    display(HTML(f'''<div style="padding:16px 20px;border-radius:12px;background:#14243a;color:#f4f7ff;font:15px system-ui">
      <b style="font-size:21px">RealESRGAN x4plus — Drive Resume Edition</b>
      <p>GPU: {gpu_name} · Không upload · Không Tunnel · Chọn video từ Drive</p>
      <small>Checkpoint MP4 theo từng đoạn: nếu Colab ngắt, mở lại notebook và chọn cùng video/cấu hình để tiếp tục.</small>
    </div>'''))

    layout = widgets.Layout(width='97%')
    sty = {'description_width': '105px'}
    folder_text = widgets.Text(value=str(watch_folder), description='Quét Drive:', layout=layout, style=sty)
    selector = widgets.Dropdown(options=[], description='Video:', layout=layout, style=sty)
    manual = widgets.Text(value='', description='Đường dẫn:', placeholder='Tùy chọn: MyDrive/Movies/example.mp4', layout=layout, style=sty)
    output_text = widgets.Text(value=str(output_folder), description='Xuất MP4:', layout=layout, style=sty)
    checkpoint_text = widgets.Text(value=str(checkpoints), description='Checkpoint:', layout=layout, style=sty, disabled=True)
    tile = widgets.Dropdown(options=[('Tự động theo GPU (khuyên dùng)', 0), ('64', 64), ('128', 128),
                                     ('192', 192), ('256', 256), ('384', 384)], value=0,
                            description='Tile GPU:', style=sty)
    quality = widgets.Dropdown(options=[('Chất lượng rất cao • CRF 12', 12),
                                        ('Khuyên dùng • CRF 14', 14),
                                        ('Cân bằng dung lượng • CRF 16', 16),
                                        ('Dung lượng thấp hơn • CRF 18', 18)], value=14,
                               description='MP4:', style=sty)
    precision = widgets.Dropdown(options=[('FP16 • nhanh (khuyên dùng)', 'fp16'),
                                          ('FP32 • chậm hơn', 'fp32')], value='fp16',
                                 description='Độ chính xác:', style=sty)
    chunk = widgets.Dropdown(options=[('30 frame • lưu rất thường xuyên', 30),
                                      ('60 frame • khuyên dùng', 60),
                                      ('120 frame • ít ghi Drive hơn', 120),
                                      ('240 frame • hiệu quả cho runtime ổn định', 240)], value=60,
                             description='Checkpoint:', style=sty)
    stage = widgets.Checkbox(value=True, description='Sao chép video về ổ đĩa Colab trước khi xử lý (tăng tốc đọc)')
    scan_btn = widgets.Button(description='Quét lại video', icon='refresh', button_style='info')
    start_btn = widgets.Button(description='Bắt đầu / Tiếp tục từ checkpoint', icon='play', button_style='success',
                               layout=widgets.Layout(width='310px'))
    jobs_btn = widgets.Button(description='Xem checkpoint đã lưu', icon='history')
    status = widgets.HTML(value='')
    logs = widgets.Output(layout=widgets.Layout(border='1px solid #ddd', padding='10px', max_height='550px', overflow='auto'))
    running = {'value': False}

    def scan(_=None):
        try:
            base = inside_drive(folder_text.value)
            if not base.is_dir():
                raise ValueError('Thư mục quét không tồn tại.')
            found = list_videos(base)
            selector.options = [(str(p.relative_to(DRIVE_ROOT)) + f' ({p.stat().st_size/1048576:.1f} MB)', str(p))
                                for p in found] or [('Không thấy video, hãy nhập đường dẫn', '')]
            status.value = f'<small>Đã tìm thấy {len(found)} video trên Drive.</small>'
        except Exception as e:
            status.value = f'<b style="color:red">Lỗi quét: {html.escape(str(e))}</b>'

    def list_jobs(_):
        with logs:
            clear_output(wait=True)
            rows = checkpoint_summary(checkpoints)
            if not rows:
                print('Chưa có checkpoint nào.')
            for x in rows[:30]:
                print(f"{x.get('updated_at','')} | {x.get('state','')} | "
                      f"{x.get('completed_frames',0)} frame | "
                      f"{x.get('source',{}).get('path','')} | job={x.get('job_id','')}")

    def start(_):
        if running['value']:
            return
        running['value'] = True
        for btn in (start_btn, scan_btn, jobs_btn):
            btn.disabled = True
        with logs:
            clear_output(wait=True)
            try:
                selected = manual.value.strip() or selector.value
                if not selected:
                    raise ValueError('Chưa chọn video nguồn.')
                src = safe_drive_file(selected)
                destination = inside_drive(output_text.value)
                status.value = '<b>Đang chạy. Sau khi mất session, khởi động lại notebook và bấm Tiếp tục.</b>'
                result = restore_from_drive(
                    src, destination, tile=int(tile.value), segment_frames=int(chunk.value),
                    crf=int(quality.value), precision=precision.value, stage_source=bool(stage.value))
                status.value = '<b style="color:green">Hoàn thành! File MP4 đã lưu trên Google Drive.</b>'
                print('Video đã lưu:', result)
                display(FileLink(str(result), result_html_prefix='Mở/tải kết quả: '))
            except Exception as e:
                status.value = '<b style="color:red">Lỗi xử lý: ' + html.escape(str(e)) + '</b>'
                traceback.print_exc()
                print('Các segment đã checkpoint vẫn còn trên Drive; có thể tiếp tục lần sau.')
            finally:
                running['value'] = False
                for btn in (start_btn, scan_btn, jobs_btn):
                    btn.disabled = False

    scan_btn.on_click(scan)
    jobs_btn.on_click(list_jobs)
    start_btn.on_click(start)
    scan()
    display(widgets.VBox([
        folder_text, widgets.HBox([scan_btn, jobs_btn]), status,
        selector, manual, output_text, checkpoint_text,
        widgets.HBox([tile, precision]), widgets.HBox([quality, chunk]),
        stage, start_btn, logs,
    ]))


if __name__ == '__main__':
    main()
