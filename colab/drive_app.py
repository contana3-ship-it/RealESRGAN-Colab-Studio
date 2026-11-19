"""Google Drive source selector for RealESRGAN x4 on Google Colab.

No browser uploads, local web servers or Cloudflare tunnels.
"""
from __future__ import annotations

import os
import shutil
import traceback
from pathlib import Path

VIDEO_TYPES = {'.mp4', '.mkv', '.mov', '.avi', '.webm', '.m4v'}
DRIVE_ROOT = Path('/content/drive/MyDrive')
OUTPUT_DEFAULT = DRIVE_ROOT / 'RealESRGAN-Studio' / 'outputs'


def list_videos(folder: Path, max_files: int = 2000):
    """Enumerate video files under a chosen Drive folder without following symlinks."""
    if not folder.is_dir():
        return []
    result = []
    for root, dirs, files in os.walk(folder, followlinks=False):
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        for filename in files:
            path = Path(root) / filename
            if path.suffix.lower() in VIDEO_TYPES and path.is_file():
                result.append(path)
                if len(result) >= max_files:
                    return sorted(result, key=lambda p: str(p).lower())
    return sorted(result, key=lambda p: str(p).lower())


def safe_drive_file(raw: str) -> Path:
    """Accept only existing videos within the mounted Drive MyDrive folder."""
    p = Path(raw.strip()).expanduser()
    if not p.is_absolute():
        p = DRIVE_ROOT / p
    p = p.resolve(strict=True)
    p.relative_to(DRIVE_ROOT.resolve(strict=True))
    if not p.is_file() or p.suffix.lower() not in VIDEO_TYPES:
        raise ValueError('Hãy chọn một file video hợp lệ trên Google Drive.')
    return p


def restore_from_drive(src: Path, output_folder: Path, tile: int = 128):
    from colab.native_app import run_restore
    if not output_folder.is_dir():
        output_folder.mkdir(parents=True, exist_ok=True)
    output_folder = output_folder.resolve()
    output_folder.relative_to(DRIVE_ROOT.resolve())
    print('Video đầu vào:', src, flush=True)
    print('Video đầu ra:', output_folder, flush=True)
    print('Đang khôi phục trực tiếp từ video Google Drive. Frame tạm trên ổ đĩa Colab.', flush=True)
    restored_local = run_restore(src, tile=tile)
    # Copy to a .partial file before atomic promotion so incomplete output is never
    # presented as a completed MP4 after an interrupted Drive transfer.
    final = output_folder / restored_local.name
    temporary = output_folder / (restored_local.name + '.partial')
    print('Đang lưu MP4 hoàn chỉnh vào Google Drive...', flush=True)
    try:
        with restored_local.open('rb') as source, temporary.open('wb') as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
        if temporary.stat().st_size != restored_local.stat().st_size:
            raise IOError('Dung lượng ghi lên Drive không khớp với file kết quả.')
        temporary.replace(final)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print('HOÀN THÀNH:', final, flush=True)
    return final


def main():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Chưa bật GPU. Chọn Runtime → Change runtime type → T4 GPU.')
    from google.colab import drive
    from IPython.display import display, HTML, clear_output
    import ipywidgets as widgets

    print('Đang kết nối Google Drive...', flush=True)
    drive.mount('/content/drive', force_remount=False)
    if not DRIVE_ROOT.is_dir():
        raise RuntimeError('Không tìm thấy MyDrive. Hãy kiểm tra quyền truy cập Google Drive.')

    watch_folder = Path(os.getenv('DRIVE_VIDEO_DIR', str(DRIVE_ROOT)))
    output_folder = Path(os.getenv('DRIVE_OUTPUT_DIR', str(OUTPUT_DEFAULT)))
    videos = list_videos(watch_folder)

    display(HTML('''<div style="padding:14px 18px;border-radius:12px;background:#16253b;color:white;font:15px system-ui">
      <b style="font-size:20px">RealESRGAN x4 — Google Drive</b>
      <p>Không upload. Chọn video đã lưu trong Drive, ứng dụng sẽ tự bắt đầu khôi phục khi nhấn <b>Khôi phục</b>.</p>
      <small>Video đầu vào từ MyDrive · Frame tạm tại Colab · MP4 kết quả lưu về MyDrive.</small></div>'''))
    folder_text = widgets.Text(value=str(watch_folder), description='Thư mục:', layout=widgets.Layout(width='95%'), style={'description_width':'85px'})
    options = [(str(p.relative_to(DRIVE_ROOT)) + f' ({p.stat().st_size/1048576:.1f} MB)', str(p)) for p in videos]
    selector = widgets.Dropdown(options=options or [('Không tìm thấy video, hãy nhập đường dẫn bên dưới', '')], description='Video:', layout=widgets.Layout(width='95%'), style={'description_width':'85px'})
    manual = widgets.Text(value='', placeholder='/content/drive/MyDrive/Video/input.mp4 (tùy chọn)', description='Đường dẫn:', layout=widgets.Layout(width='95%'), style={'description_width':'85px'})
    output_text = widgets.Text(value=str(output_folder), description='Xuất MP4:', layout=widgets.Layout(width='95%'), style={'description_width':'85px'})
    tile = widgets.Dropdown(options=[64,128,256], value=128, description='GPU Tile:', style={'description_width':'85px'})
    reload_btn = widgets.Button(description='Quét lại Drive', icon='refresh', button_style='info')
    start_btn = widgets.Button(description='Khôi phục video x4', icon='play', button_style='success')
    status = widgets.HTML(value=f'<small>Đã tìm thấy {len(videos)} video (tối đa 2.000 file). Chọn video và nhấn Khôi phục.</small>')
    logs = widgets.Output(layout=widgets.Layout(border='1px solid #ddd', padding='8px'))
    running = {'value': False}

    def refresh(_):
        try:
            base = Path(folder_text.value).expanduser().resolve(strict=True)
            base.relative_to(DRIVE_ROOT.resolve())
            if not base.is_dir():
                raise ValueError('Đường dẫn quét phải là thư mục.')
            choices = list_videos(base)
            selector.options = [(str(p.relative_to(DRIVE_ROOT)) + f' ({p.stat().st_size/1048576:.1f} MB)', str(p)) for p in choices] or [('Không thấy video', '')]
            status.value = f'<small>Đã tìm thấy {len(choices)} video.</small>'
        except Exception as exc:
            status.value = f'<b style="color:red">Không thể quét thư mục: {str(exc)}</b>'

    def start(_):
        if running['value']:
            return
        running['value'] = True
        start_btn.disabled = True
        reload_btn.disabled = True
        with logs:
            clear_output(wait=True)
            try:
                candidate = manual.value.strip() or selector.value
                if not candidate:
                    raise ValueError('Chưa chọn video. Hãy quét lại Drive hoặc nhập đường dẫn video.')
                src = safe_drive_file(candidate)
                target_folder = Path(output_text.value.strip()).resolve()
                target_folder.relative_to(DRIVE_ROOT.resolve())
                status.value = '<b>Đang khôi phục video bằng GPU...</b>'
                final = restore_from_drive(src, target_folder, tile=int(tile.value))
                status.value = f'<b style="color:green">Hoàn tất. MP4 nằm trong Drive: {final}</b>'
                print('\nFile kết quả:', final)
            except Exception as exc:
                status.value = f'<b style="color:red">Lỗi: {str(exc)}</b>'
                traceback.print_exc()
            finally:
                running['value'] = False
                start_btn.disabled = False
                reload_btn.disabled = False

    reload_btn.on_click(refresh)
    start_btn.on_click(start)
    display(widgets.VBox([folder_text, widgets.HBox([reload_btn, status]), selector, manual, output_text, tile, start_btn, logs]))
    if len(videos) == 1:
        status.value = '<small>Tìm thấy 1 video. Nhấn Khôi phục để xử lý.</small>'


if __name__ == '__main__':
    main()
