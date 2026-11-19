# RealESRGAN Colab Studio — Drive Checkpoint & Speed Edition

Khôi phục video người thật bằng trọng số chính thức **RealESRGAN_x4plus** (RRDBNet x4).
Dùng **Google Drive đã mount** để đọc video, lưu checkpoint dạng đoạn MP4 và xuất file cuối.
**Không dùng Cloudflare Tunnel, không upload qua trình duyệt.**

## Chạy với một ô Colab

1. Mở [notebook trên Google Colab](https://colab.research.google.com/github/contana3-ship-it/RealESRGAN-Colab-Studio/blob/main/colab/RealESRGAN_Studio.ipynb).
2. Chọn **Runtime → Change runtime type → T4 GPU** (hoặc GPU khác), chạy ô code, cấp quyền mount Google Drive.
3. Chọn video từ `MyDrive` và bấm **Bắt đầu / Tiếp tục từ checkpoint**.
4. Nhận MP4 tại `MyDrive/RealESRGAN-Studio/outputs`.

Để dán thẳng vào một ô notebook mới:

```python
import os, sys, subprocess, shutil, importlib
repo = 'https://github.com/contana3-ship-it/RealESRGAN-Colab-Studio.git'
folder = '/content/realesrgan-colab-studio'
if os.path.isdir(folder + '/.git'):
    subprocess.run(['git', '-C', folder, 'pull', '--ff-only'], check=True)
else:
    if os.path.exists(folder): shutil.rmtree(folder)
    subprocess.run(['git', 'clone', repo, folder], check=True)
subprocess.run(['bash', folder + '/colab/install_native.sh'], check=True)
if folder not in sys.path: sys.path.insert(0, folder)
importlib.invalidate_caches()
from colab.drive_app import main
main()
```

**Lưu ý:** Nếu notebook đã import bản cũ trong cùng Python runtime, chọn `Runtime → Restart session` trước khi chạy lại để tải các mô-đun đã cập nhật.

## Phục hồi sau khi mất Colab runtime

- Đầu vào: video trên Drive, không mất sau khi ngắt.
- Checkpoint: `MyDrive/RealESRGAN-Studio/checkpoints/<job_id>/` (manifest `checkpoint.json` và `segments/*.mp4`).
- Đầu ra: `MyDrive/RealESRGAN-Studio/outputs/<tên_video>_RealESRGAN_x4_<jobid>.mp4`.
- Mỗi **60 khung hình** (có thể đổi 30/120/240), chương trình encode thành MP4 rồi xác nhận bản sao trên Drive; chỉ sau đó mới cập nhật checkpoint.
- Nếu Colab bị ngắt giữa chừng, đoạn MP4 đang xử lý có thể mất, nhưng **các đoạn đã hoàn tất trên Drive được giữ lại**.
- Khởi động lại notebook, chọn **cùng video, cùng tile, cùng FP16/FP32, cùng CRF và checkpoint size**, nhấn **Bắt đầu / Tiếp tục từ checkpoint**. Không xử lý lại các đoạn đã lưu. Nếu lần trước đã xử lý xong toàn bộ, chỉ cần ghép MP4.
- Không di chuyển, sửa video nguồn hoặc thay đổi cấu hình trong lúc đang chạy, vì fingerprint/Job ID sẽ thay đổi và tạo job mới.
- Nếu cuối cùng có file `.uploading`, app có thể tiếp tục sao chép **nếu** file kết quả cục bộ được tạo lại có nội dung khớp với phần đã tải.

## Tối ưu chất lượng và tốc độ

| Thành phần | Cấu hình mặc định | Lý do |
|---|---|---|
| Model | RealESRGAN_x4plus | Model x4 dành cho hình ảnh thực, không dùng model anime |
| Tính toán | FP16 + `torch.inference_mode()` | Tăng tốc GPU, tương thích mặc định dự án gốc |
| Tile | Auto (thường 256 trên GPU T4) | Giảm số lượt gọi mạng; tự giảm tile khi thiếu VRAM |
| Tile padding | 16 px | Giảm vết nối tile |
| Mã hóa | H.264 x264 CRF 14, preset veryfast | Độ nét cao và giảm CPU-encode bottleneck |
| Đệm tạm | Chỉ đoạn video chưa commit | Không lưu hàng nghìn frame JPEG lên Drive |
| Đọc video | Sao chép 1 lần vào SSD Colab nếu đủ dung lượng | Giảm đọc ngẫu nhiên từ Drive |
| MP4 kết quả | FFmpeg concat `-c:v copy` | Không nén video thêm lần thứ hai |
| Âm thanh | `-c:a copy` nếu codec MP4 hỗ trợ | Giữ nguyên audio, fallback AAC 192k |

**FP16** có sai khác số học nhỏ so với FP32. Muốn ưu tiên độ chính xác số học, chọn FP32 (chậm hơn). Chọn CRF 12 cho MP4 ít mất chi tiết hơn nhưng kích thước tệp lớn hơn. Các tốc độ thực tế phụ thuộc GPU, độ phân giải và codec, chưa được benchmark trên Colab thật trong phiên này.

**Video VFR (frame rate thay đổi):** bản hiện tại dùng FPS trung bình cho MP4 kết quả; nguồn VFR có thể có sai lệch đồng bộ audio/video. Nên chuẩn hóa video đầu vào về CFR trước khi xử lý nếu yêu cầu đồng bộ thời gian chính xác.

## Giảm lỗi/mất session

- Checkpoint hoàn chỉnh **trên Drive**, không phụ thuộc `/content`.
- Retry I/O khi mạng/Drive lỗi tạm thời; file `.uploading` không được nhận là hoàn tất.
- Tự giảm tile GPU khi CUDA out-of-memory.
- Resume an toàn bằng nhận diện video/thiết lập/mã SHA model; đoạn hỏng bị bỏ qua.
- Không dùng script giả lập thao tác để giữ Colab chạy. Chính sách Colab có thể kết thúc runtime kể cả khi notebook đang hoạt động: https://research.google.com/colaboratory/faq.html
- Nếu kết nối trình duyệt đứt nhưng runtime chưa chết, kiểm tra tiến trình đang chạy **trước khi chạy lại notebook**, tránh tạo hai job cùng lúc.

## Kiểm thử

```bash
python -m unittest discover -s tests -v
```

Bài test tích hợp (CPU, mô hình giả) tạo video ngắn bằng FFmpeg, mô phỏng mất session ở giữa đoạn, sau đó chạy lại; kiểm tra số frame, kết quả MP4 và âm thanh. Chưa kiểm thử thực tế CUDA T4/A100.

## Ghi công

- [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- Xem `THIRD_PARTY.md` về bản quyền mã kiến trúc mạng và bộ trọng số.
