# RealESRGAN Colab Studio — Google Drive Mode

Đọc video từ Google Drive đã mount, xử lý bằng `RealESRGAN_x4plus`, lưu MP4 kết quả lên Google Drive. Không upload, không Cloudflare Tunnel.

## Chạy trên Colab

Bật GPU T4, dùng notebook `colab/RealESRGAN_Studio.ipynb` hoặc chạy:

```python
!git clone https://github.com/contana3-ship-it/RealESRGAN-Colab-Studio.git /content/realesrgan-colab-studio
!bash /content/realesrgan-colab-studio/colab/install_native.sh
import sys
sys.path.insert(0, '/content/realesrgan-colab-studio')
from colab.drive_app import main
main()
```

Nếu đã clone trước đó, dùng `git -C /content/realesrgan-colab-studio pull --ff-only` thay vì clone lại.

Sau khi cấp quyền Google Drive, chọn video từ MyDrive (hoặc nhập đường dẫn) và nhấn **Khôi phục video x4**. Tiến trình frame/FPS/ETA hiển thị trong notebook; kết quả: `/content/drive/MyDrive/RealESRGAN-Studio/outputs`.

Cấu hình: `DRIVE_VIDEO_DIR`, `DRIVE_OUTPUT_DIR`, `STUDIO_DATA` (mặc định lưu frame tạm trong `/content`). Giảm GPU tile xuống 64 nếu thiếu VRAM.

## Nguồn mô hình

RealESRGAN_x4plus từ [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN). Đọc `THIRD_PARTY.md` về ghi công.

**Lưu ý**: Đã kiểm tra tĩnh cú pháp, chưa chạy xử lý GPU thật trên Colab trong phiên này. Những file giao diện web cũ được giữ lại để tương thích, nhưng không dùng trong quy trình Google Drive.
