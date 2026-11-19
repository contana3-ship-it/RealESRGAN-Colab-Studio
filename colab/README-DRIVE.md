# Chạy RealESRGAN x4 trên Google Drive

Mở `RealESRGAN_Studio.ipynb` trong Google Colab và bật GPU. Notebook tự `git clone/pull`, cài dependency và mở form chọn video trên Drive.

- **Chọn video:** từ `MyDrive` hoặc điền trực tiếp đường dẫn tuyệt đối.
- **Chọn chất lượng:** CRF 14 mặc định (rất cao); CRF 12 nếu ưu tiên cực nét, file lớn.
- **Tile:** Auto theo GPU và giảm tự động khi gặp lỗi CUDA OOM.
- **Checkpoint:** mỗi 60 frame mặc định, lưu MP4 từng đoạn tại `MyDrive/RealESRGAN-Studio/checkpoints/`.
- **Tiếp tục:** khi Colab mất runtime, chạy notebook lần nữa với cùng nguồn/cấu hình rồi bấm `Bắt đầu / Tiếp tục từ checkpoint`.
- **Kết quả:** `MyDrive/RealESRGAN-Studio/outputs/`.

Sau khi model hoàn tất một đoạn, màn hình in `✓ Checkpoint N: ... frame` cùng trạng thái Drive. Không xóa thư mục checkpoint khi tác vụ đang chạy.

Không có cách đảm bảo phiên Colab miễn phí chạy mãi; ứng dụng không giả lập hoạt động để lách idle timeout.
