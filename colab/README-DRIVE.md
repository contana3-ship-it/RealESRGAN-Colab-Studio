# Google Drive mode

1. Bật GPU T4 trong Google Colab.
2. Chạy notebook `colab/RealESRGAN_Studio.ipynb` (hoặc clone + install_native.sh + `from colab.drive_app import main; main()`).
3. Xác nhận quyền mount Google Drive. Chọn video qua menu hoặc nhập đường dẫn trong MyDrive, nhấn **Khôi phục video x4**.
4. Theo dõi `tqdm` trong ô log. MP4 được lưu tại `MyDrive/RealESRGAN-Studio/outputs`.

**Biến môi trường:** `DRIVE_VIDEO_DIR` (thư mục quét), `DRIVE_OUTPUT_DIR` (thư mục kết quả), `STUDIO_DATA` (cache/frame tạm tại Colab, mặc định `/content/realesrgan-native-data`). Khuyên dùng cache trên ổ Colab để tránh chậm vì số lượng frame lớn. Video nguồn được đọc trực tiếp từ Google Drive, không upload trình duyệt. Không cần Tunnel.

**Lưu ý:** Cache/frame và trọng số có thể mất khi runtime Colab đặt lại; MP4 đã hoàn thành nằm trên Drive. Bản xử lý video đang giả định khung hình có tốc độ cố định. Một video quá lớn có thể hết bộ nhớ hoặc dung lượng ổ đĩa, hãy chọn GPU tile 64 nếu cần.
