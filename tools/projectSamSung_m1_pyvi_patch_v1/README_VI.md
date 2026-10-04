# Patch V1.1 — M1 PyVi thử nghiệm cho văn bản thô

**Chỉ cài trong `C:\projectSamSung_v1`.** Không sửa checkpoint, tokenizer, manifest, adapter, threshold hoặc project nghiên cứu gốc. Đây là thay đổi ở lớp API và không thay thế API V1 gốc nếu chưa có bản sao lưu.

## Cài đặt

Giải nén ZIP đến vị trí bất kỳ. Mở CMD trong thư mục chứa `install_patch.py` và chạy:

```cmd
cd /d <thu_muc_da_giai_nen>\projectSamSung_m1_pyvi_patch_v1
C:\projectSamSung_v1\.venv\Scripts\python.exe install_patch.py --project C:\projectSamSung_v1
```

Installer so SHA-256 của ba file API cũ với bản V1 gốc trước khi chỉnh sửa; nếu mã trên máy khác, installer **từ chối ghi đè** và báo tên file. Mỗi lần cài thành công tạo backup dưới `system\reports\api_patch_backups\...`. Các file mới chỉ được tạo nếu chưa tồn tại để không làm mất code bạn đang phát triển.

Nếu bạn đã cài patch này rồi, không chạy lại installer. Không cần cài lại thư viện: PyVi đã có trong venv của project V1 theo bước thử nghiệm trước.

## Kiểm thử (không nạp GPU)

```cmd
cd /d C:\projectSamSung_v1
.\.venv\Scripts\python.exe -m unittest discover -s system\backend\tests -v
```

Giữ nguyên 14 unit test cũ; thêm 11 bài kiểm thử cho endpoint experimental. Chỉ khởi động API nếu unit test đạt.

## Khởi động

```cmd
cd /d C:\projectSamSung_v1
set SAMSUNG_DEVICE=cuda
.\.venv\Scripts\python.exe -m uvicorn system.backend.app.api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Mở http://127.0.0.1:8000/docs. Không dùng `--reload` hoặc nhiều GPU worker. Chỉ phục vụ localhost.

## Thử HTTP thật

CMD khác:

```cmd
cd /d C:\projectSamSung_v1
.\.venv\Scripts\python.exe system\scripts\09_smoke_test_http_api.py
.\.venv\Scripts\python.exe system\scripts\16_smoke_m1_pyvi_http.py
```

`09` bảo đảm chức năng cũ còn chạy, `16` chạy đường dẫn PyVi thử nghiệm trên một câu tiếng Việt ngắn cố định với checkpoint thật. Kiểm thử HTTP **không đánh giá độ chính xác**.

## Hai đường dẫn đầu vào

**Đường dẫn tiêu chuẩn (giữ nguyên):**

`POST /api/v1/spelling`:

```json
{"article_segmented":"người_dân đang đọc báo"}
```

`POST /api/v1/analyze` với `article_segmented` đã phân đoạn: M1 hoạt động chuẩn cũ, M2/M3 dùng raw title/article.

**Thử nghiệm (phải chọn rõ ràng):**

`POST /api/v1/spelling/raw`:

```json
{"article_text":"Người dân đang đọc báo."}
```

`POST /api/v1/analyze`:

```json
{
  "title":"Tiêu đề",
  "description":"Mô tả ngắn",
  "article_text":"Người dân đang đọc báo.",
  "m1_input_mode":"pyvi_experimental"
}
```

Không được gửi đồng thời `article_segmented` và `m1_input_mode=pyvi_experimental`. Nếu chỉ gửi raw article và không chọn mode, M1 vẫn **skipped**, M2/M3 vẫn chạy như cũ. API trả thêm `input_mode` và `warning` trong kết quả M1 thử nghiệm. Bật mode PyVi nghĩa là đồng ý với tiền xử lý chưa tương đương dữ liệu huấn luyện; nó từng làm 8/60 mẫu validation đổi nhãn, và 2 mẫu thay đổi số chunk.

PyVi phải giữ nguyên các ký tự nhìn thấy của bài báo (theo NFC, bỏ qua khoảng trắng và `_` phân đoạn). Nếu làm mất/thay nội dung hoặc đặt dấu `_` tại vị trí không có ranh giới khoảng trắng trong văn bản gốc, API từ chối mẫu với HTTP 422 trước khi suy luận. Nếu tài liệu chứa `_` gốc, sử dụng đường dẫn presegmented thay vì experimental.

## Rollback

Dừng server. Từ thư mục backup được in ra sau khi cài, sao chép ba file `main.py`, `schemas.py`, `service.py` vào `system\backend\app\api\` trong V1. Các file experimental/test mới có thể để nguyên nhưng không được import từ bản API gốc; hoặc xóa riêng chúng. Sau rollback chạy 14 unit test cũ và `09_smoke_test_http_api.py`.
