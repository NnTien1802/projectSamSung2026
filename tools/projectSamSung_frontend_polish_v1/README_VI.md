# Bước 19 — Hoàn thiện giao diện React, projectSamSung V1

Bản cập nhật chỉ thay đổi **giao diện React đã gửi ở bước 17**, không cài lại thư viện, không sửa FastAPI, `.venv`, threshold hay checkpoint.

## Chức năng mới

- Việt hóa cảnh báo `document_level_only` và cảnh báo PyVi/M3 phổ biến. Nếu xuất hiện cảnh báo kỹ thuật khác chưa biết, giữ nguyên nội dung gốc và đặt nhãn tiếng Việt; có thể xem JSON để đối chiếu.
- Hiển thị thông báo khi chưa kết nối được API hoặc khi mô hình đang khởi động, có nút **Kiểm tra lại**.
- Hiển thị gợi ý nếu thiếu tiêu đề, nội dung hoặc bản phân đoạn bắt buộc; thông báo tiếng Việt khi API trả lỗi 422, 429, 500, 502, 503 hoặc 504.
- Bốn tình huống minh họa **tự tạo**: bài trung tính; M1 có lỗi gõ; M2 tiêu đề giật tít; M3 tiêu đề khác chủ đề. Đây là **kịch bản trình diễn, không phải tập benchmark, không bảo đảm mô hình phải dự đoán nhãn tương ứng**. M1 mặc định tắt trừ mẫu lỗi gõ chủ động chọn PyVi thử nghiệm.

## 1. Đặt gói đúng chỗ

Giải nén ZIP vào `C:\projectSamSung_v1\tools`, đảm bảo có:

`C:\projectSamSung_v1\tools\projectSamSung_frontend_polish_v1\install_patch.py`

Không sao chép thủ công overlay vào các thư mục khác. Script kiểm tra SHA-256 của 3 file React V1 gốc **trước** khi ghi đè và sẽ dừng nếu người dùng đã sửa file. Trường hợp mã nguồn đang tùy chỉnh, hãy gửi lại các file đã sửa để làm patch tương thích.

## 2. Cài patch bằng môi trường V1 hiện tại

```cmd
cd /d C:\projectSamSung_v1
.\.venv\Scripts\python.exe tools\projectSamSung_frontend_polish_v1\install_patch.py --project C:\projectSamSung_v1 --dry-run
.\.venv\Scripts\python.exe tools\projectSamSung_frontend_polish_v1\install_patch.py --project C:\projectSamSung_v1
```

Bản sao lưu 3 file React gốc nằm tại `system\reports\frontend_patch_backups\pre_polish_v1_<timestamp>`.

## 3. Build và chạy

Giữ backend đang hoạt động ở `127.0.0.1:8000` (chỉ **một** tiến trình Uvicorn). Mở CMD thứ hai:

```cmd
cd /d C:\projectSamSung_v1\frontend
npm run build
npm run dev
```

Mở `http://127.0.0.1:5173`. Nếu Vite đang chạy từ trước, chỉ cần tải lại trang; nếu giao diện chưa cập nhật, dừng Vite bằng Ctrl+C rồi chạy lại `npm run dev`.

## 4. Kịch bản QA nhanh

1. Kiểm tra nhãn trạng thái API, thử tắt backend rồi bấm **Kiểm tra lại**; bật lại backend và bấm nút một lần nữa.
2. Chọn từng tình huống minh họa, nhấn **Điền bài mẫu** rồi **Phân tích bài báo**; ghi lại **nhãn mà mô hình thực sự dự đoán**, không gán nhãn kỳ vọng tùy tiện.
3. Mẫu lỗi gõ tự chọn M1 `PyVi thử nghiệm`. Kiểm tra cảnh báo M1 và thông điệp tiếng Việt về dự đoán cấp độ bài báo.
4. Thử nhập thiếu tiêu đề hoặc nội dung; nút phân tích cần bị vô hiệu hóa và giao diện phải nêu trường thiếu.
5. Chọn M1 **Phân đoạn có sẵn** mà không điền bản phân đoạn; kiểm tra gợi ý.
6. Thử các kết quả khác nhau, mở **Xem dữ liệu kết quả** để đối chiếu JSON trả về, đặc biệt `probability_consistent` của M3.

**Giới hạn:** Các kịch bản minh họa ngắn có thể nằm ngoài phân phối dữ liệu huấn luyện; kết quả của chúng không phải số liệu độ chính xác. Các mẫu không thay đổi hành vi hoặc cách tiền xử lý của model.
