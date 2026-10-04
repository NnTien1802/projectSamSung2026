# ProjectSamSung V1 — Bộ tài liệu chốt demo (Bước 22)

**Ngày tổng hợp:** 24/09/2026. **Phạm vi:** ứng dụng demo Windows chạy localhost, backend FastAPI V1.1, frontend React V1 đã cập nhật. Đây là bộ tài liệu/bằng chứng; **không chứa** checkpoint, dataset, `.venv`, thư viện Node hay mã nguồn cần cài đè.

## Đặt tài liệu trong project

Giải nén ZIP vào `C:\projectSamSung_v1\docs` để có cấu trúc:

```text
C:\projectSamSung_v1\docs\release_v1\
  Tai_lieu_ky_thuat_ProjectSamSung_V1.docx
  README_Buoc22.md
  Kich_ban_demo.md
  Checklist_release_v1.md
  manifest_docs_sha256.json
  evidence\
  screenshots\
```

Không di chuyển các checkpoint, không cài lại phụ thuộc và không thay đổi mô hình. File `manifest_docs_sha256.json` chỉ kiểm tra **các file trong gói tài liệu**; nó không thay thế `system\release\model_lock_manifest_v1.json`.

## Đã xác nhận từ log

| Hạng mục | Trạng thái |
|---|---|
| Model artifact SHA-256 | 12 file giữ nguyên — PASS |
| Backend | 25/25 unit tests — PASS |
| HTTP smoke | Ba module và PyVi thử nghiệm — PASS (log CMD trong trao đổi) |
| Frontend | `tsc -b && vite build` — PASS |
| Hồi quy M1 | 24 mẫu; MAE = 0; 0 đổi nhãn và số chunk — PASS |
| Hồi quy M2 | 32 mẫu; max sai lệch 0,002441406; 0 đổi nhãn — PASS |
| Hồi quy M3 | 24 mẫu; description, MAX-MIL, fusion đều PASS |

**Phạm vi diễn giải:** các bài kiểm thử trên chứng minh sự ổn định và tính nhất quán của bản tích hợp với dữ liệu validation tham chiếu; không phải phép đánh giá accuracy/F1 mới trên tập test.

## Chạy demo bằng môi trường V1 hiện tại

**CMD 1 — FastAPI** (chỉ một tiến trình, một worker):

```cmd
cd /d C:\projectSamSung_v1
set SAMSUNG_DEVICE=cuda
.\.venv\Scripts\python.exe -m uvicorn system.backend.app.api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

**CMD 2 — React:**

```cmd
cd /d C:\projectSamSung_v1\frontend
npm run dev
```

Mở `http://127.0.0.1:5173` để demo; Swagger API tại `http://127.0.0.1:8000/docs`. Nếu báo cổng `8000` đang dùng, kiểm tra tiến trình FastAPI hiện có, không mở thêm một backend. Lệnh `npm run build` chỉ chạy từ thư mục `frontend`.

## Tài liệu chính

- `Tai_lieu_ky_thuat_ProjectSamSung_V1.docx`: kiến trúc, đặc tả ba module, API, xử lý PyVi, bằng chứng nghiệm thu, vận hành và giới hạn.
- `Kich_ban_demo.md`: kịch bản trình bày 5–7 phút và trả lời các câu hỏi về phạm vi mô hình.
- `Checklist_release_v1.md`: checklist đóng băng bản demo và hướng xử lý lỗi thường gặp.
- `evidence/`: bản sao năm log người dùng cung cấp; log HTTP trong trao đổi chưa có file đính kèm riêng.
- `screenshots/`: các ảnh giao diện người dùng cung cấp.

**Giới hạn quan trọng:** M1 chỉ phân loại lỗi ở cấp độ bài báo; PyVi là phân đoạn thử nghiệm. Trên 60 mẫu validation đã đối chiếu, đầu vào qua PyVi làm thay đổi 8 nhãn M1 so với phân đoạn gốc. Ba module đưa ra ba kết quả độc lập, không có điểm chất lượng tin tổng hợp và không xác minh sự thật.
