# Checklist chốt bản demo V1

## A. Đã có bằng chứng

- [x] Kiểm tra model lock SHA-256 — 12 file PASS (`evidence/model_lock_verify.txt`).
- [x] Backend unit tests — 25/25 PASS (`evidence/backend_tests.txt`).
- [x] React production build — PASS (`evidence/frontend_build.txt`).
- [x] M1 regression — 24 mẫu, 0 sai lệch/đổi nhãn/đổi chunk (`evidence/module1_regression.txt`).
- [x] M2 & M3 regression — tất cả nhánh PASS (`evidence/module23_regression.txt`).
- [x] HTTP smoke của ba module và PyVi — đã có log PASS trong trao đổi; **chưa có file log HTTP riêng trong gói**.
- [x] Đã có ảnh frontend nhập liệu và ba thẻ kết quả (`screenshots/`).

## B. Kiểm tra ngay trước buổi demo

- [ ] Đang chạy đúng project `C:\projectSamSung_v1` và `.venv` của V1.
- [ ] Chỉ một tiến trình FastAPI tại `127.0.0.1:8000`; `GET /api/v1/health` báo `models_ready=true`.
- [ ] React chạy từ `frontend`, mở được `http://127.0.0.1:5173`.
- [ ] Mẫu demo M1 ở PyVi thử nghiệm có cảnh báo; kết quả không bị trình bày như phân đoạn chuẩn.
- [ ] Thử chế độ bỏ qua M1: M2 và M3 vẫn có kết quả.
- [ ] Có phương án nếu không kết nối: kiểm tra cửa sổ backend, reload React, không khởi động thêm backend trên cùng cổng.
- [ ] Có sẵn bản Word và ảnh minh họa để trình bày khi demo mất kết nối.

## C. Không thay đổi trong bản đã chốt

- Checkpoint và ngưỡng: M1 = 0,57; M2 = 0,61; M3 = 0,45.
- M1: max length = 256, stride = 64, tối đa 3 chunk; tiêu chuẩn `article_segmented`.
- M3: description và MAX-MIL trọng số 50/50; MAX-MIL chunk 120 token, overlap 24, tối đa 6 chunk.
- Không thay đổi nhánh nghiên cứu cũ `C:\1_projectSamSung` hoặc các bản sao lưu ổ E.
- Không đưa nhánh nghiên cứu đạo văn cũ vào luồng demo V1.

## D. Những việc không được suy ra từ log hiện có

- Không gọi 25/25 test hay hồi quy PASS là độ chính xác 100%.
- Không so sánh trực tiếp số mẫu PyVi 60/60 với Underthesea 54/60 như hai tập đánh giá tương đương.
- Không bảo đảm mọi tiêu đề giật tít trong mẫu tự tạo sẽ được dự đoán clickbait.
- Không triển khai public nếu chưa bổ sung xác thực, giới hạn yêu cầu và đánh giá vận hành.

## E. Tài liệu chứng cứ bổ sung khi cần

Nếu hội đồng yêu cầu chứng cứ chi tiết ngoài gói hiện tại, sao chép các báo cáo JSON/CSV hồi quy từ `system\reports\module1_regression_v1` và `module2_headline_integrity\results\integration\regression_v1` sang bản tài liệu độc lập. Không chuyển hoặc chỉnh sửa checkpoint.
