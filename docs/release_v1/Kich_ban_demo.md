# Kịch bản demo ProjectSamSung V1 (5–7 phút)

Các tình huống minh họa trong frontend là ví dụ tự tạo, **không có nhãn kiểm chứng**. Không cần buộc mô hình trả về kết quả như tên tình huống; đọc và giải thích đầu ra thật.

| Thời lượng dự kiến | Thao tác trình diễn | Thông điệp kỹ thuật |
|---|---|---|
| 0:00–0:45 | Mở `http://127.0.0.1:5173`, giới thiệu ba trường nhập và trạng thái kết nối | Giao diện React gửi bài báo đến FastAPI trên máy local. |
| 0:45–1:45 | Chọn “Bài báo trung tính”, nhấn **Điền bài mẫu**, chọn M1 bỏ qua, phân tích | M2 và M3 vẫn chạy khi M1 không có phân đoạn chuẩn. |
| 1:45–3:00 | Chọn “M1 · Văn bản chứa lỗi cố ý” và PyVi thử nghiệm, phân tích | M1 phân loại xác suất có lỗi cấp bài; PyVi có thể gây lệch so với phân đoạn nghiên cứu. |
| 3:00–4:00 | Chọn ví dụ M2 giật tít, phân tích | M2 chỉ nhận tiêu đề, không dùng toàn văn. |
| 4:00–5:00 | Chọn ví dụ M3 tiêu đề khác chủ đề, phân tích | M3 xét title–description và title–article MAX-MIL; kết hợp 50/50. |
| 5:00–6:00 | Mở “Xem dữ liệu kết quả” của từng thẻ | Phân biệt label, xác suất mô hình, threshold và warning. |
| 6:00–7:00 | Trình bày log nghiệm thu, nêu giới hạn | 12 SHA-256 PASS; 25 test PASS; frontend build PASS; hồi quy ba module PASS. |

## Chuẩn bị trước khi trình diễn

1. Bật FastAPI với `SAMSUNG_DEVICE=cuda` và `--workers 1`. Đợi startup hoàn tất.
2. Bật Vite trong `C:\projectSamSung_v1\frontend`. Mở trang React và kiểm tra kết nối.
3. Giữ sẵn `docs\release_v1\Tai_lieu_ky_thuat_ProjectSamSung_V1.docx` và thư mục `evidence` để đối chiếu khi được hỏi.
4. Trình diễn đủ ba chế độ M1 nếu còn thời gian: bỏ qua, bản phân đoạn có sẵn và PyVi thử nghiệm. Với chế độ phân đoạn chuẩn, cần văn bản đã được phân đoạn phù hợp nghiên cứu; không dán văn bản thô vào `article_segmented`.

## Giải thích kết quả thận trọng

- **M1:** “Mô hình ước lượng xác suất toàn bài có dấu hiệu lỗi; không chỉ ra từ nào sai hay tự sửa từ.”
- **M2:** “Đây là dự đoán dựa vào tiêu đề, không quyết định thông tin trong bài báo là thật hay giả.”
- **M3:** “Đây là dự đoán về sự nhất quán giữa tiêu đề và nội dung, không phải bộ máy fact-checking.”
- **PyVi:** “Đây là chế độ tiện lợi để nhập tiếng Việt thông thường nhưng khác phân đoạn gốc; trên mẫu validation giới hạn 8/60 nhãn đã thay đổi.”
- **Hồi quy:** “Kiểm thử hồi quy xác nhận adapter và pipeline tích hợp giữ hành vi của mô hình tham chiếu; không thay cho đánh giá accuracy/F1 chính thức.”

## Câu hỏi hội đồng có thể đặt

**Tại sao có ba module?** Mỗi module phân tích một thuộc tính riêng của bài báo. Hệ thống không gộp ba xác suất thành một điểm tổng hợp.

**M1 có sửa lỗi chính tả không?** Không. M1 là bộ phân loại lỗi ở cấp độ văn bản; chức năng định vị/sửa lỗi nằm ngoài phạm vi bản demo.

**Vì sao PyVi chỉ là thử nghiệm?** Dữ liệu huấn luyện đã được phân đoạn từ trước. Các công cụ phân đoạn mới không tái tạo hoàn toàn ranh giới từ ban đầu, và sự khác biệt này có thể làm thay đổi dự đoán.

**Vì sao chỉ chạy một Uvicorn worker?** Để nạp mô hình một lần và tuần tự hóa suy luận qua cùng một GPU, tránh nhân đôi bộ nhớ mô hình trong bản demo.

**Bài mẫu có chứng minh model chính xác không?** Không. Bài mẫu phục vụ kiểm tra trải nghiệm và trình bày cơ chế; các chỉ số đánh giá mô hình cần dựa trên dữ liệu gán nhãn và quy trình đánh giá riêng.

**Có thể triển khai API lên internet ngay không?** Chưa nên. Bản V1 tập trung demo local, chưa có xác thực và các lớp bảo vệ vận hành cho môi trường public.
