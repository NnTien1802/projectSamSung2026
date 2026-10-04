# API V1 — hợp đồng đầu vào/đầu ra

Địa chỉ local: `http://127.0.0.1:8000`; tài liệu OpenAPI tự động: `/docs`.

## Health

`GET /api/v1/health`

Khi nạp xong ba module:

```json
{"status":"ok","models_ready":true}
```

## Kiểm tra ba module

`GET /api/v1/models` trả về ngưỡng đã khóa M1=0.57, M2=0.61, M3=0.45. Không cung cấp giao diện sửa threshold.

## Phân tích kết hợp

`POST /api/v1/analyze`:

```json
{
  "title": "Ví dụ tiêu đề",
  "description": "Tóm tắt ngắn",
  "article_text": "Nội dung bài báo tiếng Việt.",
  "article_segmented": "Nội_dung bài_báo tiếng_Việt ."
}
```

Chỉ gửi `article_segmented` nếu **đã** có đầu ra từ bộ phân đoạn từ tương thích với dữ liệu huấn luyện M1. Nếu không có, bỏ hẳn trường này: M1 được trả về với `status="skipped"`, M2/M3 vẫn chạy.

Dạng kết quả (giá trị dưới đây chỉ **minh họa cấu trúc**, không phải dự đoán của checkpoint):

```json
{
  "api_version": "v1",
  "module1_spelling": {"status": "skipped", "reason": "article_segmented is required ..."},
  "module2_clickbait": {
    "status": "completed",
    "result": {"probability_clickbait": 0.2, "threshold": 0.61, "prediction": 0, "label": "NORMAL"}
  },
  "module3_consistency": {
    "status": "completed",
    "result": {
      "probability_consistent": 0.8, "threshold": 0.45, "prediction": 1, "label": "CONSISTENT",
      "branches": {"description": {}, "max_mil": {}}
    }
  },
  "notes": ["Module outputs are independent; no aggregate news-quality score is calculated."]
}
```

## Endpoint riêng

- `POST /api/v1/spelling`: `{"article_segmented":"người_dân ..."}`. Trả kết quả trực tiếp của `SpellingAdapter`: `probability_error`, `threshold`, `prediction`, `label`, `chunk_count`, `warnings`.
- `POST /api/v1/clickbait`: `{"title":"..."}`. Trả kết quả trực tiếp `ClickbaitAdapter`.
- `POST /api/v1/consistency`: `{"title":"...", "description":"...", "article_text":"..."}`. Trả kết quả trực tiếp `HeadlineConsistencyAdapter` với hai nhánh description/MAX-MIL. Nếu `description` vắng mặt hoặc để trống, adapter hiện tại vẫn chạy và có cảnh báo; xác suất khi thiếu mô tả có thể kém đại diện hơn so với dữ liệu huấn luyện.

## Mã lỗi

- `422`: sai cấu trúc, để trống trường bắt buộc hoặc vượt giới hạn từng trường.
- `503`: model chưa sẵn sàng hoặc GPU worker bận quá 30 giây.
- `500`: suy luận lỗi; nội dung ngoại lệ nội bộ không được trả về client.

**Bảo mật:** Chỉ chạy localhost trong V1. Chưa có xác thực hay lớp bảo vệ cho deployment trên mạng công khai.
