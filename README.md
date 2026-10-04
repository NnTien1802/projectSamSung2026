# ProjectSamSung V1

Hệ thống hỗ trợ đánh giá bài báo tiếng Việt bằng các mô hình phân loại văn bản.

ProjectSamSung V1 là workspace triển khai/tích hợp được tách từ project nghiên cứu gốc, tập trung vào ba chức năng đang được giữ lại trong bản V1:

- **Module 1 — Spelling/Textual Error Detection:** phân loại bài báo ở mức tài liệu bằng **PhoBERT MIL**.
- **Module 2 — Clickbait Detection:** phân loại tiêu đề clickbait bằng **mBERT title-only**.
- **Module 3 — Headline–Content Consistency:** đánh giá tính nhất quán giữa tiêu đề và nội dung bằng mô hình kết hợp **description + full-article MAX-MIL**.

> Nhánh nghiên cứu cũ `module3_plagiarism_detection` không thuộc phạm vi V1 và không được xem là Module 3 hiện tại.

---

## 1. Kiến trúc tổng quan

Luồng triển khai hiện tại:

```text
Browser / Internet
        |
        v
Cloudflare Tunnel (tùy chọn khi public demo)
        |
        v
Nginx + React Frontend :80
        |
        v
FastAPI Backend :8000
        |
        v
Inference Service / Model Registry
        |
        +--> Module 1: PhoBERT MIL
        +--> Module 2: mBERT Clickbait
        +--> Module 3: mBERT Consistency
        |
        v
NVIDIA GPU / CUDA
```

Frontend gọi backend thông qua các đường dẫn `/api/...` và khi chạy bằng Docker Compose, Nginx phục vụ giao diện đồng thời proxy request tới FastAPI.

---

## 2. Phạm vi và nguyên tắc của V1

### Module 1

Module 1 thực hiện **document-level classification** cho bài báo có/không có lỗi văn bản.

- Mô hình: PhoBERT MIL.
- Đầu vào tham chiếu: văn bản đã được word segmentation (`article_segmented`).
- Không định vị chính xác từ sai.
- Không tự động sửa lỗi chính tả.
- Không nên mô tả Module 1 như một hệ thống correction hoặc sequence labeling.

V1 hỗ trợ ba cách sử dụng ở lớp ứng dụng:

1. Bỏ qua Module 1.
2. Gửi văn bản đã phân đoạn từ trước.
3. Dùng PyVi ở chế độ **experimental** để phân đoạn văn bản tự nhiên trước khi đưa vào M1.

PyVi chỉ là bước tiền xử lý, không phải mô hình phát hiện lỗi và không được xem là tương đương tuyệt đối với segmentation tham chiếu dùng trong nghiên cứu.

### Module 2

Module 2 phát hiện tiêu đề clickbait.

- Kiến trúc: `bert-base-multilingual-cased`.
- Input: tiêu đề (`title`) בלבד.
- Threshold khóa: `0.61`.

### Module 3

Module 3 đánh giá tính nhất quán giữa tiêu đề và nội dung bài báo.

- Kiến trúc: mBERT hybrid.
- Nhánh 1: headline + description.
- Nhánh 2: full-article MAX-MIL.
- Trọng số fusion: `0.5 / 0.5`.
- Threshold khóa: `0.45`.

M2 và M3 có thể dùng chung một đối tượng runtime `HeadlineIntegrityPipeline`, nhưng vẫn sử dụng các checkpoint/tác vụ riêng.

### Kết quả hệ thống

Ba module trả về **ba kết quả độc lập**. V1 không tạo một “điểm chất lượng báo chí tổng hợp”.

---

## 3. Cấu trúc chính của project

```text
projectSamSung_v1/
|
|-- frontend/                 # React/Vite frontend
|-- module1/                  # M1 PhoBERT MIL và artifact liên quan
|-- module2/                  # M2 clickbait + M3 consistency
|-- system/
|   |-- backend/              # FastAPI, adapters, services, registry
|   `-- scripts/              # smoke test, regression, release verification
|-- docs/                     # API, báo cáo, bằng chứng release
|
|-- Dockerfile                # Backend Docker image
|-- docker-compose.yml        # API + frontend
|-- requirements_docker.txt   # Runtime dependencies trong container
|-- requirements-v1-locked.txt# Snapshot môi trường V1
|-- .dockerignore
|-- .gitignore
`-- README.md
```

Không di chuyển các checkpoint, tokenizer, threshold hoặc lock file đã khóa nếu chưa thực hiện một release migration riêng và chạy lại regression.

---

## 4. Yêu cầu môi trường

Môi trường V1 đã được chạy với cấu hình tương đương:

```text
Python       : 3.10.x
PyTorch      : 2.14.0+cu132
Transformers : 5.16.1
CUDA         : 13.2
GPU          : NVIDIA GeForce RTX 4060 Laptop GPU
```

Docker cần hỗ trợ NVIDIA GPU passthrough.

Kiểm tra GPU Docker:

```cmd
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

Nếu GPU được hiển thị bình thường thì có thể tiếp tục build/chạy hệ thống.

---

## 5. Chạy toàn bộ hệ thống bằng Docker Compose

Đây là cách được khuyến nghị cho V1.

Mở CMD tại thư mục project:

```cmd
cd /d C:\projectSamSung_v1
```

Build image:

```cmd
docker compose build
```

Khởi động:

```cmd
docker compose up -d
```

Kiểm tra container:

```cmd
docker compose ps
```

Dự kiến có hai service:

- `api` — FastAPI, cổng `8000`.
- `frontend` — Nginx + React, cổng `80`.

Mở giao diện local:

```text
http://127.0.0.1/
```

Kiểm tra API health trực tiếp:

```cmd
curl http://127.0.0.1:8000/api/v1/health
```

Hoặc qua Nginx:

```cmd
curl http://127.0.0.1/api/v1/health
```

Kết quả mong đợi:

```json
{"status":"ok","models_ready":true}
```

Xem log API:

```cmd
docker compose logs -f api
```

Khi khởi động thành công, log cần cho thấy pipeline sẵn sàng và GPU CUDA được nhận diện.

---

## 6. Chạy backend không dùng Docker

Có thể chạy trực tiếp bằng virtual environment để phát triển hoặc kiểm thử.

Cài dependency API tối thiểu nếu cần:

```cmd
cd /d C:\projectSamSung_v1
.\.venv\Scripts\python.exe -m pip install -r requirements_api_v1.txt
```

Khởi động FastAPI:

```cmd
set SAMSUNG_DEVICE=cuda
.\.venv\Scripts\python.exe -m uvicorn system.backend.app.api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

> Với runtime GPU V1, chỉ dùng **một Uvicorn worker**. Không dùng `--workers 2` hoặc nhiều worker nếu chưa thiết kế lại cơ chế quản lý VRAM/model instance.

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

## 7. API chính

Các endpoint chính:

```text
GET  /api/v1/health
GET  /api/v1/models
POST /api/v1/spelling
POST /api/v1/clickbait
POST /api/v1/consistency
POST /api/v1/analyze
```

Ý nghĩa:

- `GET /api/v1/health` — trạng thái sẵn sàng của backend/model.
- `GET /api/v1/models` — metadata các model đang phục vụ.
- `POST /api/v1/spelling` — chạy M1.
- `POST /api/v1/clickbait` — chạy M2 từ tiêu đề.
- `POST /api/v1/consistency` — chạy M3 từ tiêu đề, description và article text.
- `POST /api/v1/analyze` — endpoint tổng hợp luồng gọi, nhưng vẫn trả ba kết quả độc lập.

Các request model được điều phối tuần tự qua lock để tránh nhiều inference đồng thời tranh chấp GPU trong cấu hình V1.

---

## 8. Kiểm thử và xác minh release

### Unit/API test

```cmd
cd /d C:\projectSamSung_v1
.\.venv\Scripts\python.exe -m unittest discover -s system\backend\tests -v
```

### Verify artifact/model lock

```cmd
.\.venv\Scripts\python.exe system\scripts\01_model_release.py verify
```

### Smoke test adapter

```cmd
.\.venv\Scripts\python.exe system\scripts\05_smoke_test_adapters.py --device cuda
```

### Regression Module 1

```cmd
.\.venv\Scripts\python.exe system\scripts\06_regression_test_module1.py --device cuda
```

### Regression Module 2 / Module 3

Tùy cấu trúc hiện tại của repository, chạy script regression của unified headline pipeline trong thư mục `module2`/`module2_headline_integrity` tương ứng với phiên bản V1 đang giữ.

### HTTP smoke test

```cmd
.\.venv\Scripts\python.exe system\scripts\09_smoke_test_http_api.py
```

Smoke/regression test dùng để xác minh routing, integration và tính tương thích với artifact đã khóa; không được diễn giải thành accuracy benchmark mới.

---

## 9. Docker runtime notes

Runtime Docker dùng GPU thông qua:

```yaml
gpus: all
```

và biến môi trường:

```text
SAMSUNG_DEVICE=cuda
HF_HOME=/opt/hf-cache
HF_HUB_DISABLE_TELEMETRY=1
```

Hugging Face cache được đặt trong Docker volume để tránh tải lại base model/tokenizer mỗi lần container khởi động.

Lần chạy container đầu tiên có thể cần tải thêm artifact từ Hugging Face Hub nếu cache chưa có.

Không retrain model hoặc retune threshold trong quá trình build/deploy V1.

---

## 10. Public demo bằng Cloudflare Tunnel

Khi frontend local đang chạy tại cổng `80`, có thể public tạm thời bằng Cloudflare Quick Tunnel:

```cmd
C:\Cloudflared\cloudflared.exe tunnel --url http://localhost:80
```

Cloudflare sẽ trả về URL dạng:

```text
https://<random-name>.trycloudflare.com
```

Quick Tunnel phù hợp cho demo/thử nghiệm. URL `trycloudflare.com` có thể thay đổi sau khi tunnel được tạo lại.

Luồng truy cập:

```text
Internet
   |
   v
Cloudflare Tunnel
   |
   v
localhost:80
   |
   v
Nginx / React
   |
   v
FastAPI
   |
   v
GPU inference
```

Nếu cần hostname cố định, cần chuyển sang Cloudflare Named Tunnel và domain riêng.

---

## 11. Những file/artifact cần giữ nguyên

Các artifact runtime quan trọng gồm:

```text
module1/
  models/module1_phobert_mil_full/
  results/module1_phobert_mil_full/threshold_tuning/

module2/
  models/clickbait/mbert_title_only_v1/
  models/consistency/baseline_title_description_mbert/
  models/consistency/full_article_max_mil_mbert/
  results/clickbait/final_title_only_v1/
  results/consistency/final_hybrid_v1/
```

Tên thư mục thực tế có thể phản ánh lịch sử refactor của project; không đổi đường dẫn của artifact đã khóa nếu chưa chạy lại toàn bộ verification/regression.

---

## 12. Giới hạn của V1

- M1 là phân loại ở mức tài liệu, không định vị/sửa từng lỗi.
- PyVi chỉ là preprocessing experimental cho M1.
- M2 chỉ đánh giá tiêu đề clickbait.
- M3 đánh giá consistency giữa tiêu đề và nội dung, không phải plagiarism detection.
- Ba module không tạo một điểm chất lượng tổng hợp.
- Runtime V1 được tối ưu cho một GPU worker.
- Quick Tunnel không phải hạ tầng production và không đảm bảo URL cố định.
- Nếu đổi đường dẫn model, checkpoint, tokenizer, dependency hoặc threshold, cần chạy lại release verification và regression.

---

## 13. Ghi chú tương thích

Workspace V1 được tạo từ project nghiên cứu gốc với mục tiêu giữ nguyên hành vi suy luận của các model đã chọn.

Không chỉnh sửa checkpoint, tokenizer, threshold hoặc model lock chỉ để làm gọn cấu trúc thư mục.

Khi cần refactor lớn, nên thực hiện theo quy trình:

```text
Backup
  -> Refactor
  -> Verify model locks
  -> Unit/API tests
  -> Adapter smoke tests
  -> Regression M1
  -> Regression M2/M3
  -> HTTP smoke test
  -> Frontend build
  -> Docker build
```

Chỉ sau khi tất cả bước trên PASS mới nên coi cấu trúc mới là release thay thế hợp lệ.

---

## 14. Tóm tắt

ProjectSamSung V1 hiện cung cấp một pipeline hoàn chỉnh từ giao diện người dùng tới inference GPU:

```text
React UI
  -> Nginx
  -> FastAPI
  -> Model Registry / Adapters
  -> PhoBERT + mBERT models
  -> NVIDIA GPU
```

Hệ thống tập trung hỗ trợ đánh giá ba khía cạnh riêng biệt của bài báo tiếng Việt: lỗi văn bản ở mức tài liệu, tiêu đề clickbait và tính nhất quán tiêu đề–nội dung.
