# Đề cương báo cáo đồ án tốt nghiệp — ProjectSamSung V1

Ngày khảo sát: 24/09/2026. Phạm vi: workspace `C:/projectSamSung_v1`.

**Trạng thái: đề cương và bản đồ bằng chứng; chưa viết nội dung đầy đủ.** Số trang dưới đây là dự kiến biên tập, không phải số liệu thực nghiệm. Tên đề tài làm việc: **Hệ thống hỗ trợ đánh giá bài báo tiếng Việt qua phân loại lỗi văn bản, phát hiện tiêu đề giật gân và đánh giá nhất quán tiêu đề–nội dung**; tên đề tài chính thức theo hồ sơ nhà trường: **[CHƯA ĐỦ DỮ LIỆU]**.

## A. Kết quả khảo sát và nguyên tắc sử dụng bằng chứng

### A.1. Phạm vi thực tế của workspace

- Đây là workspace triển khai/tích hợp V1 được sao chép từ dự án nghiên cứu, có mã suy luận, checkpoint, báo cáo nghiên cứu và bằng chứng regression; không phải bản lưu đầy đủ quy trình nghiên cứu. Căn cứ: S01 và kiểm kê file hiện có.
- **M1: document-level classification** — phân loại tài liệu có/không có lỗi văn bản bằng PhoBERT MIL; không sửa lỗi, không định vị từ lỗi. Đầu vào tham chiếu là `article_segmented`. Căn cứ: S02–S05.
- **M2: phân loại clickbait từ tiêu đề** bằng mBERT title-only. **M3: đánh giá nhất quán tiêu đề–nội dung**, kết hợp nhánh description và MAX-MIL. Căn cứ: S07–S15.
- Thư mục lịch sử `module2_headline_integrity` chứa cả Task 2A và Task 2B; trong hệ thống V1 chúng tương ứng M2 và M3. Không gọi M3 hiện tại là phát hiện đạo văn. Căn cứ: S01, S14, S15.
- M2 và M3 dùng chung một đối tượng `HeadlineIntegrityPipeline`; điều này không có nghĩa hai tác vụ dùng chung một checkpoint. Registry và pipeline là nguồn mô tả quyền sở hữu mô hình. Căn cứ: S14, S15, S18.
- Giao diện hiện tại có lựa chọn bỏ qua M1, nhập văn bản đã phân đoạn hoặc dùng PyVi thử nghiệm. API mặc định bỏ qua M1 trong `/analyze` khi thiếu `article_segmented` và không bật PyVi. Căn cứ: S16, S17, S25.
- Có ba file `official_test_metrics.json` riêng cho M1, M2, M3. Chỉ sử dụng chúng trong đúng bối cảnh dataset, split, nhãn và cấu hình đã ghi; không lấy kết quả kiểm thử tích hợp thay thế. Căn cứ: S05, S09, S13.

### A.2. Phân loại bằng chứng bắt buộc

| Nhóm | Nguồn chính | Có thể chứng minh | Không được suy diễn |
|---|---|---|---|
| Nghiên cứu chính thức | S05, S09, S13; cấu hình và dữ liệu tại S03–S12 | Kết quả test được lưu của từng tác vụ; điều kiện đánh giá và ngưỡng đã chọn | Độ chính xác chung của cả hệ thống, hiệu quả trên mọi bài báo thực tế |
| Phát triển mô hình trên validation | S04, S07, S08, S11, S12, S27 | Lịch sử huấn luyện được lưu, so sánh cấu hình và lựa chọn trên validation | Kết quả official test; một lần huấn luyện mới thực hiện trong bước khảo sát |
| Regression/tích hợp | S19–S22 | Hành vi API, định tuyến, tương đương đầu ra với reference trong dung sai | Accuracy benchmark; tỷ lệ PASS không phải accuracy |
| PyVi thử nghiệm | S24, S25 | Khác biệt phân đoạn/dự đoán trên mẫu validation đã khảo sát và cơ chế bật thử nghiệm | Tương đương bộ phân đoạn gốc; kế thừa trực tiếp metric test M1 |
| Khóa artifact và tài nguyên | S18, S23 | Hash hoặc thông tin bộ nhớ/thời gian trong phạm vi báo cáo lưu | Hash PASS chứng minh chất lượng mô hình; thời gian adapter là độ trễ HTTP |

Quy tắc khi viết: gắn mỗi bảng, hình, thông số và khẳng định kỹ thuật với đường dẫn nguồn, trường JSON/hàm tương ứng, split và phạm vi đo. Phân biệt “artifact ghi nhận” với “đã tái kiểm chứng trong workspace”. Bước này chỉ đọc file, không chạy huấn luyện, suy luận, benchmark, regression hoặc build.

### A.3. Khoảng trống và điểm cần đối chiếu

| Nội dung | Phát hiện và cách xử lý trong báo cáo |
|---|---|
| Mã huấn luyện gốc | Trong `scripts/` hiện chỉ thấy nguồn Python `16_evaluate_phobert_test.py`; chưa thấy các script huấn luyện/tạo dữ liệu gốc trong cây dự án được khảo sát. Chương 3 mô tả từ cấu hình và lịch sử đã lưu; chi tiết thuật toán không được lưu: **[CHƯA ĐỦ DỮ LIỆU]**. |
| Dữ liệu M1 | Có `validation.jsonl` và `split_summary.json`; chưa có `train.jsonl`, `test.jsonl` tại thư mục dataset M1 hiện tại. Quy mô train/test được trích từ báo cáo, không khẳng định đã đếm hoặc đánh giá lại toàn bộ. |
| Dữ liệu M3 | Có các file cặp train/validation/test; ở `data/interim/consistency/articles_v1` của nhánh headline chỉ thấy `validation.parquet`. Manifest nêu cặp không materialize toàn văn. Khả năng tái chạy huấn luyện/test đầy đủ chỉ từ V1: **[CHƯA ĐỦ DỮ LIỆU]**. |
| Nguồn dữ liệu thô và sinh lỗi M1 | Summary/EDA và audit có thông tin lịch sử; thiếu dữ liệu thô và mã tạo dữ liệu đầy đủ để kiểm chứng toàn bộ chuỗi tạo lập. Không biến trường `input_path` hay trích đoạn trong audit thành bằng chứng file gốc hiện tồn tại. |
| Bộ phân đoạn gốc M1 | Audit không xác định được thuật toán/phiên bản gốc. **[CHƯA ĐỦ DỮ LIỆU]**; không khẳng định PyVi hay underthesea là bộ phân đoạn huấn luyện. |
| Đánh giá M3 trong thực tế | S10, S13 nêu negative tổng hợp và positive gốc được giả định nhất quán. Benchmark test được con người xác minh đầy đủ trên tin thực tế: **[CHƯA ĐỦ DỮ LIỆU]**. |
| Thiết kế cũ so với mã chạy | `SYSTEM_DESIGN_V1.md` tự ghi là bản thiết kế, có SQLite/lịch sử và `/api/v1/analyses`. Mã S16 hiện có `/api/v1/analyze`, không có các route lịch sử đó; không đưa SQLite/lịch sử vào kiến trúc đã triển khai. Thiết kế giữ kết quả từng module khi module khác lỗi cũng cần phân biệt với `service.py` hiện tại, nơi lỗi trong chuỗi có thể làm cả request thất bại. |
| Tài liệu API trước bản PyVi | S26 là tài liệu hỗ trợ; S16, S17, S25 mới là căn cứ cho nhánh PyVi hiện tại. Các file trong `tools/.../overlay` và backup không thay thế mã đang dùng. |
| Kết quả HTTP smoke | Có script S22, nhưng chưa xác lập được log kết quả HTTP smoke riêng trong các nguồn đã đọc. Trạng thái chạy thành công: **[CHƯA ĐỦ DỮ LIỆU]**. Không dùng log smoke của pipeline Python để chứng minh HTTP đã chạy. |
| Tái lập, thống kê và vận hành | Đánh giá nhiều seed, khoảng tin cậy cho mọi official metric, kiểm thử tải HTTP và khảo sát người dùng: **[CHƯA ĐỦ DỮ LIỆU]**. Chỉ trình bày phân tích thống kê cụ thể nếu có artifact riêng, như so sánh validation M2 tại S08. |
| Tài liệu học thuật | Chưa xác lập tài liệu tham khảo học thuật đầy đủ từ các file đã khảo sát. Tác giả, năm, DOI và phát biểu lý thuyết vượt quá mã triển khai: **[CHƯA ĐỦ DỮ LIỆU]**; không tự điền trích dẫn. |

## B. Phân bổ dung lượng báo cáo

Giả định biên tập: khổ A4, cỡ chữ khoảng 13, giãn dòng khoảng 1,3–1,5; đây chỉ là cơ sở ước lượng. Quy định trình bày của trường chưa được cung cấp. Tổng dưới đây bao gồm bảng/hình, mục lục và tài liệu tham khảo; chưa tính bìa, lời cảm ơn hoặc biểu mẫu hành chính ngoài cấu trúc yêu cầu.

| Thứ tự | Phần | Trang dự kiến | Vị trí dự kiến trong 37 trang |
|---|---|---:|---|
| 1 | Mục lục | 1 | 1 |
| 2 | Đặt vấn đề | 2 | 2–3 |
| 3 | Chương 1: Cơ sở lý thuyết | 6 | 4–9 |
| 4 | Chương 2: Phân tích và thiết kế hệ thống | 7 | 10–16 |
| 5 | Chương 3: Huấn luyện mô hình và tích hợp hệ thống | 9 | 17–25 |
| 6 | Chương 4: Kết quả, kiểm thử và đánh giá hệ thống | 8 | 26–33 |
| 7 | Kết luận và hướng phát triển | 2 | 34–35 |
| 8 | Tài liệu tham khảo | 2 | 36–37 |
| | **Tổng** | **37** | |

Mục tiêu sau dàn trang là 30–40 trang. Không kéo dài phần thiếu dữ liệu để đủ trang; nếu tài liệu tham khảo chưa đủ hai trang, điều chỉnh dung lượng trong khoảng mục tiêu. Các mục A, B, D, E của file này là ghi chú chuẩn bị, không phải các chương bổ sung của báo cáo cuối.

## C. Đề cương chi tiết tới cấp 3

Các mã Sxx dẫn đến danh mục file tại mục D. Mỗi mục cấp 3 dưới đây là nhiệm vụ viết dự kiến, chưa phải nội dung báo cáo hoàn chỉnh.

### Mục lục — 1 trang

- Liệt kê đúng tám phần được yêu cầu; hiển thị đề mục tới cấp 3 của các chương, ví dụ `1.1.1`.
- Sinh số trang tự động sau dàn trang; các vị trí tại mục B chỉ là kế hoạch.

### Đặt vấn đề — 2 trang

**ĐV.1. Bối cảnh và bài toán — 0,75 trang**

- **ĐV.1.1. Bối cảnh sử dụng:** nhu cầu phân tích bài báo qua ba tác vụ trong phạm vi sản phẩm; không thêm số liệu thị trường hoặc mức độ phổ biến lỗi chưa có nguồn. Nguồn: S01, S16, S17.
- **ĐV.1.2. Định nghĩa vấn đề:** lỗi văn bản cấp tài liệu, clickbait và nhất quán tiêu đề–nội dung; tách chúng khỏi xác minh sự thật. Nguồn: S02, S09, S10, S16.

**ĐV.2. Mục tiêu và phạm vi — 0,75 trang**

- **ĐV.2.1. Mục tiêu hệ thống:** cung cấp ba kết quả riêng qua ứng dụng web; không có điểm chất lượng báo chí tổng hợp. Nguồn: S14–S17.
- **ĐV.2.2. Phạm vi nghiên cứu và triển khai:** kết quả nghiên cứu đã lưu, tích hợp V1 và tiền xử lý thử nghiệm; chỉ rõ phần tài nguyên không có trong V1. Nguồn: S01, S05, S09, S13, S24, S25.

**ĐV.3. Phương pháp thực hiện và tổ chức báo cáo — 0,5 trang**

- **ĐV.3.1. Phương pháp dựa trên bằng chứng:** đối chiếu code, cấu hình, báo cáo test, regression và hạn chế tái lập. Nguồn: S02–S25.
- **ĐV.3.2. Bố cục và đóng góp trong phạm vi chứng minh được:** dẫn sang bốn chương; không tự nhận đóng góp thuật toán mới nếu chưa có căn cứ.

### Chương 1. Cơ sở lý thuyết — 6 trang

**1.1. Ba bài toán phân loại và biểu diễn đầu vào — 1,5 trang**

- **1.1.1. Phân loại lỗi ở cấp tài liệu:** nhãn M1, `article_segmented`, phân biệt dự đoán tài liệu với phát hiện vị trí/sửa lỗi. Nguồn: S02–S05.
- **1.1.2. Phân loại clickbait và nhất quán:** nhãn, đầu vào khác nhau và chiều lớp dương của M2/M3; không đồng nhất clickbait với thiếu nhất quán. Nguồn: S06, S09, S10, S13, S14.
- **1.1.3. Phân đoạn từ và token hóa mô hình:** phân biệt văn bản natural/segmented, dấu `_` và tokenizer; giới hạn truy vết phân đoạn gốc. Nguồn: S02, S24, S25. Thuật toán phân đoạn gốc: **[CHƯA ĐỦ DỮ LIỆU]**.

**1.2. Encoder và đầu phân loại trong project — 1,5 trang**

- **1.2.1. PhoBERT trong M1:** trình bày encoder, biểu diễn token đầu, dropout và lớp tuyến tính theo `PhoBERTMIL`; không suy ra chi tiết pretraining chỉ từ tên model. Nguồn: S02, S04.
- **1.2.2. mBERT trong M2/M3:** trình bày bộ phân loại và cách biểu diễn tiêu đề/cặp văn bản theo pipeline và cấu hình. Nguồn: S07, S11, S14.
- **1.2.3. Phạm vi lý thuyết Transformer:** dự kiến giải thích các khái niệm cần dùng; tài liệu gốc cho attention, pretraining và lịch sử BERT/PhoBERT: **[CHƯA ĐỦ DỮ LIỆU]**. Không tự bổ sung thông số kiến trúc nền chưa kiểm chứng.

**1.3. Xử lý văn bản dài và kết hợp mô hình — 1,5 trang**

- **1.3.1. Chia chunk và MIL của M1:** `DocumentCollator`, chọn chunk và max pooling logit theo tài liệu; không diễn giải chunk thành vị trí từ lỗi. Nguồn: S02, S04.
- **1.3.2. MAX-MIL của M3:** chính sách chunk và điểm nhị phân từ chênh lệch logit; phân biệt với cách chia chunk M1. Nguồn: S11, S12, S14.
- **1.3.3. Fusion description và MAX-MIL:** phương thức kết hợp score và ngưỡng quyết định đã lưu; không gọi là cơ chế attention nếu nguồn không thể hiện. Nguồn: S12–S14.

**1.4. Nguyên tắc đánh giá và tái lập — 1,5 trang**

- **1.4.1. Train, validation và test:** đơn vị chia tập, lựa chọn cấu hình trên validation, khóa trước test theo artifact. Nguồn: S03, S04, S06, S09, S10, S12, S13.
- **1.4.2. Các chỉ số được project sử dụng:** confusion matrix, accuracy, precision, recall, F1, macro-F1, ROC-AUC và average precision; nêu rõ lớp dương. Nguồn: hàm tính metric trong S02 và các trường S05, S09, S13. Chứng minh lý thuyết ngoài mã: **[CHƯA ĐỦ DỮ LIỆU]**.
- **1.4.3. Ba loại kết luận khác nhau:** kết quả nghiên cứu, kiểm thử hành vi/tương đương số học và thử nghiệm thay tiền xử lý. Nguồn: S18–S25.

### Chương 2. Phân tích và thiết kế hệ thống — 7 trang

**2.1. Yêu cầu và ranh giới chức năng — 1 trang**

- **2.1.1. Dữ liệu người dùng nhập:** tiêu đề, mô tả, nội dung và lựa chọn đầu vào M1; trường bắt buộc/tùy chọn theo schema. Nguồn: S16, S17.
- **2.1.2. Đầu ra và giới hạn:** ba kết quả độc lập, nhãn/score/ngưỡng/cảnh báo; không sửa từ, không tổng hợp điểm. Nguồn: S02, S14–S17.

**2.2. Kiến trúc đã triển khai — 2 trang**

- **2.2.1. Các lớp của hệ thống:** React/TypeScript → FastAPI → `InferenceService` → registry/adapters → pipeline/checkpoint; dự kiến một sơ đồ thành phần. Nguồn: S14–S18.
- **2.2.2. Tổ chức ba module logic:** một adapter M1, hai adapter M2/M3 và pipeline headline dùng chung; minh họa quan hệ với các checkpoint. Nguồn: S02, S14, S15, S18.
- **2.2.3. Đối chiếu thiết kế cũ:** những nội dung S26 mới là đề xuất như SQLite/lịch sử; chỉ vẽ thành phần tồn tại trong mã hiện tại. Nguồn: S16, S17, S26.

**2.3. Luồng xử lý và hợp đồng API — 1,5 trang**

- **2.3.1. Khởi tạo và điều phối:** lifecycle, readiness, nạp registry và khóa suy luận; dự kiến sơ đồ tuần tự từ mã. Nguồn: S15, S16.
- **2.3.2. Endpoint và kiểm tra đầu vào:** bảng route, trường dữ liệu, giới hạn và lỗi HTTP theo code; không lấy endpoint dự kiến của tài liệu cũ. Nguồn: S16, S19.
- **2.3.3. Các nhánh M1 và xử lý lỗi:** presegmented, skipped, PyVi bật chủ động; hành vi lỗi toàn request theo service hiện tại. Nguồn: S16, S19, S25.

**2.4. Thiết kế giao diện — 1,5 trang**

- **2.4.1. Nhập liệu và trạng thái phân tích:** biểu mẫu, kiểm tra dữ liệu, trạng thái kết nối và gửi yêu cầu. Nguồn: S17.
- **2.4.2. Hiển thị kết quả theo module:** nhãn, score, ngưỡng, cảnh báo thiếu description và giới hạn M1. Nguồn: S15–S17.
- **2.4.3. Giao diện PyVi và ví dụ demo:** lựa chọn rõ ràng, cảnh báo thử nghiệm; demo tự tạo không phải dữ liệu đánh giá. Nguồn: S17, S25. Ảnh giao diện thực tế để chèn báo cáo: **[CHƯA ĐỦ DỮ LIỆU]** trong nguồn ảnh đã khảo sát.

**2.5. Tổ chức artifact và điều kiện vận hành — 1 trang**

- **2.5.1. Quản lý release:** checkpoint, tokenizer, threshold và SHA-256 theo manifest; phân biệt khóa artifact với kiểm chứng hành vi. Nguồn: S18, S20, S21.
- **2.5.2. Môi trường triển khai:** dependency, Vite proxy, cấu hình API và báo cáo tài nguyên; không đồng nhất môi trường ghi trong release với môi trường huấn luyện mọi thí nghiệm. Nguồn: S07, S11, S16–S18, S23.

### Chương 3. Huấn luyện mô hình và tích hợp hệ thống — 9 trang

**3.1. Dữ liệu và quy trình chuẩn bị được ghi nhận — 2 trang**

- **3.1.1. VN-NewsSpell v1 cho M1:** thống kê split, nhóm `source_id`, clean/noisy, mức độ và loại lỗi từ summary; nêu rõ phần dữ liệu/mã tạo không có. Nguồn: S03, S24.
- **3.1.2. ViClickbait v1 cho M2:** thống kê nhãn, chia tập theo normalized title và các kiểm tra overlap được lưu. Nguồn: S06, S08.
- **3.1.3. Headline consistency v1 cho M3:** cặp positive, negative tổng hợp cùng category, chính sách chọn ứng viên và giới hạn manual review. Nguồn: S10, S27.

**3.2. Huấn luyện và lựa chọn M1 từ artifact — 2 trang**

- **3.2.1. Cấu hình huấn luyện:** model, text field, số epoch, batch, learning rate và tham số chunk theo `training_config.json`. Nguồn: S04. Optimizer/chi tiết vòng lặp chưa có nguồn đầy đủ: **[CHƯA ĐỦ DỮ LIỆU]**.
- **3.2.2. Lịch sử huấn luyện và checkpoint:** trình bày đường cong từ `training_history.json`, đối chiếu best validation và checkpoint epoch trong báo cáo; không tự dựng các epoch không có. Nguồn: S04, S05.
- **3.2.3. Chọn threshold và chuẩn bị đánh giá:** quy tắc chọn trên validation, cấu hình khóa và mã evaluation. Nguồn: S02, S04, S05.

**3.3. Huấn luyện và lựa chọn M2 từ artifact — 1,5 trang**

- **3.3.1. Cấu hình mBERT title-only:** đầu vào, max length, class weights và lịch sử epoch theo run config/history. Nguồn: S07.
- **3.3.2. So sánh title-only với title+lead:** kết quả validation và phân tích thống kê đã lưu; không suy ra ưu thế tuyệt đối trên mọi dữ liệu. Nguồn: S08.
- **3.3.3. Cấu hình cuối trước test:** checkpoint, best epoch và threshold theo final lock; chi tiết script huấn luyện không có: **[CHƯA ĐỦ DỮ LIỆU]**. Nguồn: S09.

**3.4. Huấn luyện và kết hợp M3 từ artifact — 2 trang**

- **3.4.1. Các nhánh baseline:** title–description và title–article-head; sử dụng run config và báo cáo so sánh đúng tập validation. Nguồn: S11, S27.
- **3.4.2. Nhánh full-article MAX-MIL:** cấu hình khởi tạo, loss, batch/gradient accumulation, giới hạn chunk theo run config và pipeline. Nguồn: S11, S14. Checkpoint khởi tạo được nhắc trong config nhưng chưa thấy bản tương ứng trong V1; không khẳng định tái lập đầy đủ.
- **3.4.3. Chọn fusion và khóa mô hình:** phân chia calibration/holdout bên trong validation, trọng số và threshold cuối; phân biệt với official test. Nguồn: S12, S13.

**3.5. Tích hợp mô hình vào ứng dụng — 1,5 trang**

- **3.5.1. Tái sử dụng mã suy luận:** adapter M1 nạp lớp/collator từ evaluation; M2/M3 gọi pipeline cũ; registry giữ instance. Nguồn: S02, S14, S15.
- **3.5.2. Kết nối API và frontend:** schema, service, client, readiness và cảnh báo được hiển thị. Nguồn: S16, S17, S19.
- **3.5.3. Bổ sung PyVi thử nghiệm:** nhánh opt-in, kiểm tra bề mặt Unicode và ranh giới nối; không thay checkpoint/ngưỡng và không nhận là preprocessing chính thức. Nguồn: S24, S25.

### Chương 4. Kết quả, kiểm thử và đánh giá hệ thống — 8 trang

**4.1. Thiết lập và cách đọc kết quả — 1 trang**

- **4.1.1. Nguồn và điều kiện đánh giá:** bảng dataset/split/đơn vị mẫu/nhãn/cấu hình cho từng tác vụ. Nguồn: S03, S05, S06, S09, S10, S13.
- **4.1.2. Phân tách các bảng kết quả:** bảng official test, bảng regression, bảng kiểm thử phần mềm và bảng PyVi riêng; không xếp hạng ba module qua metric của các tập khác nhau. Nguồn: S05, S09, S13, S19–S25.

**4.2. Kết quả nghiên cứu chính thức — 2,5 trang**

- **4.2.1. M1 trên test VN-NewsSpell:** metric và confusion matrix từ official JSON; phân tích mức độ lỗi từ trường có thật và error analysis. Nhắc đây là phân loại cấp tài liệu với phân đoạn tham chiếu. Nguồn: S05, S28.
- **4.2.2. M2 trên test ViClickbait:** metric và confusion matrix; phân tích theo source/category trong giới hạn kích thước nhóm được báo cáo. Nguồn: S09, S28.
- **4.2.3. M3 trên test headline consistency:** metric fusion cuối; nếu dùng metric từng nhánh phải ghi đúng `branch_reference_metrics_fixed_at_final_threshold`. Nêu rõ negative tổng hợp, không phải benchmark thực tế được con người xác minh đầy đủ. Nguồn: S10, S13, S28.

**4.3. Regression và kiểm thử tích hợp — 1,5 trang**

- **4.3.1. Regression M1:** mẫu validation cố định, reference, sai lệch xác suất/nhãn/chunk và dung sai từ báo cáo; đánh giá tương đương định tuyến số học, không đo accuracy. Nguồn: S20.
- **4.3.2. Regression M2/M3:** kiểm tra clickbait, description, MAX-MIL và fusion so với artifact validation; báo cáo token/chunk mismatch nếu có trường tương ứng. Nguồn: S21.
- **4.3.3. Kiểm thử phần mềm và smoke:** phân biệt unit/API tests dùng fake/mock, build frontend, HTTP smoke với checkpoint thật và smoke pipeline Python. Chỉ báo PASS khi có log phù hợp; kết quả HTTP smoke hiện **[CHƯA ĐỦ DỮ LIỆU]**. Nguồn: S19, S22.

**4.4. Khảo sát PyVi thử nghiệm — 1 trang**

- **4.4.1. Mục đích và phương pháp:** đối chiếu phân đoạn trên mẫu validation, kiểm tra bề mặt và thay đổi đầu vào; không truy cập test để chọn bộ phân đoạn. Nguồn: S24.
- **4.4.2. Độ lệch dự đoán:** trình bày số mẫu, số lần đổi nhãn, sai lệch xác suất và thay đổi chunk đúng theo `prediction_drift_pyvi.json`; không gọi tỷ lệ giữ nguyên nhãn là accuracy. Nguồn: S24.
- **4.4.3. Ý nghĩa với ứng dụng:** cảnh báo, bật chủ động, từ chối bề mặt bị thay đổi và giới hạn tương thích; không dùng metric chính thức M1 để quảng bá nhánh PyVi. Nguồn: S17, S25.

**4.5. Tài nguyên và giới hạn vận hành — 1 trang**

- **4.5.1. Profile đã lưu:** mô tả thiết bị, chế độ chạy, số lần lặp, thời gian adapter và bộ nhớ từ script/report; nêu đường dẫn môi trường gốc trong báo cáo khi cần. Nguồn: S18, S23.
- **4.5.2. Giới hạn suy luận về hiệu năng:** không gọi số đo profile là latency HTTP end-to-end, throughput nhiều người dùng hoặc cam kết production. Các đánh giá đó: **[CHƯA ĐỦ DỮ LIỆU]**.

**4.6. Thảo luận và giới hạn độ tin cậy — 1 trang**

- **4.6.1. Giới hạn dữ liệu và đầu vào:** lỗi tạo lập M1 theo nguồn đã lưu, shortcut nguồn/phong cách trong báo cáo M2, negative tổng hợp M3 và sai khác segmentation. Nguồn: S03, S08, S10, S13, S24.
- **4.6.2. Giới hạn tái lập và khái quát:** thiếu script/dữ liệu đầy đủ, giới hạn chunk, log kiểm thử cục bộ; phân biệt kết luận đã có với đánh giá còn thiếu. Nguồn: S01–S04, S11, S14, S19–S25.

### Kết luận và hướng phát triển — 2 trang

**KL.1. Tổng kết trong phạm vi bằng chứng — 1 trang**

- **KL.1.1. Kết quả đạt được:** tóm tắt ba mô hình và ứng dụng tích hợp trên cơ sở Chương 3–4; không thêm metric mới. Nguồn: S05, S09, S13–S21.
- **KL.1.2. Hạn chế còn tồn tại:** M1 chỉ phân loại tài liệu, hạn chế kiểm chứng thực tế của M3, PyVi thử nghiệm và tài nguyên tái lập thiếu. Nguồn: S01, S02, S10, S13, S24, S25.

**KL.2. Hướng phát triển đề xuất — 1 trang**

- **KL.2.1. Dữ liệu và đánh giá:** đề xuất bổ sung tài liệu nguồn, bộ test được người xác minh, kiểm chứng segmentation và lưu đầy đủ quy trình; ghi rõ đây là công việc tương lai.
- **KL.2.2. Tích hợp và vận hành:** đề xuất kiểm thử tải HTTP, khảo sát người dùng và hoàn thiện vận hành; không trình bày tính năng dự kiến như đã triển khai.

### Tài liệu tham khảo — 2 trang

**TK.1. Nguồn nội bộ có thể trích dẫn — khoảng 1 trang**

- **TK.1.1. Mã và cấu hình:** chọn các file S01–S18, S25 làm nguồn triển khai; ghi đường dẫn và ngày khảo sát, không tự gán tác giả hoặc mã commit.
- **TK.1.2. Báo cáo thực nghiệm/kiểm thử:** S05, S09, S13, S19–S24, S27, S28; ghi đúng loại bằng chứng, phiên bản và thời điểm nếu artifact có trường này.

**TK.2. Nguồn học thuật cần bổ sung — khoảng 1 trang dự phòng**

- **TK.2.1. Nền tảng mô hình và đánh giá:** tài liệu gốc về Transformer, BERT/mBERT, PhoBERT, MIL và metric: **[CHƯA ĐỦ DỮ LIỆU]** để lập danh mục học thuật hoàn chỉnh từ repository hiện có.
- **TK.2.2. Nguồn dataset và công cụ:** nguồn công bố dataset, điều kiện sử dụng và thông tin thư mục học thuật của PyVi: **[CHƯA ĐỦ DỮ LIỆU]**. Tên thư viện trong requirements không thay thế trích dẫn học thuật.

## D. Danh mục file nguồn và ánh xạ sang chương

Đường dẫn liên kết bên dưới tương đối với `docs/thesis/`. Các file đã được xác nhận tồn tại khi khảo sát; mức bằng chứng khác nhau: mã thể hiện triển khai, JSON/CSV là kết quả lưu, tài liệu mô tả là nguồn bổ trợ. Các file checkpoint được xác nhận hiện diện nhưng không nạp để thực thi ở bước này.

| Mã | File nguồn có thật | Phần sử dụng và phạm vi bằng chứng |
|---|---|---|
| S01 | [README_V1.md](../../README_V1.md); [PROJECT_V1_CREATE_REPORT.json](../../PROJECT_V1_CREATE_REPORT.json) | Đặt vấn đề, Ch.2–3, kết luận: phạm vi V1, nguồn sao chép và loại tài nguyên không được mang sang; số file trong báo cáo tạo V1 không phải kiểm kê hiện tại. |
| S02 | [16_evaluate_phobert_test.py](../../scripts/16_evaluate_phobert_test.py); [spelling.py](../../system/backend/app/adapters/spelling.py) | Ch.1–3: `DocumentCollator`, `PhoBERTMIL`, hàm đánh giá và `SpellingAdapter.predict`; chứng minh M1 là document-level. |
| S03 | [split_summary.json](../../data/processed/vn_newsspell_v1/split_summary.json); [validation.jsonl](../../data/processed/vn_newsspell_v1/validation.jsonl) | Ch.1, 3–4: split/nhãn/thống kê được lưu và file validation hiện có. Không suy ra train/test M1 hiện diện. |
| S04 | [training_config.json](../../results/module1_phobert_mil_full/training_config.json); [training_history.json](../../results/module1_phobert_mil_full/training_history.json); [best_validation_metrics.json](../../results/module1_phobert_mil_full/best_validation_metrics.json); [best_threshold.json](../../results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) | Ch.1, 3: cấu hình, lịch sử, chọn threshold M1; không thay thế mã huấn luyện đầy đủ. |
| S05 | [official_test_metrics.json — M1](../../results/module1_phobert_mil_full/test_evaluation/official_test_metrics.json); [test_predictions.csv](../../results/module1_phobert_mil_full/test_evaluation/test_predictions.csv) | Ch.4, kết luận: kết quả nghiên cứu M1 đã lưu; dùng các trường `test_records`, `text_field`, `threshold_selected_on`, `threshold_tuned_on_test`. |
| S06 | [prepare_report.json](../../module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json); [train.parquet](../../module2_headline_integrity/data/processed/clickbait/viclickbait_v1/train.parquet); [validation.parquet](../../module2_headline_integrity/data/processed/clickbait/viclickbait_v1/validation.parquet); [test.parquet](../../module2_headline_integrity/data/processed/clickbait/viclickbait_v1/test.parquet) | Ch.1, 3–4: dữ liệu M2, đơn vị chia tập và overlap theo báo cáo. Sự hiện diện của Parquet chưa phải kiểm định lại nội dung. |
| S07 | [run_config.json — title-only](../../module2_headline_integrity/results/clickbait/mbert_title_only_v1/run_config.json); [training_history.csv](../../module2_headline_integrity/results/clickbait/mbert_title_only_v1/training_history.csv); [validation_metrics.json](../../module2_headline_integrity/results/clickbait/mbert_title_only_v1/validation_metrics.json) | Ch.1, 3: cấu hình và tiến trình M2 đã lưu, tách số epoch cấu hình với best epoch. |
| S08 | [comparison_report.json](../../module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json); [diagnostic_report.json](../../module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json); [FINAL_EXPERIMENT_CARD.md — M2](../../module2_headline_integrity/results/clickbait/final_title_only_v1/experiment_summary/FINAL_EXPERIMENT_CARD.md) | Ch.3–4: lựa chọn title-only, baseline/shortcut và giới hạn so sánh validation. Card là bản tổng hợp, ưu tiên JSON khi lấy số. |
| S09 | [final_model_lock.json — M2](../../module2_headline_integrity/results/clickbait/final_title_only_v1/final_model_lock.json); [official_test_metrics.json — M2](../../module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_metrics.json) | Ch.1, 3–4, kết luận: `frozen_configuration`, `official_final_metrics`, quy mô test và trạng thái không tune test theo artifact. |
| S10 | [dataset_manifest.json — M3](../../module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json); [validation_report.json](../../module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json); [validation.parquet — cặp](../../module2_headline_integrity/data/processed/consistency/headline_consistency_v1/validation.parquet); [validation.parquet — bài báo](../../module2_headline_integrity/data/interim/consistency/articles_v1/validation.parquet) | Ch.1, 3–4: nhãn, cách tạo cặp, chính sách negative, structural validation và cảnh báo synthetic evaluation. |
| S11 | [run_config.json — description](../../module2_headline_integrity/results/consistency/baseline_title_description_mbert/run_config.json); [training_history.csv — description](../../module2_headline_integrity/results/consistency/baseline_title_description_mbert/training_history.csv); [run_config.json — article head](../../module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/run_config.json); [run_config.json — MAX-MIL](../../module2_headline_integrity/results/consistency/full_article_max_mil_mbert/run_config.json); [training_history.csv — MAX-MIL](../../module2_headline_integrity/results/consistency/full_article_max_mil_mbert/training_history.csv) | Ch.1, 3: cấu hình các nhánh M3; trường `initialization`, `loss`, `chunk_binary_score`, `test_loaded`. Không tự điền chi tiết mã huấn luyện thiếu. |
| S12 | [fusion_report.json](../../module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json); [final_model_lock.json — M3](../../module2_headline_integrity/results/consistency/final_hybrid_v1/final_model_lock.json); [FINAL_EXPERIMENT_CARD.md — M3](../../module2_headline_integrity/results/consistency/final_hybrid_v1/experiment_summary/FINAL_EXPERIMENT_CARD.md) | Ch.1, 3: validation calibration/holdout, trọng số fusion, threshold và chunk policy. |
| S13 | [official_test_metrics.json — M3](../../module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json); [official_test_predictions.parquet](../../module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_predictions.parquet) | Ch.4, kết luận: `official_final_metrics`, metric nhánh tham chiếu và `methodological_warning`; không phải human-gold benchmark. |
| S14 | [headline_integrity_pipeline.py](../../module2_headline_integrity/inference/headline_integrity_pipeline.py); [UNIFIED_MODULE2_CARD.md](../../module2_headline_integrity/results/integration/unified_module2_v1/UNIFIED_MODULE2_CARD.md) | Ch.1–3: `HeadlineIntegrityPipeline`, `predict_clickbait`, `_build_article_chunks`, `predict_mil_consistency`, `predict_consistency`; chính sách suy luận và quan hệ Task 2A/2B. |
| S15 | [model_registry.py](../../system/backend/app/services/model_registry.py); [clickbait.py](../../system/backend/app/adapters/clickbait.py); [headline_consistency.py](../../system/backend/app/adapters/headline_consistency.py) | Ch.2–3: instance dùng chung, adapter logic và cảnh báo thiếu description. |
| S16 | [main.py](../../system/backend/app/api/main.py); [schemas.py](../../system/backend/app/api/schemas.py); [service.py](../../system/backend/app/api/service.py) | Ch.2–3: route, schema, default M1, lifecycle, khóa và lỗi; ưu tiên hơn tài liệu thiết kế cũ. |
| S17 | [App.tsx](../../frontend/src/App.tsx); [api.ts](../../frontend/src/api.ts); [presentation.ts](../../frontend/src/presentation.ts); [demo_cases.ts](../../frontend/src/demo_cases.ts); [package.json](../../frontend/package.json); [vite.config.ts](../../frontend/vite.config.ts) | Ch.2–3: giao diện, contract frontend, cảnh báo và cấu hình build/proxy. Demo không phải nhãn ground truth. |
| S18 | [model_lock_manifest_v1.json](../../system/release/model_lock_manifest_v1.json); [model_lock_config.json](../../system/release/model_lock_config.json); [environment_v1.json](../../system/release/environment_v1.json); [requirements-v1-locked.txt](../../requirements-v1-locked.txt); [requirements_api_v1.txt](../../requirements_api_v1.txt); [model_lock_verify.txt](../../system/reports/final_validation_v1/model_lock_verify.txt) | Ch.2–4: artifact/threshold/hash và môi trường ghi nhận; log xác minh lưu sẵn không chứng minh kiểm tra lại trong bước này. |
| S19 | [test_adapters.py](../../system/backend/tests/test_adapters.py); [test_api_v1.py](../../system/backend/tests/test_api_v1.py); [test_api_m1_pyvi_v1.py](../../system/backend/tests/test_api_m1_pyvi_v1.py); [backend_tests.txt](../../system/reports/final_validation_v1/backend_tests.txt); [frontend_build.txt](../../system/reports/final_validation_v1/frontend_build.txt) | Ch.4: phạm vi fake/mock và kết quả unit/API tests, build được lưu; không phải đo chất lượng mô hình hay kiểm thử trình duyệt đầy đủ. |
| S20 | [06_regression_test_module1.py](../../system/scripts/06_regression_test_module1.py); [reference_v1.json](../../system/reports/module1_regression_v1/reference_v1.json); [regression_report.json — M1](../../system/reports/module1_regression_v1/regression_report.json); [module1_regression.txt](../../system/reports/final_validation_v1/module1_regression.txt) | Ch.4: parity M1 trên validation, reference và dung sai, không phải official test. |
| S21 | [04_regression_check_unified_pipeline.py](../../module2_headline_integrity/inference/04_regression_check_unified_pipeline.py); [regression_report.json — M2/M3](../../module2_headline_integrity/results/integration/regression_v1/regression_report.json); [module23_regression.txt](../../system/reports/final_validation_v1/module23_regression.txt) | Ch.4: parity M2/M3 với artifact validation. |
| S22 | [09_smoke_test_http_api.py](../../system/scripts/09_smoke_test_http_api.py); [16_smoke_m1_pyvi_http.py](../../system/scripts/16_smoke_m1_pyvi_http.py); [smoke_test_report.json — pipeline](../../module2_headline_integrity/results/integration/unified_module2_v1/smoke_test/smoke_test_report.json) | Ch.4: kịch bản HTTP và báo cáo smoke pipeline riêng; không tráo nguồn kết quả giữa hai loại. |
| S23 | [07_profile_gpu_memory.py](../../system/scripts/07_profile_gpu_memory.py); [summary.json — profile](../../system/reports/gpu_profile_v1/summary.json); [combined.json](../../system/reports/gpu_profile_v1/combined.json) | Ch.2, 4: cách đo, bộ nhớ và latency ở cấp adapter trong báo cáo; không đo HTTP end-to-end. |
| S24 | [11_trace_m1_segmentation_origin.py](../../system/scripts/11_trace_m1_segmentation_origin.py); [12_benchmark_m1_segmenters.py](../../system/scripts/12_benchmark_m1_segmenters.py); [13_measure_m1_segmentation_drift.py](../../system/scripts/13_measure_m1_segmentation_drift.py); [origin_trace_v1.json](../../system/reports/m1_preprocessing_audit_v1/origin_trace_v1.json); [segmenter_benchmark_pyvi.json](../../system/reports/m1_preprocessing_audit_v1/segmenter_benchmark_pyvi.json); [prediction_drift_pyvi.json](../../system/reports/m1_preprocessing_audit_v1/prediction_drift_pyvi.json) | Ch.1, 3–4: truy vết và khảo sát tương thích. Trường `exists` trong audit lịch sử không chứng minh file được trích hiện có trong V1. Từ “benchmark” trong tên file không biến đây thành official accuracy benchmark. |
| S25 | [experimental_segmenter.py](../../system/backend/app/api/experimental_segmenter.py); [service.py](../../system/backend/app/api/service.py); [test_api_m1_pyvi_v1.py](../../system/backend/tests/test_api_m1_pyvi_v1.py) | Ch.2–4: `segment_raw_pyvi`, kiểm tra surface/join, opt-in và cảnh báo thử nghiệm; tests có segmenter tiêm vào, không xác nhận tương thích với phân đoạn gốc. |
| S26 | [SYSTEM_DESIGN_V1.md](../../system/scripts/SYSTEM_DESIGN_V1.md); [README_VI.md](../../README_VI.md); [API_V1.md](../API_V1.md) | Ch.2: tham khảo chủ đích thiết kế và tài liệu API trước cập nhật; luôn đối chiếu S16–S17, không coi phần đề xuất là đã triển khai. |
| S27 | [analysis_report.json — manual review](../../module2_headline_integrity/results/consistency/manual_annotation_analysis_v1/analysis_report.json); [comparison_summary.json — baseline A/B](../../module2_headline_integrity/results/consistency/baseline_a_b_comparison/comparison_summary.json) | Ch.3–4: bằng chứng phụ cho chính sách negative và so sánh baseline, không thay thế kết quả test cuối. |
| S28 | [test_error_analysis.json — M1](../../results/module1_phobert_mil_full/test_evaluation/test_error_analysis.json); [official_test_by_source.csv — M2](../../module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_by_source.csv); [official_test_by_category.csv — M2](../../module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_by_category.csv); [official_test_by_category.csv — M3](../../module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_by_category.csv); [official_test_negative_by_tier.csv — M3](../../module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_negative_by_tier.csv) | Ch.4: phân tích lỗi/phân nhóm từ kết quả lưu; phải kèm kích thước nhóm và hạn chế suy rộng. |

## E. Bảng và hình dự kiến, điều kiện chuyển sang viết đầy đủ

| Vị trí | Bảng/hình dự kiến | Nguồn | Điều kiện sử dụng |
|---|---|---|---|
| Ch.1 | Sơ đồ document → chunks → score → nhãn M1; sơ đồ hai nhánh M3 | S02, S12, S14 | Vẽ từ mã; không thêm tầng hay cơ chế attention chưa được chứng minh. |
| Ch.2 | Sơ đồ thành phần, sơ đồ tuần tự `/analyze`, bảng endpoint và đầu vào | S15–S17 | Bám mã hiện tại, tách nhánh PyVi; không thêm CSDL từ thiết kế cũ. |
| Ch.2 | Ảnh giao diện nhập liệu/kết quả | S17 làm căn cứ chức năng | Ảnh chụp thực tế phù hợp phiên bản: **[CHƯA ĐỦ DỮ LIỆU]**; không dựng ảnh giả kết quả chạy. |
| Ch.3 | Bảng dữ liệu, bảng cấu hình và đường cong huấn luyện đã lưu | S03, S04, S06, S07, S10–S12 | Phân biệt số trong report với số tự đếm; chỉ vẽ các epoch có dữ liệu. |
| Ch.4 | Bảng official test và confusion matrix riêng từng module | S05, S09, S13 | Ghi dataset, split, lớp dương, threshold; không tính “accuracy hệ thống” từ ba số. |
| Ch.4 | Bảng regression và bảng kiểm thử phần mềm | S19–S22 | Ghi sample count/dung sai và nguồn log; HTTP smoke chưa có kết quả thì giữ nhãn thiếu dữ liệu. |
| Ch.4 | Bảng drift PyVi; bảng profile tài nguyên | S23–S25 | Tách khỏi official test, ghi quy mô mẫu và cấp đo. |

Trước khi viết đầy đủ, đối chiếu từng con số với trường nguồn, giải quyết hoặc giữ nhãn các khoảng trống ở A.3, và xác định định dạng báo cáo của trường. Không cần chỉnh model, checkpoint, dataset, threshold hoặc source code để hoàn thành công việc biên soạn. Mọi file biên soạn tiếp theo chỉ nằm trong `docs/thesis/`.
