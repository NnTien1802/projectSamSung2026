# Đề cương báo cáo đồ án tốt nghiệp — phiên bản 2

Ngày khảo sát: 24/09/2026. Kế thừa [00_outline.md](00_outline.md); mở rộng bằng chứng sang project nghiên cứu gốc. Đây là kế hoạch viết, chưa phải nội dung các chương.

## A. Mục tiêu và nguyên tắc viết báo cáo

### A.1. Định hướng báo cáo

Báo cáo dự kiến **37 trang**, trình bày theo mạch: nền tảng học thuật → bài toán và kiến trúc hệ thống → xây dựng dữ liệu, baseline, huấn luyện và lựa chọn mô hình → kết quả nghiên cứu và kiểm thử triển khai → giới hạn và hướng phát triển. Chương 3–4 dành tổng cộng 18 trang. Nội dung truy vết file nằm chủ yếu trong bản đồ bằng chứng của đề cương, không biến thành mạch kể chính của báo cáo.

Hai gốc nguồn được khảo sát chỉ đọc:

- **V** = `C:/projectSamSung_v1`: nguồn ưu tiên cho hệ thống đang triển khai, release và kiểm thử tích hợp.
- **R** = `C:/1_projectSamSung`: nguồn bổ sung cho quá trình nghiên cứu, tạo dữ liệu, baseline, huấn luyện, chọn ngưỡng và official test.

Chỉ tạo file đề cương V2 trong `V/docs/thesis/`. Không sửa V1, source code, dataset, checkpoint, model, threshold hay release manifest; không thực thi script nghiên cứu, kiểm thử hoặc build ở bước này.

### A.2. Phạm vi kỹ thuật đã xác nhận

| Thành phần | Phạm vi sử dụng trong báo cáo | Nguồn |
|---|---|---|
| M1 | **Document-level classification**, PhoBERT MIL, nhãn tài liệu có/không có lỗi, input tham chiếu `article_segmented`, threshold **0.57**. Không định vị từ lỗi, không sửa lỗi. | S02–S05, S32 |
| M2 | Phát hiện clickbait bằng **mBERT title-only**, input tiêu đề, threshold **0.61**. | S07–S09, S34–S35 |
| M3 | Nhất quán tiêu đề–nội dung bằng **mBERT hybrid**, fusion nhánh description và MAX-MIL, threshold **0.45**. | S10–S14, S38–S40 |
| Runtime M2/M3 | Hai adapter có thể dùng chung một đối tượng pipeline; các nhánh có checkpoint riêng. Task 2A/2B trong thư mục lịch sử `module2_headline_integrity` tương ứng M2/M3 hiện tại. | S01, S14–S15, S18 |
| Ứng dụng | React frontend, FastAPI backend, service/registry/adapters; có cấu hình CUDA và báo cáo chạy GPU. Trả ba kết quả độc lập, không có điểm chất lượng báo chí tổng hợp. | S15–S18, S23 |
| PyVi | Nhánh tiền xử lý **experimental**, bật chủ động cho M1; không phải phân đoạn tham chiếu và không phải đóng góp nghiên cứu chính. | S16–S17, S24–S25 |

Project R còn chứa các nhánh `all_fields`, `word_detector` và correction. Không lấy kiến trúc, threshold hoặc metric của các nhánh này thay cho `module1_phobert_mil_full` được khóa trong V. Không đưa nhánh đạo văn cũ vào M3. Phạm vi release được xác định bởi S01, S18, không bởi việc một script có mặt trong project gốc.

### A.3. Quy tắc lập luận và trích nguồn

1. **Chương 1 dùng nguồn học thuật làm nền tảng.** Mã nội bộ chỉ minh họa cách project áp dụng lý thuyết. Nguồn học thuật còn thiếu được quản lý riêng tại G; không tự điền paper, tác giả, năm hoặc DOI.
2. **Chương 2 mô tả triển khai hiện có.** Không thêm SQLite, lịch sử phân tích, database hoặc route chỉ có trong bản thiết kế cũ S26.
3. **Chương 3 trình bày quá trình nghiên cứu.** Dùng script gốc để giải thích phương pháp, dùng run config/history để xác định lần chạy thực tế. Default trong script không tự động là cấu hình đã chạy; ví dụ batch của nhánh description phải lấy từ S11, không chỉ từ default tại S38.
4. **Chương 4 phân tách bảy nhóm:** official test; validation/model selection; regression; unit/API test; HTTP smoke; frontend build; thử nghiệm PyVi. Profile tài nguyên là nhóm số đo vận hành riêng.
5. Mọi bảng kết quả có dataset, split, sample count/đơn vị đo, threshold, positive class, loại metric và nguồn. Với build/unit test, các trường không có ý nghĩa ghi **“không áp dụng”**, không tự gán dataset hay lớp dương.
6. Regression PASS chỉ xác nhận mức tương đương được kiểm tra; HTTP smoke PASS chỉ xác nhận kịch bản gọi API; build PASS chỉ xác nhận biên dịch/đóng gói. Không chuyển chúng thành accuracy, không tạo “accuracy toàn hệ thống”.
7. Kết quả PyVi và dữ liệu demo không thay thế official test. Với M3, phải giữ cảnh báo negative tổng hợp và positive được giả định nhất quán; không gọi là benchmark thực tế được người xác minh đầy đủ.
8. Giữ số liệu gốc; nếu làm tròn phải ghi quy ước. Có script không đồng nghĩa đã chạy; có báo cáo lưu không đồng nghĩa đã tái chạy trong bước biên soạn này.

## B. Khoảng trống dữ liệu còn lại

### B.1. Những khoảng trống của V1 đã được thu hẹp

| Phát hiện mới | Ý nghĩa với V2 | Nguồn |
|---|---|---|
| Tìm thấy chuỗi tạo dữ liệu, validate, chia tập, baseline, train, tune threshold và evaluate M1 trong R. | Có thể giải thích phương pháp và vòng lặp huấn luyện; không còn ghi “thiếu script train M1” như khi chỉ đọc V. | S29–S32 |
| Có `train.jsonl`, `validation.jsonl`, `test.jsonl` M1 trong R. | Thiếu train/test là giới hạn bản tích hợp V, không phải kết luận về cả hai project. Chỉ xác nhận hiện diện, chưa kiểm định lại toàn bộ bản ghi. | S30 |
| Có script chuẩn bị dữ liệu, baseline, train, so sánh, khóa và official evaluation M2. | Có thể kể quá trình chọn title-only từ đối chứng title+lead trên validation. | S33–S35 |
| Có chuỗi xây cặp, kiểm tra dataset, train baseline/MAX-MIL, fusion, chuẩn bị test và evaluation M3. | Có thể mô tả nguồn gốc synthetic negatives, động cơ xử lý toàn bài và lựa chọn fusion. | S36–S40 |
| Có toàn văn train/validation/test M3 và checkpoint baseline article-head trong R. | Không lặp lại kết luận V1 rằng tài nguyên này chưa thấy ở toàn bộ phạm vi được phép đọc. Không khẳng định đã tái lập mọi thí nghiệm. | S38–S39 |
| `20_prepare_test_mil_chunk_cache.py` ghi final lock M3 trước bước chuẩn bị cache test. | Có nguồn mã cho bước khóa M3, không chỉ artifact final lock. | S40, S12 |
| Ba file official test của V và R trùng khớp SHA-256 khi đối chiếu. | Xác nhận đang dùng cùng bản báo cáo lưu cho ba module; không phải lần đánh giá test mới. | S05, S09, S13 |
| Có ba ảnh giao diện thật và tài liệu release mô tả nguồn ảnh. | Có thể lập placeholder hình giao diện từ ảnh có sẵn; kết quả trong ảnh là minh họa, không phải benchmark. | S42 |
| Có README và download manifest của dataset M2/M3. | Đã xác định được nguồn tải; citation học thuật vẫn cần hoàn thiện. README ViClickbait lưu trong R nói chưa có DOI, không suy diễn tình trạng công bố hiện tại ngoài repository. | S33, S41 |

### B.2. Các mục thiếu dữ liệu kỹ thuật/thực nghiệm

Mỗi mã Bxx dưới đây là **một mục thiếu dữ liệu riêng**. Các phần D–H dẫn lại mã, không nhân đôi số mục thiếu.

| Mã | Nội dung còn thiếu | Bằng chứng đã có và cách viết |
|---|---|---|
| B01 | **[CHƯA ĐỦ DỮ LIỆU]** — thuật toán và phiên bản phân đoạn upstream tạo đầu vào gốc của M1. | Generator bảo toàn biểu diễn segmented và tạo natural từ nó; audit không xác định được segmenter gốc. Không gọi PyVi là preprocessing chính thức. S24, S29. |
| B02 | **[CHƯA ĐỦ DỮ LIỆU]** — log chạy riêng đầy đủ của `05_validate_spelling_dataset.py` cho bộ hybrid cuối. | Có mã kiểm tra và summary sinh/chia dữ liệu; script validator in kết quả ra console. Mô tả những gì script kiểm tra, không tự ghi mọi check đã PASS. S29–S30. |
| B03 | **[CHƯA ĐỦ DỮ LIỆU]** — official test của các baseline cổ điển M1 để so sánh trực tiếp với official test PhoBERT. | Có `baseline_comparison.json` trên validation. Đưa baseline vào bảng validation cùng split; lưu ý baseline dùng `article_natural`, PhoBERT dùng `article_segmented`, không gọi đây là đối chứng chỉ thay kiến trúc. S04, S31. |
| B04 | **[CHƯA ĐỦ DỮ LIỆU]** — benchmark M3 trên tin thực tế có nhãn nhất quán được người xác minh đầy đủ. | Có synthetic test và manual review ứng viên TRAIN; chúng không thay thế human-gold test. S10, S13, S27, S36. |
| B05 | **[CHƯA ĐỦ DỮ LIỆU]** — file log HTTP smoke gốc lưu riêng để kiểm tra lại từng response. | S42 ghi nhận HTTP smoke PASS từ trao đổi và nói rõ chưa có file log riêng. Có thể trích là ghi nhận thứ cấp; không viết “đã kiểm tra log HTTP gốc”. S22, S42. |
| B06 | **[CHƯA ĐỦ DỮ LIỆU]** — độ biến thiên official metric qua nhiều seed/lần train, benchmark tải HTTP end-to-end và khảo sát người dùng. | Có so sánh thống kê validation M2, regression và profile adapter; không mở rộng thành chứng minh các nội dung còn thiếu. S08, S19–S24. |
| B07 | **[CHƯA ĐỦ DỮ LIỆU]** — tên đề tài được duyệt, mẫu trình bày và quy định trích dẫn của trường. | Sử dụng tên làm việc “Hệ thống hỗ trợ đánh giá bài báo tiếng Việt bằng các mô hình phân loại văn bản”; số trang ở C chỉ là kế hoạch. |

Thiếu nguồn học thuật được liệt kê riêng theo G01–G11 ở mục G. Chưa có khoảng trống “chỉ artifact, không có script train” đối với ba cấu hình chính sau khi mở rộng sang R; vẫn phải phân biệt mã phương pháp với bằng chứng một lần chạy cụ thể.

## C. Phân bổ số trang

Phân bổ trung tâm là **37 trang**, mục tiêu biên tập 35–38 trang, giới hạn cuối 30–40 trang. Số trang gồm bảng/hình; chưa tính bìa và các biểu mẫu ngoài tám phần yêu cầu. Khoảng trang ở từng mục D là biên độ biên tập, không cộng đồng thời tất cả mức tối đa.

| Phần | Trang kế hoạch | Khoảng điều chỉnh tham khảo |
|---|---:|---:|
| Mục lục | 1 | 1–1,5 |
| Đặt vấn đề | 2 | 2–2,5 |
| Chương 1: Cơ sở lý thuyết | 7 | 6,5–7,5 |
| Chương 2: Phân tích và thiết kế hệ thống | 5 | 4,5–5,5 |
| Chương 3: Huấn luyện mô hình và tích hợp hệ thống | 10 | 9–10 |
| Chương 4: Kết quả, kiểm thử và đánh giá hệ thống | 8 | 7,5–8,5 |
| Kết luận và hướng phát triển | 2 | 1,5–2 |
| Tài liệu tham khảo | 2 | 1,5–2,5 |
| **Tổng kế hoạch** | **37** | **Ưu tiên giữ 35–38** |

Chương 1 không kéo dài bằng mô tả code khi thiếu tài liệu học thuật. Chương 2 dùng sơ đồ/bảng ngắn để nhường dung lượng cho phương pháp và đánh giá. Phần PyVi dành khoảng 0,25 trang ở 3.8 và 0,35 trang ở 4.10; chỉ nhắc thêm khi cần trong giao diện và hạn chế.

## D. Đề cương chi tiết đến cấp 3

Mỗi mục lớn ghi mục tiêu, nguồn chính, loại bằng chứng và dung lượng; các mục cấp 3 là nội dung dự kiến viết. Sxx tra tại E; Gxx tra tại G. Các nguồn Sxx ở Chương 1 chỉ dùng cho liên hệ ứng dụng, không thay tài liệu học thuật.

### Mục lục

**Mục tiêu:** giúp theo dõi tám phần và đề mục tới cấp 3. **Nội dung:** mục lục tự động theo cấu trúc dưới đây, cập nhật sau dàn trang. **Nguồn:** cấu trúc được yêu cầu, đề cương này. **Loại:** biên tập. **Trang:** 1 (khoảng 1–1,5).

### Đặt vấn đề

**Mục tiêu:** xác lập vấn đề, mục tiêu và phạm vi nghiên cứu. **Nguồn:** S01, S05, S09, S13, S16–S17; nguồn học thuật bối cảnh từ G08–G10 khi hoàn thiện. **Loại:** bài toán ứng dụng và bằng chứng nghiên cứu. **Trang:** 2 (khoảng 2–2,5).

- **ĐV.1. Bối cảnh và vấn đề nghiên cứu.**
  - **ĐV.1.1.** Nhu cầu hỗ trợ phân tích văn bản báo chí qua lỗi tài liệu, clickbait và nhất quán; không thêm tỷ lệ phổ biến lỗi chưa có nguồn.
  - **ĐV.1.2.** Phân biệt ba tác vụ và giới hạn so với xác minh sự thật hoặc đánh giá chất lượng báo chí toàn diện.
- **ĐV.2. Mục tiêu, phạm vi và phương pháp.**
  - **ĐV.2.1.** Xây dựng/đánh giá các mô hình theo từng tác vụ, tích hợp thành ứng dụng trả kết quả riêng.
  - **ĐV.2.2.** Phạm vi dữ liệu, đối chứng và kiểm thử; phân biệt nghiên cứu chính thức với nhánh triển khai thử nghiệm.
- **ĐV.3. Đóng góp trong phạm vi chứng minh được và bố cục.**
  - **ĐV.3.1.** Tổ chức dữ liệu, thực nghiệm lựa chọn mô hình và tích hợp; không tự nhận thuật toán mới.
  - **ĐV.3.2.** Vai trò của bốn chương và cách dẫn sang phần kết luận.

### Chương 1. Cơ sở lý thuyết — 7 trang

**Mục tiêu chương:** xây dựng nền tảng giải thích thiết kế và đánh giá AI/ML. **Loại bằng chứng chính:** tài liệu học thuật, hiện còn các mục G01–G07; phải bổ sung trước khi viết khẳng định lý thuyết. Không lấy thân hàm Python làm bố cục chương.

#### 1.1. Bài toán phân loại văn bản

**Mục tiêu:** định nghĩa bài toán và đơn vị dự đoán. **Nguồn chính:** G07–G10; liên hệ project S02, S09–S10. **Loại:** lý thuyết và định nghĩa tác vụ. **Trang:** 0,7 (0,6–0,8).

- **1.1.1.** Phân loại nhị phân, mẫu đầu vào, nhãn và score; phân biệt document classification với sequence labeling/correction.
- **1.1.2.** Phân loại một văn bản và phân loại quan hệ giữa hai phần văn bản; liên hệ M1, M2 và M3.

#### 1.2. Transformer

**Mục tiêu:** giải thích cơ chế nền cho encoder được dùng. **Nguồn chính:** G01. **Loại:** học thuật, đang thiếu nguồn đã kiểm chứng trong repository. **Trang:** 1,3 (1,1–1,4).

- **1.2.1.** Biểu diễn token, thông tin vị trí và self-attention; dự kiến giải thích bằng ký hiệu sau khi có nguồn.
- **1.2.2.** Multi-head attention, feed-forward, residual và normalization trong khối Transformer; chỉ trình bày mức cần thiết cho báo cáo.

#### 1.3. BERT và multilingual BERT

**Mục tiêu:** nối nền tảng Transformer với fine-tuning phân loại. **Nguồn chính:** G02–G03; liên hệ S07, S11, S14. **Loại:** học thuật; code chỉ xác nhận model mà project sử dụng. **Trang:** 1,2 (1,1–1,3).

- **1.3.1.** Tiền huấn luyện và fine-tuning BERT, biểu diễn đầu vào đơn/cặp và đầu phân loại.
- **1.3.2.** mBERT, biểu diễn đa ngôn ngữ và phạm vi áp dụng; không tự điền số ngôn ngữ, corpus hoặc ưu thế định lượng.

#### 1.4. PhoBERT

**Mục tiêu:** giải thích lựa chọn encoder tiếng Việt và yêu cầu biểu diễn đầu vào. **Nguồn chính:** G04; liên hệ S02, S04, S29, S32. **Loại:** học thuật và ứng dụng. **Trang:** 1 (0,9–1,1).

- **1.4.1.** Nền tảng PhoBERT và biểu diễn tiếng Việt theo tài liệu gốc cần bổ sung.
- **1.4.2.** Phân đoạn từ so với token hóa subword; liên hệ `vinai/phobert-base-v2` và `article_segmented`, giữ khoảng trống B01.

#### 1.5. Multiple Instance Learning

**Mục tiêu:** giải thích học với nhãn tài liệu và nhiều đoạn đầu vào. **Nguồn chính:** G05; liên hệ S02, S14, S32, S39. **Loại:** học thuật và phương pháp áp dụng. **Trang:** 1 (0,9–1,1).

- **1.5.1.** Khái niệm bag, instance, nhãn bag và tổng hợp thông tin.
- **1.5.2.** Max pooling và cách áp dụng trong M1/M3; nêu rõ bag score không tự tạo bằng chứng định vị từ lỗi.

#### 1.6. Xử lý văn bản dài bằng chunking

**Mục tiêu:** giải thích giới hạn độ dài và lựa chọn đoạn. **Nguồn chính:** G06; liên hệ S02, S12, S14, S39. **Loại:** học thuật và thiết kế phương pháp. **Trang:** 0,7 (0,6–0,8).

- **1.6.1.** Chunk, overlap, bước dịch và chọn vị trí trải đều; phân biệt tham số overlap với tên `stride` trong từng triển khai.
- **1.6.2.** Độ phủ đầu vào và giới hạn số chunk; không mặc định tên “full article” có nghĩa mọi token đều được encoder xử lý.

#### 1.7. Các chỉ số đánh giá mô hình

**Mục tiêu:** chuẩn hóa cách đọc kết quả Chương 4. **Nguồn chính:** G07; liên hệ S05, S09, S13. **Loại:** học thuật; artifact xác nhận tập chỉ số được dùng. **Trang:** 1,1 (1–1,2).

- **1.7.1.** Confusion matrix, accuracy, precision, recall, F1 và macro-F1; định nghĩa positive class theo từng tác vụ.
- **1.7.2.** ROC-AUC và average precision, vai trò của score và threshold; không đồng nhất average precision với mọi cách tính diện tích đường precision–recall.
- **1.7.3.** Tách train/validation/test, lựa chọn mô hình và ngưỡng; khác biệt giữa đánh giá phân loại và kiểm thử phần mềm.

### Chương 2. Phân tích và thiết kế hệ thống — 5 trang

#### 2.1. Phân tích yêu cầu

**Mục tiêu:** xác định tác nhân, dữ liệu và kết quả mong đợi. **Nguồn chính:** S01, S16–S17. **Loại:** triển khai. **Trang:** 0,5 (0,4–0,6).

- **2.1.1.** Yêu cầu chức năng: nhập tiêu đề/mô tả/nội dung, lựa chọn đầu vào M1, nhận ba kết quả riêng.
- **2.1.2.** Ràng buộc đầu vào, trạng thái xử lý và cảnh báo; các yêu cầu ngoài phạm vi release.

#### 2.2. Kiến trúc tổng thể

**Mục tiêu:** thể hiện trách nhiệm các lớp đã triển khai. **Nguồn chính:** S14–S18. **Loại:** triển khai. **Trang:** 0,7 (0,6–0,8).

- **2.2.1.** React → FastAPI → service → registry/adapters → mô hình.
- **2.2.2.** Ba module logic, pipeline M2/M3 dùng chung và các checkpoint riêng; không có database/lịch sử phân tích trong sơ đồ hiện tại.

#### 2.3. Thiết kế Module 1

**Mục tiêu:** mô tả pipeline phân loại tài liệu. **Nguồn chính:** S02, S04, S18. **Loại:** mô hình được triển khai. **Trang:** 0,6 (0,5–0,7).

- **2.3.1.** `article_segmented` → tokenizer → chọn chunk → PhoBERT → max pooling → score tài liệu.
- **2.3.2.** Nhãn, threshold 0.57, cảnh báo không sửa/định vị lỗi và giới hạn độ phủ chunk.

#### 2.4. Thiết kế Module 2

**Mục tiêu:** mô tả đường suy luận title-only. **Nguồn chính:** S07, S09, S14–S15. **Loại:** mô hình được triển khai. **Trang:** 0,5 (0,4–0,6).

- **2.4.1.** Tiêu đề → tokenizer mBERT → bộ phân loại clickbait.
- **2.4.2.** Positive class, score, threshold 0.61 và đầu ra độc lập với M3.

#### 2.5. Thiết kế Module 3

**Mục tiêu:** mô tả hai nhánh và fusion. **Nguồn chính:** S11–S15. **Loại:** mô hình được triển khai. **Trang:** 0,8 (0,7–0,9).

- **2.5.1.** Cặp tiêu đề–description và cặp tiêu đề–chunk nội dung; MAX-MIL và chính sách chọn chunk.
- **2.5.2.** Fusion score hai nhánh, threshold 0.45, ý nghĩa lớp `CONSISTENT` và cảnh báo thiếu description.

#### 2.6. Thiết kế Backend FastAPI

**Mục tiêu:** giải thích contract và điều phối suy luận. **Nguồn chính:** S15–S16, S19. **Loại:** triển khai và đặc tả kiểm thử. **Trang:** 0,7 (0,6–0,8).

- **2.6.1.** Schema, endpoint thật, lifespan/readiness và registry; không dùng route `/api/v1/analyses` của thiết kế cũ.
- **2.6.2.** Khóa xử lý, validation và lỗi; mô tả đúng việc exception có thể làm cả `/analyze` thất bại, không hứa giữ kết quả từng module khi chưa có triển khai đó.

#### 2.7. Thiết kế Frontend React

**Mục tiêu:** mô tả trải nghiệm nhập và đọc kết quả. **Nguồn chính:** S17, S42. **Loại:** triển khai và ảnh minh họa có sẵn. **Trang:** 0,7 (0,6–0,8).

- **2.7.1.** Biểu mẫu, trạng thái kết nối, kiểm tra đầu vào và gửi request.
- **2.7.2.** Ba thẻ kết quả, nhãn/score/ngưỡng/cảnh báo; ảnh kết quả minh họa có PyVi phải giữ chú thích experimental.

#### 2.8. Luồng xử lý tổng thể

**Mục tiêu:** nối giao diện với các đường xử lý backend. **Nguồn chính:** S15–S17, S25. **Loại:** triển khai. **Trang:** 0,5 (0,4–0,6).

- **2.8.1.** Luồng nhập → kiểm tra → gọi API → điều phối → trả và hiển thị kết quả.
- **2.8.2.** Các trường hợp M1 presegmented/skipped/experimental và xử lý lỗi.

### Chương 3. Huấn luyện mô hình và tích hợp hệ thống — 10 trang

Mạch nghiên cứu dự kiến: **nguồn dữ liệu → cách xây bài toán/nhãn → baseline → mô hình ứng viên → lựa chọn trên validation → khóa cấu hình → chuẩn bị official test → tích hợp**. Chương này giải thích quyết định; Chương 4 trình bày số liệu đánh giá. Không trộn nhánh nghiên cứu ngoài release vào cấu hình chính.

#### 3.1. Môi trường thực nghiệm

**Mục tiêu:** nêu điều kiện tái lập trong phạm vi nguồn ghi nhận. **Nguồn chính:** S04, S07, S11, S18, S23, S32, S34, S39. **Loại:** cấu hình lần chạy và mã huấn luyện. **Trang:** 0,6 (0,5–0,7).

- **3.1.1.** Phần cứng GPU, device và môi trường thư viện theo từng nguồn; không dùng environment release để khẳng định mọi lần train có cùng phiên bản.
- **3.1.2.** Seed, batch/effective batch, mixed precision, optimizer/scheduler theo script và run config; ưu tiên cấu hình được lưu của lần chạy.

#### 3.2. Dữ liệu và tiền xử lý

**Mục tiêu:** trình bày nguyên tắc chung trước các pipeline cụ thể. **Nguồn chính:** S03, S06, S10, S29–S30, S33, S36–S37, S41. **Loại:** nghiên cứu dữ liệu. **Trang:** 0,8 (0,7–0,9).

- **3.2.1.** Nguồn corpus và vai trò từng dataset; tách VietNews của M1 với VietOnlineNews của M3, dataset gốc với dataset dẫn xuất.
- **3.2.2.** Đơn vị chia tập, chống trùng/overlap, khóa test và quy ước nhãn; không mặc định ba tác vụ dùng chung dataset.
- **3.2.3.** Bảng tổng quan split và đơn vị mẫu; phân biệt số tài liệu, số tiêu đề, số anchor và số cặp.

#### 3.3. Module 1

**Mục tiêu:** giải thích từ dữ liệu lỗi đến PhoBERT MIL được chọn. **Nguồn chính:** S03–S05, S29–S32. **Loại:** script nghiên cứu + artifact thực nghiệm. **Trang:** 2,4 (2,2–2,6).

- **3.3.1. Dataset.** Nguồn VietNews, khai thác mẫu lỗi, trusted lookup và sinh clean/noisy hybrid; dedup theo source và chia nhóm. Đối chiếu phạm vi lỗi title/abstract/article của generator với đầu vào article của mô hình, không mặc định mọi lỗi nhãn dương đều nằm trong article. Nguồn: S29–S30.
- **3.3.2. Tiền xử lý.** Bảo toàn segmentation, biểu diễn natural, chuẩn hóa/kiểm tra dữ liệu và collator chunk; phân biệt mã validate với log chạy còn thiếu B02. Nguồn: S02, S24, S29–S30.
- **3.3.3. Kiến trúc PhoBERT MIL.** Encoder, biểu diễn token đầu, dropout, tuyến tính, max logit theo tài liệu; đối chiếu giới hạn chunk với phạm vi phát hiện. Nguồn: S02, S32.
- **3.3.4. Huấn luyện.** Điểm xuất phát word TF-IDF + logistic regression và char TF-IDF + LinearSVC trên validation; chuyển sang PhoBERT MIL. Giải thích loss, optimizer, freeze layers, scheduler và chọn checkpoint theo script, minh họa lịch sử đã lưu. Nguồn: S04, S31–S32.
- **3.3.5. Lựa chọn threshold.** `15_tune_phobert_threshold.py`, tiêu chí validation macro-F1 và threshold cuối 0.57; không dùng test tìm ngưỡng. Nguồn: S04, S32.
- **3.3.6. Khóa mô hình.** Checkpoint, tokenizer, cấu hình chunk, threshold và release; script official evaluation đọc cấu hình khóa. Không nêu có script khóa M1 chuyên biệt nếu chưa có nguồn. Nguồn: S02, S04–S05, S18, S32.

#### 3.4. Module 2

**Mục tiêu:** kể quá trình chọn mô hình clickbait và đầu vào title-only. **Nguồn chính:** S06–S09, S33–S35. **Loại:** script nghiên cứu + validation/model selection + cấu hình khóa. **Trang:** 2 (1,8–2,2).

- **3.4.1. Dataset.** ViClickbait, nhãn và phân bố train/validation/test từ prepare report; nguồn tải/citation tại S33 và G09.
- **3.4.2. Tiền xử lý.** Chuẩn hóa, nhóm theo normalized title, stratification, overlap và biểu diễn title/title+lead trong hai thí nghiệm. Nguồn: S06, S33–S34.
- **3.4.3. Kiến trúc.** Baseline majority/source/surface/TF-IDF làm đối chứng; mBERT title-only và biến thể title+lead. Nguồn: S08, S34.
- **3.4.4. Huấn luyện.** Class weights, weighted cross-entropy, AdamW, lịch sử epoch và tiêu chí chọn checkpoint; phân biệt số epoch cấu hình với best epoch. Nguồn: S07, S34.
- **3.4.5. Lựa chọn mô hình và threshold.** So sánh validation, McNemar/bootstrap nếu dùng đúng báo cáo; lý do giữ title-only và ngưỡng 0.61, không tuyên bố vượt trội phổ quát. Nguồn: S08, S35.
- **3.4.6. Khóa mô hình.** Script `09_lock_final_clickbait_model.py`, checkpoint/hash, cấu hình khóa và chuẩn bị official evaluation. Nguồn: S09, S35.

#### 3.5. Module 3

**Mục tiêu:** giải thích từ xây cặp nhất quán đến mô hình hybrid. **Nguồn chính:** S10–S13, S27, S36–S40. **Loại:** nghiên cứu dữ liệu + huấn luyện + model selection. **Trang:** 2,5 (2,3–2,7).

- **3.5.1. Dataset.** VietOnlineNews → articles → positive pairs → candidate negatives → manual review TRAIN → negative policy → dataset cuối; không coi ViANLI được tải là bằng chứng nó được dùng huấn luyện mô hình cuối. Nguồn: S33, S36–S37, S41.
- **3.5.2. Tiền xử lý.** Chuẩn hóa/join title–description–content, kiểm tra cặp và split; cache chunk, chọn vị trí theo chính sách đã khóa. Nguồn: S10, S36–S37, S39.
- **3.5.3. Kiến trúc.** Baseline A title–description và B title–article-head; nhánh MAX-MIL khởi tạo từ B; hybrid kết hợp description và MAX-MIL. Nguồn: S11–S14, S38–S39.
- **3.5.4. Huấn luyện.** Cấu hình từng nhánh, chênh lệch logit, max pooling, BCE của MAX-MIL, effective batch và lịch sử chạy; không dùng một bộ tham số đại diện cho mọi nhánh. Nguồn: S11, S38–S39.
- **3.5.5. Lựa chọn fusion và threshold.** Động cơ kết hợp từ phân tích sai khác lỗi; calibration/holdout bên trong validation; trọng số cuối hai nhánh và ngưỡng 0.45. Oracle complementarity nếu nhắc chỉ là chẩn đoán, không là score mô hình triển khai. Nguồn: S12, S27, S40.
- **3.5.6. Khóa mô hình.** `20_prepare_test_mil_chunk_cache.py` ghi lock trước tiền xử lý test; `21_evaluate_final_hybrid_test.py` kiểm tra lock; không tune lại trên test. Nguồn: S12–S13, S40.

#### 3.6. Tích hợp FastAPI

**Mục tiêu:** chuyển pipeline nghiên cứu thành API có contract rõ ràng. **Nguồn chính:** S02, S14–S16, S18. **Loại:** triển khai. **Trang:** 0,9 (0,8–1).

- **3.6.1.** Tái sử dụng collator/model M1 và pipeline M2/M3 qua adapters/registry, giữ checkpoint/ngưỡng.
- **3.6.2.** Schema, service, nạp model, điều phối GPU và xử lý lỗi; chuẩn bị các lớp kiểm thử ở Chương 4.

#### 3.7. Tích hợp React

**Mục tiêu:** nối form và phần hiển thị với API thật. **Nguồn chính:** S17, S19, S42. **Loại:** triển khai và bằng chứng build. **Trang:** 0,55 (0,45–0,65).

- **3.7.1.** Client API, mapping field, trạng thái chờ/thành công/lỗi và cấu hình proxy.
- **3.7.2.** Hiển thị ba kết quả, threshold, cảnh báo và demo; build không chứng minh toàn bộ luồng trình duyệt đã được kiểm thử.

#### 3.8. Xử lý đầu vào M1 trong ứng dụng

**Mục tiêu:** xử lý giới hạn segmentation khi triển khai. **Nguồn chính:** S16–S17, S24–S25. **Loại:** triển khai experimental. **Trang:** 0,25 (0,2–0,3).

- **3.8.1.** Chế độ presegmented tham chiếu và skipped mặc định khi thiếu đầu vào phù hợp.
- **3.8.2.** PyVi opt-in, kiểm tra surface/join và cảnh báo; chuyển khảo sát drift sang 4.10, không trình bày thành mô hình nghiên cứu thứ tư.

### Chương 4. Kết quả, kiểm thử và đánh giá hệ thống — 8 trang

#### 4.1. Thiết lập đánh giá

**Mục tiêu:** xác định protocol và cách đọc các bảng. **Nguồn chính:** S03–S13, S19–S24; bảng quy ước E.3. **Loại:** protocol nghiên cứu/kiểm thử. **Trang:** 0,6 (0,5–0,7).

- **4.1.1.** Dataset, split, sample count, positive class, threshold và cấu hình khóa cho từng official test.
- **4.1.2.** Phân biệt validation/model selection, official test, regression và kiểm thử ứng dụng; nguồn môi trường và giới hạn tái kiểm chứng.

#### 4.2. Kết quả Module 1

**Mục tiêu:** đánh giá phân loại lỗi cấp tài liệu. **Nguồn chính:** S05, S28. **Loại:** official test. **Trang:** 1,1 (1–1,2).

- **4.2.1.** Các metric và confusion matrix trên test VN-NewsSpell tại ngưỡng 0.57.
- **4.2.2.** Phân tích theo severity/nhóm lỗi trong artifact; liên hệ FN/FP với giới hạn đầu vào/chunk ở mức bằng chứng cho phép, không tự suy ra nguyên nhân nhân quả.

#### 4.3. Kết quả Module 2

**Mục tiêu:** đánh giá clickbait title-only. **Nguồn chính:** S09, S28. **Loại:** official test. **Trang:** 0,95 (0,85–1,05).

- **4.3.1.** Metric và confusion matrix trên test ViClickbait tại ngưỡng 0.61.
- **4.3.2.** Phân tích theo source/category, kèm kích thước nhóm; không suy rộng từ nhóm ít mẫu.

#### 4.4. Kết quả Module 3

**Mục tiêu:** đánh giá hybrid trong protocol synthetic consistency. **Nguồn chính:** S10, S13, S28. **Loại:** official test. **Trang:** 1,2 (1,1–1,3).

- **4.4.1.** Metric, confusion matrix và kết quả fusion tại threshold 0.45; lớp dương là `CONSISTENT`.
- **4.4.2.** Kết quả theo category/negative tier; các metric nhánh ở `branch_reference_metrics_fixed_at_final_threshold` phải ghi rõ cùng ngưỡng cuối, không tráo với ngưỡng tune validation.
- **4.4.3.** Ý nghĩa và giới hạn của synthetic negatives/presumed-consistent positives; giữ B04.

#### 4.5. So sánh với baseline

**Mục tiêu:** giải thích giá trị thực nghiệm của lựa chọn mô hình trong từng tác vụ. **Nguồn chính:** S04, S07–S08, S11–S12, S27, S31, S34–S40. **Loại:** validation/model selection. **Trang:** 1,15 (1–1,3).

- **4.5.1.** M1: baseline cổ điển và PhoBERT trên validation; ghi khác biệt natural/segmented và ngưỡng mặc định/tuned. Không dùng baseline validation để tính cải thiện so với official test.
- **4.5.2.** M2: diagnostic baselines, title-only/title+lead và thống kê so sánh đúng báo cáo validation.
- **4.5.3.** M3: A/B/MAX-MIL trên validation và fusion trên calibration/holdout; tách bảng theo protocol, không xếp các tập khác nhau thành một bảng xếp hạng đồng nhất.

#### 4.6. Regression test

**Mục tiêu:** xác nhận tính tương đương với reference khi tích hợp. **Nguồn chính:** S20–S21. **Loại:** regression. **Trang:** 0,6 (0,5–0,7).

- **4.6.1.** M1: sai lệch xác suất, nhãn/chunk mismatch, mẫu cố định và dung sai.
- **4.6.2.** M2/M3: parity các nhánh và fusion, token/chunk mismatch; PASS không phải accuracy.

#### 4.7. Unit/API test

**Mục tiêu:** đánh giá contract và điều phối phần mềm. **Nguồn chính:** S19, S25. **Loại:** unit/API test với fake/mock. **Trang:** 0,4 (0,3–0,5).

- **4.7.1.** Kiểm thử adapter, schema, đầu vào không hợp lệ, lock và lỗi.
- **4.7.2.** Kiểm thử nhánh experimental bằng thành phần tiêm vào; kết quả lưu không phải đo accuracy của checkpoint/PyVi thực.

#### 4.8. HTTP smoke test

**Mục tiêu:** mô tả kiểm tra API đang chạy với checkpoint thật. **Nguồn chính:** S22, S42. **Loại:** HTTP smoke; ghi nhận kết quả thứ cấp, thiếu log gốc B05. **Trang:** 0,25 (0,2–0,3).

- **4.8.1.** Kịch bản gọi health/analyze hoặc spelling/raw, kiểm tra status/threshold/cảnh báo.
- **4.8.2.** Phân biệt script, ghi nhận PASS trong release README và file log gốc; không gọi routing smoke là numerical regression hay benchmark.

#### 4.9. Frontend build / kiểm thử tích hợp

**Mục tiêu:** đánh giá phần giao diện và phạm vi tích hợp có bằng chứng. **Nguồn chính:** S17, S19, S42. **Loại:** frontend build và minh họa tích hợp. **Trang:** 0,3 (0,25–0,4).

- **4.9.1.** Kết quả `tsc -b && vite build` trong log, nêu phạm vi compile/bundle.
- **4.9.2.** Ảnh nhập liệu/kết quả và nối contract frontend–backend; không nâng ảnh demo thành bộ kiểm thử E2E có coverage đầy đủ.

#### 4.10. Thử nghiệm PyVi

**Mục tiêu:** khảo sát ảnh hưởng thay preprocessing khi triển khai M1. **Nguồn chính:** S24–S25. **Loại:** experimental trên mẫu validation. **Trang:** 0,35 (0,3–0,4).

- **4.10.1.** Đối chiếu segmentation tham chiếu với PyVi trên mẫu đã lưu; số đổi nhãn, độ lệch score và chunk.
- **4.10.2.** Giới hạn suy luận: tỷ lệ giữ nguyên nhãn không phải accuracy; không dùng metric official M1 đại diện cho chế độ này.

#### 4.11. Đánh giá tài nguyên

**Mục tiêu:** mô tả khả năng chạy các adapter trong điều kiện được đo. **Nguồn chính:** S18, S23. **Loại:** profile tài nguyên. **Trang:** 0,45 (0,4–0,55).

- **4.11.1.** GPU, RAM/VRAM, thời gian nạp và latency adapter; ghi chế độ chạy, số lặp và mẫu short/long theo script/report.
- **4.11.2.** Phân biệt latency adapter với HTTP end-to-end, warm-up, phạm vi bộ nhớ và hạn chế suy rộng tải nhiều người dùng.

#### 4.12. Hạn chế của hệ thống

**Mục tiêu:** đánh giá độ tin cậy và phạm vi khái quát của kết quả. **Nguồn chính:** S03, S08, S10, S13, S24, S29, S31; các khoảng trống B. **Loại:** thảo luận dựa trên bằng chứng. **Trang:** 0,65 (0,6–0,75).

- **4.12.1.** Lỗi sinh/tổng hợp, phạm vi nhãn và input M1; shortcut dataset M2; nhãn giả định/synthetic M3; giới hạn chunk.
- **4.12.2.** Segmentation upstream chưa xác định, thiếu benchmark thực tế, nhiều seed và kiểm thử vận hành; không biến giả thuyết nguyên nhân thành kết luận đã kiểm chứng.

### Kết luận và hướng phát triển

**Mục tiêu:** trả lời mục tiêu đề tài và chỉ ra công việc tiếp theo. **Nguồn chính:** các kết quả đã xác nhận ở S05, S09, S13, S19–S24 và hạn chế B. **Loại:** tổng hợp nghiên cứu/triển khai và đề xuất tương lai. **Trang:** 2 (1,5–2).

- **KL.1. Kết luận.**
  - **KL.1.1.** Tổng kết kết quả từng tác vụ và các quyết định mô hình có bằng chứng.
  - **KL.1.2.** Giá trị của tích hợp ứng dụng; giữ ranh giới giữa chất lượng dự đoán và độ ổn định phần mềm.
- **KL.2. Hướng phát triển.**
  - **KL.2.1.** Bổ sung nhãn thực tế, xác minh segmentation, đánh giá nhiều seed và phân tích lỗi sâu hơn.
  - **KL.2.2.** Đánh giá tải HTTP, trải nghiệm người dùng và hoàn thiện vận hành; ghi là đề xuất, không là tính năng đã có.

### Tài liệu tham khảo

**Mục tiêu:** lập bibliography học thuật đúng nguồn. **Nguồn chính:** các nguồn cần bổ sung G01–G11; S33/S41 chỉ giúp nhận diện nguồn dataset. **Loại:** học thuật/dataset/documentation. **Trang:** 2 (1,5–2,5).

- **TK.1. Nguồn nền tảng và phương pháp.**
  - **TK.1.1.** Transformer, BERT/mBERT, PhoBERT, MIL và xử lý văn bản dài.
  - **TK.1.2.** Metric và phương pháp đánh giá/lựa chọn mô hình.
- **TK.2. Dataset và công cụ.**
  - **TK.2.1.** Nguồn dataset gốc và mô tả dataset dẫn xuất, không gán DOI/citation chưa có.
  - **TK.2.2.** Documentation công cụ cần thiết; nguồn code/artifact nội bộ được dẫn để truy vết, không thay bibliography học thuật chính.

## E. Bảng ánh xạ nguồn bằng chứng

### E.1. Quy ước và nguồn nghiên cứu

Giữ các mã **S01–S28 theo nhóm chủ đề của V1**, bổ sung S29–S42 cho khảo sát mở rộng. Các liên kết có đường dẫn đầy đủ hoặc tương đối đến file thật. “Đã khảo sát” nghĩa đã đọc nội dung liên quan bằng đọc tĩnh/truy vấn, không thực thi script. File dữ liệu lớn/checkpoint được kiểm tra hiện diện; không nạp checkpoint hay đọc lại toàn bộ dataset.

| Mã | File nguồn chính | Dùng cho / loại bằng chứng |
|---|---|---|
| S01 | [README V1](../../README_V1.md); [báo cáo tạo V1](../../PROJECT_V1_CREATE_REPORT.json) | Đặt vấn đề, Ch.2–3: phạm vi release và nguồn sao chép. |
| S02 | [evaluation M1 trong V](../../scripts/16_evaluate_phobert_test.py); [adapter M1](../../system/backend/app/adapters/spelling.py) | Ch.2–3: kiến trúc/collator và đường suy luận tham chiếu. |
| S03 | [split summary M1](../../data/processed/vn_newsspell_v1/split_summary.json); [validation M1 trong V](../../data/processed/vn_newsspell_v1/validation.jsonl) | Ch.3–4: thống kê chia tập được lưu; dữ liệu tích hợp. |
| S04 | [training config M1](../../results/module1_phobert_mil_full/training_config.json); [history M1](../../results/module1_phobert_mil_full/training_history.json); [best validation](../../results/module1_phobert_mil_full/best_validation_metrics.json); [best threshold](../../results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) | Ch.3, 4.5: cấu hình và validation/model selection M1. |
| S05 | [official test M1 — V](../../results/module1_phobert_mil_full/test_evaluation/official_test_metrics.json); [official test M1 — R](/C:/1_projectSamSung/results/module1_phobert_mil_full/test_evaluation/official_test_metrics.json) | 4.2: official test; đã đọc và đối chiếu hash hai bản. |
| S06 | [prepare report M2](../../module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) | Ch.3–4: dataset, split, nhãn và overlap được ghi nhận. |
| S07 | [run config title-only](../../module2_headline_integrity/results/clickbait/mbert_title_only_v1/run_config.json); [training history](../../module2_headline_integrity/results/clickbait/mbert_title_only_v1/training_history.csv); [validation metrics](../../module2_headline_integrity/results/clickbait/mbert_title_only_v1/validation_metrics.json) | 3.4, 4.5: cấu hình lần chạy và validation M2. |
| S08 | [title-only/title+lead comparison](../../module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json); [diagnostic baselines](../../module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json); [experiment card M2](../../module2_headline_integrity/results/clickbait/final_title_only_v1/experiment_summary/FINAL_EXPERIMENT_CARD.md) | 3.4, 4.5, 4.12: model selection và giới hạn dataset M2. |
| S09 | [lock M2](../../module2_headline_integrity/results/clickbait/final_title_only_v1/final_model_lock.json); [official test M2 — V](../../module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_metrics.json); [official test M2 — R](/C:/1_projectSamSung/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_metrics.json) | 3.4, 4.3: cấu hình khóa và official test; hai bản metric đã đối chiếu hash. |
| S10 | [dataset manifest M3](../../module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json); [dataset validation report](../../module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) | 3.2, 3.5, 4.4: nhãn, synthetic negative policy, split và kiểm tra cấu trúc. |
| S11 | [config description](../../module2_headline_integrity/results/consistency/baseline_title_description_mbert/run_config.json); [config article-head](../../module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/run_config.json); [config MAX-MIL](../../module2_headline_integrity/results/consistency/full_article_max_mil_mbert/run_config.json); [history MAX-MIL](../../module2_headline_integrity/results/consistency/full_article_max_mil_mbert/training_history.csv) | 3.1, 3.5: tham số lần chạy từng nhánh, khác với default CLI nếu có. |
| S12 | [fusion report](../../module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json); [final lock M3](../../module2_headline_integrity/results/consistency/final_hybrid_v1/final_model_lock.json) | 3.5, 4.5: calibration/holdout, trọng số và threshold cuối. |
| S13 | [official test M3 — V](../../module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json); [official test M3 — R](/C:/1_projectSamSung/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) | 4.4: official fusion, nhánh tham chiếu và cảnh báo phương pháp; đã đối chiếu hash. |
| S27 | [manual annotation analysis](../../module2_headline_integrity/results/consistency/manual_annotation_analysis_v1/analysis_report.json); [baseline A/B comparison](../../module2_headline_integrity/results/consistency/baseline_a_b_comparison/comparison_summary.json) | 3.5, 4.5: lựa chọn negative và chẩn đoán baseline; oracle không là score triển khai. |
| S28 | [M1 test error analysis](../../results/module1_phobert_mil_full/test_evaluation/test_error_analysis.json); [M2 test by source](../../module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_by_source.csv); [M2 test by category](../../module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_by_category.csv); [M3 test by category](../../module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_by_category.csv); [M3 negative by tier](../../module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_negative_by_tier.csv) | 4.2–4.4: nguồn chi tiết phân tích lỗi/phân nhóm; không tự bịa ví dụ hoặc nguyên nhân. |
| S29 | [01_download_raw.py](/C:/1_projectSamSung/scripts/01_download_raw.py); [06_mine_real_error_patterns.py](/C:/1_projectSamSung/scripts/06_mine_real_error_patterns.py); [08_build_trusted_error_lookup.py](/C:/1_projectSamSung/scripts/08_build_trusted_error_lookup.py); [09_generate_hybrid_spelling_dataset.py](/C:/1_projectSamSung/scripts/09_generate_hybrid_spelling_dataset.py); [generation summary](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) | **Mới từ R**, 3.2–3.3: nguồn, sinh lỗi hybrid, dedup, natural/segmented. |
| S30 | [05_validate_spelling_dataset.py](/C:/1_projectSamSung/scripts/05_validate_spelling_dataset.py); [12_split_spelling_dataset.py](/C:/1_projectSamSung/scripts/12_split_spelling_dataset.py); [train M1](/C:/1_projectSamSung/data/processed/vn_newsspell_v1/train.jsonl); [validation M1](/C:/1_projectSamSung/data/processed/vn_newsspell_v1/validation.jsonl); [test M1](/C:/1_projectSamSung/data/processed/vn_newsspell_v1/test.jsonl) | **Mới từ R**, 3.3: mã validate và split; các file dữ liệu đã xác nhận hiện diện, chưa tái chạy kiểm định. |
| S31 | [13_train_classical_baselines.py](/C:/1_projectSamSung/scripts/13_train_classical_baselines.py); [baseline comparison M1](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) | **Mới từ R**, 3.3, 4.5: word TF-IDF/LR và char TF-IDF/SVM, validation, input natural. |
| S32 | [14_train_phobert_mil.py](/C:/1_projectSamSung/scripts/14_train_phobert_mil.py); [15_tune_phobert_threshold.py](/C:/1_projectSamSung/scripts/15_tune_phobert_threshold.py); [16_evaluate_phobert_test.py — R](/C:/1_projectSamSung/scripts/16_evaluate_phobert_test.py) | **Mới từ R**, 3.3: vòng lặp train, optimizer/loss/scheduler, chọn checkpoint/ngưỡng và official evaluation. |
| S33 | [download manifest](/C:/1_projectSamSung/module2_headline_integrity/data/raw/download_manifest.json); [README ViClickbait](/C:/1_projectSamSung/module2_headline_integrity/data/raw/clickbait/viclickbait_2025/README.md); [03_prepare_clickbait_dataset.py](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/03_prepare_clickbait_dataset.py) | **Mới từ R**, 3.2, 3.4: nguồn dataset và script chia tập; README không có citation hoàn chỉnh. |
| S34 | [05_train_diagnostic_baselines.py](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/05_train_diagnostic_baselines.py); [06_train_mbert_title_only.py](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/06_train_mbert_title_only.py); [07_train_mbert_title_lead.py](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/07_train_mbert_title_lead.py) | **Mới từ R**, 3.4, 4.5: baseline, hai cách biểu diễn input và huấn luyện. |
| S35 | [08_compare_mbert_title_vs_title_lead.py](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/08_compare_mbert_title_vs_title_lead.py); [09_lock_final_clickbait_model.py](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/09_lock_final_clickbait_model.py); [10_evaluate_final_clickbait_test.py](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/10_evaluate_final_clickbait_test.py) | **Mới từ R**, 3.4, 4.3, 4.5: so sánh validation → lock → official test. |
| S36 | [03_prepare_articles.py](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/03_prepare_articles.py); [04_build_positive_pairs.py](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/04_build_positive_pairs.py); [05_mine_hard_negative_candidates.py](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/05_mine_hard_negative_candidates.py); [07_analyze_manual_annotations.py](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/07_analyze_manual_annotations.py); [08_build_negative_pairs.py](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/08_build_negative_pairs.py) | **Mới từ R**, 3.5: nguồn bài báo, cặp dương/âm, manual review TRAIN và chính sách negative. |
| S37 | [09_build_consistency_dataset.py](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/09_build_consistency_dataset.py); [10_validate_consistency_dataset.py](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/10_validate_consistency_dataset.py) | **Mới từ R**, 3.5: dựng dataset cuối và kiểm tra cấu trúc/join/split; đối chiếu S10. |
| S38 | [11_train_baseline_title_description.py](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/11_train_baseline_title_description.py); [13_train_baseline_title_article_head.py](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/13_train_baseline_title_article_head.py); [14_compare_baselines_a_b.py](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/14_compare_baselines_a_b.py); [checkpoint baseline B](/C:/1_projectSamSung/module2_headline_integrity/models/consistency/baseline_title_article_head_mbert/best_model.pt) | **Mới từ R**, 3.5, 4.5: huấn luyện/so sánh baseline; checkpoint B xác nhận hiện diện, không nạp. |
| S39 | [17_prepare_mil_chunk_cache.py](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/17_prepare_mil_chunk_cache.py); [18_train_full_article_max_mil.py](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/18_train_full_article_max_mil.py); [articles train](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/train.parquet); [articles validation](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/validation.parquet); [articles test](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/test.parquet) | **Mới từ R**, 3.5: cache/chunk policy và huấn luyện MAX-MIL; toàn văn các split có trong R. |
| S40 | [19_compare_and_fuse_description_mil.py](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/19_compare_and_fuse_description_mil.py); [20_prepare_test_mil_chunk_cache.py](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/20_prepare_test_mil_chunk_cache.py); [21_evaluate_final_hybrid_test.py](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/21_evaluate_final_hybrid_test.py) | **Mới từ R**, 3.5, 4.4–4.5: fusion selection, ghi final lock trước tiền xử lý test và đánh giá cuối. |
| S41 | [README VietOnlineNews](/C:/1_projectSamSung/module2_headline_integrity/data/raw/consistency/viet_online_news/README.md); [VietNews dataset metadata](/C:/1_projectSamSung/data/raw/vietnews/train/dataset_info.json) | **Mới từ R**, 3.2 và G: nguồn dữ liệu/documentation; trường citation của metadata VietNews rỗng. Không thay số dataset dẫn xuất bằng quy mô corpus gốc. |

### E.2. Nguồn triển khai và kiểm thử

| Mã | File nguồn chính | Dùng cho / loại bằng chứng |
|---|---|---|
| S14 | [headline_integrity_pipeline.py](../../module2_headline_integrity/inference/headline_integrity_pipeline.py); [unified pipeline card](../../module2_headline_integrity/results/integration/unified_module2_v1/UNIFIED_MODULE2_CARD.md) | Ch.2–3: suy luận M2/M3 và thông số pipeline được khóa. |
| S15 | [model_registry.py](../../system/backend/app/services/model_registry.py); [clickbait adapter](../../system/backend/app/adapters/clickbait.py); [consistency adapter](../../system/backend/app/adapters/headline_consistency.py) | Ch.2–3: quyền sở hữu instance và adapter logic. |
| S16 | [main.py](../../system/backend/app/api/main.py); [schemas.py](../../system/backend/app/api/schemas.py); [service.py](../../system/backend/app/api/service.py) | Ch.2–3: FastAPI hiện tại, contract, lifecycle, khóa và xử lý lỗi. |
| S17 | [App.tsx](../../frontend/src/App.tsx); [api.ts](../../frontend/src/api.ts); [presentation.ts](../../frontend/src/presentation.ts); [demo_cases.ts](../../frontend/src/demo_cases.ts); [package.json](../../frontend/package.json); [vite.config.ts](../../frontend/vite.config.ts) | Ch.2–3, 4.9: React, client, cảnh báo, demo và build/proxy. |
| S18 | [release manifest](../../system/release/model_lock_manifest_v1.json); [lock config](../../system/release/model_lock_config.json); [environment](../../system/release/environment_v1.json); [model lock verify log](../../system/reports/final_validation_v1/model_lock_verify.txt) | Ch.2–3: checkpoint/tokenizer/threshold và môi trường release; hash verification là kiểm tra artifact, không là metric chất lượng. |
| S19 | [test_adapters.py](../../system/backend/tests/test_adapters.py); [test_api_v1.py](../../system/backend/tests/test_api_v1.py); [test_api_m1_pyvi_v1.py](../../system/backend/tests/test_api_m1_pyvi_v1.py); [backend tests log](../../system/reports/final_validation_v1/backend_tests.txt); [frontend build log](../../system/reports/final_validation_v1/frontend_build.txt) | 4.7, 4.9: unit/API fake/mock và frontend build; hai loại kết quả trình bày riêng. |
| S20 | [M1 regression script](../../system/scripts/06_regression_test_module1.py); [reference M1](../../system/reports/module1_regression_v1/reference_v1.json); [regression report M1](../../system/reports/module1_regression_v1/regression_report.json) | 4.6: parity M1 với reference validation. |
| S21 | [M2/M3 regression script](../../module2_headline_integrity/inference/04_regression_check_unified_pipeline.py); [regression report M2/M3](../../module2_headline_integrity/results/integration/regression_v1/regression_report.json) | 4.6: parity clickbait, description, MAX-MIL, fusion. |
| S22 | [HTTP smoke script](../../system/scripts/09_smoke_test_http_api.py); [HTTP smoke PyVi script](../../system/scripts/16_smoke_m1_pyvi_http.py); [pipeline smoke report](../../module2_headline_integrity/results/integration/unified_module2_v1/smoke_test/smoke_test_report.json) | 4.8: kịch bản HTTP; report cuối là smoke pipeline Python, không phải log HTTP. |
| S23 | [profile script](../../system/scripts/07_profile_gpu_memory.py); [profile summary](../../system/reports/gpu_profile_v1/summary.json); [combined profile](../../system/reports/gpu_profile_v1/combined.json) | 3.1, 4.11: tài nguyên GPU và latency adapter, không phải tải HTTP. |
| S24 | [segmentation origin audit](../../system/reports/m1_preprocessing_audit_v1/origin_trace_v1.json); [segmenter comparison script](../../system/scripts/12_benchmark_m1_segmenters.py); [drift script](../../system/scripts/13_measure_m1_segmentation_drift.py); [PyVi segmentation report](../../system/reports/m1_preprocessing_audit_v1/segmenter_benchmark_pyvi.json); [PyVi prediction drift](../../system/reports/m1_preprocessing_audit_v1/prediction_drift_pyvi.json) | 3.8, 4.10, 4.12: experimental compatibility, không phải official accuracy benchmark. |
| S25 | [experimental_segmenter.py](../../system/backend/app/api/experimental_segmenter.py); [PyVi API tests](../../system/backend/tests/test_api_m1_pyvi_v1.py) | 3.8, 4.7, 4.10: opt-in/surface/join và tests có thành phần giả lập. |
| S26 | [bản thiết kế cũ](../../system/scripts/SYSTEM_DESIGN_V1.md); [API document](../API_V1.md) | Chỉ đối chiếu lịch sử thiết kế; không làm nguồn khẳng định SQLite/lịch sử/route dự kiến đã triển khai. |
| S42 | [release README](../release_v1/README_Buoc22.md); [release checklist](../release_v1/Checklist_release_v1.md); [ảnh nhập liệu](../release_v1/screenshots/react_input_top.png); [ảnh lựa chọn M1](../release_v1/screenshots/react_input_bottom.png); [ảnh kết quả](../release_v1/screenshots/react_results.png) | Ch.2, 4.8–4.9: ảnh thật đã xem; release ghi nhận HTTP PASS thứ cấp và thiếu log riêng. Ảnh là snapshot minh họa, không xác nhận mọi chi tiết giao diện hiện tại. |

### E.3. Quy ước bắt buộc cho từng loại kết quả

Các số đếm/ngưỡng dưới đây trích từ nguồn được chỉ ra, để chuẩn bị viết Chương 4; chưa phải bảng phân tích kết quả hoàn chỉnh. Positive class áp dụng cho metric phân loại; đối với parity chỉ mô tả chiều score, không tính lại accuracy.

| Loại kết quả | Dataset / split | Sample count hoặc đơn vị kiểm tra | Threshold / positive class | Loại metric và nguồn |
|---|---|---|---|---|
| Official M1 | VN-NewsSpell v1 / test | 28.764 tài liệu | 0.57 / 1 = HAS_ERROR | Accuracy, precision/recall/F1 lỗi, macro-F1, specificity, ROC-AUC, AP, confusion matrix; S05. |
| Official M2 | viclickbait_v1 / test | 512 tiêu đề | 0.61 / 1 = CLICKBAIT | `official_final_metrics`; S09. |
| Official M3 | headline_consistency_v1 / test | 57.376 cặp, 28.688 anchors | 0.45 / 1 = CONSISTENT | `official_final_metrics`, confusion matrix, nhánh tham chiếu và cảnh báo synthetic; S13. |
| Validation M1 / baseline | VN-NewsSpell v1 / validation | 28.762 tài liệu | Theo model/quy tắc predict trong S31; PhoBERT phân biệt 0.5 với threshold tuned 0.57 / HAS_ERROR | Không gán threshold 0.57 cho baseline; S03–S04, S31–S32. |
| Validation M2 / model selection | viclickbait_v1 / validation | 512 tiêu đề | Mỗi model có ngưỡng riêng; title-only 0.61 / CLICKBAIT | Metric mặc định/tuned, kiểm định so sánh đúng report; S07–S08, S34–S35. |
| Validation M3 / baseline | headline_consistency_v1 / validation | 57.426 cặp | Theo từng baseline, không áp chung 0.45 / CONSISTENT | S10–S11, S27, S38–S39. |
| Validation M3 / fusion selection | Cùng validation, chia nội bộ theo anchor | Calibration 28.712 cặp; holdout 28.714 cặp | Ngưỡng/trọng số theo phase, fusion cuối 0.45 / CONSISTENT | Phân biệt calibration, holdout và full validation; S12, S40. |
| Regression M1 | VN-NewsSpell validation, mẫu cố định | 24 mẫu | 0.57; score hướng HAS_ERROR | MAE/max error, mismatch nhãn/chunk; tolerance 0.005; S20. |
| Regression M2/M3 | Validation của từng tác vụ | M2: 32 mẫu; M3: 24 cặp | 0.61 / 0.45; score hướng CLICKBAIT / CONSISTENT | Sai lệch xác suất, nhãn/token/chunk, tolerance 0.005; S21. |
| Unit/API tests | Không áp dụng dataset ML/split | Log ghi 25 tests | Threshold chỉ là assertion contract khi có; positive class không áp dụng | PASS/FAIL từng test; fake/mock, không GPU model benchmark; S19. |
| HTTP smoke | Mẫu validation M1 + demo headline; script raw PyVi dùng ví dụ cố định | Script chuẩn dựng một payload analyze; script PyVi một payload raw. Số lần thực thi thực tế cần log B05 | Kiểm tra 0.57/0.61/0.45 tùy route; không có metric phân loại theo positive class | Status, threshold, warning; hai phần trong payload chuẩn không bắt buộc cùng một bài báo. S22, S42. |
| Frontend build | Không áp dụng dataset/split | Một log build đã lưu, không phải số mẫu ML | Không áp dụng | Compile/bundle thành công theo log; S19. |
| PyVi drift | VN-NewsSpell / mẫu validation | 60 mẫu, 30 mỗi nhãn | 0.57; score hướng HAS_ERROR | Prediction flips, sai lệch score và chunk; report ghi 8 nhãn đổi, không diễn giải thành accuracy; S24. |
| GPU profile | Mẫu validation/demo theo script; không phải benchmark phân loại | Theo tên call, chế độ và `repeat_count` trong report | Dùng ngưỡng mô hình khóa; positive class không áp dụng cho metric tài nguyên | Load time, RAM/VRAM, latency adapter; S23. |

Khi viết số liệu official, lấy đúng trường JSON: M1 ở cấp gốc; M2/M3 ở `official_final_metrics`. Việc đánh dấu xác nhận trong H chỉ nghĩa đã đọc các metric và đối chiếu bản V/R, không nghĩa đã chạy lại inference hoặc xác minh độc lập toàn bộ quy trình lịch sử.

## F. Danh sách bảng/hình dự kiến

Chỉ đặt chỗ và chỉ định nguồn tạo. Không tạo ảnh giả, confusion matrix suy đoán hay số liệu minh họa thay kết quả thật. Tổng dung lượng đã nằm trong C; có thể ghép hình/bảng nhỏ khi dàn trang.

| Vị trí | Placeholder | Nguồn dùng để tạo và điều kiện |
|---|---|---|
| 1.2 | [HÌNH 1.1: Khối Transformer và cơ chế attention] | Chỉ vẽ sau khi bổ sung G01; không sao chép hình học thuật khi chưa xác định nguồn/quyền sử dụng. |
| 1.5–1.6 | [HÌNH 1.2: Bag, instance và chunk trong phân loại tài liệu] | G05–G06 cho lý thuyết; S02, S39 cho phần liên hệ ứng dụng. |
| 2.2 | [HÌNH 2.1: Kiến trúc tổng thể React–FastAPI–service–registry–adapters] | S14–S18; chỉ thành phần đang triển khai. |
| 2.3 | [HÌNH 2.2: Pipeline M1 PhoBERT MIL ở cấp tài liệu] | S02, S04, S32; không thêm bước sửa/định vị từ lỗi. |
| 2.4 | [HÌNH 2.3: Pipeline M2 mBERT title-only] | S09, S14, S34. |
| 2.5 | [HÌNH 2.4: Pipeline M3 description và MAX-MIL fusion] | S11–S14, S39–S40; ghi hai nhánh/checkpoint riêng. |
| 2.6 | [BẢNG 2.1: Endpoint, trường đầu vào và kết quả của API] | S16; không chép route dự kiến từ S26. |
| 2.7 | [HÌNH 2.5: Giao diện nhập bài báo và lựa chọn đầu vào M1] | S42, `react_input_top.png`, `react_input_bottom.png`; ảnh có sẵn, đối chiếu S17. |
| 2.7 | [HÌNH 2.6: Giao diện ba kết quả độc lập] | S42, `react_results.png`; giữ nhãn PyVi experimental và chú thích kết quả minh họa. |
| 2.8 | [HÌNH 2.7: Luồng xử lý một yêu cầu phân tích] | S15–S17, S25; thể hiện các nhánh M1 và hành vi lỗi đúng code. |
| 3.2 | [BẢNG 3.1: Dataset, split, đơn vị mẫu và phân bố nhãn] | S03, S06, S10; ghi nguồn số lưu, không gộp anchor với pair. |
| 3.3–3.5 | [HÌNH 3.1: Quy trình xây dữ liệu, huấn luyện, lựa chọn và khóa mô hình] | S29–S40; ba nhánh có bước riêng, test không tham gia lựa chọn. |
| 3.3 | [BẢNG 3.2: Cấu hình và lựa chọn checkpoint M1] | S04, S32; đối chiếu config/history. |
| 3.4–3.5 | [BẢNG 3.3: Cấu hình huấn luyện các nhánh mBERT] | S07, S11, S34, S38–S39; mỗi nhánh một hàng riêng. |
| 3.3–3.5 | [HÌNH 3.2: Diễn biến huấn luyện và validation đã lưu] | S04, S07, S11; chỉ vẽ epoch có record, không nối điểm giả. |
| 3.6–3.7 | [HÌNH 3.3: Quy trình tích hợp pipeline nghiên cứu vào API và frontend] | S02, S14–S19; tách bước triển khai với bằng chứng kiểm thử. |
| 4.2–4.4 | [BẢNG 4.1: Metric official test theo từng tác vụ] | S05, S09, S13 và metadata E.3; không xếp hạng trực tiếp ba bài toán. |
| 4.2 | [HÌNH 4.1: Confusion matrix official test M1] | `confusion_matrix` của S05, lớp dương HAS_ERROR. |
| 4.3 | [HÌNH 4.2: Confusion matrix official test M2] | `official_final_metrics.confusion_matrix` của S09, lớp dương CLICKBAIT. |
| 4.4 | [HÌNH 4.3: Confusion matrix official test M3] | `official_final_metrics.confusion_matrix` của S13, lớp dương CONSISTENT. |
| 4.5 | [BẢNG 4.2: So sánh baseline và mô hình trên validation] | S04, S07–S08, S11, S27, S31; chia nhóm theo dataset/protocol/input/ngưỡng. |
| 4.5 | [BẢNG 4.3: Lựa chọn fusion M3 trên calibration/holdout] | S12, S40; không trộn vào bảng official test. |
| 4.6 | [BẢNG 4.4: Regression M1/M2/M3] | S20–S21; số mẫu, tolerance, probability error và mismatch. |
| 4.7–4.9 | [BẢNG 4.5: Unit/API tests, HTTP smoke và frontend build] | S19, S22, S42; ba nhóm hàng riêng, HTTP đánh dấu thiếu log gốc theo B05. |
| 4.10 | [BẢNG 4.6: PyVi prediction drift trên mẫu validation] | S24; bảng ngắn, không gọi là accuracy của PyVi. |
| 4.11 | [BẢNG 4.7: GPU memory, thời gian nạp và latency adapter] | S23; ghi thiết bị, chế độ, sample/call, số lặp và đơn vị đo. |

## G. Danh sách tài liệu học thuật còn cần bổ sung

Đây là **11 mục nguồn còn thiếu**, chưa phải bibliography. Mỗi mã có một nhãn trạng thái; Chương 1 và phần tài liệu tham khảo dẫn tới các mã này. Việc tìm trong hai project chưa cung cấp đủ nguồn học thuật cho các chủ đề dưới đây. Không tự tạo citation từ tên thư viện/model hoặc từ trí nhớ.

| Mã | Chủ đề cần nguồn | Dấu vết hiện có / tài liệu cần bổ sung | Trạng thái |
|---|---|---|---|
| G01 | Transformer | Cần paper nền tảng có metadata xác minh để viết self-attention, kiến trúc và ký hiệu ở 1.2. | **[CHƯA ĐỦ DỮ LIỆU HỌC THUẬT]** |
| G02 | BERT | Cần paper gốc về pretraining/fine-tuning; import/model name trong S34 không thay thế nguồn này. | **[CHƯA ĐỦ DỮ LIỆU HỌC THUẬT]** |
| G03 | multilingual BERT | Cần tài liệu gốc/chính thức mô tả mBERT và nguồn học thuật phù hợp cho các nhận định đa ngôn ngữ. | **[CHƯA ĐỦ DỮ LIỆU HỌC THUẬT]** |
| G04 | PhoBERT | Cần paper và documentation/model card tương ứng biến thể `vinai/phobert-base-v2`; không suy ra cấu hình mọi phiên bản từ tên chung PhoBERT. | **[CHƯA ĐỦ DỮ LIỆU HỌC THUẬT]** |
| G05 | Multiple Instance Learning | Cần nguồn gốc/tổng quan có định nghĩa bag/instance và aggregation phù hợp với phương pháp áp dụng. | **[CHƯA ĐỦ DỮ LIỆU HỌC THUẬT]** |
| G06 | Chunking và phân loại văn bản dài | Cần nguồn phù hợp cho động cơ chia đoạn, overlap, lựa chọn đoạn và giới hạn độ phủ; code chỉ chứng minh chính sách riêng của project. | **[CHƯA ĐỦ DỮ LIỆU HỌC THUẬT]** |
| G07 | Phân loại, metric và protocol đánh giá | Cần nguồn cho confusion matrix/F1/ROC/AP, model selection và diễn giải thống kê; nếu dùng McNemar/bootstrap phải có nguồn phương pháp. | **[CHƯA ĐỦ DỮ LIỆU HỌC THUẬT]** |
| G08 | Dataset gốc và dẫn xuất M1 | S29 nhận diện `nam194/vietnews`, `nguyenthanhasia/vsec-vietnamese-spell-correction`, `coung21/vi-spelling-correction`; S41 có citation rỗng. Cần nguồn công bố/citation được xác nhận; VN-NewsSpell là dataset dẫn xuất theo generator, không tự gán paper riêng. | **[CHƯA ĐỦ DỮ LIỆU HỌC THUẬT]** |
| G09 | Dataset M2 | S33 xác định `blanatole/ViClickbait-2025`; README lưu chưa có DOI. Cần thông tin tác giả/phiên bản/cách trích dẫn phù hợp, không tạo DOI giả và không coi tên GitHub là tên tác giả học thuật. | **[CHƯA ĐỦ DỮ LIỆU HỌC THUẬT]** |
| G10 | Dataset M3 | S33/S41 nhận diện `VLUS06/VietOnlineNews`; dataset consistency dẫn xuất do S36–S37 xây dựng. Cần citation corpus gốc và mô tả provenance; không tự gán công bố độc lập cho `headline_consistency_v1`. | **[CHƯA ĐỦ DỮ LIỆU HỌC THUẬT]** |
| G11 | Documentation công cụ | Cần nguồn chính thức phù hợp cho PyTorch/Transformers, FastAPI/React và PyVi khi thực sự trích trong báo cáo; requirements/code nội bộ chỉ xác nhận công cụ được dùng. | **[CHƯA ĐỦ DỮ LIỆU HỌC THUẬT]** |

Khi bổ sung nguồn, ghi đủ thông tin có thể xác minh và dùng một chuẩn trích dẫn thống nhất theo B07. Chỉ tài liệu thực sự được dùng trong lập luận mới đưa vào bibliography cuối; không dùng danh mục file S01–S42 thay danh mục học thuật.

## H. Checklist trước khi bắt đầu viết Chương 1

- [ ] Đã có nguồn học thuật Transformer — G01.
- [ ] Đã có nguồn học thuật BERT — G02.
- [ ] Đã có nguồn học thuật mBERT — G03.
- [ ] Đã có nguồn học thuật PhoBERT và đối chiếu biến thể sử dụng — G04.
- [ ] Đã có nguồn MIL — G05.
- [ ] Đã có nguồn chunking/văn bản dài — G06.
- [ ] Đã có nguồn metric và phương pháp đánh giá — G07.
- [ ] Đã có citation dataset gốc M1 — G08.
- [ ] Đã có citation dataset M2 — G09; mới có README/source identity, chưa đủ metadata citation.
- [ ] Đã có citation dataset M3 nếu có — G10; phân biệt corpus gốc và cặp consistency dẫn xuất.
- [x] Đã xác nhận metric official test của M1 — đọc S05 và đối chiếu hash hai bản V/R.
- [x] Đã xác nhận metric official test của M2 — đọc S09 và đối chiếu hash hai bản V/R.
- [x] Đã xác nhận metric official test của M3 — đọc S13, bao gồm cảnh báo synthetic và đối chiếu hash hai bản V/R.
- [x] Đã tách official test, validation/model selection, regression, unit/API, HTTP smoke, frontend build và PyVi trong đề cương.
- [x] Đã xác định đúng M1 document-level; các nhánh định vị/sửa lỗi ngoài release không nhập vào báo cáo chính.
- [x] Đã bổ sung nguồn script nghiên cứu từ R thay cho những khoảng trống chỉ do thiếu file trong V.
- [ ] Đã hoàn thiện các khoảng trống B01–B07 hoặc giữ giới hạn tương ứng trong phần sẽ viết.
- [ ] Đã có quy định trình bày/trích dẫn của trường và tên đề tài được duyệt.

**Dừng ở đề cương V2. Không tự động viết Chương 1 hoặc nội dung đầy đủ của báo cáo.**
