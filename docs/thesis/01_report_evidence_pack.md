# 1. Thông tin đề tài
Tên làm việc: Hệ thống hỗ trợ đánh giá bài báo tiếng Việt bằng các mô hình phân loại văn bản. Tên được duyệt: [CHƯA CÓ TÊN ĐỀ TÀI CHÍNH THỨC].
Mục đích pack: bằng chứng phục vụ báo cáo 30–40 trang; không phải luận văn hoàn chỉnh. Chỉ đọc tĩnh source/artifact và đếm bản ghi để đối chiếu; không train, tune, model selection, official evaluation, inference hoặc build mới. V = C:/projectSamSung_v1; R = C:/1_projectSamSung. Sxx là nhóm nguồn của outline; Pxxx trong mục 26 là từng file đã đọc trong pack.
Quy ước số: giữ nguyên giá trị JSON/CSV; float dài là biểu diễn artifact, không tự làm tròn ngầm. Đếm lại row/label/confusion chỉ là kiểm tra tĩnh được gắn nguồn; không tính metric mới không có trong artifact.
| Giai đoạn | Tên/cách tổ chức cũ | Tên module trong V1 | Trạng thái |
| --- | --- | --- | --- |
| Nghiên cứu | Module 1 spelling | Module 1 | Giữ lại: document-level PhoBERT MIL |
| Nghiên cứu | Task 2A clickbait trong module2_headline_integrity | Module 2 | Giữ lại |
| Nghiên cứu | Task 2B consistency trong module2_headline_integrity | Module 3 | Tách thành module logic độc lập |
| Nghiên cứu cũ | Module 3 cũ / plagiarism branch | Không thuộc V1 | EXCLUDED |

Nguồn: `C:/projectSamSung_v1/README_V1.md` — [P002](/C:/projectSamSung_v1/README_V1.md). Field/CSV columns/source code: `Scope; compatibility rule`.

Ba đầu ra độc lập; không có điểm chất lượng báo chí tổng hợp. M1 không localize/correct/sequence-label; PyVi chỉ experimental. M2 title-only; M3 headline-content consistency. M2/M3 dùng chung runtime pipeline object không có nghĩa dùng chung checkpoint. Không dùng nhánh all_fields/4chunks/word detector/correction/plagiarism thay mô hình cuối.
| Mã | Evidence type | Cách diễn giải |
| --- | --- | --- |
| A | Dataset construction | Nguồn, sinh dữ liệu, split/nhãn |
| B | Training configuration | Phân biệt default và actual run |
| C | Training history | Chỉ epoch đã lưu |
| D | Validation / model selection | Không trộn official test |
| E | Official test | Theo từng task/dataset |
| F | Regression test | REGRESSION != ACCURACY |
| G | Unit/API test | Fake/mock không phải model benchmark |
| H | HTTP smoke | Routing check, không benchmark |
| I | Frontend build | Build PASS != E2E correctness |
| J | GPU/resource profile | Adapter timing không là HTTP timing |
| K | PyVi experimental | Giữ nguyên nhãn != accuracy |
| L | Model lock/release verification | SHA256 verification != model accuracy |

# 2. Kiến trúc hệ thống
React → FastAPI.create_app → InferenceService → ModelRegistry → adapters → PhoBERT MIL / HeadlineIntegrityPipeline. Không đưa database, SQLite, lịch sử phân tích hay route dự kiến vào kiến trúc đang chạy.
| File/class/function | Trách nhiệm | Source |
| --- | --- | --- |
| create_app / lifespan | Nạp M1 và headline adapters tại startup; ready false trước nạp, true sau thành công; shutdown xóa service/registry | [P007](/C:/projectSamSung_v1/system/backend/app/api/main.py) |
| get_service / health / models | Readiness; 503 nếu model chưa sẵn sàng; metadata models | [P007](/C:/projectSamSung_v1/system/backend/app/api/main.py) |
| InferenceService._locked | threading.Lock; timeout 30 s → 503, Retry-After 3; finally release; ValueError→422, ImportError→503, lỗi khác→500 không lộ bài báo | [P009](/C:/projectSamSung_v1/system/backend/app/api/service.py) |
| InferenceService.analyze | Một lock bao segmentation và chuỗi M1/M2/M3; exception có thể làm cả request thất bại | [P009](/C:/projectSamSung_v1/system/backend/app/api/service.py) |
| ModelRegistry.headline_adapters | Lazy/shared HeadlineIntegrityPipeline; hai adapter logic trỏ cùng instance | [P003](/C:/projectSamSung_v1/system/backend/app/services/model_registry.py) |
| SpellingAdapter.predict | Tái sử dụng DocumentCollator/PhoBERTMIL từ script evaluation; sigmoid score và ngưỡng | [P004](/C:/projectSamSung_v1/system/backend/app/adapters/spelling.py) |

Nguồn: `C:/projectSamSung_v1/system/backend/app/api/main.py` — [P007](/C:/projectSamSung_v1/system/backend/app/api/main.py). Field/CSV columns/source code: `create_app, lifespan, get_service; routes`.

Nguồn: `C:/projectSamSung_v1/system/backend/app/api/service.py` — [P009](/C:/projectSamSung_v1/system/backend/app/api/service.py). Field/CSV columns/source code: `InferenceService._locked, analyze`.

| Method / endpoint | Request schema | Response contract từ mã | Source |
| --- | --- | --- | --- |
| GET /api/v1/health | Không body | status, models_ready | [P007](/C:/projectSamSung_v1/system/backend/app/api/main.py) |
| GET /api/v1/models | Không body | ready, models[], shared_pipeline | [P007](/C:/projectSamSung_v1/system/backend/app/api/main.py) |
| POST /api/v1/spelling | SpellingRequest: article_segmented | probability_error, threshold, prediction, label, chunk_count, warnings | [P007](/C:/projectSamSung_v1/system/backend/app/api/main.py); [P008](/C:/projectSamSung_v1/system/backend/app/api/schemas.py) |
| POST /api/v1/spelling/raw | RawSpellingRequest: article_text | status, input_mode=pyvi_experimental, warning, result | [P007](/C:/projectSamSung_v1/system/backend/app/api/main.py); [P009](/C:/projectSamSung_v1/system/backend/app/api/service.py) |
| POST /api/v1/clickbait | ClickbaitRequest: title | probability_clickbait, prediction, label, threshold theo pipeline | [P007](/C:/projectSamSung_v1/system/backend/app/api/main.py) |
| POST /api/v1/consistency | ConsistencyRequest: title, description, article_text | probability_consistent, prediction, label, threshold, branches; warnings khi thiếu description | [P007](/C:/projectSamSung_v1/system/backend/app/api/main.py) |
| POST /api/v1/analyze | ArticleRequest | api_version, module1_spelling, module2_clickbait, module3_consistency, notes | [P007](/C:/projectSamSung_v1/system/backend/app/api/main.py); [P009](/C:/projectSamSung_v1/system/backend/app/api/service.py) |

Backend trả dict, không khai báo Pydantic response_model trong các route này; TypeScript response interfaces ở frontend là contract phía client, không phải backend response validation.
Nguồn: `C:/projectSamSung_v1/system/backend/app/api/schemas.py` — [P008](/C:/projectSamSung_v1/system/backend/app/api/schemas.py).
~~~text
"""API input contracts. Frozen models and model thresholds are unchanged."""
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ArticleRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=1000, description="Vietnamese news headline")
    description: str = Field(default="", max_length=5000, description="Article lead/summary")
    article_text: str = Field(..., min_length=1, max_length=120000, description="Full raw article")
    article_segmented: Optional[str] = Field(
        default=None, min_length=1, max_length=120000,
        description="Optional presegmented M1 input; always takes the standard M1 path."
    )
    m1_input_mode: Literal["segmented_only", "pyvi_experimental"] = Field(
        default="segmented_only",
        description="Defaults to the original API. Select pyvi_experimental explicitly to run M1 on article_text."
    )


class SpellingRequest(BaseModel):
    article_segmented: str = Field(
        ..., min_length=1, max_length=120000,
        description="Presegmented Vietnamese text (e.g., người_dân); not raw text."
    )


class RawSpellingRequest(BaseModel):
    article_text: str = Field(
        ..., min_length=1, max_length=120000,
        description="Ordinary Vietnamese article; experimental PyVi segmentation is explicitly opted into by calling this endpoint."
    )


class ClickbaitRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=1000)


class ConsistencyRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=1000)
    description: str = Field(default="", max_length=5000)
    article_text: str = Field(..., min_length=1, max_length=120000)
~~~

M1 mặc định segmented_only: thiếu article_segmented thì skipped; presegmented và pyvi_experimental đồng thời bị từ chối 422. CORS chỉ các origin local tại port 5173 trong main.py; response Cache-Control no-store. Device mặc định CUDA qua SAMSUNG_DEVICE; không suy ra concurrency production từ khóa nội bộ.
# 3. Module 1 — Dữ liệu
## Nguồn tải corpus
| Config/call trong mã (không tự coi là actual run) | Biểu thức | Vị trí | Source |
| --- | --- | --- | --- |
| DATASETS | {'vietnews': {'name': 'nam194/vietnews'}, 'vsec': {'name': 'nguyenthanhasia/vsec-vietnamese-spell-correction'}, 'vi_spelling_correction': {'name': 'coung21/vi-spelling-correction'}} | line 19 | [P010](/C:/1_projectSamSung/scripts/01_download_raw.py) |

Nguồn: `C:/1_projectSamSung/scripts/01_download_raw.py` — [P010](/C:/1_projectSamSung/scripts/01_download_raw.py). Field/CSV columns/source code: `DATASETS`.

Corpus: nam194/vietnews; nguồn khai thác cặp lỗi: nguyenthanhasia/vsec-vietnamese-spell-correction và coung21/vi-spelling-correction theo DATASETS. Chuỗi mine → filter → trusted lookup → generator hybrid; không gọi các bản noisy là lỗi tự nhiên được người gán nhãn trực tiếp.
## Metadata corpus VietNews (khác dataset dẫn xuất)
| Field JSON | Giá trị lưu trong file | Source |
| --- | --- | --- |
| builder_name | parquet | [P017](/C:/1_projectSamSung/data/raw/vietnews/train/dataset_info.json) |
| citation |  | [P017](/C:/1_projectSamSung/data/raw/vietnews/train/dataset_info.json) |
| config_name | default | [P017](/C:/1_projectSamSung/data/raw/vietnews/train/dataset_info.json) |
| dataset_name | vietnews | [P017](/C:/1_projectSamSung/data/raw/vietnews/train/dataset_info.json) |
| dataset_size | 473370709 | [P017](/C:/1_projectSamSung/data/raw/vietnews/train/dataset_info.json) |
| description |  | [P017](/C:/1_projectSamSung/data/raw/vietnews/train/dataset_info.json) |
| download_size | 246524136 | [P017](/C:/1_projectSamSung/data/raw/vietnews/train/dataset_info.json) |
| features.guid.dtype | int64 | [P017](/C:/1_projectSamSung/data/raw/vietnews/train/dataset_info.json) |
| features.guid._type | Value | [P017](/C:/1_projectSamSung/data/raw/vietnews/train/dataset_info.json) |
| features.title.dtype | string | [P017](/C:/1_projectSamSung/data/raw/vietnews/train/dataset_info.json) |
| features.title._type | Value | [P017](/C:/1_projectSamSung/data/raw/vietnews/train/dataset_info.json) |
| features.abstract.dtype | string | [P017](/C:/1_projectSamSung/data/raw/vietnews/train/dataset_info.json) |
| features.abstract._type | Value | [P017](/C:/1_projectSamSung/data/raw/vietnews/train/dataset_info.json) |
| features.article.dtype | string | [P017](/C:/1_projectSamSung/data/raw/vietnews/train/dataset_info.json) |
| features.article._type | Value | [P017](/C:/1_projectSamSung/data/raw/vietnews/train/dataset_info.json) |
| homepage |  | [P017](/C:/1_projectSamSung/data/raw/vietnews/train/dataset_info.json) |
| license |  | [P017](/C:/1_projectSamSung/data/raw/vietnews/train/dataset_info.json) |
| size_in_bytes | 719894845 | [P017](/C:/1_projectSamSung/data/raw/vietnews/train/dataset_info.json) |
| splits.train.name | train | [P017](/C:/1_projectSamSung/data/raw/vietnews/train/dataset_info.json) |
| splits.train.num_bytes | 325430847 | [P017](/C:/1_projectSamSung/data/raw/vietnews/train/dataset_info.json) |
| splits.train.num_examples | 99134 | [P017](/C:/1_projectSamSung/data/raw/vietnews/train/dataset_info.json) |
| splits.train.dataset_name | vietnews | [P017](/C:/1_projectSamSung/data/raw/vietnews/train/dataset_info.json) |
| splits.validation.name | validation | [P017](/C:/1_projectSamSung/data/raw/vietnews/train/dataset_info.json) |
| splits.validation.num_bytes | 73400090 | [P017](/C:/1_projectSamSung/data/raw/vietnews/train/dataset_info.json) |
| splits.validation.num_examples | 22184 | [P017](/C:/1_projectSamSung/data/raw/vietnews/train/dataset_info.json) |
| splits.validation.dataset_name | vietnews | [P017](/C:/1_projectSamSung/data/raw/vietnews/train/dataset_info.json) |
| splits.test.name | test | [P017](/C:/1_projectSamSung/data/raw/vietnews/train/dataset_info.json) |
| splits.test.num_bytes | 74539772 | [P017](/C:/1_projectSamSung/data/raw/vietnews/train/dataset_info.json) |
| splits.test.num_examples | 22498 | [P017](/C:/1_projectSamSung/data/raw/vietnews/train/dataset_info.json) |
| splits.test.dataset_name | vietnews | [P017](/C:/1_projectSamSung/data/raw/vietnews/train/dataset_info.json) |
| version.version_str | 0.0.0 | [P017](/C:/1_projectSamSung/data/raw/vietnews/train/dataset_info.json) |
| version.major | 0 | [P017](/C:/1_projectSamSung/data/raw/vietnews/train/dataset_info.json) |
| version.minor | 0 | [P017](/C:/1_projectSamSung/data/raw/vietnews/train/dataset_info.json) |
| version.patch | 0 | [P017](/C:/1_projectSamSung/data/raw/vietnews/train/dataset_info.json) |

Nguồn: `C:/1_projectSamSung/data/raw/vietnews/train/dataset_info.json` — [P017](/C:/1_projectSamSung/data/raw/vietnews/train/dataset_info.json). Field/CSV columns/source code: `Các field ghi ở cột trái; không tính lại metric`.

## Trusted lookup summary
| Field JSON | Giá trị lưu trong file | Source |
| --- | --- | --- |
| input_patterns | 63754 | [P018](/C:/1_projectSamSung/data/interim/real_error_patterns_v03/trusted_error_summary.json) |
| trusted_before_variant_cap | 22655 | [P018](/C:/1_projectSamSung/data/interim/real_error_patterns_v03/trusted_error_summary.json) |
| trusted_final_patterns | 15449 | [P018](/C:/1_projectSamSung/data/interim/real_error_patterns_v03/trusted_error_summary.json) |
| lookup_words | 3176 | [P018](/C:/1_projectSamSung/data/interim/real_error_patterns_v03/trusted_error_summary.json) |
| rejected_patterns | 41099 | [P018](/C:/1_projectSamSung/data/interim/real_error_patterns_v03/trusted_error_summary.json) |
| error_type_distribution.diacritic_confusion | 1874 | [P018](/C:/1_projectSamSung/data/interim/real_error_patterns_v03/trusted_error_summary.json) |
| error_type_distribution.delete_character | 3167 | [P018](/C:/1_projectSamSung/data/interim/real_error_patterns_v03/trusted_error_summary.json) |
| error_type_distribution.onset_confusion | 797 | [P018](/C:/1_projectSamSung/data/interim/real_error_patterns_v03/trusted_error_summary.json) |
| error_type_distribution.insert_character | 5718 | [P018](/C:/1_projectSamSung/data/interim/real_error_patterns_v03/trusted_error_summary.json) |
| error_type_distribution.transpose_character | 3893 | [P018](/C:/1_projectSamSung/data/interim/real_error_patterns_v03/trusted_error_summary.json) |
| reject_reasons.untrusted_generic_edit_type | 16299 | [P018](/C:/1_projectSamSung/data/interim/real_error_patterns_v03/trusted_error_summary.json) |
| reject_reasons.possible_real_word | 5040 | [P018](/C:/1_projectSamSung/data/interim/real_error_patterns_v03/trusted_error_summary.json) |
| reject_reasons.too_many_edits | 3 | [P018](/C:/1_projectSamSung/data/interim/real_error_patterns_v03/trusted_error_summary.json) |
| reject_reasons.weak_correct_word_support | 509 | [P018](/C:/1_projectSamSung/data/interim/real_error_patterns_v03/trusted_error_summary.json) |
| reject_reasons.low_frequency | 19248 | [P018](/C:/1_projectSamSung/data/interim/real_error_patterns_v03/trusted_error_summary.json) |
| config.trusted_error_types | ["delete_character", "diacritic_confusion", "insert_character", "onset_confusion", "transpose_character"] | [P018](/C:/1_projectSamSung/data/interim/real_error_patterns_v03/trusted_error_summary.json) |
| config.min_pattern_frequency | 3 | [P018](/C:/1_projectSamSung/data/interim/real_error_patterns_v03/trusted_error_summary.json) |
| config.min_correct_vietnews_frequency | 5 | [P018](/C:/1_projectSamSung/data/interim/real_error_patterns_v03/trusted_error_summary.json) |
| config.max_wrong_vietnews_frequency | 100 | [P018](/C:/1_projectSamSung/data/interim/real_error_patterns_v03/trusted_error_summary.json) |
| config.max_variants_per_word | 20 | [P018](/C:/1_projectSamSung/data/interim/real_error_patterns_v03/trusted_error_summary.json) |

Nguồn: `C:/1_projectSamSung/data/interim/real_error_patterns_v03/trusted_error_summary.json` — [P018](/C:/1_projectSamSung/data/interim/real_error_patterns_v03/trusted_error_summary.json). Field/CSV columns/source code: `Các field ghi ở cột trái; không tính lại metric`.

## Sinh dataset hybrid và thống kê lỗi
| Field JSON | Giá trị lưu trong file | Source |
| --- | --- | --- |
| version | hybrid_v1_2_dedup | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| raw_articles_available | 143816 | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| raw_articles_scanned | 143816 | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| duplicate_sources_skipped | 1 | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| unique_sources_written | 143815 | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| clean_records | 143815 | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| noisy_records | 143815 | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| total_records | 287630 | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| total_errors | 336850 | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| average_errors_per_noisy_article | 2.342245245628064 | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| error_source_distribution.rule_based | 147581 | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| error_source_distribution.mined | 189269 | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| error_type_distribution.wrong_tone | 26476 | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| error_type_distribution.transpose_character | 75567 | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| error_type_distribution.delete_character | 58849 | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| error_type_distribution.insert_character | 28125 | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| error_type_distribution.onset_confusion | 17921 | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| error_type_distribution.duplicate_character | 17647 | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| error_type_distribution.telex_unconverted | 10933 | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| error_type_distribution.diacritic_confusion | 32302 | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| error_type_distribution.missing_diacritic | 43148 | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| error_type_distribution.keyboard_neighbor | 25882 | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| severity_distribution.minor | 71772 | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| severity_distribution.moderate | 50387 | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| severity_distribution.severe | 21656 | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| field_distribution.article | 307052 | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| field_distribution.abstract | 21746 | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| field_distribution.title | 8052 | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| trusted_lookup_words | 3176 | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| configuration.max_articles | null | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| configuration.noisy_variants_per_article | 1 | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| configuration.mined_pattern_probability | 0.6 | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| configuration.min_mined_transpose_length | 3 | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| configuration.mined_insert_filter | duplicate_like_only | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| configuration.canonical_deduplication | true | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| configuration.canonical_normalization | NFC | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| configuration.source_id | SHA1(normalized_article)[:20] | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| configuration.random_seed | 20260918 | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| configuration.rule_error_weights.missing_diacritic | 0.24 | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| configuration.rule_error_weights.wrong_tone | 0.18 | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| configuration.rule_error_weights.delete_character | 0.12 | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| configuration.rule_error_weights.duplicate_character | 0.1 | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| configuration.rule_error_weights.transpose_character | 0.1 | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| configuration.rule_error_weights.keyboard_neighbor | 0.1 | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| configuration.rule_error_weights.onset_confusion | 0.1 | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| configuration.rule_error_weights.telex_unconverted | 0.06 | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| configuration.severity_weights.minor | 0.5 | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| configuration.severity_weights.moderate | 0.35 | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| configuration.severity_weights.severe | 0.15 | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| configuration.error_density.minor | [1, 3] | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| configuration.error_density.moderate | [4, 8] | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| configuration.error_density.severe | [9, 15] | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| configuration.preserve_underscore_segmentation | true | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |
| configuration.protect_probable_proper_nouns | true | [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json) |

Nguồn: `C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json` — [P019](/C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json). Field/CSV columns/source code: `Các field ghi ở cột trái; không tính lại metric`.

## Split và label/severity/error distributions
| Field JSON | Giá trị lưu trong file | Source |
| --- | --- | --- |
| dataset | VN-NewsSpell v1 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| input_path | C:\1_projectSamSung\data\interim\vn_newsspell_hybrid_v03_full\vn_newsspell_all.jsonl | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| random_seed | 20260918 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| split_strategy | grouped_by_source_id | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| ratios.train | 0.8 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| ratios.validation | 0.1 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| ratios.test | 0.1 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| total_unique_sources | 143815 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| total_records | 287630 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| leakage_check.train_validation | 0 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| leakage_check.train_test | 0 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| leakage_check.validation_test | 0 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.train.source_count | 115052 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.train.record_count | 230104 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.train.label_distribution.0 | 115052 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.train.label_distribution.1 | 115052 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.train.severity_distribution.minor | 57507 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.train.severity_distribution.severe | 17292 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.train.severity_distribution.moderate | 40253 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.train.error_type_distribution.delete_character | 47014 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.train.error_type_distribution.transpose_character | 60267 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.train.error_type_distribution.duplicate_character | 14049 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.train.error_type_distribution.onset_confusion | 14217 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.train.error_type_distribution.missing_diacritic | 34426 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.train.error_type_distribution.insert_character | 22585 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.train.error_type_distribution.diacritic_confusion | 25835 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.train.error_type_distribution.wrong_tone | 21048 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.train.error_type_distribution.keyboard_neighbor | 20720 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.train.error_type_distribution.telex_unconverted | 8659 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.train.error_source_distribution.mined | 151248 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.train.error_source_distribution.rule_based | 117572 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.train.original_source_split_distribution.train | 158756 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.train.original_source_split_distribution.validation | 35344 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.train.original_source_split_distribution.test | 36004 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.train.path | C:\1_projectSamSung\data\processed\vn_newsspell_v1\train.jsonl | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.validation.source_count | 14381 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.validation.record_count | 28762 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.validation.label_distribution.0 | 14381 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.validation.label_distribution.1 | 14381 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.validation.severity_distribution.minor | 7132 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.validation.severity_distribution.moderate | 5067 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.validation.severity_distribution.severe | 2182 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.validation.error_type_distribution.onset_confusion | 1914 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.validation.error_type_distribution.diacritic_confusion | 3261 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.validation.error_type_distribution.transpose_character | 7643 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.validation.error_type_distribution.missing_diacritic | 4344 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.validation.error_type_distribution.delete_character | 5885 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.validation.error_type_distribution.keyboard_neighbor | 2508 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.validation.error_type_distribution.telex_unconverted | 1130 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.validation.error_type_distribution.wrong_tone | 2720 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.validation.error_type_distribution.insert_character | 2751 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.validation.error_type_distribution.duplicate_character | 1813 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.validation.error_source_distribution.mined | 19020 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.validation.error_source_distribution.rule_based | 14949 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.validation.original_source_split_distribution.test | 4536 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.validation.original_source_split_distribution.train | 19624 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.validation.original_source_split_distribution.validation | 4602 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.validation.path | C:\1_projectSamSung\data\processed\vn_newsspell_v1\validation.jsonl | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.test.source_count | 14382 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.test.record_count | 28764 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.test.label_distribution.0 | 14382 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.test.label_distribution.1 | 14382 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.test.severity_distribution.minor | 7133 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.test.severity_distribution.moderate | 5067 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.test.severity_distribution.severe | 2182 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.test.error_type_distribution.duplicate_character | 1785 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.test.error_type_distribution.telex_unconverted | 1144 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.test.error_type_distribution.delete_character | 5950 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.test.error_type_distribution.diacritic_confusion | 3206 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.test.error_type_distribution.transpose_character | 7657 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.test.error_type_distribution.missing_diacritic | 4378 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.test.error_type_distribution.keyboard_neighbor | 2654 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.test.error_type_distribution.insert_character | 2789 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.test.error_type_distribution.onset_confusion | 1790 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.test.error_type_distribution.wrong_tone | 2708 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.test.error_source_distribution.rule_based | 15060 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.test.error_source_distribution.mined | 19001 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.test.original_source_split_distribution.train | 19888 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.test.original_source_split_distribution.test | 4454 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.test.original_source_split_distribution.validation | 4422 | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |
| splits.test.path | C:\1_projectSamSung\data\processed\vn_newsspell_v1\test.jsonl | [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json) |

Nguồn: `C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json` — [P020](/C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json). Field/CSV columns/source code: `Các field ghi ở cột trái; không tính lại metric`.

Document count là số bản ghi clean/noisy; source count là số bài nguồn; error type/source counts là số lỗi hoặc số liên quan lỗi, không phải các nhóm tài liệu loại trừ nhau. Không cộng các error-type counts để thay document count.
Generator dùng source_id từ SHA1 bài normalized và dedup; summary ghi normalization NFC. Giữ article_segmented từ fields[article]; article_natural = thay dấu gạch dưới bằng khoảng trắng. Model cuối chỉ đọc article_segmented. Generator có lỗi trên title/abstract/article; không suy ra mọi nhãn HAS_ERROR đều có lỗi trong phần article được model đọc.
Nguồn: `C:/1_projectSamSung/scripts/09_generate_hybrid_spelling_dataset.py` — [P014](/C:/1_projectSamSung/scripts/09_generate_hybrid_spelling_dataset.py). Field/CSV columns/source code: `create_article_id, to_natural, output record fields; main`.

Nguồn: `C:/1_projectSamSung/scripts/12_split_spelling_dataset.py` — [P016](/C:/1_projectSamSung/scripts/12_split_spelling_dataset.py). Field/CSV columns/source code: `grouping source_id; split assertions`.

Có source validator 05_validate_spelling_dataset.py kiểm tra id/group/error metadata/offset/NFC/underscore, nhưng chưa xác lập log chạy riêng của validator cuối: [CHƯA ĐỦ DỮ LIỆU]. Không tự đánh dấu tất cả checks PASS từ việc có mã kiểm tra.
# 4. Module 1 — Baseline và huấn luyện
## Baseline trên validation; input article_natural
| Field JSON | Giá trị lưu trong file | Source |
| --- | --- | --- |
| text_field | article_natural | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| evaluation_split | validation | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline1.model | word_tfidf_logistic_regression | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline1.accuracy | 0.6828454210416521 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline1.error_precision | 0.7881012380848034 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline1.error_recall | 0.500173840483972 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline1.error_f1 | 0.6119618853156372 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline1.macro_f1 | 0.671896982168637 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline1.roc_auc | 0.7265096489676496 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline1.average_precision | 0.7677016465386188 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline1.confusion_matrix.tn | 12447 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline1.confusion_matrix.fp | 1934 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline1.confusion_matrix.fn | 7188 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline1.confusion_matrix.tp | 7193 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline1.severity_recall.minor.count | 7132 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline1.severity_recall.minor.recall | 0.3651149747616377 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline1.severity_recall.moderate.count | 5067 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline1.severity_recall.moderate.recall | 0.5664101045983817 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline1.severity_recall.severe.count | 2182 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline1.severity_recall.severe.recall | 0.7878093492208983 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline1.classification_report.NO_ERROR.precision | 0.6339190221543163 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline1.classification_report.NO_ERROR.recall | 0.8655170015993324 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline1.classification_report.NO_ERROR.f1-score | 0.7318320790216368 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline1.classification_report.NO_ERROR.support | 14381.0 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline1.classification_report.HAS_ERROR.precision | 0.7881012380848034 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline1.classification_report.HAS_ERROR.recall | 0.500173840483972 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline1.classification_report.HAS_ERROR.f1-score | 0.6119618853156372 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline1.classification_report.HAS_ERROR.support | 14381.0 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline1.classification_report.accuracy | 0.6828454210416521 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline1.classification_report.macro avg.precision | 0.7110101301195598 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline1.classification_report.macro avg.recall | 0.6828454210416521 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline1.classification_report.macro avg.f1-score | 0.671896982168637 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline1.classification_report.macro avg.support | 28762.0 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline1.classification_report.weighted avg.precision | 0.7110101301195598 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline1.classification_report.weighted avg.recall | 0.6828454210416521 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline1.classification_report.weighted avg.f1-score | 0.671896982168637 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline1.classification_report.weighted avg.support | 28762.0 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline1.training_seconds | 11.78338885307312 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline1.text_field | article_natural | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline1.tfidf_config.analyzer | word | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline1.tfidf_config.ngram_range | [1, 2] | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline1.tfidf_config.min_df | 3 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline1.tfidf_config.max_features | 120000 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline2.model | char_tfidf_linear_svm | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline2.accuracy | 0.8114873791808637 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline2.error_precision | 0.9258484646829547 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline2.error_recall | 0.6772129893609624 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline2.error_f1 | 0.7822489959839357 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline2.macro_f1 | 0.8080261654760287 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline2.roc_auc | 0.877903807582149 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline2.average_precision | 0.9033464757708022 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline2.confusion_matrix.tn | 13601 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline2.confusion_matrix.fp | 780 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline2.confusion_matrix.fn | 4642 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline2.confusion_matrix.tp | 9739 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline2.severity_recall.minor.count | 7132 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline2.severity_recall.minor.recall | 0.5113572630398205 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline2.severity_recall.moderate.count | 5067 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline2.severity_recall.moderate.recall | 0.7935662127491613 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline2.severity_recall.severe.count | 2182 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline2.severity_recall.severe.recall | 0.9491292392300642 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline2.classification_report.NO_ERROR.precision | 0.7455462369127884 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline2.classification_report.NO_ERROR.recall | 0.9457617690007649 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline2.classification_report.NO_ERROR.f1-score | 0.8338033349681216 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline2.classification_report.NO_ERROR.support | 14381.0 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline2.classification_report.HAS_ERROR.precision | 0.9258484646829547 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline2.classification_report.HAS_ERROR.recall | 0.6772129893609624 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline2.classification_report.HAS_ERROR.f1-score | 0.7822489959839357 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline2.classification_report.HAS_ERROR.support | 14381.0 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline2.classification_report.accuracy | 0.8114873791808637 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline2.classification_report.macro avg.precision | 0.8356973507978716 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline2.classification_report.macro avg.recall | 0.8114873791808637 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline2.classification_report.macro avg.f1-score | 0.8080261654760287 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline2.classification_report.macro avg.support | 28762.0 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline2.classification_report.weighted avg.precision | 0.8356973507978716 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline2.classification_report.weighted avg.recall | 0.8114873791808637 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline2.classification_report.weighted avg.f1-score | 0.8080261654760286 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline2.classification_report.weighted avg.support | 28762.0 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline2.training_seconds | 316.1796615123749 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline2.text_field | article_natural | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline2.tfidf_config.analyzer | char_wb | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline2.tfidf_config.ngram_range | [3, 5] | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline2.tfidf_config.min_df | 5 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |
| baseline2.tfidf_config.max_features | 120000 | [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json) |

Nguồn: `C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json` — [P021](/C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json). Field/CSV columns/source code: `Các field ghi ở cột trái; không tính lại metric`.

## Baseline config/calls
| Config/call trong mã (không tự coi là actual run) | Biểu thức | Vị trí | Source |
| --- | --- | --- | --- |
| TEXT_FIELD | 'article_natural' | line 97 | [P022](/C:/1_projectSamSung/scripts/13_train_classical_baselines.py) |
| RANDOM_SEED | 20260918 | line 104 | [P022](/C:/1_projectSamSung/scripts/13_train_classical_baselines.py) |
| WORD_MAX_FEATURES | 120000 | line 112 | [P022](/C:/1_projectSamSung/scripts/13_train_classical_baselines.py) |
| WORD_MIN_DF | 3 | line 114 | [P022](/C:/1_projectSamSung/scripts/13_train_classical_baselines.py) |
| WORD_NGRAM_RANGE | (1, 2) | line 116 | [P022](/C:/1_projectSamSung/scripts/13_train_classical_baselines.py) |
| CHAR_MAX_FEATURES | 120000 | line 130 | [P022](/C:/1_projectSamSung/scripts/13_train_classical_baselines.py) |
| CHAR_MIN_DF | 5 | line 132 | [P022](/C:/1_projectSamSung/scripts/13_train_classical_baselines.py) |
| CHAR_NGRAM_RANGE | (3, 5) | line 134 | [P022](/C:/1_projectSamSung/scripts/13_train_classical_baselines.py) |
| TfidfVectorizer | TfidfVectorizer(analyzer='word', ngram_range=WORD_NGRAM_RANGE, min_df=WORD_MIN_DF, max_df=0.995, max_features=WORD_MAX_FEATURES, lowercase=True, token_pattern='(?u)\\b\\w+\\b', sublinear_tf=True, norm='l2', dtype=np.float32) | line 708 | [P022](/C:/1_projectSamSung/scripts/13_train_classical_baselines.py) |
| LogisticRegression | LogisticRegression(C=1.0, solver='saga', max_iter=500, tol=0.001, random_state=RANDOM_SEED, verbose=1) | line 780 | [P022](/C:/1_projectSamSung/scripts/13_train_classical_baselines.py) |
| TfidfVectorizer | TfidfVectorizer(analyzer='char_wb', ngram_range=CHAR_NGRAM_RANGE, min_df=CHAR_MIN_DF, max_features=CHAR_MAX_FEATURES, lowercase=True, sublinear_tf=True, norm='l2', dtype=np.float32) | line 998 | [P022](/C:/1_projectSamSung/scripts/13_train_classical_baselines.py) |
| LinearSVC | LinearSVC(C=1.0, dual='auto', max_iter=5000, tol=0.0001, random_state=RANDOM_SEED) | line 1061 | [P022](/C:/1_projectSamSung/scripts/13_train_classical_baselines.py) |

Nguồn: `C:/1_projectSamSung/scripts/13_train_classical_baselines.py` — [P022](/C:/1_projectSamSung/scripts/13_train_classical_baselines.py). Field/CSV columns/source code: `TEXT_FIELD, WORD_MAX_FEATURES, WORD_MIN_DF, WORD_NGRAM_RANGE, CHAR_MAX_FEATURES, CHAR_MIN_DF, CHAR_NGRAM_RANGE, RANDOM_SEED, TfidfVectorizer, LogisticRegression, LinearSVC`.

## Actual training_config M1 — ưu tiên so với comment/default
| Field JSON | Giá trị lưu trong file | Source |
| --- | --- | --- |
| model_name | vinai/phobert-base-v2 | [P023](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_config.json) |
| text_field | article_segmented | [P023](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_config.json) |
| pilot_mode | false | [P023](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_config.json) |
| epochs | 2 | [P023](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_config.json) |
| max_length | 256 | [P023](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_config.json) |
| stride | 64 | [P023](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_config.json) |
| max_chunks_per_document | 3 | [P023](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_config.json) |
| document_batch_size | 12 | [P023](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_config.json) |
| validation_document_batch_size | 16 | [P023](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_config.json) |
| gradient_accumulation_steps | 1 | [P023](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_config.json) |
| effective_document_batch_size | 12 | [P023](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_config.json) |
| encoder_lr | 2e-05 | [P023](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_config.json) |
| head_lr | 0.0001 | [P023](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_config.json) |
| weight_decay | 0.01 | [P023](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_config.json) |
| warmup_ratio | 0.1 | [P023](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_config.json) |
| freeze_bottom_layers | 6 | [P023](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_config.json) |
| gradient_checkpointing | false | [P023](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_config.json) |
| num_workers | 2 | [P023](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_config.json) |
| random_seed | 20260918 | [P023](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_config.json) |

Nguồn: `C:/projectSamSung_v1/results/module1_phobert_mil_full/training_config.json` — [P023](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_config.json). Field/CSV columns/source code: `Các field ghi ở cột trái; không tính lại metric`.

## Script config và các lời gọi huấn luyện M1
| Config/call trong mã (không tự coi là actual run) | Biểu thức | Vị trí | Source |
| --- | --- | --- | --- |
| MODEL_NAME | 'vinai/phobert-base-v2' | line 60 | [P024](/C:/1_projectSamSung/scripts/14_train_phobert_mil.py) |
| TEXT_FIELD | 'article_segmented' | line 62 | [P024](/C:/1_projectSamSung/scripts/14_train_phobert_mil.py) |
| PILOT_MODE | False | line 117 | [P024](/C:/1_projectSamSung/scripts/14_train_phobert_mil.py) |
| MAX_LENGTH | 256 | line 128 | [P024](/C:/1_projectSamSung/scripts/14_train_phobert_mil.py) |
| STRIDE | 64 | line 131 | [P024](/C:/1_projectSamSung/scripts/14_train_phobert_mil.py) |
| MAX_CHUNKS_PER_DOCUMENT | 3 | line 138 | [P024](/C:/1_projectSamSung/scripts/14_train_phobert_mil.py) |
| EPOCHS | 2 | line 145 | [P024](/C:/1_projectSamSung/scripts/14_train_phobert_mil.py) |
| DOCUMENT_BATCH_SIZE | 12 | line 159 | [P024](/C:/1_projectSamSung/scripts/14_train_phobert_mil.py) |
| VALIDATION_DOCUMENT_BATCH_SIZE | 16 | line 161 | [P024](/C:/1_projectSamSung/scripts/14_train_phobert_mil.py) |
| GRADIENT_ACCUMULATION_STEPS | 1 | line 176 | [P024](/C:/1_projectSamSung/scripts/14_train_phobert_mil.py) |
| ENCODER_LR | 2e-05 | line 183 | [P024](/C:/1_projectSamSung/scripts/14_train_phobert_mil.py) |
| HEAD_LR | 0.0001 | line 185 | [P024](/C:/1_projectSamSung/scripts/14_train_phobert_mil.py) |
| WEIGHT_DECAY | 0.01 | line 187 | [P024](/C:/1_projectSamSung/scripts/14_train_phobert_mil.py) |
| WARMUP_RATIO | 0.1 | line 189 | [P024](/C:/1_projectSamSung/scripts/14_train_phobert_mil.py) |
| MAX_GRAD_NORM | 1.0 | line 191 | [P024](/C:/1_projectSamSung/scripts/14_train_phobert_mil.py) |
| FREEZE_BOTTOM_LAYERS | 6 | line 202 | [P024](/C:/1_projectSamSung/scripts/14_train_phobert_mil.py) |
| RANDOM_SEED | 20260918 | line 229 | [P024](/C:/1_projectSamSung/scripts/14_train_phobert_mil.py) |
| USE_AMP | DEVICE.type == 'cuda' | line 242 | [P024](/C:/1_projectSamSung/scripts/14_train_phobert_mil.py) |
| torch.amp.autocast | torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP) | line 1194 | [P024](/C:/1_projectSamSung/scripts/14_train_phobert_mil.py) |
| nn.BCEWithLogitsLoss | nn.BCEWithLogitsLoss() | line 2037 | [P024](/C:/1_projectSamSung/scripts/14_train_phobert_mil.py) |
| get_linear_schedule_with_warmup | get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_optimizer_steps) | line 2062 | [P024](/C:/1_projectSamSung/scripts/14_train_phobert_mil.py) |
| torch.amp.GradScaler | torch.amp.GradScaler('cuda', enabled=USE_AMP) | line 2073 | [P024](/C:/1_projectSamSung/scripts/14_train_phobert_mil.py) |
| nn.Dropout | nn.Dropout(0.1) | line 788 | [P024](/C:/1_projectSamSung/scripts/14_train_phobert_mil.py) |
| nn.Linear | nn.Linear(hidden_size, 1) | line 792 | [P024](/C:/1_projectSamSung/scripts/14_train_phobert_mil.py) |
| torch.optim.AdamW | torch.optim.AdamW(optimizer_groups, weight_decay=WEIGHT_DECAY, fused=DEVICE.type == 'cuda') | line 2005 | [P024](/C:/1_projectSamSung/scripts/14_train_phobert_mil.py) |
| torch.optim.AdamW | torch.optim.AdamW(optimizer_groups, weight_decay=WEIGHT_DECAY) | line 2025 | [P024](/C:/1_projectSamSung/scripts/14_train_phobert_mil.py) |

Nguồn: `C:/1_projectSamSung/scripts/14_train_phobert_mil.py` — [P024](/C:/1_projectSamSung/scripts/14_train_phobert_mil.py). Field/CSV columns/source code: `MODEL_NAME, TEXT_FIELD, PILOT_MODE, MAX_LENGTH, STRIDE, MAX_CHUNKS_PER_DOCUMENT, EPOCHS, DOCUMENT_BATCH_SIZE, VALIDATION_DOCUMENT_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS, ENCODER_LR, HEAD_LR, WEIGHT_DECAY, WARMUP_RATIO, MAX_GRAD_NORM, FREEZE_BOTTOM_LAYERS, RANDOM_SEED, USE_AMP, Dropout, Linear, BCEWithLogitsLoss, AdamW, get_linear_schedule_with_warmup, autocast, GradScaler`.

Source code: DocumentCollator chia token không special tokens rồi thêm special tokens; step_size=content_length−STRIDE, chọn tối đa max_chunks bằng evenly_spaced_indices. PhoBERTMIL lấy last_hidden_state[:,0,:], dropout→linear một logit/chunk, max logit mỗi document. Không có sequence-labeling head. Loss BCEWithLogitsLoss() không truyền class/positive weight trong mã; không tự ghi một hệ số class weight đã train nếu config không lưu.
Nguồn: `C:/1_projectSamSung/scripts/14_train_phobert_mil.py` — [P024](/C:/1_projectSamSung/scripts/14_train_phobert_mil.py). Field/CSV columns/source code: `DocumentCollator; evenly_spaced_indices; PhoBERTMIL.forward; freeze_lower_layers; main`.

## Training history M1 — toàn bộ epoch đã lưu
| Field JSON | Giá trị lưu trong file | Source |
| --- | --- | --- |
| [0].accuracy | 0.908316528753216 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [0].error_precision | 0.9504449217551396 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [0].error_recall | 0.861553438564773 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [0].error_f1 | 0.9038187985556406 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [0].macro_f1 | 0.9081155971719977 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [0].roc_auc | 0.9492152339376507 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [0].average_precision | 0.9623412157468804 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [0].confusion_matrix.tn | 13735 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [0].confusion_matrix.fp | 646 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [0].confusion_matrix.fn | 1991 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [0].confusion_matrix.tp | 12390 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [0].severity_recall.minor.count | 7132 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [0].severity_recall.minor.recall | 0.7774817722938867 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [0].severity_recall.moderate.count | 5067 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [0].severity_recall.moderate.recall | 0.9265837773830669 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [0].severity_recall.severe.count | 2182 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [0].severity_recall.severe.recall | 0.9853345554537122 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [0].validation_loss | 0.268471171381211 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [0].validation_seconds | 175.91592502593994 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [0].epoch | 1 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [0].average_train_loss | 0.3310996303027888 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [0].epoch_training_seconds | 3029.914954185486 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [0].documents_per_second | 75.9440457832446 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [0].pilot_mode | false | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [0].max_length | 256 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [0].stride | 64 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [0].max_chunks_per_document | 3 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [0].document_batch_size | 12 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [0].gradient_accumulation_steps | 1 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [1].accuracy | 0.9142271052082609 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [1].error_precision | 0.9446849805912213 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [1].error_recall | 0.8799805298657951 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [1].error_f1 | 0.9111855131943695 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [1].macro_f1 | 0.9141263902409136 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [1].roc_auc | 0.9520655119235859 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [1].average_precision | 0.9650803662450511 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [1].confusion_matrix.tn | 13640 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [1].confusion_matrix.fp | 741 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [1].confusion_matrix.fn | 1726 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [1].confusion_matrix.tp | 12655 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [1].severity_recall.minor.count | 7132 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [1].severity_recall.minor.recall | 0.8037016264722378 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [1].severity_recall.moderate.count | 5067 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [1].severity_recall.moderate.recall | 0.9398065916716005 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [1].severity_recall.severe.count | 2182 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [1].severity_recall.severe.recall | 0.9903758020164987 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [1].validation_loss | 0.25995274605302016 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [1].validation_seconds | 169.8191568851471 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [1].epoch | 2 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [1].average_train_loss | 0.2552648423489272 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [1].epoch_training_seconds | 3018.289467573166 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [1].documents_per_second | 76.23655798163503 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [1].pilot_mode | false | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [1].max_length | 256 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [1].stride | 64 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [1].max_chunks_per_document | 3 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [1].document_batch_size | 12 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |
| [1].gradient_accumulation_steps | 1 | [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json) |

Nguồn: `C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json` — [P025](/C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json). Field/CSV columns/source code: `Các field ghi ở cột trái; không tính lại metric`.

## Best validation M1 trước tuning
| Field JSON | Giá trị lưu trong file | Source |
| --- | --- | --- |
| accuracy | 0.9142271052082609 | [P026](/C:/projectSamSung_v1/results/module1_phobert_mil_full/best_validation_metrics.json) |
| error_precision | 0.9446849805912213 | [P026](/C:/projectSamSung_v1/results/module1_phobert_mil_full/best_validation_metrics.json) |
| error_recall | 0.8799805298657951 | [P026](/C:/projectSamSung_v1/results/module1_phobert_mil_full/best_validation_metrics.json) |
| error_f1 | 0.9111855131943695 | [P026](/C:/projectSamSung_v1/results/module1_phobert_mil_full/best_validation_metrics.json) |
| macro_f1 | 0.9141263902409136 | [P026](/C:/projectSamSung_v1/results/module1_phobert_mil_full/best_validation_metrics.json) |
| roc_auc | 0.9520655119235859 | [P026](/C:/projectSamSung_v1/results/module1_phobert_mil_full/best_validation_metrics.json) |
| average_precision | 0.9650803662450511 | [P026](/C:/projectSamSung_v1/results/module1_phobert_mil_full/best_validation_metrics.json) |
| confusion_matrix.tn | 13640 | [P026](/C:/projectSamSung_v1/results/module1_phobert_mil_full/best_validation_metrics.json) |
| confusion_matrix.fp | 741 | [P026](/C:/projectSamSung_v1/results/module1_phobert_mil_full/best_validation_metrics.json) |
| confusion_matrix.fn | 1726 | [P026](/C:/projectSamSung_v1/results/module1_phobert_mil_full/best_validation_metrics.json) |
| confusion_matrix.tp | 12655 | [P026](/C:/projectSamSung_v1/results/module1_phobert_mil_full/best_validation_metrics.json) |
| severity_recall.minor.count | 7132 | [P026](/C:/projectSamSung_v1/results/module1_phobert_mil_full/best_validation_metrics.json) |
| severity_recall.minor.recall | 0.8037016264722378 | [P026](/C:/projectSamSung_v1/results/module1_phobert_mil_full/best_validation_metrics.json) |
| severity_recall.moderate.count | 5067 | [P026](/C:/projectSamSung_v1/results/module1_phobert_mil_full/best_validation_metrics.json) |
| severity_recall.moderate.recall | 0.9398065916716005 | [P026](/C:/projectSamSung_v1/results/module1_phobert_mil_full/best_validation_metrics.json) |
| severity_recall.severe.count | 2182 | [P026](/C:/projectSamSung_v1/results/module1_phobert_mil_full/best_validation_metrics.json) |
| severity_recall.severe.recall | 0.9903758020164987 | [P026](/C:/projectSamSung_v1/results/module1_phobert_mil_full/best_validation_metrics.json) |
| validation_loss | 0.25995274605302016 | [P026](/C:/projectSamSung_v1/results/module1_phobert_mil_full/best_validation_metrics.json) |
| validation_seconds | 169.8191568851471 | [P026](/C:/projectSamSung_v1/results/module1_phobert_mil_full/best_validation_metrics.json) |
| epoch | 2 | [P026](/C:/projectSamSung_v1/results/module1_phobert_mil_full/best_validation_metrics.json) |
| average_train_loss | 0.2552648423489272 | [P026](/C:/projectSamSung_v1/results/module1_phobert_mil_full/best_validation_metrics.json) |
| epoch_training_seconds | 3018.289467573166 | [P026](/C:/projectSamSung_v1/results/module1_phobert_mil_full/best_validation_metrics.json) |
| documents_per_second | 76.23655798163503 | [P026](/C:/projectSamSung_v1/results/module1_phobert_mil_full/best_validation_metrics.json) |
| pilot_mode | false | [P026](/C:/projectSamSung_v1/results/module1_phobert_mil_full/best_validation_metrics.json) |
| max_length | 256 | [P026](/C:/projectSamSung_v1/results/module1_phobert_mil_full/best_validation_metrics.json) |
| stride | 64 | [P026](/C:/projectSamSung_v1/results/module1_phobert_mil_full/best_validation_metrics.json) |
| max_chunks_per_document | 3 | [P026](/C:/projectSamSung_v1/results/module1_phobert_mil_full/best_validation_metrics.json) |
| document_batch_size | 12 | [P026](/C:/projectSamSung_v1/results/module1_phobert_mil_full/best_validation_metrics.json) |
| gradient_accumulation_steps | 1 | [P026](/C:/projectSamSung_v1/results/module1_phobert_mil_full/best_validation_metrics.json) |

Nguồn: `C:/projectSamSung_v1/results/module1_phobert_mil_full/best_validation_metrics.json` — [P026](/C:/projectSamSung_v1/results/module1_phobert_mil_full/best_validation_metrics.json). Field/CSV columns/source code: `Các field ghi ở cột trái; không tính lại metric`.

## Threshold tuning M1 trên validation
| Field JSON | Giá trị lưu trong file | Source |
| --- | --- | --- |
| model | vinai/phobert-base-v2 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| checkpoint | C:\1_projectSamSung\models\module1_phobert_mil_full\best_model.pt | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| checkpoint_epoch | 2 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| text_field | article_segmented | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| max_length | 256 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| stride | 64 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| max_chunks_per_document | 3 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| validation_records | 28762 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| validation_loss | 0.25995274605302016 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| inference_seconds | 175.94606637954712 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| roc_auc | 0.9520655119235859 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| average_precision | 0.9650803662450511 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| threshold_sweep.minimum | 0.1 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| threshold_sweep.maximum | 0.9 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| threshold_sweep.step | 0.01 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| primary_selection_rule | maximum_validation_macro_f1 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| selected_threshold | 0.57 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| selected_metrics.threshold | 0.57 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| selected_metrics.accuracy | 0.914783394756971 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| selected_metrics.error_precision | 0.9479573445479123 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| selected_metrics.error_recall | 0.8777553716709547 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| selected_metrics.error_f1 | 0.9115066613712677 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| selected_metrics.macro_f1 | 0.9146663960291146 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| selected_metrics.specificity | 0.9518114178429873 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| selected_metrics.tn | 13688 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| selected_metrics.fp | 693 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| selected_metrics.fn | 1758 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| selected_metrics.tp | 12623 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| selected_metrics.minor_recall | 0.8008973639932697 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| selected_metrics.moderate_recall | 0.9378330373001776 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| selected_metrics.severe_recall | 0.9894592117323556 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.default_threshold_0_5.threshold | 0.5 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.default_threshold_0_5.accuracy | 0.9142271052082609 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.default_threshold_0_5.error_precision | 0.9446849805912213 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.default_threshold_0_5.error_recall | 0.8799805298657951 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.default_threshold_0_5.error_f1 | 0.9111855131943695 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.default_threshold_0_5.macro_f1 | 0.9141263902409136 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.default_threshold_0_5.specificity | 0.9484736805507267 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.default_threshold_0_5.tn | 13640 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.default_threshold_0_5.fp | 741 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.default_threshold_0_5.fn | 1726 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.default_threshold_0_5.tp | 12655 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.default_threshold_0_5.minor_recall | 0.8037016264722378 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.default_threshold_0_5.moderate_recall | 0.9398065916716005 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.default_threshold_0_5.severe_recall | 0.9903758020164987 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.best_macro_f1.threshold | 0.57 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.best_macro_f1.accuracy | 0.914783394756971 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.best_macro_f1.error_precision | 0.9479573445479123 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.best_macro_f1.error_recall | 0.8777553716709547 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.best_macro_f1.error_f1 | 0.9115066613712677 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.best_macro_f1.macro_f1 | 0.9146663960291146 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.best_macro_f1.specificity | 0.9518114178429873 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.best_macro_f1.tn | 13688 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.best_macro_f1.fp | 693 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.best_macro_f1.fn | 1758 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.best_macro_f1.tp | 12623 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.best_macro_f1.minor_recall | 0.8008973639932697 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.best_macro_f1.moderate_recall | 0.9378330373001776 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.best_macro_f1.severe_recall | 0.9894592117323556 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.best_error_f1.threshold | 0.57 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.best_error_f1.accuracy | 0.914783394756971 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.best_error_f1.error_precision | 0.9479573445479123 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.best_error_f1.error_recall | 0.8777553716709547 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.best_error_f1.error_f1 | 0.9115066613712677 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.best_error_f1.macro_f1 | 0.9146663960291146 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.best_error_f1.specificity | 0.9518114178429873 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.best_error_f1.tn | 13688 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.best_error_f1.fp | 693 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.best_error_f1.fn | 1758 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.best_error_f1.tp | 12623 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.best_error_f1.minor_recall | 0.8008973639932697 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.best_error_f1.moderate_recall | 0.9378330373001776 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.best_error_f1.severe_recall | 0.9894592117323556 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.recall_at_least_90.threshold | 0.15 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.recall_at_least_90.accuracy | 0.9051874000417217 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.recall_at_least_90.error_precision | 0.9083964115503224 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.recall_at_least_90.error_recall | 0.9012586051039566 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.recall_at_least_90.error_f1 | 0.9048134315333869 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.recall_at_least_90.macro_f1 | 0.9051859365459143 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.recall_at_least_90.specificity | 0.9091161949794868 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.recall_at_least_90.tn | 13074 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.recall_at_least_90.fp | 1307 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.recall_at_least_90.fn | 1420 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.recall_at_least_90.tp | 12961 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.recall_at_least_90.minor_recall | 0.8359506449803702 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.recall_at_least_90.moderate_recall | 0.9540161831458457 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.recall_at_least_90.severe_recall | 0.9922089825847846 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.recall_at_least_92.threshold | 0.1 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.recall_at_least_92.accuracy | 0.8591196717891663 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.recall_at_least_92.error_precision | 0.8168599300570587 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.recall_at_least_92.error_recall | 0.92580488144079 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.recall_at_least_92.error_f1 | 0.8679269882659714 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.recall_at_least_92.macro_f1 | 0.8584903902736503 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.recall_at_least_92.specificity | 0.7924344621375425 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.recall_at_least_92.tn | 11396 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.recall_at_least_92.fp | 2985 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.recall_at_least_92.fn | 1067 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.recall_at_least_92.tp | 13314 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.recall_at_least_92.minor_recall | 0.8708637128435222 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.recall_at_least_92.moderate_recall | 0.9723702388000789 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| operating_points.recall_at_least_92.severe_recall | 0.997250229147571 | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |
| test_set_used | false | [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json) |

Nguồn: `C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json` — [P028](/C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json). Field/CSV columns/source code: `Các field ghi ở cột trái; không tính lại metric`.

Checkpoint selection theo validation macro-F1 trong main; threshold selection ở bước riêng theo maximum_validation_macro_f1. Final threshold 0.57 được artifact chọn; không tune lại. Training loss/validation loss và macro-F1 history không đồng nhất kết quả sau tuning threshold.
# 5. Module 1 — Official test
## Official M1 — dataset VN-NewsSpell v1 / test / HAS_ERROR
| Field JSON | Giá trị lưu trong file | Source |
| --- | --- | --- |
| threshold | 0.57 | [P029](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/official_test_metrics.json) |
| accuracy | 0.9148240856626338 | [P029](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/official_test_metrics.json) |
| error_precision | 0.9442955019362526 | [P029](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/official_test_metrics.json) |
| error_recall | 0.8816576275900431 | [P029](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/official_test_metrics.json) |
| error_f1 | 0.9119021934555915 | [P029](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/official_test_metrics.json) |
| macro_f1 | 0.9147302877904514 | [P029](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/official_test_metrics.json) |
| specificity | 0.9479905437352246 | [P029](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/official_test_metrics.json) |
| roc_auc | 0.9519615544670721 | [P029](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/official_test_metrics.json) |
| average_precision | 0.9639383022801679 | [P029](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/official_test_metrics.json) |
| confusion_matrix.tn | 13634 | [P029](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/official_test_metrics.json) |
| confusion_matrix.fp | 748 | [P029](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/official_test_metrics.json) |
| confusion_matrix.fn | 1702 | [P029](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/official_test_metrics.json) |
| confusion_matrix.tp | 12680 | [P029](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/official_test_metrics.json) |
| severity_recall.minor.count | 7133 | [P029](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/official_test_metrics.json) |
| severity_recall.minor.recall | 0.8028879854198794 | [P029](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/official_test_metrics.json) |
| severity_recall.moderate.count | 5067 | [P029](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/official_test_metrics.json) |
| severity_recall.moderate.recall | 0.9457272547858694 | [P029](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/official_test_metrics.json) |
| severity_recall.severe.count | 2182 | [P029](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/official_test_metrics.json) |
| severity_recall.severe.recall | 0.9903758020164987 | [P029](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/official_test_metrics.json) |
| test_loss | 0.2650065634666929 | [P029](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/official_test_metrics.json) |
| inference_seconds | 175.75773000717163 | [P029](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/official_test_metrics.json) |
| test_records | 28764 | [P029](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/official_test_metrics.json) |
| clean_records | 14382 | [P029](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/official_test_metrics.json) |
| noisy_records | 14382 | [P029](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/official_test_metrics.json) |
| model_name | vinai/phobert-base-v2 | [P029](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/official_test_metrics.json) |
| checkpoint_epoch | 2 | [P029](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/official_test_metrics.json) |
| text_field | article_segmented | [P029](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/official_test_metrics.json) |
| max_length | 256 | [P029](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/official_test_metrics.json) |
| stride | 64 | [P029](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/official_test_metrics.json) |
| max_chunks_per_document | 3 | [P029](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/official_test_metrics.json) |
| threshold_selection_rule | maximum_validation_macro_f1 | [P029](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/official_test_metrics.json) |
| threshold_selected_on | validation | [P029](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/official_test_metrics.json) |
| threshold_tuned_on_test | false | [P029](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/official_test_metrics.json) |

Nguồn: `C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/official_test_metrics.json` — [P029](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/official_test_metrics.json). Field/CSV columns/source code: `Các field ghi ở cột trái; không tính lại metric`.

Macro precision và macro recall không có field trong official JSON này: [CHƯA ĐỦ DỮ LIỆU]. Không tự suy ra từ confusion matrix để bổ sung vào bảng official. Positive class HAS_ERROR (1); ngưỡng 0.57, article_segmented. Không dùng metric word_detector/all_fields/correction.
## M1 test error analysis
| Field JSON | Giá trị lưu trong file | Source |
| --- | --- | --- |
| threshold | 0.57 | [P031](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/test_error_analysis.json) |
| false_negative_count | 1702 | [P031](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/test_error_analysis.json) |
| false_positive_count | 748 | [P031](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/test_error_analysis.json) |
| false_negative_by_severity.minor | 1406 | [P031](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/test_error_analysis.json) |
| false_negative_by_severity.moderate | 275 | [P031](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/test_error_analysis.json) |
| false_negative_by_severity.severe | 21 | [P031](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/test_error_analysis.json) |
| false_negative_by_error_count.1 | 1483 | [P031](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/test_error_analysis.json) |
| false_negative_by_error_count.2 | 179 | [P031](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/test_error_analysis.json) |
| false_negative_by_error_count.3 | 23 | [P031](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/test_error_analysis.json) |
| false_negative_by_error_count.4 | 7 | [P031](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/test_error_analysis.json) |
| false_negative_by_error_count.5 | 7 | [P031](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/test_error_analysis.json) |
| false_negative_by_error_count.6 | 1 | [P031](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/test_error_analysis.json) |
| false_negative_by_error_count.8 | 2 | [P031](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/test_error_analysis.json) |
| false_negative_error_types.missing_diacritic | 346 | [P031](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/test_error_analysis.json) |
| false_negative_error_types.delete_character | 338 | [P031](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/test_error_analysis.json) |
| false_negative_error_types.transpose_character | 293 | [P031](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/test_error_analysis.json) |
| false_negative_error_types.wrong_tone | 261 | [P031](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/test_error_analysis.json) |
| false_negative_error_types.diacritic_confusion | 168 | [P031](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/test_error_analysis.json) |
| false_negative_error_types.keyboard_neighbor | 161 | [P031](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/test_error_analysis.json) |
| false_negative_error_types.onset_confusion | 134 | [P031](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/test_error_analysis.json) |
| false_negative_error_types.duplicate_character | 100 | [P031](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/test_error_analysis.json) |
| false_negative_error_types.insert_character | 89 | [P031](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/test_error_analysis.json) |
| false_negative_error_types.telex_unconverted | 53 | [P031](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/test_error_analysis.json) |
| false_negative_error_sources.rule_based | 1025 | [P031](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/test_error_analysis.json) |
| false_negative_error_sources.mined | 804 | [P031](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/test_error_analysis.json) |
| probability_statistics.clean.mean | 0.118462785496166 | [P031](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/test_error_analysis.json) |
| probability_statistics.clean.median | 0.05987548828125 | [P031](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/test_error_analysis.json) |
| probability_statistics.clean.p95 | 0.6410888671874995 | [P031](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/test_error_analysis.json) |
| probability_statistics.noisy.mean | 0.8855744066001012 | [P031](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/test_error_analysis.json) |
| probability_statistics.noisy.median | 0.998046875 | [P031](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/test_error_analysis.json) |
| probability_statistics.noisy.p05 | 0.0710540771484375 | [P031](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/test_error_analysis.json) |

Nguồn: `C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/test_error_analysis.json` — [P031](/C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/test_error_analysis.json). Field/CSV columns/source code: `Các field ghi ở cột trái; không tính lại metric`.

FN/FP và phân bố lỗi là mô tả artifact; một tài liệu có nhiều loại/nguồn lỗi nên các nhóm có thể chồng lấn. Không suy ra nguyên nhân nhân quả hoặc định vị từ sai từ các thống kê này.
# 6. Module 2 — Dataset
## Nguồn download các dataset headline (vai trò theo task)
| Field JSON | Giá trị lưu trong file | Source |
| --- | --- | --- |
| module | module2_headline_integrity | [P032](/C:/1_projectSamSung/module2_headline_integrity/data/raw/download_manifest.json) |
| created_at | 2026-09-19T02:45:54.876745+07:00 | [P032](/C:/1_projectSamSung/module2_headline_integrity/data/raw/download_manifest.json) |
| datasets.viclickbait_2025.task | clickbait | [P032](/C:/1_projectSamSung/module2_headline_integrity/data/raw/download_manifest.json) |
| datasets.viclickbait_2025.source_type | github | [P032](/C:/1_projectSamSung/module2_headline_integrity/data/raw/download_manifest.json) |
| datasets.viclickbait_2025.source | blanatole/ViClickbait-2025 | [P032](/C:/1_projectSamSung/module2_headline_integrity/data/raw/download_manifest.json) |
| datasets.viclickbait_2025.local_path | data\raw\clickbait\viclickbait_2025 | [P032](/C:/1_projectSamSung/module2_headline_integrity/data/raw/download_manifest.json) |
| datasets.viclickbait_2025.size_bytes | 3997013 | [P032](/C:/1_projectSamSung/module2_headline_integrity/data/raw/download_manifest.json) |
| datasets.viclickbait_2025.files[0].name | clickbait_dataset_vietnamese.csv | [P032](/C:/1_projectSamSung/module2_headline_integrity/data/raw/download_manifest.json) |
| datasets.viclickbait_2025.files[0].size_bytes | 1769245 | [P032](/C:/1_projectSamSung/module2_headline_integrity/data/raw/download_manifest.json) |
| datasets.viclickbait_2025.files[0].sha256 | a8c346648666155e97a4c0aecd5a1a01131cc94631d78965b83da3739158182e | [P032](/C:/1_projectSamSung/module2_headline_integrity/data/raw/download_manifest.json) |
| datasets.viclickbait_2025.files[1].name | clickbait_dataset_vietnamese.jsonl | [P032](/C:/1_projectSamSung/module2_headline_integrity/data/raw/download_manifest.json) |
| datasets.viclickbait_2025.files[1].size_bytes | 2225898 | [P032](/C:/1_projectSamSung/module2_headline_integrity/data/raw/download_manifest.json) |
| datasets.viclickbait_2025.files[1].sha256 | 177584fdbcf8ce41dd5611061f138a0613039de9f5bffbd5deaf35e0921e6b2e | [P032](/C:/1_projectSamSung/module2_headline_integrity/data/raw/download_manifest.json) |
| datasets.viclickbait_2025.files[2].name | README.md | [P032](/C:/1_projectSamSung/module2_headline_integrity/data/raw/download_manifest.json) |
| datasets.viclickbait_2025.files[2].size_bytes | 1870 | [P032](/C:/1_projectSamSung/module2_headline_integrity/data/raw/download_manifest.json) |
| datasets.viclickbait_2025.files[2].sha256 | f9c70b095910c9d0228d621b7ad61cff12ba73e7913ce0eb5013d2d019c434ac | [P032](/C:/1_projectSamSung/module2_headline_integrity/data/raw/download_manifest.json) |
| datasets.vianli.task | consistency_auxiliary | [P032](/C:/1_projectSamSung/module2_headline_integrity/data/raw/download_manifest.json) |
| datasets.vianli.source_type | huggingface | [P032](/C:/1_projectSamSung/module2_headline_integrity/data/raw/download_manifest.json) |
| datasets.vianli.source | uitnlp/ViANLI | [P032](/C:/1_projectSamSung/module2_headline_integrity/data/raw/download_manifest.json) |
| datasets.vianli.local_path | data\raw\consistency\vianli | [P032](/C:/1_projectSamSung/module2_headline_integrity/data/raw/download_manifest.json) |
| datasets.vianli.size_bytes | 3627875 | [P032](/C:/1_projectSamSung/module2_headline_integrity/data/raw/download_manifest.json) |
| datasets.viet_online_news.task | consistency_corpus | [P032](/C:/1_projectSamSung/module2_headline_integrity/data/raw/download_manifest.json) |
| datasets.viet_online_news.source_type | huggingface | [P032](/C:/1_projectSamSung/module2_headline_integrity/data/raw/download_manifest.json) |
| datasets.viet_online_news.source | VLUS06/VietOnlineNews | [P032](/C:/1_projectSamSung/module2_headline_integrity/data/raw/download_manifest.json) |
| datasets.viet_online_news.local_path | data\raw\consistency\viet_online_news | [P032](/C:/1_projectSamSung/module2_headline_integrity/data/raw/download_manifest.json) |
| datasets.viet_online_news.size_bytes | 2512653369 | [P032](/C:/1_projectSamSung/module2_headline_integrity/data/raw/download_manifest.json) |

Nguồn: `C:/1_projectSamSung/module2_headline_integrity/data/raw/download_manifest.json` — [P032](/C:/1_projectSamSung/module2_headline_integrity/data/raw/download_manifest.json). Field/CSV columns/source code: `Các field ghi ở cột trái; không tính lại metric`.

## ViClickbait v1 — split, nhãn và overlap
| Field JSON | Giá trị lưu trong file | Source |
| --- | --- | --- |
| stage | 03_prepare_clickbait_dataset | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| task | 2A_clickbait_detection | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| dataset | viclickbait_v1 | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| source | data\raw\clickbait\viclickbait_2025\clickbait_dataset_vietnamese.csv | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| raw_data_modified | false | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| seed | 20260921 | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| raw_rows | 3414 | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| raw_label_distribution.normal | 2349 | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| raw_label_distribution.clickbait | 1065 | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| raw_label_distribution.clickbait_rate | 0.31195079086115995 | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| missing.empty_title | 0 | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| missing.empty_lead | 3 | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| duplicates.normalized_title.duplicate_rows | 2 | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| duplicates.normalized_title.duplicate_groups | 1 | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| duplicates.normalized_title.conflicting_groups | 0 | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| duplicates.normalized_title.max_group_size | 2 | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| duplicates.duplicate_nonempty_url_rows | 0 | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| split_policy.train_target_ratio | 0.7 | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| split_policy.validation_target_ratio | 0.15 | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| split_policy.test_target_ratio | 0.15 | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| split_policy.split_unit | normalized_title | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| split_policy.stratification | clickbait label | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| split_policy.method | stratified group split by normalized title | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| splits.train.split | train | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| splits.train.rows | 2390 | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| splits.train.title_groups | 2389 | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| splits.train.normal | 1645 | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| splits.train.clickbait | 745 | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| splits.train.normal_rate | 0.6882845188284519 | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| splits.train.clickbait_rate | 0.3117154811715481 | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| splits.train.empty_lead | 2 | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| splits.validation.split | validation | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| splits.validation.rows | 512 | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| splits.validation.title_groups | 512 | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| splits.validation.normal | 352 | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| splits.validation.clickbait | 160 | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| splits.validation.normal_rate | 0.6875 | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| splits.validation.clickbait_rate | 0.3125 | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| splits.validation.empty_lead | 0 | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| splits.test.split | test | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| splits.test.rows | 512 | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| splits.test.title_groups | 512 | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| splits.test.normal | 352 | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| splits.test.clickbait | 160 | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| splits.test.normal_rate | 0.6875 | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| splits.test.clickbait_rate | 0.3125 | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| splits.test.empty_lead | 1 | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| cross_split_sample_overlap.train_validation | 0 | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| cross_split_sample_overlap.train_test | 0 | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| cross_split_sample_overlap.validation_test | 0 | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| cross_split_title_overlap.train_validation | 0 | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| cross_split_title_overlap.train_test | 0 | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| cross_split_title_overlap.validation_test | 0 | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| cross_split_url_overlap.train_validation | 0 | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| cross_split_url_overlap.train_test | 0 | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| cross_split_url_overlap.validation_test | 0 | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| test_policy.status | LOCKED | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| test_policy.allowed_during_model_development | false | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |
| test_policy.usage | Official final evaluation after architecture and threshold are frozen. | [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json) |

Nguồn: `C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json` — [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json). Field/CSV columns/source code: `Các field ghi ở cột trái; không tính lại metric`.

Nguồn tải blanatole/ViClickbait-2025 được manifest ghi nhận. Dataset dẫn xuất viclickbait_v1; grouping normalized title và stratified label theo script/prepare report. Input final title_only; title+lead là đối chứng validation, không phải final runtime M2. Nhãn dương CLICKBAIT; nhãn âm runtime NORMAL. README lưu chưa có DOI; không suy ra tình trạng công bố hiện tại ngoài repository.
Nguồn: `C:/1_projectSamSung/module2_headline_integrity/data/raw/download_manifest.json` — [P032](/C:/1_projectSamSung/module2_headline_integrity/data/raw/download_manifest.json). Field/CSV columns/source code: `datasets.viclickbait_2025`.

Nguồn: `C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json` — [P034](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json). Field/CSV columns/source code: `split_policy, splits, cross_split_*_overlap, test_policy`.

# 7. Module 2 — Baseline và training
## Diagnostic baselines M2: mỗi baseline và threshold riêng
| Field JSON | Giá trị lưu trong file | Source |
| --- | --- | --- |
| stage | 05_train_diagnostic_baselines | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| task | 2A_clickbait_detection | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| dataset | viclickbait_v1 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| train_rows | 2390 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| validation_rows | 512 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| test_loaded | false | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[0].model | majority | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[0].macro_f1_05 | 0.4074074074074074 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[0].accuracy_05 | 0.6875 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[0].precision_clickbait_05 | 0.0 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[0].recall_clickbait_05 | 0.0 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[0].f1_clickbait_05 | 0.0 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[0].roc_auc | 0.5 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[0].average_precision | 0.3125 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[0].tuned_threshold | 0.4999999999999998 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[0].macro_f1_tuned | 0.4074074074074074 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[0].accuracy_tuned | 0.6875 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[0].precision_clickbait_tuned | 0.0 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[0].recall_clickbait_tuned | 0.0 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[0].f1_clickbait_tuned | 0.0 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[0].confusion_matrix_tuned.tn | 352 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[0].confusion_matrix_tuned.fp | 0 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[0].confusion_matrix_tuned.fn | 160 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[0].confusion_matrix_tuned.tp | 0 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[1].model | source_only | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[1].macro_f1_05 | 0.6153072269977714 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[1].accuracy_05 | 0.654296875 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[1].precision_clickbait_05 | 0.455026455026455 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[1].recall_clickbait_05 | 0.5375 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[1].f1_clickbait_05 | 0.49283667621776506 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[1].roc_auc | 0.6847212357954545 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[1].average_precision | 0.4823611470349379 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[1].tuned_threshold | 0.5099999999999998 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[1].macro_f1_tuned | 0.6279470987281004 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[1].accuracy_tuned | 0.697265625 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[1].precision_clickbait_tuned | 0.5190839694656488 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[1].recall_clickbait_tuned | 0.425 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[1].f1_clickbait_tuned | 0.46735395189003437 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[1].confusion_matrix_tuned.tn | 289 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[1].confusion_matrix_tuned.fp | 63 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[1].confusion_matrix_tuned.fn | 92 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[1].confusion_matrix_tuned.tp | 68 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[2].model | source_category | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[2].macro_f1_05 | 0.6431970500041813 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[2].accuracy_05 | 0.658203125 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[2].precision_clickbait_05 | 0.46963562753036436 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[2].recall_clickbait_05 | 0.725 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[2].f1_clickbait_05 | 0.5700245700245701 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[2].roc_auc | 0.7110440340909091 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[2].average_precision | 0.5503939698061331 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[2].tuned_threshold | 0.5099999999999998 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[2].macro_f1_tuned | 0.6534686971235195 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[2].accuracy_tuned | 0.671875 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[2].precision_clickbait_tuned | 0.4829059829059829 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[2].recall_clickbait_tuned | 0.70625 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[2].f1_clickbait_tuned | 0.5736040609137056 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[2].confusion_matrix_tuned.tn | 231 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[2].confusion_matrix_tuned.fp | 121 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[2].confusion_matrix_tuned.fn | 47 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[2].confusion_matrix_tuned.tp | 113 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[3].model | surface_features | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[3].macro_f1_05 | 0.7152838798684071 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[3].accuracy_05 | 0.744140625 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[3].precision_clickbait_05 | 0.5767195767195767 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[3].recall_clickbait_05 | 0.68125 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[3].f1_clickbait_05 | 0.6246418338108882 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[3].roc_auc | 0.779403409090909 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[3].average_precision | 0.6072744374337427 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[3].tuned_threshold | 0.4999999999999998 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[3].macro_f1_tuned | 0.7152838798684071 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[3].accuracy_tuned | 0.744140625 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[3].precision_clickbait_tuned | 0.5767195767195767 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[3].recall_clickbait_tuned | 0.68125 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[3].f1_clickbait_tuned | 0.6246418338108882 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[3].confusion_matrix_tuned.tn | 272 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[3].confusion_matrix_tuned.fp | 80 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[3].confusion_matrix_tuned.fn | 51 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[3].confusion_matrix_tuned.tp | 109 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[4].model | tfidf_title | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[4].macro_f1_05 | 0.7315945670775297 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[4].accuracy_05 | 0.763671875 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[4].precision_clickbait_05 | 0.6114285714285714 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[4].recall_clickbait_05 | 0.66875 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[4].f1_clickbait_05 | 0.6388059701492538 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[4].roc_auc | 0.8238813920454545 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[4].average_precision | 0.6865930902240073 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[4].tuned_threshold | 0.5199999999999998 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[4].macro_f1_tuned | 0.7418611351195621 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[4].accuracy_tuned | 0.78125 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[4].precision_clickbait_tuned | 0.6578947368421053 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[4].recall_clickbait_tuned | 0.625 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[4].f1_clickbait_tuned | 0.6410256410256411 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[4].confusion_matrix_tuned.tn | 300 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[4].confusion_matrix_tuned.fp | 52 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[4].confusion_matrix_tuned.fn | 60 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[4].confusion_matrix_tuned.tp | 100 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[5].model | tfidf_title_lead | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[5].macro_f1_05 | 0.755619196389829 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[5].accuracy_05 | 0.78515625 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[5].precision_clickbait_05 | 0.6436781609195402 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[5].recall_clickbait_05 | 0.7 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[5].f1_clickbait_05 | 0.6706586826347305 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[5].roc_auc | 0.8568181818181819 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[5].average_precision | 0.7442995947129062 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[5].tuned_threshold | 0.45999999999999985 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[5].macro_f1_tuned | 0.7569583469204955 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[5].accuracy_tuned | 0.775390625 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[5].precision_clickbait_tuned | 0.6066350710900474 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[5].recall_clickbait_tuned | 0.8 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[5].f1_clickbait_tuned | 0.6900269541778976 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[5].confusion_matrix_tuned.tn | 269 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[5].confusion_matrix_tuned.fp | 83 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[5].confusion_matrix_tuned.fn | 32 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| baselines[5].confusion_matrix_tuned.tp | 128 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| comparisons.tfidf_title_lead_minus_title_macro_f1 | 0.015097211800933374 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| comparisons.source_only_macro_f1 | 0.6279470987281004 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| comparisons.surface_only_macro_f1 | 0.7152838798684071 | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| interpretation_policy.source_models | Diagnostic only. Publisher identity will not be used as a production model input. | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| interpretation_policy.surface_models | Diagnostic only. Used to measure possible shallow shortcuts. | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |
| interpretation_policy.test_policy | TEST remains locked until final architecture and threshold are frozen. | [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json) |

Nguồn: `C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json` — [P036](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json). Field/CSV columns/source code: `Các field ghi ở cột trái; không tính lại metric`.

## Model: mbert_title_only_v1
## Actual run config mbert_title_only_v1
| Field JSON | Giá trị lưu trong file | Source |
| --- | --- | --- |
| stage | 06_train_mbert_title_only | [P038](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/run_config.json) |
| task | 2A_clickbait_detection | [P038](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/run_config.json) |
| dataset | viclickbait_v1 | [P038](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/run_config.json) |
| model | bert-base-multilingual-cased | [P038](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/run_config.json) |
| input | title_only | [P038](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/run_config.json) |
| max_length | 160 | [P038](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/run_config.json) |
| train_rows | 2390 | [P038](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/run_config.json) |
| validation_rows | 512 | [P038](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/run_config.json) |
| test_loaded | false | [P038](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/run_config.json) |
| batch_size | 16 | [P038](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/run_config.json) |
| epochs | 5 | [P038](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/run_config.json) |
| early_stopping_patience | 2 | [P038](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/run_config.json) |
| learning_rate | 2e-05 | [P038](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/run_config.json) |
| weight_decay | 0.01 | [P038](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/run_config.json) |
| warmup_ratio | 0.1 | [P038](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/run_config.json) |
| seed | 20260921 | [P038](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/run_config.json) |
| class_counts.normal | 1645 | [P038](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/run_config.json) |
| class_counts.clickbait | 745 | [P038](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/run_config.json) |
| class_weights.normal | 0.7264437689969605 | [P038](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/run_config.json) |
| class_weights.clickbait | 1.604026845637584 | [P038](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/run_config.json) |

Nguồn: `C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/run_config.json` — [P038](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/run_config.json). Field/CSV columns/source code: `Các field ghi ở cột trái; không tính lại metric`.

## Script defaults/calls mbert_title_only_v1
| Config/call trong mã (không tự coi là actual run) | Biểu thức | Vị trí | Source |
| --- | --- | --- | --- |
| MODEL_NAME | 'bert-base-multilingual-cased' | line 105 | [P039](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/06_train_mbert_title_only.py) |
| MAX_LENGTH | 160 | line 109 | [P039](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/06_train_mbert_title_only.py) |
| DEFAULT_BATCH_SIZE | 16 | line 111 | [P039](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/06_train_mbert_title_only.py) |
| DEFAULT_EPOCHS | 5 | line 112 | [P039](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/06_train_mbert_title_only.py) |
| DEFAULT_PATIENCE | 2 | line 113 | [P039](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/06_train_mbert_title_only.py) |
| DEFAULT_LR | 2e-05 | line 115 | [P039](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/06_train_mbert_title_only.py) |
| DEFAULT_WEIGHT_DECAY | 0.01 | line 116 | [P039](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/06_train_mbert_title_only.py) |
| DEFAULT_WARMUP_RATIO | 0.1 | line 117 | [P039](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/06_train_mbert_title_only.py) |
| DEFAULT_MAX_GRAD_NORM | 1.0 | line 118 | [P039](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/06_train_mbert_title_only.py) |
| DEFAULT_SEED | 20260921 | line 120 | [P039](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/06_train_mbert_title_only.py) |
| THRESHOLDS | np.arange(0.05, 0.951, 0.01) | line 123 | [P039](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/06_train_mbert_title_only.py) |
| AutoTokenizer.from_pretrained | AutoTokenizer.from_pretrained(MODEL_NAME) | line 1085 | [P039](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/06_train_mbert_title_only.py) |
| AutoModelForSequenceClassification.from_pretrained | AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2) | line 1160 | [P039](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/06_train_mbert_title_only.py) |
| torch.nn.CrossEntropyLoss | torch.nn.CrossEntropyLoss(weight=loss_weights_tensor) | line 1178 | [P039](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/06_train_mbert_title_only.py) |
| torch.optim.AdamW | torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) | line 1189 | [P039](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/06_train_mbert_title_only.py) |
| get_linear_schedule_with_warmup | get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps) | line 1212 | [P039](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/06_train_mbert_title_only.py) |
| torch.amp.GradScaler | torch.amp.GradScaler('cuda', enabled=use_amp) | line 1223 | [P039](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/06_train_mbert_title_only.py) |
| torch.amp.autocast | torch.amp.autocast(device_type=device.type, enabled=use_amp) | line 737 | [P039](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/06_train_mbert_title_only.py) |
| torch.amp.autocast | torch.amp.autocast(device_type=device.type, enabled=use_amp) | line 1411 | [P039](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/06_train_mbert_title_only.py) |

Nguồn: `C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/06_train_mbert_title_only.py` — [P039](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/06_train_mbert_title_only.py). Field/CSV columns/source code: `MODEL_NAME, MAX_LENGTH, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_PATIENCE, DEFAULT_LR, DEFAULT_WEIGHT_DECAY, DEFAULT_WARMUP_RATIO, DEFAULT_MAX_GRAD_NORM, DEFAULT_SEED, THRESHOLDS, AutoTokenizer.from_pretrained, AutoModelForSequenceClassification.from_pretrained, CrossEntropyLoss, AdamW, get_linear_schedule_with_warmup, autocast, GradScaler`.

## Training history mbert_title_only_v1
| epoch | train_loss | validation_loss | validation_accuracy_05 | validation_macro_f1_05 | validation_precision_clickbait_05 | validation_recall_clickbait_05 | validation_f1_clickbait_05 | validation_roc_auc | elapsed_seconds | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.5763107772152793 | 0.5757721005938947 | 0.787109375 | 0.7501152976470324 | 0.6645161290322581 | 0.64375 | 0.653968253968254 | 0.8478249289772727 | 42.97813439369202 | [P040](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/training_history.csv) |
| 2 | 0.3942097845935422 | 0.5234215962700546 | 0.8046875 | 0.776438738974762 | 0.6764705882352942 | 0.71875 | 0.696969696969697 | 0.8743519176136364 | 42.76021885871887 | [P040](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/training_history.csv) |
| 3 | 0.28246701255004275 | 0.5299397739581764 | 0.810546875 | 0.7841662212139782 | 0.6820809248554913 | 0.7375 | 0.7087087087087087 | 0.8777965198863636 | 41.80023264884949 | [P040](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/training_history.csv) |
| 4 | 0.16965578430055575 | 0.7689522467553616 | 0.8203125 | 0.7887144523190095 | 0.7207792207792207 | 0.69375 | 0.7070063694267515 | 0.8832563920454547 | 43.73925805091858 | [P040](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/training_history.csv) |
| 5 | 0.09796882704071794 | 0.8686952276621014 | 0.8125 | 0.7810679352183441 | 0.7025316455696202 | 0.69375 | 0.6981132075471698 | 0.8840465198863636 | 43.59174966812134 | [P040](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/training_history.csv) |

Nguồn: `C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/training_history.csv` — [P040](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/training_history.csv). Field/CSV columns/source code: `epoch, train_loss, validation_loss, validation_accuracy_05, validation_macro_f1_05, validation_precision_clickbait_05, validation_recall_clickbait_05, validation_f1_clickbait_05, validation_roc_auc, elapsed_seconds`.

## Validation metrics mbert_title_only_v1
| Field JSON | Giá trị lưu trong file | Source |
| --- | --- | --- |
| stage | 06_train_mbert_title_only | [P041](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/validation_metrics.json) |
| task | 2A_clickbait_detection | [P041](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/validation_metrics.json) |
| dataset | viclickbait_v1 | [P041](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/validation_metrics.json) |
| model | bert-base-multilingual-cased | [P041](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/validation_metrics.json) |
| input | title_only | [P041](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/validation_metrics.json) |
| test_used | false | [P041](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/validation_metrics.json) |
| best_epoch | 4 | [P041](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/validation_metrics.json) |
| total_training_seconds | 221.47967052459717 | [P041](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/validation_metrics.json) |
| validation_loss | 0.7689522467553616 | [P041](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/validation_metrics.json) |
| metrics_at_0_5.threshold | 0.5 | [P041](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/validation_metrics.json) |
| metrics_at_0_5.accuracy | 0.8203125 | [P041](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/validation_metrics.json) |
| metrics_at_0_5.precision_clickbait | 0.7207792207792207 | [P041](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/validation_metrics.json) |
| metrics_at_0_5.recall_clickbait | 0.69375 | [P041](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/validation_metrics.json) |
| metrics_at_0_5.f1_clickbait | 0.7070063694267515 | [P041](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/validation_metrics.json) |
| metrics_at_0_5.macro_f1 | 0.7887144523190095 | [P041](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/validation_metrics.json) |
| metrics_at_0_5.roc_auc | 0.8832563920454547 | [P041](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/validation_metrics.json) |
| metrics_at_0_5.average_precision | 0.7554753254554589 | [P041](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/validation_metrics.json) |
| metrics_at_0_5.confusion_matrix.tn | 309 | [P041](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/validation_metrics.json) |
| metrics_at_0_5.confusion_matrix.fp | 43 | [P041](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/validation_metrics.json) |
| metrics_at_0_5.confusion_matrix.fn | 49 | [P041](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/validation_metrics.json) |
| metrics_at_0_5.confusion_matrix.tp | 111 | [P041](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/validation_metrics.json) |
| best_validation_threshold | 0.6100000000000001 | [P041](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/validation_metrics.json) |
| metrics_at_tuned_threshold.threshold | 0.6100000000000001 | [P041](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/validation_metrics.json) |
| metrics_at_tuned_threshold.accuracy | 0.82421875 | [P041](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/validation_metrics.json) |
| metrics_at_tuned_threshold.precision_clickbait | 0.7302631578947368 | [P041](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/validation_metrics.json) |
| metrics_at_tuned_threshold.recall_clickbait | 0.69375 | [P041](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/validation_metrics.json) |
| metrics_at_tuned_threshold.f1_clickbait | 0.7115384615384616 | [P041](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/validation_metrics.json) |
| metrics_at_tuned_threshold.macro_f1 | 0.7925669835782195 | [P041](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/validation_metrics.json) |
| metrics_at_tuned_threshold.roc_auc | 0.8832563920454547 | [P041](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/validation_metrics.json) |
| metrics_at_tuned_threshold.average_precision | 0.7554753254554589 | [P041](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/validation_metrics.json) |
| metrics_at_tuned_threshold.confusion_matrix.tn | 311 | [P041](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/validation_metrics.json) |
| metrics_at_tuned_threshold.confusion_matrix.fp | 41 | [P041](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/validation_metrics.json) |
| metrics_at_tuned_threshold.confusion_matrix.fn | 49 | [P041](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/validation_metrics.json) |
| metrics_at_tuned_threshold.confusion_matrix.tp | 111 | [P041](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/validation_metrics.json) |
| note | Checkpoint selected on validation Macro-F1 @0.5. Threshold tuned on validation after checkpoint selection. TEST remains untouched. | [P041](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/validation_metrics.json) |

Nguồn: `C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/validation_metrics.json` — [P041](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/validation_metrics.json). Field/CSV columns/source code: `Các field ghi ở cột trái; không tính lại metric`.

## Model: mbert_title_lead_v1
## Actual run config mbert_title_lead_v1
| Field JSON | Giá trị lưu trong file | Source |
| --- | --- | --- |
| stage | 07_train_mbert_title_lead | [P042](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/run_config.json) |
| task | 2A_clickbait_detection | [P042](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/run_config.json) |
| dataset | viclickbait_v1 | [P042](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/run_config.json) |
| model | bert-base-multilingual-cased | [P042](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/run_config.json) |
| input | title_plus_lead | [P042](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/run_config.json) |
| truncation | only_second | [P042](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/run_config.json) |
| max_length | 160 | [P042](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/run_config.json) |
| train_rows | 2390 | [P042](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/run_config.json) |
| validation_rows | 512 | [P042](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/run_config.json) |
| test_loaded | false | [P042](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/run_config.json) |
| batch_size | 16 | [P042](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/run_config.json) |
| epochs | 5 | [P042](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/run_config.json) |
| early_stopping_patience | 2 | [P042](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/run_config.json) |
| learning_rate | 2e-05 | [P042](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/run_config.json) |
| weight_decay | 0.01 | [P042](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/run_config.json) |
| warmup_ratio | 0.1 | [P042](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/run_config.json) |
| seed | 20260921 | [P042](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/run_config.json) |
| class_counts.normal | 1645 | [P042](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/run_config.json) |
| class_counts.clickbait | 745 | [P042](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/run_config.json) |
| class_weights.normal | 0.7264437689969605 | [P042](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/run_config.json) |
| class_weights.clickbait | 1.604026845637584 | [P042](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/run_config.json) |
| controlled_comparison_with_step06 | true | [P042](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/run_config.json) |

Nguồn: `C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/run_config.json` — [P042](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/run_config.json). Field/CSV columns/source code: `Các field ghi ở cột trái; không tính lại metric`.

## Script defaults/calls mbert_title_lead_v1
| Config/call trong mã (không tự coi là actual run) | Biểu thức | Vị trí | Source |
| --- | --- | --- | --- |
| MODEL_NAME | 'bert-base-multilingual-cased' | line 105 | [P043](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/07_train_mbert_title_lead.py) |
| MAX_LENGTH | 160 | line 107 | [P043](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/07_train_mbert_title_lead.py) |
| DEFAULT_BATCH_SIZE | 16 | line 109 | [P043](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/07_train_mbert_title_lead.py) |
| DEFAULT_EPOCHS | 5 | line 110 | [P043](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/07_train_mbert_title_lead.py) |
| DEFAULT_PATIENCE | 2 | line 111 | [P043](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/07_train_mbert_title_lead.py) |
| DEFAULT_LR | 2e-05 | line 113 | [P043](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/07_train_mbert_title_lead.py) |
| DEFAULT_WEIGHT_DECAY | 0.01 | line 114 | [P043](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/07_train_mbert_title_lead.py) |
| DEFAULT_WARMUP_RATIO | 0.1 | line 115 | [P043](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/07_train_mbert_title_lead.py) |
| DEFAULT_MAX_GRAD_NORM | 1.0 | line 116 | [P043](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/07_train_mbert_title_lead.py) |
| DEFAULT_SEED | 20260921 | line 118 | [P043](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/07_train_mbert_title_lead.py) |
| THRESHOLDS | np.arange(0.05, 0.951, 0.01) | line 121 | [P043](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/07_train_mbert_title_lead.py) |
| AutoTokenizer.from_pretrained | AutoTokenizer.from_pretrained(MODEL_NAME) | line 1098 | [P043](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/07_train_mbert_title_lead.py) |
| AutoModelForSequenceClassification.from_pretrained | AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2) | line 1159 | [P043](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/07_train_mbert_title_lead.py) |
| torch.nn.CrossEntropyLoss | torch.nn.CrossEntropyLoss(weight=loss_weights_tensor) | line 1176 | [P043](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/07_train_mbert_title_lead.py) |
| torch.optim.AdamW | torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) | line 1184 | [P043](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/07_train_mbert_title_lead.py) |
| get_linear_schedule_with_warmup | get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps) | line 1205 | [P043](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/07_train_mbert_title_lead.py) |
| torch.amp.GradScaler | torch.amp.GradScaler('cuda', enabled=use_amp) | line 1216 | [P043](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/07_train_mbert_title_lead.py) |
| torch.amp.autocast | torch.amp.autocast(device_type=device.type, enabled=use_amp) | line 763 | [P043](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/07_train_mbert_title_lead.py) |
| torch.amp.autocast | torch.amp.autocast(device_type=device.type, enabled=use_amp) | line 1410 | [P043](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/07_train_mbert_title_lead.py) |

Nguồn: `C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/07_train_mbert_title_lead.py` — [P043](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/07_train_mbert_title_lead.py). Field/CSV columns/source code: `MODEL_NAME, MAX_LENGTH, DEFAULT_BATCH_SIZE, DEFAULT_EPOCHS, DEFAULT_PATIENCE, DEFAULT_LR, DEFAULT_WEIGHT_DECAY, DEFAULT_WARMUP_RATIO, DEFAULT_MAX_GRAD_NORM, DEFAULT_SEED, THRESHOLDS, AutoTokenizer.from_pretrained, AutoModelForSequenceClassification.from_pretrained, CrossEntropyLoss, AdamW, get_linear_schedule_with_warmup, autocast, GradScaler`.

## Training history mbert_title_lead_v1
| epoch | train_loss | validation_loss | validation_accuracy_05 | validation_macro_f1_05 | validation_precision_clickbait_05 | validation_recall_clickbait_05 | validation_f1_clickbait_05 | validation_roc_auc | elapsed_seconds | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.5731875725869853 | 0.5707225997466594 | 0.794921875 | 0.7389974511469839 | 0.7391304347826086 | 0.53125 | 0.6181818181818182 | 0.8678799715909089 | 44.470383167266846 | [P044](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/training_history.csv) |
| 2 | 0.40560634490336334 | 0.5030835955403745 | 0.798828125 | 0.757672672465686 | 0.7021276595744681 | 0.61875 | 0.6578073089700996 | 0.8833362926136364 | 49.1506404876709 | [P044](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/training_history.csv) |
| 3 | 0.24235684896301027 | 0.5350931259454228 | 0.8125 | 0.7853811894157716 | 0.6882352941176471 | 0.73125 | 0.7090909090909091 | 0.8918678977272728 | 49.33427906036377 | [P044](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/training_history.csv) |
| 4 | 0.14119989767701804 | 0.8554151765929419 | 0.810546875 | 0.7760290065526307 | 0.7086092715231788 | 0.66875 | 0.6881028938906752 | 0.8878373579545455 | 47.25757718086243 | [P044](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/training_history.csv) |
| 5 | 0.07752968653038955 | 1.0258033621321374 | 0.818359375 | 0.7867956209282021 | 0.7161290322580646 | 0.69375 | 0.7047619047619048 | 0.8896750710227275 | 49.09709072113037 | [P044](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/training_history.csv) |

Nguồn: `C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/training_history.csv` — [P044](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/training_history.csv). Field/CSV columns/source code: `epoch, train_loss, validation_loss, validation_accuracy_05, validation_macro_f1_05, validation_precision_clickbait_05, validation_recall_clickbait_05, validation_f1_clickbait_05, validation_roc_auc, elapsed_seconds`.

## Validation metrics mbert_title_lead_v1
| Field JSON | Giá trị lưu trong file | Source |
| --- | --- | --- |
| stage | 07_train_mbert_title_lead | [P045](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/validation_metrics.json) |
| task | 2A_clickbait_detection | [P045](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/validation_metrics.json) |
| dataset | viclickbait_v1 | [P045](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/validation_metrics.json) |
| model | bert-base-multilingual-cased | [P045](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/validation_metrics.json) |
| input | title_plus_lead | [P045](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/validation_metrics.json) |
| truncation | only_second | [P045](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/validation_metrics.json) |
| test_used | false | [P045](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/validation_metrics.json) |
| best_epoch | 5 | [P045](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/validation_metrics.json) |
| total_training_seconds | 246.7133469581604 | [P045](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/validation_metrics.json) |
| validation_loss | 1.0258033621321374 | [P045](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/validation_metrics.json) |
| metrics_at_0_5.threshold | 0.5 | [P045](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/validation_metrics.json) |
| metrics_at_0_5.accuracy | 0.818359375 | [P045](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/validation_metrics.json) |
| metrics_at_0_5.precision_clickbait | 0.7161290322580646 | [P045](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/validation_metrics.json) |
| metrics_at_0_5.recall_clickbait | 0.69375 | [P045](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/validation_metrics.json) |
| metrics_at_0_5.f1_clickbait | 0.7047619047619048 | [P045](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/validation_metrics.json) |
| metrics_at_0_5.macro_f1 | 0.7867956209282021 | [P045](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/validation_metrics.json) |
| metrics_at_0_5.roc_auc | 0.8896750710227275 | [P045](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/validation_metrics.json) |
| metrics_at_0_5.average_precision | 0.764046008488936 | [P045](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/validation_metrics.json) |
| metrics_at_0_5.confusion_matrix.tn | 308 | [P045](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/validation_metrics.json) |
| metrics_at_0_5.confusion_matrix.fp | 44 | [P045](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/validation_metrics.json) |
| metrics_at_0_5.confusion_matrix.fn | 49 | [P045](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/validation_metrics.json) |
| metrics_at_0_5.confusion_matrix.tp | 111 | [P045](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/validation_metrics.json) |
| best_validation_threshold | 0.23000000000000004 | [P045](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/validation_metrics.json) |
| metrics_at_tuned_threshold.threshold | 0.23000000000000004 | [P045](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/validation_metrics.json) |
| metrics_at_tuned_threshold.accuracy | 0.8203125 | [P045](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/validation_metrics.json) |
| metrics_at_tuned_threshold.precision_clickbait | 0.7048192771084337 | [P045](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/validation_metrics.json) |
| metrics_at_tuned_threshold.recall_clickbait | 0.73125 | [P045](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/validation_metrics.json) |
| metrics_at_tuned_threshold.f1_clickbait | 0.7177914110429447 | [P045](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/validation_metrics.json) |
| metrics_at_tuned_threshold.macro_f1 | 0.7929931267249108 | [P045](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/validation_metrics.json) |
| metrics_at_tuned_threshold.roc_auc | 0.8896750710227275 | [P045](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/validation_metrics.json) |
| metrics_at_tuned_threshold.average_precision | 0.764046008488936 | [P045](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/validation_metrics.json) |
| metrics_at_tuned_threshold.confusion_matrix.tn | 303 | [P045](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/validation_metrics.json) |
| metrics_at_tuned_threshold.confusion_matrix.fp | 49 | [P045](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/validation_metrics.json) |
| metrics_at_tuned_threshold.confusion_matrix.fn | 43 | [P045](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/validation_metrics.json) |
| metrics_at_tuned_threshold.confusion_matrix.tp | 117 | [P045](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/validation_metrics.json) |
| controlled_comparison_step06.same_model | true | [P045](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/validation_metrics.json) |
| controlled_comparison_step06.same_max_length | true | [P045](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/validation_metrics.json) |
| controlled_comparison_step06.same_batch_size | true | [P045](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/validation_metrics.json) |
| controlled_comparison_step06.same_learning_rate | true | [P045](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/validation_metrics.json) |
| controlled_comparison_step06.same_class_weight_policy | true | [P045](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/validation_metrics.json) |
| controlled_comparison_step06.same_checkpoint_policy | true | [P045](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/validation_metrics.json) |
| controlled_comparison_step06.changed_variable | lead_added_as_second_sequence | [P045](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/validation_metrics.json) |
| note | Checkpoint selected using validation Macro-F1 @0.5. Threshold tuned on validation after checkpoint selection. TEST remains untouched. | [P045](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/validation_metrics.json) |

Nguồn: `C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/validation_metrics.json` — [P045](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/validation_metrics.json). Field/CSV columns/source code: `Các field ghi ở cột trái; không tính lại metric`.

Hai mBERT run dùng weighted CrossEntropyLoss theo class_weights được tính từ train; bảng config ghi hệ số thực tế. Checkpoint được chọn theo validation macro-F1 ở ngưỡng 0.5; sau đó tune threshold trên validation của checkpoint được chọn. Không lấy best epoch của title+lead làm best epoch final title-only.
Nguồn: `C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/06_train_mbert_title_only.py` — [P039](/C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/06_train_mbert_title_only.py). Field/CSV columns/source code: `compute_class_weights; tune_threshold; main`.

## So sánh title-only/title+lead, McNemar và paired bootstrap
| Field JSON | Giá trị lưu trong file | Source |
| --- | --- | --- |
| stage | 08_compare_mbert_title_vs_title_lead | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| task | 2A_clickbait_detection | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| dataset | viclickbait_v1 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| split | validation | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| test_used | false | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.A_title_only.threshold | 0.6100000000000001 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.A_title_only.metrics_at_0_5.threshold | 0.5 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.A_title_only.metrics_at_0_5.accuracy | 0.8203125 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.A_title_only.metrics_at_0_5.precision_clickbait | 0.7207792207792207 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.A_title_only.metrics_at_0_5.recall_clickbait | 0.69375 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.A_title_only.metrics_at_0_5.f1_clickbait | 0.7070063694267515 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.A_title_only.metrics_at_0_5.macro_f1 | 0.7887144523190095 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.A_title_only.metrics_at_0_5.roc_auc | 0.8832563920454547 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.A_title_only.metrics_at_0_5.average_precision | 0.7554753254554589 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.A_title_only.metrics_at_0_5.confusion_matrix.tn | 309 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.A_title_only.metrics_at_0_5.confusion_matrix.fp | 43 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.A_title_only.metrics_at_0_5.confusion_matrix.fn | 49 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.A_title_only.metrics_at_0_5.confusion_matrix.tp | 111 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.A_title_only.metrics_at_saved_tuned_threshold.threshold | 0.6100000000000001 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.A_title_only.metrics_at_saved_tuned_threshold.accuracy | 0.82421875 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.A_title_only.metrics_at_saved_tuned_threshold.precision_clickbait | 0.7302631578947368 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.A_title_only.metrics_at_saved_tuned_threshold.recall_clickbait | 0.69375 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.A_title_only.metrics_at_saved_tuned_threshold.f1_clickbait | 0.7115384615384616 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.A_title_only.metrics_at_saved_tuned_threshold.macro_f1 | 0.7925669835782195 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.A_title_only.metrics_at_saved_tuned_threshold.roc_auc | 0.8832563920454547 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.A_title_only.metrics_at_saved_tuned_threshold.average_precision | 0.7554753254554589 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.A_title_only.metrics_at_saved_tuned_threshold.confusion_matrix.tn | 311 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.A_title_only.metrics_at_saved_tuned_threshold.confusion_matrix.fp | 41 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.A_title_only.metrics_at_saved_tuned_threshold.confusion_matrix.fn | 49 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.A_title_only.metrics_at_saved_tuned_threshold.confusion_matrix.tp | 111 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.B_title_plus_lead.threshold | 0.23000000000000004 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.B_title_plus_lead.metrics_at_0_5.threshold | 0.5 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.B_title_plus_lead.metrics_at_0_5.accuracy | 0.818359375 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.B_title_plus_lead.metrics_at_0_5.precision_clickbait | 0.7161290322580646 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.B_title_plus_lead.metrics_at_0_5.recall_clickbait | 0.69375 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.B_title_plus_lead.metrics_at_0_5.f1_clickbait | 0.7047619047619048 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.B_title_plus_lead.metrics_at_0_5.macro_f1 | 0.7867956209282021 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.B_title_plus_lead.metrics_at_0_5.roc_auc | 0.8896750710227275 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.B_title_plus_lead.metrics_at_0_5.average_precision | 0.764046008488936 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.B_title_plus_lead.metrics_at_0_5.confusion_matrix.tn | 308 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.B_title_plus_lead.metrics_at_0_5.confusion_matrix.fp | 44 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.B_title_plus_lead.metrics_at_0_5.confusion_matrix.fn | 49 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.B_title_plus_lead.metrics_at_0_5.confusion_matrix.tp | 111 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.B_title_plus_lead.metrics_at_saved_tuned_threshold.threshold | 0.23000000000000004 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.B_title_plus_lead.metrics_at_saved_tuned_threshold.accuracy | 0.8203125 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.B_title_plus_lead.metrics_at_saved_tuned_threshold.precision_clickbait | 0.7048192771084337 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.B_title_plus_lead.metrics_at_saved_tuned_threshold.recall_clickbait | 0.73125 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.B_title_plus_lead.metrics_at_saved_tuned_threshold.f1_clickbait | 0.7177914110429447 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.B_title_plus_lead.metrics_at_saved_tuned_threshold.macro_f1 | 0.7929931267249108 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.B_title_plus_lead.metrics_at_saved_tuned_threshold.roc_auc | 0.8896750710227275 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.B_title_plus_lead.metrics_at_saved_tuned_threshold.average_precision | 0.764046008488936 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.B_title_plus_lead.metrics_at_saved_tuned_threshold.confusion_matrix.tn | 303 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.B_title_plus_lead.metrics_at_saved_tuned_threshold.confusion_matrix.fp | 49 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.B_title_plus_lead.metrics_at_saved_tuned_threshold.confusion_matrix.fn | 43 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| models.B_title_plus_lead.metrics_at_saved_tuned_threshold.confusion_matrix.tp | 117 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| deltas_b_minus_a.macro_f1_at_0_5 | -0.0019188313908073962 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| deltas_b_minus_a.macro_f1_tuned | 0.0004261431466913157 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| deltas_b_minus_a.accuracy_tuned | -0.00390625 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| deltas_b_minus_a.precision_clickbait_tuned | -0.025443880786303108 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| deltas_b_minus_a.recall_clickbait_tuned | 0.03749999999999998 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| deltas_b_minus_a.f1_clickbait_tuned | 0.006252949504483185 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| deltas_b_minus_a.roc_auc | 0.006418678977272774 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| deltas_b_minus_a.average_precision | 0.008570683033477078 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| paired_error_overlap.BOTH_CORRECT.count | 390 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| paired_error_overlap.BOTH_CORRECT.rate | 0.76171875 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| paired_error_overlap.TITLE_ONLY_CORRECT.count | 32 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| paired_error_overlap.TITLE_ONLY_CORRECT.rate | 0.0625 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| paired_error_overlap.TITLE_LEAD_CORRECT.count | 30 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| paired_error_overlap.TITLE_LEAD_CORRECT.rate | 0.05859375 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| paired_error_overlap.BOTH_WRONG.count | 60 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| paired_error_overlap.BOTH_WRONG.rate | 0.1171875 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| mcnemar_exact.title_only_correct_title_lead_wrong | 32 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| mcnemar_exact.title_lead_correct_title_only_wrong | 30 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| mcnemar_exact.discordant_samples | 62 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| mcnemar_exact.two_sided_p_value | 0.8990763136528589 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| paired_bootstrap.iterations | 10000 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| paired_bootstrap.metric | Macro-F1 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| paired_bootstrap.thresholds_reoptimized_per_bootstrap | false | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| paired_bootstrap.mean_delta_b_minus_a | 0.0006206081849212814 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| paired_bootstrap.median_delta_b_minus_a | 0.0008655270742222165 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| paired_bootstrap.ci95_low | -0.0346898374069771 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| paired_bootstrap.ci95_high | 0.0353566535572874 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| paired_bootstrap.fraction_delta_greater_than_zero | 0.5191 | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| paired_bootstrap.warning | Thresholds were already tuned on this validation set. Bootstrap therefore measures sample stability, not performance on an untouched external population. | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |
| interpretation_note | Primary comparison remains Macro-F1. Subgroup and bootstrap analyses are diagnostic only. TEST remains untouched. | [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json) |

Nguồn: `C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json` — [P046](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json). Field/CSV columns/source code: `Các field ghi ở cột trái; không tính lại metric`.

Bootstrap dùng threshold đã tune trên chính validation, không tối ưu lại mỗi bootstrap; đo độ ổn định mẫu, không là khoảng tin cậy hiệu năng trên quần thể độc lập. Chênh lệch validation nhỏ và báo cáo không chứng minh title+lead vượt trội tin cậy; quyết định giữ title-only được khóa trước test theo các nguồn dưới.
## Final lock M2
| Field JSON | Giá trị lưu trong file | Source |
| --- | --- | --- |
| stage | 09_lock_final_clickbait_model | [P048](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/final_model_lock.json) |
| task | 2A_clickbait_detection | [P048](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/final_model_lock.json) |
| dataset | viclickbait_v1 | [P048](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/final_model_lock.json) |
| status | LOCKED_BEFORE_TEST | [P048](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/final_model_lock.json) |
| test_opened | false | [P048](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/final_model_lock.json) |
| test_metrics_computed | false | [P048](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/final_model_lock.json) |
| final_model.architecture | bert-base-multilingual-cased | [P048](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/final_model_lock.json) |
| final_model.input | title_only | [P048](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/final_model_lock.json) |
| final_model.input_description | headline/title only | [P048](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/final_model_lock.json) |
| final_model.max_length | 160 | [P048](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/final_model_lock.json) |
| final_model.checkpoint | models\clickbait\mbert_title_only_v1\best_model.pt | [P048](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/final_model_lock.json) |
| final_model.checkpoint_sha256 | 6c10d84f012cf05a0917e35abee4d8f006688315ab0570a3739bebea737efdf7 | [P048](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/final_model_lock.json) |
| final_model.best_epoch | 4 | [P048](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/final_model_lock.json) |
| final_model.decision_threshold | 0.6100000000000001 | [P048](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/final_model_lock.json) |
| final_model.label_mapping.0 | NORMAL | [P048](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/final_model_lock.json) |
| final_model.label_mapping.1 | CLICKBAIT | [P048](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/final_model_lock.json) |
| validation_reference.threshold_0_5.accuracy | 0.8203125 | [P048](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/final_model_lock.json) |
| validation_reference.threshold_0_5.macro_f1 | 0.7887144523190095 | [P048](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/final_model_lock.json) |
| validation_reference.threshold_0_5.precision_clickbait | 0.7207792207792207 | [P048](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/final_model_lock.json) |
| validation_reference.threshold_0_5.recall_clickbait | 0.69375 | [P048](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/final_model_lock.json) |
| validation_reference.threshold_0_5.f1_clickbait | 0.7070063694267515 | [P048](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/final_model_lock.json) |
| validation_reference.threshold_0_5.roc_auc | 0.8832563920454547 | [P048](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/final_model_lock.json) |
| validation_reference.threshold_0_5.average_precision | 0.7554753254554589 | [P048](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/final_model_lock.json) |
| validation_reference.frozen_threshold.threshold | 0.6100000000000001 | [P048](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/final_model_lock.json) |
| validation_reference.frozen_threshold.accuracy | 0.82421875 | [P048](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/final_model_lock.json) |
| validation_reference.frozen_threshold.macro_f1 | 0.7925669835782195 | [P048](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/final_model_lock.json) |
| validation_reference.frozen_threshold.precision_clickbait | 0.7302631578947368 | [P048](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/final_model_lock.json) |
| validation_reference.frozen_threshold.recall_clickbait | 0.69375 | [P048](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/final_model_lock.json) |
| validation_reference.frozen_threshold.f1_clickbait | 0.7115384615384616 | [P048](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/final_model_lock.json) |
| validation_reference.frozen_threshold.roc_auc | 0.8832563920454547 | [P048](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/final_model_lock.json) |
| validation_reference.frozen_threshold.average_precision | 0.7554753254554589 | [P048](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/final_model_lock.json) |
| model_selection_evidence.compared_against | mBERT title + lead | [P048](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/final_model_lock.json) |
| model_selection_evidence.title_lead_minus_title_macro_f1 | 0.0004261431466913157 | [P048](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/final_model_lock.json) |
| model_selection_evidence.mcnemar_exact_p_value | 0.8990763136528589 | [P048](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/final_model_lock.json) |
| model_selection_evidence.paired_bootstrap_macro_f1_delta_ci95 | [-0.0346898374069771, 0.0353566535572874] | [P048](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/final_model_lock.json) |
| model_selection_evidence.reason_for_title_only_selection | Title+lead produced essentially the same validation Macro-F1. Paired McNemar and bootstrap analyses did not show reliable superiority. Title-only is therefore retained as the simpler final architecture. | [P048](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/final_model_lock.json) |
| frozen_test_protocol.checkpoint_selection | No checkpoint may be selected using TEST. | [P048](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/final_model_lock.json) |
| frozen_test_protocol.threshold_selection | Decision threshold is frozen at 0.61 before TEST. | [P048](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/final_model_lock.json) |
| frozen_test_protocol.architecture_selection | Architecture is frozen as mBERT title-only before TEST. | [P048](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/final_model_lock.json) |
| frozen_test_protocol.test_evaluation | Run once using the frozen checkpoint and threshold. | [P048](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/final_model_lock.json) |

Nguồn: `C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/final_model_lock.json` — [P048](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/final_model_lock.json). Field/CSV columns/source code: `Các field ghi ở cột trái; không tính lại metric`.

# 8. Module 2 — Official test
## Official M2 — ViClickbait test / CLICKBAIT
| Field JSON | Giá trị lưu trong file | Source |
| --- | --- | --- |
| stage | 10_evaluate_final_clickbait_test | [P050](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_metrics.json) |
| task | 2A_clickbait_detection | [P050](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_metrics.json) |
| dataset | viclickbait_v1 | [P050](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_metrics.json) |
| evaluation_status | OFFICIAL_TEST_COMPLETE | [P050](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_metrics.json) |
| configuration_frozen_before_test | true | [P050](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_metrics.json) |
| test_tuning_performed | false | [P050](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_metrics.json) |
| test_samples | 512 | [P050](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_metrics.json) |
| test_class_distribution.normal | 352 | [P050](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_metrics.json) |
| test_class_distribution.clickbait | 160 | [P050](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_metrics.json) |
| test_class_distribution.clickbait_rate | 0.3125 | [P050](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_metrics.json) |
| frozen_configuration.architecture | bert-base-multilingual-cased | [P050](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_metrics.json) |
| frozen_configuration.input | title_only | [P050](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_metrics.json) |
| frozen_configuration.max_length | 160 | [P050](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_metrics.json) |
| frozen_configuration.decision_threshold | 0.6100000000000001 | [P050](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_metrics.json) |
| frozen_configuration.checkpoint | models\clickbait\mbert_title_only_v1\best_model.pt | [P050](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_metrics.json) |
| frozen_configuration.checkpoint_sha256 | 6c10d84f012cf05a0917e35abee4d8f006688315ab0570a3739bebea737efdf7 | [P050](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_metrics.json) |
| frozen_configuration.best_epoch | 4 | [P050](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_metrics.json) |
| official_final_metrics.threshold | 0.6100000000000001 | [P050](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_metrics.json) |
| official_final_metrics.accuracy | 0.83203125 | [P050](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_metrics.json) |
| official_final_metrics.precision_clickbait | 0.7569444444444444 | [P050](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_metrics.json) |
| official_final_metrics.recall_clickbait | 0.68125 | [P050](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_metrics.json) |
| official_final_metrics.f1_clickbait | 0.7171052631578947 | [P050](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_metrics.json) |
| official_final_metrics.macro_f1 | 0.7988304093567251 | [P050](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_metrics.json) |
| official_final_metrics.roc_auc | 0.8674183238636364 | [P050](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_metrics.json) |
| official_final_metrics.average_precision | 0.8028530848536526 | [P050](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_metrics.json) |
| official_final_metrics.confusion_matrix.tn | 317 | [P050](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_metrics.json) |
| official_final_metrics.confusion_matrix.fp | 35 | [P050](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_metrics.json) |
| official_final_metrics.confusion_matrix.fn | 51 | [P050](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_metrics.json) |
| official_final_metrics.confusion_matrix.tp | 109 | [P050](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_metrics.json) |
| inference.batch_size | 32 | [P050](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_metrics.json) |
| inference.device | cuda | [P050](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_metrics.json) |
| inference.seconds | 0.480712890625 | [P050](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_metrics.json) |
| methodology_note | TEST was evaluated using the checkpoint, architecture, input representation, max length and decision threshold frozen in Step 09. No TEST-based search or tuning was performed. | [P050](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_metrics.json) |

Nguồn: `C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_metrics.json` — [P050](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_metrics.json). Field/CSV columns/source code: `Các field ghi ở cột trái; không tính lại metric`.

Macro precision/macro recall và specificity không được lưu trong official_final_metrics: [CHƯA ĐỦ DỮ LIỆU]. Không tự điền. Macro-F1/ROC-AUC/AP dùng đúng field đã có. Thời gian inference của test script không phải HTTP latency.
## M2 official_test_by_source.csv
| source | n | normal_n | clickbait_n | clickbait_rate | accuracy | macro_f1 | precision_clickbait | recall_clickbait | f1_clickbait | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VnExpress | 101 | 80 | 21 | 0.2079207920792079 | 0.801980198019802 | 0.6882716049382716 | 0.5263157894736842 | 0.47619047619047616 | 0.5 | [P052](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_by_source.csv) |
| Thanh Niên | 92 | 73 | 19 | 0.20652173913043478 | 0.8260869565217391 | 0.7346791636625811 | 0.5789473684210527 | 0.5789473684210527 | 0.5789473684210527 | [P052](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_by_source.csv) |
| Tuổi Trẻ | 42 | 31 | 11 | 0.2619047619047619 | 0.8095238095238095 | 0.7536656891495601 | 0.6363636363636364 | 0.6363636363636364 | 0.6363636363636364 | [P052](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_by_source.csv) |
| VietNamNet | 51 | 34 | 17 | 0.3333333333333333 | 0.8235294117647058 | 0.7831837505904582 | 0.8333333333333334 | 0.5882352941176471 | 0.6896551724137931 | [P052](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_by_source.csv) |
| Báo Mới | 94 | 73 | 21 | 0.22340425531914893 | 0.8723404255319149 | 0.8021052631578948 | 0.7647058823529411 | 0.6190476190476191 | 0.6842105263157895 | [P052](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_by_source.csv) |
| 24h | 78 | 42 | 36 | 0.46153846153846156 | 0.8076923076923077 | 0.803790038571189 | 0.8387096774193549 | 0.7222222222222222 | 0.7761194029850746 | [P052](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_by_source.csv) |
| SaoStar | 48 | 16 | 32 | 0.6666666666666666 | 0.875 | 0.859375 | 0.90625 | 0.90625 | 0.90625 | [P052](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_by_source.csv) |

Nguồn: `C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_by_source.csv` — [P052](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_by_source.csv). Field/CSV columns/source code: `source, n, normal_n, clickbait_n, clickbait_rate, accuracy, macro_f1, precision_clickbait, recall_clickbait, f1_clickbait`.

## M2 official_test_by_category.csv
| category | n | normal_n | clickbait_n | clickbait_rate | accuracy | macro_f1 | precision_clickbait | recall_clickbait | f1_clickbait | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Văn hóa & Nghệ thuật | 11 | 11 | 0 | 0.0 | 0.9090909090909091 | 0.47619047619047616 | 0.0 | 0.0 | 0.0 | [P053](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_by_category.csv) |
| Kinh doanh & Kinh tế | 27 | 19 | 8 | 0.2962962962962963 | 0.7407407407407407 | 0.6454033771106942 | 0.6 | 0.375 | 0.46153846153846156 | [P053](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_by_category.csv) |
| Giáo dục & Học đường | 27 | 22 | 5 | 0.18518518518518517 | 0.8148148148148148 | 0.6666666666666666 | 0.5 | 0.4 | 0.4444444444444444 | [P053](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_by_category.csv) |
| Thể thao | 53 | 40 | 13 | 0.24528301886792453 | 0.7735849056603774 | 0.6942307692307692 | 0.5384615384615384 | 0.5384615384615384 | 0.5384615384615384 | [P053](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_by_category.csv) |
| Tin tức tổng hợp | 29 | 27 | 2 | 0.06896551724137931 | 0.9310344827586207 | 0.7314814814814814 | 0.5 | 0.5 | 0.5 | [P053](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_by_category.csv) |
| Công nghệ & Khoa học | 28 | 22 | 6 | 0.21428571428571427 | 0.8214285714285714 | 0.7495527728085868 | 0.5714285714285714 | 0.6666666666666666 | 0.6153846153846154 | [P053](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_by_category.csv) |
| Quốc tế | 37 | 26 | 11 | 0.2972972972972973 | 0.8108108108108109 | 0.7521531100478469 | 0.75 | 0.5454545454545454 | 0.631578947368421 | [P053](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_by_category.csv) |
| Other | 170 | 122 | 48 | 0.2823529411764706 | 0.8117647058823529 | 0.761570827489481 | 0.6818181818181818 | 0.625 | 0.6521739130434783 | [P053](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_by_category.csv) |
| Chính trị & An ninh | 23 | 16 | 7 | 0.30434782608695654 | 0.8695652173913043 | 0.8207792207792208 | 1.0 | 0.5714285714285714 | 0.7272727272727273 | [P053](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_by_category.csv) |
| Du lịch & Ẩm thực | 16 | 11 | 5 | 0.3125 | 0.875 | 0.8333333333333333 | 1.0 | 0.6 | 0.75 | [P053](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_by_category.csv) |
| Xe & Vận tải | 15 | 10 | 5 | 0.3333333333333333 | 0.8666666666666667 | 0.8500000000000001 | 0.8 | 0.8 | 0.8 | [P053](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_by_category.csv) |
| Giải trí & Showbiz | 41 | 12 | 29 | 0.7073170731707317 | 0.8780487804878049 | 0.856140350877193 | 0.9285714285714286 | 0.896551724137931 | 0.9122807017543859 | [P053](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_by_category.csv) |
| Sức khỏe & Đời sống | 35 | 14 | 21 | 0.6 | 0.9142857142857143 | 0.911690496215307 | 0.95 | 0.9047619047619048 | 0.926829268292683 | [P053](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_by_category.csv) |

Nguồn: `C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_by_category.csv` — [P053](/C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_by_category.csv). Field/CSV columns/source code: `category, n, normal_n, clickbait_n, clickbait_rate, accuracy, macro_f1, precision_clickbait, recall_clickbait, f1_clickbait`.

Mỗi subgroup giữ n/normal_n/clickbait_n nếu CSV có. Các nhóm ít mẫu chỉ mô tả tập test, không xếp hạng phổ quát nhà xuất bản hoặc lĩnh vực.
# 9. Module 3 — Dataset
Chuỗi dữ liệu: VietOnlineNews → articles → positive pairs → hard-negative candidates → manual review TRAIN → frozen negative policy → headline_consistency_v1. ViANLI có trong download manifest với vai trò auxiliary; chưa có căn cứ đưa metric/dataset ViANLI vào huấn luyện final M3 từ pipeline đang dùng.
## Chuẩn bị articles: counts và category
| Field JSON | Giá trị lưu trong file | Source |
| --- | --- | --- |
| module | module2_headline_integrity | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| task | headline_consistency | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| stage | 03_prepare_articles | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| source_dataset | viet_online_news | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| policy.preserve_official_splits | true | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| policy.normalize_unicode | NFC | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| policy.cross_split_title_policy | remove_all_rows_for_any_normalized_title_hash_appearing_in_multiple_splits | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| policy.within_split_duplicate_titles | retained; later negative mining must forbid same title_hash pairs | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| policy.raw_data_modified | false | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| records_scanned.train | 186010 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| records_scanned.validation | 39859 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| records_scanned.test | 39860 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| missing_titles_first_pass | {} | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| cross_split_title_groups | 195 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| excluded_records.train | 222 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| excluded_records.validation | 125 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| excluded_records.test | 101 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| excluded_total | 448 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| retained_records.train | 185788 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| retained_records.validation | 39734 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| retained_records.test | 39759 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| retained_total | 265281 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| invalid_required_records | {} | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| duplicate_content_rows | 0 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| final_cross_split_overlap.title.train__validation | 0 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| final_cross_split_overlap.title.train__test | 0 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| final_cross_split_overlap.title.validation__test | 0 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| final_cross_split_overlap.content.train__validation | 0 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| final_cross_split_overlap.content.train__test | 0 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| final_cross_split_overlap.content.validation__test | 0 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| category_distribution.train.Sức khỏe | 11623 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| category_distribution.train.Bất động sản | 4645 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| category_distribution.train.Thời sự | 19178 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| category_distribution.train.Xe | 13157 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| category_distribution.train.Giáo dục | 15676 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| category_distribution.train.Kinh doanh | 23193 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| category_distribution.train.Giải trí | 28581 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| category_distribution.train.Thể thao | 22573 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| category_distribution.train.Khoa học công nghệ | 14696 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| category_distribution.train.Du lịch | 6420 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| category_distribution.train.Đời sống | 13377 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| category_distribution.train.Thế giới | 10390 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| category_distribution.train.Pháp luật | 2279 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| category_distribution.validation.Thế giới | 2227 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| category_distribution.validation.Giáo dục | 3351 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| category_distribution.validation.Kinh doanh | 4939 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| category_distribution.validation.Bất động sản | 995 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| category_distribution.validation.Giải trí | 6117 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| category_distribution.validation.Sức khỏe | 2481 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| category_distribution.validation.Khoa học công nghệ | 3144 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| category_distribution.validation.Xe | 2815 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| category_distribution.validation.Thời sự | 4107 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| category_distribution.validation.Đời sống | 2866 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| category_distribution.validation.Du lịch | 1374 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| category_distribution.validation.Thể thao | 4829 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| category_distribution.validation.Pháp luật | 489 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| category_distribution.test.Bất động sản | 995 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| category_distribution.test.Thời sự | 4108 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| category_distribution.test.Kinh doanh | 4954 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| category_distribution.test.Sức khỏe | 2487 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| category_distribution.test.Xe | 2819 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| category_distribution.test.Đời sống | 2865 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| category_distribution.test.Thể thao | 4832 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| category_distribution.test.Giáo dục | 3350 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| category_distribution.test.Giải trí | 6117 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| category_distribution.test.Thế giới | 2225 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| category_distribution.test.Khoa học công nghệ | 3143 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| category_distribution.test.Du lịch | 1376 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| category_distribution.test.Pháp luật | 488 | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| outputs.train | data\interim\consistency\articles_v1\train.parquet | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| outputs.validation | data\interim\consistency\articles_v1\validation.parquet | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| outputs.test | data\interim\consistency\articles_v1\test.parquet | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| outputs.article_index | data\interim\consistency\articles_v1\article_index.parquet | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| outputs.excluded_titles | data\interim\consistency\articles_v1\excluded_cross_split_titles.csv | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |
| validation_passed | true | [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json) |

Nguồn: `C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json` — [P064](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json). Field/CSV columns/source code: `Các field ghi ở cột trái; không tính lại metric`.

## Positive pairs
| Field JSON | Giá trị lưu trong file | Source |
| --- | --- | --- |
| module | module2_headline_integrity | [P065](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/positive_pairs_v1/build_report.json) |
| task | headline_consistency | [P065](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/positive_pairs_v1/build_report.json) |
| stage | 04_build_positive_pairs | [P065](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/positive_pairs_v1/build_report.json) |
| definition.positive_label | 1 | [P065](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/positive_pairs_v1/build_report.json) |
| definition.positive_meaning | original headline paired with its own original article | [P065](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/positive_pairs_v1/build_report.json) |
| definition.pair_type | original_positive | [P065](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/positive_pairs_v1/build_report.json) |
| definition.full_article_content_copied | false | [P065](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/positive_pairs_v1/build_report.json) |
| definition.pair_storage | reference-based via article_uid | [P065](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/positive_pairs_v1/build_report.json) |
| counts.train | 185788 | [P065](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/positive_pairs_v1/build_report.json) |
| counts.validation | 39734 | [P065](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/positive_pairs_v1/build_report.json) |
| counts.test | 39759 | [P065](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/positive_pairs_v1/build_report.json) |
| expected_counts.train | 185788 | [P065](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/positive_pairs_v1/build_report.json) |
| expected_counts.test | 39759 | [P065](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/positive_pairs_v1/build_report.json) |
| expected_counts.validation | 39734 | [P065](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/positive_pairs_v1/build_report.json) |
| total_positive_pairs | 265281 | [P065](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/positive_pairs_v1/build_report.json) |
| outputs.train | data\interim\consistency\positive_pairs_v1\train.parquet | [P065](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/positive_pairs_v1/build_report.json) |
| outputs.validation | data\interim\consistency\positive_pairs_v1\validation.parquet | [P065](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/positive_pairs_v1/build_report.json) |
| outputs.test | data\interim\consistency\positive_pairs_v1\test.parquet | [P065](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/positive_pairs_v1/build_report.json) |
| validation_passed | true | [P065](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/positive_pairs_v1/build_report.json) |

Nguồn: `C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/positive_pairs_v1/build_report.json` — [P065](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/positive_pairs_v1/build_report.json). Field/CSV columns/source code: `Các field ghi ở cột trái; không tính lại metric`.

## Candidate mining
| Field JSON | Giá trị lưu trong file | Source |
| --- | --- | --- |
| module | module2_headline_integrity | [P066](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/candidate_pairs_v1/mining_report.json) |
| task | headline_consistency | [P066](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/candidate_pairs_v1/mining_report.json) |
| stage | 05_mine_hard_negative_candidates | [P066](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/candidate_pairs_v1/mining_report.json) |
| important_note | These records are candidate mismatches only. They are NOT yet gold negative labels. | [P066](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/candidate_pairs_v1/mining_report.json) |
| retrieval.method | same_split_same_category_tfidf_word_1_2 | [P066](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/candidate_pairs_v1/mining_report.json) |
| retrieval.scope | same split + same category | [P066](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/candidate_pairs_v1/mining_report.json) |
| retrieval.text | title repeated twice + description | [P066](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/candidate_pairs_v1/mining_report.json) |
| retrieval.vectorizer | TF-IDF | [P066](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/candidate_pairs_v1/mining_report.json) |
| retrieval.analyzer | word | [P066](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/candidate_pairs_v1/mining_report.json) |
| retrieval.ngram_range | [1, 2] | [P066](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/candidate_pairs_v1/mining_report.json) |
| retrieval.top_k | 10 | [P066](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/candidate_pairs_v1/mining_report.json) |
| retrieval.max_features | 120000 | [P066](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/candidate_pairs_v1/mining_report.json) |
| retrieval.min_similarity | 0.1 | [P066](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/candidate_pairs_v1/mining_report.json) |
| retrieval.title_repeat | 2 | [P066](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/candidate_pairs_v1/mining_report.json) |
| filters.same_article_forbidden | true | [P066](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/candidate_pairs_v1/mining_report.json) |
| filters.same_title_hash_forbidden | true | [P066](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/candidate_pairs_v1/mining_report.json) |
| filters.same_content_hash_forbidden | true | [P066](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/candidate_pairs_v1/mining_report.json) |
| filters.cross_split_candidates | false | [P066](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/candidate_pairs_v1/mining_report.json) |
| filters.cross_category_candidates | false | [P066](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/candidate_pairs_v1/mining_report.json) |
| anchor_counts.train | 185788 | [P066](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/candidate_pairs_v1/mining_report.json) |
| anchor_counts.validation | 39734 | [P066](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/candidate_pairs_v1/mining_report.json) |
| anchor_counts.test | 39759 | [P066](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/candidate_pairs_v1/mining_report.json) |
| candidate_counts.train | 1835464 | [P066](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/candidate_pairs_v1/mining_report.json) |
| candidate_counts.validation | 385204 | [P066](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/candidate_pairs_v1/mining_report.json) |
| candidate_counts.test | 385589 | [P066](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/candidate_pairs_v1/mining_report.json) |
| validated_row_counts.train | 1835464 | [P066](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/candidate_pairs_v1/mining_report.json) |
| validated_row_counts.validation | 385204 | [P066](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/candidate_pairs_v1/mining_report.json) |
| validated_row_counts.test | 385589 | [P066](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/candidate_pairs_v1/mining_report.json) |
| total_candidates | 2606257 | [P066](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/candidate_pairs_v1/mining_report.json) |
| outputs.train | data\interim\consistency\candidate_pairs_v1\train.parquet | [P066](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/candidate_pairs_v1/mining_report.json) |
| outputs.validation | data\interim\consistency\candidate_pairs_v1\validation.parquet | [P066](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/candidate_pairs_v1/mining_report.json) |
| outputs.test | data\interim\consistency\candidate_pairs_v1\test.parquet | [P066](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/candidate_pairs_v1/mining_report.json) |
| group_stats | data\interim\consistency\candidate_pairs_v1\group_stats.csv | [P066](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/candidate_pairs_v1/mining_report.json) |
| validation_passed | true | [P066](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/candidate_pairs_v1/mining_report.json) |

Nguồn: `C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/candidate_pairs_v1/mining_report.json` — [P066](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/candidate_pairs_v1/mining_report.json). Field/CSV columns/source code: `Các field ghi ở cột trái; không tính lại metric`.

## Negative pairs theo policy
| Field JSON | Giá trị lưu trong file | Source |
| --- | --- | --- |
| module | module2_headline_integrity | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| task | headline_consistency | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| stage | 08_build_negative_pairs | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| label.value | 0 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| label.meaning | INCONSISTENT | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| policy_source | Step 07 TRAIN-only manual annotation analysis | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| policy.rank_min | 4 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| policy.rank_max | 10 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| policy.primary_similarity.min_inclusive | 0.2 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| policy.primary_similarity.max_exclusive | 0.4 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| policy.fallback_similarity.min_inclusive | 0.15 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| policy.fallback_similarity.max_exclusive | 0.2 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| policy.max_negative_per_anchor | 1 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| policy.primary_before_fallback | true | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| policy.within_tier_selection | highest similarity, then lowest rank, then stable ID | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| policy.same_split_only | true | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| policy.same_category_only | true | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| policy.same_title_forbidden | true | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| policy.same_content_forbidden | true | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.train.articles | 185788 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.train.eligible_primary_candidates | 461019 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.train.eligible_fallback_candidates | 431521 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.train.negative_pairs | 153002 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.train.coverage | 0.823530044997524 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.train.primary_selected | 93787 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.train.fallback_selected | 59215 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.train.anchors_without_negative | 32786 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.train.similarity.mean | 0.23279953002929688 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.train.similarity.median | 0.21669349074363708 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.train.similarity.p05 | 0.15684226155281067 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.train.similarity.p95 | 0.3678925335407257 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.train.similarity.min | 0.15000014007091522 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.train.similarity.max | 0.3999975025653839 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.train.rank_distribution.4 | 148509 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.train.rank_distribution.5 | 1600 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.train.rank_distribution.6 | 1041 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.train.rank_distribution.8 | 471 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.train.rank_distribution.7 | 689 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.train.rank_distribution.10 | 309 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.train.rank_distribution.9 | 383 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.train.headline_reuse.split | train | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.train.headline_reuse.unique_headlines_used | 90308 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.train.headline_reuse.mean_reuse | 1.6942242104796916 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.train.headline_reuse.median_reuse | 1.0 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.train.headline_reuse.p95_reuse | 4.0 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.train.headline_reuse.p99_reuse | 5.0 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.train.headline_reuse.max_reuse | 16 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.train.validation.rows | 153002 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.train.validation.label_ok | true | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.train.validation.split_ok | true | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.train.validation.unique_pair_ids | true | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.train.validation.unique_anchors | true | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.train.validation.no_self_pairs | true | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.train.validation.no_same_title | true | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.train.validation.no_same_content | true | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.train.validation.same_category | true | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.train.validation.valid_tiers | true | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.train.validation.passed | true | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.validation.articles | 39734 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.validation.eligible_primary_candidates | 59087 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.validation.eligible_fallback_candidates | 86586 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.validation.negative_pairs | 28713 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.validation.coverage | 0.7226304927769668 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.validation.primary_selected | 13867 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.validation.fallback_selected | 14846 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.validation.anchors_without_negative | 11021 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.validation.similarity.mean | 0.21479836106300354 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.validation.similarity.median | 0.1977849006652832 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.validation.similarity.p05 | 0.1540554165840149 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.validation.similarity.p95 | 0.340209037065506 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.validation.similarity.min | 0.15000146627426147 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.validation.similarity.max | 0.3999117612838745 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.validation.rank_distribution.7 | 70 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.validation.rank_distribution.4 | 28207 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.validation.rank_distribution.10 | 34 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.validation.rank_distribution.5 | 173 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.validation.rank_distribution.8 | 60 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.validation.rank_distribution.9 | 50 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.validation.rank_distribution.6 | 119 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.validation.headline_reuse.split | validation | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.validation.headline_reuse.unique_headlines_used | 18072 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.validation.headline_reuse.mean_reuse | 1.5888114209827358 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.validation.headline_reuse.median_reuse | 1.0 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.validation.headline_reuse.p95_reuse | 3.0 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.validation.headline_reuse.p99_reuse | 5.0 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.validation.headline_reuse.max_reuse | 9 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.validation.validation.rows | 28713 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.validation.validation.label_ok | true | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.validation.validation.split_ok | true | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.validation.validation.unique_pair_ids | true | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.validation.validation.unique_anchors | true | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.validation.validation.no_self_pairs | true | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.validation.validation.no_same_title | true | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.validation.validation.no_same_content | true | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.validation.validation.same_category | true | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.validation.validation.valid_tiers | true | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.validation.validation.passed | true | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.test.articles | 39759 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.test.eligible_primary_candidates | 60606 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.test.eligible_fallback_candidates | 86392 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.test.negative_pairs | 28688 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.test.coverage | 0.7215473226187782 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.test.primary_selected | 14049 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.test.fallback_selected | 14639 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.test.anchors_without_negative | 11071 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.test.similarity.mean | 0.21535013616085052 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.test.similarity.median | 0.19869810342788696 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.test.similarity.p05 | 0.15440930426120758 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.test.similarity.p95 | 0.3376576602458954 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.test.similarity.min | 0.1500026434659958 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.test.similarity.max | 0.3999636173248291 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.test.rank_distribution.4 | 28227 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.test.rank_distribution.8 | 47 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.test.rank_distribution.6 | 103 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.test.rank_distribution.5 | 183 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.test.rank_distribution.7 | 62 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.test.rank_distribution.9 | 41 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.test.rank_distribution.10 | 25 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.test.headline_reuse.split | test | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.test.headline_reuse.unique_headlines_used | 17891 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.test.headline_reuse.mean_reuse | 1.603487787155553 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.test.headline_reuse.median_reuse | 1.0 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.test.headline_reuse.p95_reuse | 3.0 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.test.headline_reuse.p99_reuse | 5.0 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.test.headline_reuse.max_reuse | 9 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.test.validation.rows | 28688 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.test.validation.label_ok | true | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.test.validation.split_ok | true | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.test.validation.unique_pair_ids | true | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.test.validation.unique_anchors | true | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.test.validation.no_self_pairs | true | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.test.validation.no_same_title | true | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.test.validation.no_same_content | true | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.test.validation.same_category | true | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.test.validation.valid_tiers | true | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| splits.test.validation.passed | true | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| total_negative_pairs | 210403 | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| outputs.train | data\interim\consistency\negative_pairs_v1\train.parquet | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| outputs.validation | data\interim\consistency\negative_pairs_v1\validation.parquet | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| outputs.test | data\interim\consistency\negative_pairs_v1\test.parquet | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |
| headline_reuse_stats | data\interim\consistency\negative_pairs_v1\headline_reuse_stats.csv | [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json) |

Nguồn: `C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json` — [P067](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json). Field/CSV columns/source code: `Các field ghi ở cột trái; không tính lại metric`.

## Manual review TRAIN
| Field JSON | Giá trị lưu trong file | Source |
| --- | --- | --- |
| module | module2_headline_integrity | [P068](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/manual_annotation_analysis_v1/analysis_report.json) |
| task | headline_consistency | [P068](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/manual_annotation_analysis_v1/analysis_report.json) |
| stage | 07_analyze_manual_annotations | [P068](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/manual_annotation_analysis_v1/analysis_report.json) |
| rows | 240 | [P068](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/manual_annotation_analysis_v1/analysis_report.json) |
| validation.empty_annotations | 0 | [P068](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/manual_annotation_analysis_v1/analysis_report.json) |
| validation.duplicate_pair_ids | 0 | [P068](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/manual_annotation_analysis_v1/analysis_report.json) |
| validation.invalid_labels | [] | [P068](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/manual_annotation_analysis_v1/analysis_report.json) |
| validation.passed | true | [P068](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/manual_annotation_analysis_v1/analysis_report.json) |
| overall.VALID_NEGATIVE.count | 192 | [P068](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/manual_annotation_analysis_v1/analysis_report.json) |
| overall.VALID_NEGATIVE.rate | 0.8 | [P068](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/manual_annotation_analysis_v1/analysis_report.json) |
| overall.FALSE_NEGATIVE.count | 45 | [P068](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/manual_annotation_analysis_v1/analysis_report.json) |
| overall.FALSE_NEGATIVE.rate | 0.1875 | [P068](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/manual_annotation_analysis_v1/analysis_report.json) |
| overall.AMBIGUOUS.count | 3 | [P068](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/manual_annotation_analysis_v1/analysis_report.json) |
| overall.AMBIGUOUS.rate | 0.0125 | [P068](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/manual_annotation_analysis_v1/analysis_report.json) |
| candidate_policy_v1.primary.rank | 4-10 | [P068](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/manual_annotation_analysis_v1/analysis_report.json) |
| candidate_policy_v1.primary.similarity_min | 0.2 | [P068](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/manual_annotation_analysis_v1/analysis_report.json) |
| candidate_policy_v1.primary.similarity_max_exclusive | 0.4 | [P068](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/manual_annotation_analysis_v1/analysis_report.json) |
| candidate_policy_v1.primary.manual_evidence.n | 30 | [P068](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/manual_annotation_analysis_v1/analysis_report.json) |
| candidate_policy_v1.primary.manual_evidence.valid_negative | 28 | [P068](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/manual_annotation_analysis_v1/analysis_report.json) |
| candidate_policy_v1.primary.manual_evidence.false_negative | 2 | [P068](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/manual_annotation_analysis_v1/analysis_report.json) |
| candidate_policy_v1.primary.manual_evidence.ambiguous | 0 | [P068](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/manual_annotation_analysis_v1/analysis_report.json) |
| candidate_policy_v1.primary.manual_evidence.valid_negative_rate | 0.9333333333333333 | [P068](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/manual_annotation_analysis_v1/analysis_report.json) |
| candidate_policy_v1.fallback.rank | 4-10 | [P068](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/manual_annotation_analysis_v1/analysis_report.json) |
| candidate_policy_v1.fallback.similarity_min | 0.15 | [P068](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/manual_annotation_analysis_v1/analysis_report.json) |
| candidate_policy_v1.fallback.similarity_max_exclusive | 0.2 | [P068](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/manual_annotation_analysis_v1/analysis_report.json) |
| candidate_policy_v1.fallback.manual_evidence.n | 10 | [P068](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/manual_annotation_analysis_v1/analysis_report.json) |
| candidate_policy_v1.fallback.manual_evidence.valid_negative | 10 | [P068](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/manual_annotation_analysis_v1/analysis_report.json) |
| candidate_policy_v1.fallback.manual_evidence.false_negative | 0 | [P068](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/manual_annotation_analysis_v1/analysis_report.json) |
| candidate_policy_v1.fallback.manual_evidence.ambiguous | 0 | [P068](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/manual_annotation_analysis_v1/analysis_report.json) |
| candidate_policy_v1.fallback.manual_evidence.valid_negative_rate | 1.0 | [P068](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/manual_annotation_analysis_v1/analysis_report.json) |
| candidate_policy_v1.forbidden_for_automatic_v1.rank_1 | true | [P068](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/manual_annotation_analysis_v1/analysis_report.json) |
| candidate_policy_v1.forbidden_for_automatic_v1.rank_2_3 | true | [P068](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/manual_annotation_analysis_v1/analysis_report.json) |
| candidate_policy_v1.forbidden_for_automatic_v1.similarity_0_40_0_55 | true | [P068](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/manual_annotation_analysis_v1/analysis_report.json) |
| notes | ["Rules were designed using TRAIN manual annotations only.", "Do not infer category-specific rules because category sample sizes are small and unequal.", "Similarity x rank cells contain only about 10 reviewed examples; avoid overfitting fine-grained thresholds."] | [P068](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/manual_annotation_analysis_v1/analysis_report.json) |

Nguồn: `C:/projectSamSung_v1/module2_headline_integrity/results/consistency/manual_annotation_analysis_v1/analysis_report.json` — [P068](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/manual_annotation_analysis_v1/analysis_report.json). Field/CSV columns/source code: `Các field ghi ở cột trái; không tính lại metric`.

## Final consistency dataset manifest
| Field JSON | Giá trị lưu trong file | Source |
| --- | --- | --- |
| dataset_name | headline_consistency_v1 | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| module | module2_headline_integrity | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| task | headline_article_consistency | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| version | v1 | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| labels.0 | INCONSISTENT | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| labels.1 | CONSISTENT | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| construction.positive | original headline paired with its own original article | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| construction.negative | synthetic same-category headline mismatch selected using TRAIN-derived candidate policy v1 | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| construction.balanced_per_anchor | true | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| construction.pairs_per_anchor | 2 | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| construction.positive_per_anchor | 1 | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| construction.negative_per_anchor | 1 | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| construction.full_text_materialized | false | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| construction.article_storage | data\interim\consistency\articles_v1 | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| negative_policy.primary.candidate_rank | 4-10 | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| negative_policy.primary.similarity | [0.20, 0.40) | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| negative_policy.fallback.candidate_rank | 4-10 | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| negative_policy.fallback.similarity | [0.15, 0.20) | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| negative_policy.same_split | true | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| negative_policy.same_category | true | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| negative_policy.same_title_forbidden | true | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| negative_policy.same_content_forbidden | true | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| manual_policy_evidence.source | 240 manually reviewed TRAIN candidate pairs | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| manual_policy_evidence.overall_valid_negative_rate | 0.8 | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| manual_policy_evidence.rank_4_10_valid_negative_rate | 0.9375 | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| manual_policy_evidence.primary_rule_valid_rate | 0.9333 | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| manual_policy_evidence.fallback_rule_valid_rate | 1.0 | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| manual_policy_evidence.note | These estimates are based on small stratified manual samples and should not be interpreted as exact population error rates. | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| splits.train.anchors | 153002 | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| splits.train.positive | 153002 | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| splits.train.negative | 153002 | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| splits.train.pairs | 306004 | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| splits.train.file | data\processed\consistency\headline_consistency_v1\train.parquet | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| splits.validation.anchors | 28713 | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| splits.validation.positive | 28713 | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| splits.validation.negative | 28713 | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| splits.validation.pairs | 57426 | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| splits.validation.file | data\processed\consistency\headline_consistency_v1\validation.parquet | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| splits.test.anchors | 28688 | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| splits.test.positive | 28688 | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| splits.test.negative | 28688 | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| splits.test.pairs | 57376 | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| splits.test.file | data\processed\consistency\headline_consistency_v1\test.parquet | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| total.anchors | 210403 | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| total.pairs | 420806 | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| total.positive | 210403 | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| total.negative | 210403 | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| structural_validation.passed | true | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| structural_validation.cross_split_article_overlap.train__validation | 0 | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| structural_validation.cross_split_article_overlap.train__test | 0 | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| structural_validation.cross_split_article_overlap.validation__test | 0 | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |
| evaluation_warning | The validation/test negatives are synthetic, not human-gold. Use them for controlled synthetic evaluation. A human-verified evaluation subset should be created before reporting final real-world performance. | [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json) |

Nguồn: `C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json` — [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json). Field/CSV columns/source code: `Các field ghi ở cột trái; không tính lại metric`.

## Final dataset structural validation
| Field JSON | Giá trị lưu trong file | Source |
| --- | --- | --- |
| module | module2_headline_integrity | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| task | headline_consistency | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| stage | 10_final_dataset_validation | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| dataset | headline_consistency_v1 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| semantic_review_policy.manual_join_sample_split | train | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| semantic_review_policy.validation_semantics_used | false | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| semantic_review_policy.test_semantics_used | false | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| semantic_review_policy.reason | Validation/test are kept untouched for model selection and final evaluation. | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.stats.split | train | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.stats.pairs | 306004 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.stats.anchors | 153002 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.stats.positive | 153002 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.stats.negative | 153002 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.stats.primary | 93787 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.stats.fallback | 59215 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.stats.similarity_mean | 0.23279953002929688 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.stats.similarity_median | 0.21669349074363708 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.stats.similarity_p95 | 0.3678925335407257 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.stats.headline_reuse_p95 | 4.0 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.stats.headline_reuse_p99 | 5.0 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.stats.headline_reuse_max | 16 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.stats.validation_passed | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.checks.pair_id_unique | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.checks.split_field_valid | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.checks.labels_exactly_0_1 | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.checks.label_name_valid | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.checks.all_anchor_articles_exist | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.checks.all_headline_articles_exist | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.checks.anchor_split_consistent | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.checks.headline_split_consistent | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.checks.anchor_category_matches_index | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.checks.headline_category_matches_index | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.checks.anchor_title_hash_matches_index | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.checks.headline_title_hash_matches_index | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.checks.anchor_content_hash_matches_index | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.checks.headline_content_hash_matches_index | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.checks.exact_class_balance | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.checks.exactly_two_pairs_per_anchor | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.checks.one_positive_one_negative_per_anchor | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.checks.positive_self_pair | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.checks.positive_type_valid | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.checks.positive_tier_original | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.checks.positive_similarity_null | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.checks.positive_rank_null | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.checks.positive_source_candidate_null | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.checks.negative_not_self | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.checks.negative_same_category | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.checks.negative_title_different | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.checks.negative_content_different | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.checks.negative_source_candidate_present | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.checks.negative_similarity_present | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.checks.negative_rank_present | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.checks.negative_rank_4_10 | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.checks.negative_tiers_valid | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.checks.primary_similarity_rule | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.checks.fallback_similarity_rule | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.checks.category_balance_per_label | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.similarity.min | 0.15000014007091522 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.similarity.p05 | 0.15684226155281067 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.similarity.median | 0.21669349074363708 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.similarity.mean | 0.23279953002929688 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.similarity.p95 | 0.3678925335407257 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.similarity.max | 0.3999975025653839 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.headline_reuse.unique_headlines | 90308 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.headline_reuse.mean | 1.6942242104796916 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.headline_reuse.median | 1.0 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.headline_reuse.p95 | 4.0 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.headline_reuse.p99 | 5.0 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.headline_reuse.max | 16 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.train.passed | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.stats.split | validation | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.stats.pairs | 57426 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.stats.anchors | 28713 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.stats.positive | 28713 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.stats.negative | 28713 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.stats.primary | 13867 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.stats.fallback | 14846 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.stats.similarity_mean | 0.21479836106300354 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.stats.similarity_median | 0.1977849006652832 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.stats.similarity_p95 | 0.340209037065506 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.stats.headline_reuse_p95 | 3.0 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.stats.headline_reuse_p99 | 5.0 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.stats.headline_reuse_max | 9 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.stats.validation_passed | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.checks.pair_id_unique | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.checks.split_field_valid | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.checks.labels_exactly_0_1 | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.checks.label_name_valid | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.checks.all_anchor_articles_exist | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.checks.all_headline_articles_exist | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.checks.anchor_split_consistent | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.checks.headline_split_consistent | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.checks.anchor_category_matches_index | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.checks.headline_category_matches_index | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.checks.anchor_title_hash_matches_index | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.checks.headline_title_hash_matches_index | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.checks.anchor_content_hash_matches_index | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.checks.headline_content_hash_matches_index | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.checks.exact_class_balance | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.checks.exactly_two_pairs_per_anchor | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.checks.one_positive_one_negative_per_anchor | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.checks.positive_self_pair | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.checks.positive_type_valid | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.checks.positive_tier_original | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.checks.positive_similarity_null | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.checks.positive_rank_null | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.checks.positive_source_candidate_null | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.checks.negative_not_self | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.checks.negative_same_category | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.checks.negative_title_different | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.checks.negative_content_different | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.checks.negative_source_candidate_present | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.checks.negative_similarity_present | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.checks.negative_rank_present | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.checks.negative_rank_4_10 | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.checks.negative_tiers_valid | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.checks.primary_similarity_rule | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.checks.fallback_similarity_rule | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.checks.category_balance_per_label | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.similarity.min | 0.15000146627426147 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.similarity.p05 | 0.1540554165840149 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.similarity.median | 0.1977849006652832 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.similarity.mean | 0.21479836106300354 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.similarity.p95 | 0.340209037065506 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.similarity.max | 0.3999117612838745 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.headline_reuse.unique_headlines | 18072 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.headline_reuse.mean | 1.5888114209827358 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.headline_reuse.median | 1.0 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.headline_reuse.p95 | 3.0 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.headline_reuse.p99 | 5.0 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.headline_reuse.max | 9 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.validation.passed | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.stats.split | test | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.stats.pairs | 57376 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.stats.anchors | 28688 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.stats.positive | 28688 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.stats.negative | 28688 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.stats.primary | 14049 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.stats.fallback | 14639 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.stats.similarity_mean | 0.21535013616085052 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.stats.similarity_median | 0.19869810342788696 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.stats.similarity_p95 | 0.3376576602458954 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.stats.headline_reuse_p95 | 3.0 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.stats.headline_reuse_p99 | 5.0 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.stats.headline_reuse_max | 9 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.stats.validation_passed | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.checks.pair_id_unique | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.checks.split_field_valid | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.checks.labels_exactly_0_1 | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.checks.label_name_valid | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.checks.all_anchor_articles_exist | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.checks.all_headline_articles_exist | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.checks.anchor_split_consistent | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.checks.headline_split_consistent | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.checks.anchor_category_matches_index | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.checks.headline_category_matches_index | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.checks.anchor_title_hash_matches_index | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.checks.headline_title_hash_matches_index | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.checks.anchor_content_hash_matches_index | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.checks.headline_content_hash_matches_index | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.checks.exact_class_balance | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.checks.exactly_two_pairs_per_anchor | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.checks.one_positive_one_negative_per_anchor | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.checks.positive_self_pair | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.checks.positive_type_valid | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.checks.positive_tier_original | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.checks.positive_similarity_null | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.checks.positive_rank_null | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.checks.positive_source_candidate_null | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.checks.negative_not_self | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.checks.negative_same_category | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.checks.negative_title_different | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.checks.negative_content_different | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.checks.negative_source_candidate_present | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.checks.negative_similarity_present | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.checks.negative_rank_present | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.checks.negative_rank_4_10 | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.checks.negative_tiers_valid | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.checks.primary_similarity_rule | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.checks.fallback_similarity_rule | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.checks.category_balance_per_label | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.similarity.min | 0.1500026434659958 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.similarity.p05 | 0.15440930426120758 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.similarity.median | 0.19869810342788696 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.similarity.mean | 0.21535013616085052 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.similarity.p95 | 0.3376576602458954 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.similarity.max | 0.3999636173248291 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.headline_reuse.unique_headlines | 17891 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.headline_reuse.mean | 1.603487787155553 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.headline_reuse.median | 1.0 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.headline_reuse.p95 | 3.0 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.headline_reuse.p99 | 5.0 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.headline_reuse.max | 9 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| splits.test.passed | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| cross_split.passed | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| cross_split.overlap.train__validation.article_uid | 0 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| cross_split.overlap.train__validation.anchor_uid | 0 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| cross_split.overlap.train__validation.pair_id | 0 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| cross_split.overlap.train__test.article_uid | 0 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| cross_split.overlap.train__test.anchor_uid | 0 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| cross_split.overlap.train__test.pair_id | 0 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| cross_split.overlap.validation__test.article_uid | 0 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| cross_split.overlap.validation__test.anchor_uid | 0 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| cross_split.overlap.validation__test.pair_id | 0 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| train_join_review_rows | 20 | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| overall_validation_passed | true | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| outputs.split_stats | results\consistency\final_dataset_validation_v1\split_stats.csv | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| outputs.category_stats | results\consistency\final_dataset_validation_v1\category_stats.csv | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| outputs.negative_tier_stats | results\consistency\final_dataset_validation_v1\negative_tier_stats.csv | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |
| outputs.train_join_review_sample | results\consistency\final_dataset_validation_v1\train_join_review_sample.csv | [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json) |

Nguồn: `C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json` — [P070](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json). Field/CSV columns/source code: `Các field ghi ở cột trái; không tính lại metric`.

Đơn vị: article là bài corpus; anchor là bài được giữ để tạo cặp; pair là đơn vị phân loại. Một positive và một negative cho mỗi anchor final. Positive giữ headline gốc với article của nó là presumed-consistent; negative tạo tổng hợp trong cùng category/split theo rank/similarity. Manual review ứng viên TRAIN không biến test thành human-gold. Không coi category review nhỏ là quy tắc cho từng lĩnh vực.
Nguồn: `C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json` — [P069](/C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json). Field/CSV columns/source code: `construction, negative_policy, manual_policy_evidence, splits, structural_validation, evaluation_warning`.

# 10. Module 3 — Baseline và training
## A — title-description — actual run config
| Field JSON | Giá trị lưu trong file | Source |
| --- | --- | --- |
| stage | 11_baseline_title_description | [P071](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/run_config.json) |
| model_name | bert-base-multilingual-cased | [P071](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/run_config.json) |
| input | headline_used_in_pair + anchor_description | [P071](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/run_config.json) |
| train_rows | 306004 | [P071](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/run_config.json) |
| validation_rows | 57426 | [P071](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/run_config.json) |
| test_loaded | false | [P071](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/run_config.json) |
| max_length | 160 | [P071](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/run_config.json) |
| batch_size | 8 | [P071](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/run_config.json) |
| gradient_accumulation | 4 | [P071](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/run_config.json) |
| effective_batch_size | 32 | [P071](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/run_config.json) |
| epochs | 1 | [P071](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/run_config.json) |
| learning_rate | 2e-05 | [P071](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/run_config.json) |
| weight_decay | 0.01 | [P071](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/run_config.json) |
| warmup_ratio | 0.06 | [P071](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/run_config.json) |
| seed | 20260920 | [P071](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/run_config.json) |
| amp | true | [P071](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/run_config.json) |
| device | cuda | [P071](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/run_config.json) |
| gpu | NVIDIA GeForce RTX 4060 Laptop GPU | [P071](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/run_config.json) |
| smoke_test | false | [P071](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/run_config.json) |

Nguồn: `C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/run_config.json` — [P071](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/run_config.json). Field/CSV columns/source code: `Các field ghi ở cột trái; không tính lại metric`.

## A — title-description — script defaults/calls
| Config/call trong mã (không tự coi là actual run) | Biểu thức | Vị trí | Source |
| --- | --- | --- | --- |
| DEFAULT_MODEL_NAME | 'bert-base-multilingual-cased' | line 73 | [P072](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/11_train_baseline_title_description.py) |
| DEFAULT_MAX_LENGTH | 160 | line 77 | [P072](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/11_train_baseline_title_description.py) |
| DEFAULT_BATCH_SIZE | 16 | line 79 | [P072](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/11_train_baseline_title_description.py) |
| DEFAULT_GRAD_ACCUM | 2 | line 81 | [P072](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/11_train_baseline_title_description.py) |
| DEFAULT_EPOCHS | 1 | line 83 | [P072](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/11_train_baseline_title_description.py) |
| DEFAULT_LR | 2e-05 | line 85 | [P072](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/11_train_baseline_title_description.py) |
| DEFAULT_WEIGHT_DECAY | 0.01 | line 87 | [P072](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/11_train_baseline_title_description.py) |
| DEFAULT_WARMUP_RATIO | 0.06 | line 89 | [P072](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/11_train_baseline_title_description.py) |
| DEFAULT_SEED | 20260920 | line 91 | [P072](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/11_train_baseline_title_description.py) |
| torch.optim.AdamW | torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) | line 1164 | [P072](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/11_train_baseline_title_description.py) |
| get_linear_schedule_with_warmup | get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_optimizer_steps) | line 1195 | [P072](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/11_train_baseline_title_description.py) |
| torch.cuda.amp.GradScaler | torch.cuda.amp.GradScaler(enabled=use_amp) | line 1215 | [P072](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/11_train_baseline_title_description.py) |
| AutoTokenizer.from_pretrained | AutoTokenizer.from_pretrained(args.model_name) | line 2026 | [P072](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/11_train_baseline_title_description.py) |
| AutoModelForSequenceClassification.from_pretrained | AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2) | line 2106 | [P072](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/11_train_baseline_title_description.py) |
| torch.cuda.amp.autocast | torch.cuda.amp.autocast(enabled=use_amp) | line 1059 | [P072](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/11_train_baseline_title_description.py) |
| torch.cuda.amp.autocast | torch.cuda.amp.autocast(enabled=use_amp) | line 1330 | [P072](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/11_train_baseline_title_description.py) |

Nguồn: `C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/11_train_baseline_title_description.py` — [P072](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/11_train_baseline_title_description.py). Field/CSV columns/source code: `DEFAULT_MODEL_NAME, MODEL_NAME, DEFAULT_MAX_LENGTH, MAX_LENGTH, DEFAULT_BATCH_SIZE, DEFAULT_EVAL_BATCH_SIZE, DEFAULT_GRAD_ACCUM, DEFAULT_EPOCHS, DEFAULT_LR, DEFAULT_WEIGHT_DECAY, DEFAULT_WARMUP_RATIO, DEFAULT_SEED, CACHE_MAX_CHUNKS, ARTICLE_CHAR_LIMIT, DEFAULT_ARTICLE_CHAR_LIMIT, CrossEntropyLoss, BCEWithLogitsLoss, AdamW, get_linear_schedule_with_warmup, autocast, GradScaler, AutoTokenizer.from_pretrained, AutoModelForSequenceClassification.from_pretrained`.

## A — title-description — training history
| epoch | train_loss | validation_loss | validation_accuracy_05 | validation_macro_f1_05 | validation_f1_consistent_05 | validation_roc_auc | validation_ap | elapsed_seconds | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.36268455146725675 | 0.240290025975875 | 0.8962316720649184 | 0.8960946997843049 | 0.8923221481361017 | 0.9658197126429778 | 0.9693584653833434 | 2359.9978532791138 | [P073](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/training_history.csv) |

Nguồn: `C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/training_history.csv` — [P073](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/training_history.csv). Field/CSV columns/source code: `epoch, train_loss, validation_loss, validation_accuracy_05, validation_macro_f1_05, validation_f1_consistent_05, validation_roc_auc, validation_ap, elapsed_seconds`.

## A — title-description — validation metrics
| Field JSON | Giá trị lưu trong file | Source |
| --- | --- | --- |
| dataset | headline_consistency_v1 | [P074](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/validation_metrics.json) |
| model | bert-base-multilingual-cased | [P074](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/validation_metrics.json) |
| baseline | A_title_plus_description | [P074](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/validation_metrics.json) |
| test_used | false | [P074](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/validation_metrics.json) |
| checkpoint_selection | best validation Macro-F1 at threshold 0.5 | [P074](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/validation_metrics.json) |
| validation_loss | 0.240290025975875 | [P074](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/validation_metrics.json) |
| metrics_at_0_5.threshold | 0.5 | [P074](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/validation_metrics.json) |
| metrics_at_0_5.accuracy | 0.8962316720649184 | [P074](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/validation_metrics.json) |
| metrics_at_0_5.precision_consistent | 0.9272570226828902 | [P074](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/validation_metrics.json) |
| metrics_at_0_5.recall_consistent | 0.8599240762024171 | [P074](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/validation_metrics.json) |
| metrics_at_0_5.f1_consistent | 0.8923221481361017 | [P074](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/validation_metrics.json) |
| metrics_at_0_5.macro_f1 | 0.8960946997843049 | [P074](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/validation_metrics.json) |
| metrics_at_0_5.roc_auc | 0.9658197126429778 | [P074](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/validation_metrics.json) |
| metrics_at_0_5.average_precision | 0.9693584653833434 | [P074](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/validation_metrics.json) |
| metrics_at_0_5.confusion_matrix.tn | 26776 | [P074](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/validation_metrics.json) |
| metrics_at_0_5.confusion_matrix.fp | 1937 | [P074](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/validation_metrics.json) |
| metrics_at_0_5.confusion_matrix.fn | 4022 | [P074](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/validation_metrics.json) |
| metrics_at_0_5.confusion_matrix.tp | 24691 | [P074](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/validation_metrics.json) |
| best_validation_threshold | 0.43000000000000005 | [P074](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/validation_metrics.json) |
| metrics_at_tuned_threshold.threshold | 0.43000000000000005 | [P074](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/validation_metrics.json) |
| metrics_at_tuned_threshold.accuracy | 0.8966844286560095 | [P074](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/validation_metrics.json) |
| metrics_at_tuned_threshold.precision_consistent | 0.9056267806267806 | [P074](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/validation_metrics.json) |
| metrics_at_tuned_threshold.recall_consistent | 0.8856615470344443 | [P074](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/validation_metrics.json) |
| metrics_at_tuned_threshold.f1_consistent | 0.8955329001813603 | [P074](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/validation_metrics.json) |
| metrics_at_tuned_threshold.macro_f1 | 0.8966718738837183 | [P074](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/validation_metrics.json) |
| metrics_at_tuned_threshold.roc_auc | 0.9658197126429778 | [P074](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/validation_metrics.json) |
| metrics_at_tuned_threshold.average_precision | 0.9693584653833434 | [P074](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/validation_metrics.json) |
| metrics_at_tuned_threshold.confusion_matrix.tn | 26063 | [P074](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/validation_metrics.json) |
| metrics_at_tuned_threshold.confusion_matrix.fp | 2650 | [P074](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/validation_metrics.json) |
| metrics_at_tuned_threshold.confusion_matrix.fn | 3283 | [P074](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/validation_metrics.json) |
| metrics_at_tuned_threshold.confusion_matrix.tp | 25430 | [P074](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/validation_metrics.json) |
| important_note | Threshold was tuned on validation only. Do not tune on test. | [P074](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/validation_metrics.json) |

Nguồn: `C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/validation_metrics.json` — [P074](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/validation_metrics.json). Field/CSV columns/source code: `Các field ghi ở cột trái; không tính lại metric`.

## B — title-article-head — actual run config
| Field JSON | Giá trị lưu trong file | Source |
| --- | --- | --- |
| stage | 13_baseline_title_article_head | [P075](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/run_config.json) |
| model_name | bert-base-multilingual-cased | [P075](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/run_config.json) |
| input | headline_used_in_pair + beginning_of_anchor_article | [P075](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/run_config.json) |
| max_length | 160 | [P075](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/run_config.json) |
| article_char_limit | 3000 | [P075](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/run_config.json) |
| train_rows | 306004 | [P075](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/run_config.json) |
| validation_rows | 57426 | [P075](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/run_config.json) |
| test_loaded | false | [P075](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/run_config.json) |
| batch_size | 8 | [P075](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/run_config.json) |
| grad_accum | 4 | [P075](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/run_config.json) |
| effective_batch | 32 | [P075](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/run_config.json) |
| epochs | 1 | [P075](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/run_config.json) |
| learning_rate | 2e-05 | [P075](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/run_config.json) |
| seed | 20260920 | [P075](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/run_config.json) |
| smoke_test | false | [P075](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/run_config.json) |

Nguồn: `C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/run_config.json` — [P075](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/run_config.json). Field/CSV columns/source code: `Các field ghi ở cột trái; không tính lại metric`.

## B — title-article-head — script defaults/calls
| Config/call trong mã (không tự coi là actual run) | Biểu thức | Vị trí | Source |
| --- | --- | --- | --- |
| DEFAULT_MODEL_NAME | 'bert-base-multilingual-cased' | line 80 | [P076](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/13_train_baseline_title_article_head.py) |
| DEFAULT_MAX_LENGTH | 160 | line 84 | [P076](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/13_train_baseline_title_article_head.py) |
| DEFAULT_BATCH_SIZE | 8 | line 86 | [P076](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/13_train_baseline_title_article_head.py) |
| DEFAULT_GRAD_ACCUM | 4 | line 88 | [P076](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/13_train_baseline_title_article_head.py) |
| DEFAULT_EPOCHS | 1 | line 90 | [P076](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/13_train_baseline_title_article_head.py) |
| DEFAULT_LR | 2e-05 | line 92 | [P076](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/13_train_baseline_title_article_head.py) |
| DEFAULT_WEIGHT_DECAY | 0.01 | line 94 | [P076](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/13_train_baseline_title_article_head.py) |
| DEFAULT_WARMUP_RATIO | 0.06 | line 96 | [P076](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/13_train_baseline_title_article_head.py) |
| DEFAULT_SEED | 20260920 | line 98 | [P076](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/13_train_baseline_title_article_head.py) |
| DEFAULT_ARTICLE_CHAR_LIMIT | 3000 | line 109 | [P076](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/13_train_baseline_title_article_head.py) |
| torch.optim.AdamW | torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) | line 979 | [P076](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/13_train_baseline_title_article_head.py) |
| get_linear_schedule_with_warmup | get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps) | line 1006 | [P076](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/13_train_baseline_title_article_head.py) |
| torch.amp.GradScaler | torch.amp.GradScaler('cuda', enabled=use_amp) | line 1025 | [P076](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/13_train_baseline_title_article_head.py) |
| AutoTokenizer.from_pretrained | AutoTokenizer.from_pretrained(args.model_name) | line 1736 | [P076](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/13_train_baseline_title_article_head.py) |
| AutoModelForSequenceClassification.from_pretrained | AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2) | line 1824 | [P076](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/13_train_baseline_title_article_head.py) |
| torch.amp.autocast | torch.amp.autocast(device_type='cuda', enabled=use_amp) | line 887 | [P076](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/13_train_baseline_title_article_head.py) |
| torch.amp.autocast | torch.amp.autocast(device_type='cuda', enabled=use_amp) | line 1113 | [P076](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/13_train_baseline_title_article_head.py) |

Nguồn: `C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/13_train_baseline_title_article_head.py` — [P076](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/13_train_baseline_title_article_head.py). Field/CSV columns/source code: `DEFAULT_MODEL_NAME, MODEL_NAME, DEFAULT_MAX_LENGTH, MAX_LENGTH, DEFAULT_BATCH_SIZE, DEFAULT_EVAL_BATCH_SIZE, DEFAULT_GRAD_ACCUM, DEFAULT_EPOCHS, DEFAULT_LR, DEFAULT_WEIGHT_DECAY, DEFAULT_WARMUP_RATIO, DEFAULT_SEED, CACHE_MAX_CHUNKS, ARTICLE_CHAR_LIMIT, DEFAULT_ARTICLE_CHAR_LIMIT, CrossEntropyLoss, BCEWithLogitsLoss, AdamW, get_linear_schedule_with_warmup, autocast, GradScaler, AutoTokenizer.from_pretrained, AutoModelForSequenceClassification.from_pretrained`.

## B — title-article-head — training history
| epoch | train_loss | validation_loss | validation_accuracy_05 | validation_macro_f1_05 | validation_f1_consistent_05 | validation_roc_auc | elapsed_seconds | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.4359997275003388 | 0.31843424432932743 | 0.8599240762024171 | 0.8597442927856901 | 0.8547227740653783 | 0.9410042717317337 | 2887.745434522629 | [P077](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/training_history.csv) |

Nguồn: `C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/training_history.csv` — [P077](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/training_history.csv). Field/CSV columns/source code: `epoch, train_loss, validation_loss, validation_accuracy_05, validation_macro_f1_05, validation_f1_consistent_05, validation_roc_auc, elapsed_seconds`.

## B — title-article-head — validation metrics
| Field JSON | Giá trị lưu trong file | Source |
| --- | --- | --- |
| baseline | B_title_plus_article_head | [P078](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/validation_metrics.json) |
| model | bert-base-multilingual-cased | [P078](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/validation_metrics.json) |
| dataset | headline_consistency_v1 | [P078](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/validation_metrics.json) |
| test_used | false | [P078](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/validation_metrics.json) |
| validation_loss | 0.31843424432932743 | [P078](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/validation_metrics.json) |
| metrics_at_0_5.threshold | 0.5 | [P078](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/validation_metrics.json) |
| metrics_at_0_5.accuracy | 0.8599240762024171 | [P078](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/validation_metrics.json) |
| metrics_at_0_5.precision_consistent | 0.8876842855535132 | [P078](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/validation_metrics.json) |
| metrics_at_0_5.recall_consistent | 0.8241214780761328 | [P078](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/validation_metrics.json) |
| metrics_at_0_5.f1_consistent | 0.8547227740653783 | [P078](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/validation_metrics.json) |
| metrics_at_0_5.macro_f1 | 0.8597442927856901 | [P078](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/validation_metrics.json) |
| metrics_at_0_5.roc_auc | 0.9410042717317337 | [P078](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/validation_metrics.json) |
| metrics_at_0_5.average_precision | 0.9439782284798034 | [P078](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/validation_metrics.json) |
| metrics_at_0_5.confusion_matrix.tn | 25719 | [P078](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/validation_metrics.json) |
| metrics_at_0_5.confusion_matrix.fp | 2994 | [P078](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/validation_metrics.json) |
| metrics_at_0_5.confusion_matrix.fn | 5050 | [P078](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/validation_metrics.json) |
| metrics_at_0_5.confusion_matrix.tp | 23663 | [P078](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/validation_metrics.json) |
| best_validation_threshold | 0.45000000000000007 | [P078](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/validation_metrics.json) |
| metrics_at_tuned_threshold.threshold | 0.45000000000000007 | [P078](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/validation_metrics.json) |
| metrics_at_tuned_threshold.accuracy | 0.8614390694110682 | [P078](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/validation_metrics.json) |
| metrics_at_tuned_threshold.precision_consistent | 0.8695605726087885 | [P078](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/validation_metrics.json) |
| metrics_at_tuned_threshold.recall_consistent | 0.850451015219587 | [P078](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/validation_metrics.json) |
| metrics_at_tuned_threshold.f1_consistent | 0.8598996390527335 | [P078](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/validation_metrics.json) |
| metrics_at_tuned_threshold.macro_f1 | 0.8614223379134691 | [P078](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/validation_metrics.json) |
| metrics_at_tuned_threshold.roc_auc | 0.9410042717317337 | [P078](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/validation_metrics.json) |
| metrics_at_tuned_threshold.average_precision | 0.9439782284798034 | [P078](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/validation_metrics.json) |
| metrics_at_tuned_threshold.confusion_matrix.tn | 25050 | [P078](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/validation_metrics.json) |
| metrics_at_tuned_threshold.confusion_matrix.fp | 3663 | [P078](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/validation_metrics.json) |
| metrics_at_tuned_threshold.confusion_matrix.fn | 4294 | [P078](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/validation_metrics.json) |
| metrics_at_tuned_threshold.confusion_matrix.tp | 24419 | [P078](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/validation_metrics.json) |

Nguồn: `C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/validation_metrics.json` — [P078](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/validation_metrics.json). Field/CSV columns/source code: `Các field ghi ở cột trái; không tính lại metric`.

## C — full-article MAX-MIL — actual run config
| Field JSON | Giá trị lưu trong file | Source |
| --- | --- | --- |
| stage | 18_train_full_article_max_mil | [P079](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/run_config.json) |
| dataset | headline_consistency_v1 | [P079](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/run_config.json) |
| model | bert-base-multilingual-cased | [P079](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/run_config.json) |
| initialization | baseline_title_article_head_mbert/best_model.pt | [P079](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/run_config.json) |
| pooling | max | [P079](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/run_config.json) |
| loss | BCEWithLogitsLoss | [P079](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/run_config.json) |
| chunk_binary_score | logit_consistent - logit_inconsistent | [P079](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/run_config.json) |
| train_pairs | 306004 | [P079](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/run_config.json) |
| validation_pairs | 57426 | [P079](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/run_config.json) |
| test_loaded | false | [P079](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/run_config.json) |
| max_length | 160 | [P079](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/run_config.json) |
| max_chunks | 6 | [P079](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/run_config.json) |
| pair_batch_size | 2 | [P079](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/run_config.json) |
| eval_pair_batch_size | 4 | [P079](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/run_config.json) |
| grad_accum | 8 | [P079](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/run_config.json) |
| effective_pair_batch | 16 | [P079](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/run_config.json) |
| epochs | 1 | [P079](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/run_config.json) |
| learning_rate | 1e-05 | [P079](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/run_config.json) |
| weight_decay | 0.01 | [P079](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/run_config.json) |
| warmup_ratio | 0.06 | [P079](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/run_config.json) |
| seed | 20260920 | [P079](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/run_config.json) |
| amp | true | [P079](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/run_config.json) |
| gradient_checkpointing | false | [P079](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/run_config.json) |
| smoke_test | false | [P079](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/run_config.json) |

Nguồn: `C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/run_config.json` — [P079](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/run_config.json). Field/CSV columns/source code: `Các field ghi ở cột trái; không tính lại metric`.

## C — full-article MAX-MIL — script defaults/calls
| Config/call trong mã (không tự coi là actual run) | Biểu thức | Vị trí | Source |
| --- | --- | --- | --- |
| MODEL_NAME | 'bert-base-multilingual-cased' | line 86 | [P080](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/18_train_full_article_max_mil.py) |
| MAX_LENGTH | 160 | line 88 | [P080](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/18_train_full_article_max_mil.py) |
| CACHE_MAX_CHUNKS | 6 | line 90 | [P080](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/18_train_full_article_max_mil.py) |
| DEFAULT_BATCH_SIZE | 2 | line 92 | [P080](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/18_train_full_article_max_mil.py) |
| DEFAULT_EVAL_BATCH_SIZE | 4 | line 94 | [P080](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/18_train_full_article_max_mil.py) |
| DEFAULT_GRAD_ACCUM | 8 | line 96 | [P080](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/18_train_full_article_max_mil.py) |
| DEFAULT_EPOCHS | 1 | line 98 | [P080](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/18_train_full_article_max_mil.py) |
| DEFAULT_LR | 1e-05 | line 102 | [P080](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/18_train_full_article_max_mil.py) |
| DEFAULT_WEIGHT_DECAY | 0.01 | line 104 | [P080](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/18_train_full_article_max_mil.py) |
| DEFAULT_WARMUP_RATIO | 0.06 | line 106 | [P080](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/18_train_full_article_max_mil.py) |
| DEFAULT_SEED | 20260920 | line 110 | [P080](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/18_train_full_article_max_mil.py) |
| torch.nn.BCEWithLogitsLoss | torch.nn.BCEWithLogitsLoss(reduction='sum') | line 1500 | [P080](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/18_train_full_article_max_mil.py) |
| torch.optim.AdamW | torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) | line 1634 | [P080](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/18_train_full_article_max_mil.py) |
| get_linear_schedule_with_warmup | get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_optimizer_steps) | line 1665 | [P080](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/18_train_full_article_max_mil.py) |
| torch.amp.GradScaler | torch.amp.GradScaler('cuda', enabled=use_amp) | line 1684 | [P080](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/18_train_full_article_max_mil.py) |
| torch.nn.BCEWithLogitsLoss | torch.nn.BCEWithLogitsLoss() | line 1690 | [P080](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/18_train_full_article_max_mil.py) |
| AutoTokenizer.from_pretrained | AutoTokenizer.from_pretrained(MODEL_NAME) | line 2439 | [P080](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/18_train_full_article_max_mil.py) |
| AutoModelForSequenceClassification.from_pretrained | AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2) | line 2633 | [P080](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/18_train_full_article_max_mil.py) |
| torch.amp.autocast | torch.amp.autocast(device_type=device.type, enabled=use_amp) | line 1537 | [P080](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/18_train_full_article_max_mil.py) |
| torch.amp.autocast | torch.amp.autocast(device_type=device.type, enabled=use_amp) | line 1800 | [P080](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/18_train_full_article_max_mil.py) |

Nguồn: `C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/18_train_full_article_max_mil.py` — [P080](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/18_train_full_article_max_mil.py). Field/CSV columns/source code: `DEFAULT_MODEL_NAME, MODEL_NAME, DEFAULT_MAX_LENGTH, MAX_LENGTH, DEFAULT_BATCH_SIZE, DEFAULT_EVAL_BATCH_SIZE, DEFAULT_GRAD_ACCUM, DEFAULT_EPOCHS, DEFAULT_LR, DEFAULT_WEIGHT_DECAY, DEFAULT_WARMUP_RATIO, DEFAULT_SEED, CACHE_MAX_CHUNKS, ARTICLE_CHAR_LIMIT, DEFAULT_ARTICLE_CHAR_LIMIT, CrossEntropyLoss, BCEWithLogitsLoss, AdamW, get_linear_schedule_with_warmup, autocast, GradScaler, AutoTokenizer.from_pretrained, AutoModelForSequenceClassification.from_pretrained`.

## C — full-article MAX-MIL — training history
| epoch | train_loss | validation_loss | validation_accuracy_05 | validation_macro_f1_05 | validation_f1_consistent_05 | validation_roc_auc | elapsed_seconds | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.3330300867991948 | 0.24757275384168778 | 0.897468045832898 | 0.8974022725648538 | 0.8948045451297078 | 0.9658074169699808 | 12037.039787530899 | [P081](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/training_history.csv) |

Nguồn: `C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/training_history.csv` — [P081](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/training_history.csv). Field/CSV columns/source code: `epoch, train_loss, validation_loss, validation_accuracy_05, validation_macro_f1_05, validation_f1_consistent_05, validation_roc_auc, elapsed_seconds`.

## C — full-article MAX-MIL — validation metrics
| Field JSON | Giá trị lưu trong file | Source |
| --- | --- | --- |
| stage | 18_train_full_article_max_mil | [P082](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/validation_metrics.json) |
| dataset | headline_consistency_v1 | [P082](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/validation_metrics.json) |
| model | bert-base-multilingual-cased | [P082](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/validation_metrics.json) |
| architecture | full_article_max_mil | [P082](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/validation_metrics.json) |
| test_used | false | [P082](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/validation_metrics.json) |
| validation_loss | 0.24757275384168778 | [P082](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/validation_metrics.json) |
| metrics_at_0_5.threshold | 0.5 | [P082](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/validation_metrics.json) |
| metrics_at_0_5.accuracy | 0.897468045832898 | [P082](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/validation_metrics.json) |
| metrics_at_0_5.precision_consistent | 0.9186690634285923 | [P082](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/validation_metrics.json) |
| metrics_at_0_5.recall_consistent | 0.872148504161878 | [P082](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/validation_metrics.json) |
| metrics_at_0_5.f1_consistent | 0.8948045451297078 | [P082](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/validation_metrics.json) |
| metrics_at_0_5.macro_f1 | 0.8974022725648538 | [P082](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/validation_metrics.json) |
| metrics_at_0_5.roc_auc | 0.9658074169699808 | [P082](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/validation_metrics.json) |
| metrics_at_0_5.average_precision | 0.9658258248803385 | [P082](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/validation_metrics.json) |
| metrics_at_0_5.confusion_matrix.tn | 26496 | [P082](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/validation_metrics.json) |
| metrics_at_0_5.confusion_matrix.fp | 2217 | [P082](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/validation_metrics.json) |
| metrics_at_0_5.confusion_matrix.fn | 3671 | [P082](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/validation_metrics.json) |
| metrics_at_0_5.confusion_matrix.tp | 25042 | [P082](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/validation_metrics.json) |
| best_validation_threshold | 0.4100000000000001 | [P082](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/validation_metrics.json) |
| metrics_at_tuned_threshold.threshold | 0.4100000000000001 | [P082](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/validation_metrics.json) |
| metrics_at_tuned_threshold.accuracy | 0.9002716539546547 | [P082](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/validation_metrics.json) |
| metrics_at_tuned_threshold.precision_consistent | 0.9020780856423174 | [P082](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/validation_metrics.json) |
| metrics_at_tuned_threshold.recall_consistent | 0.8980252847142409 | [P082](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/validation_metrics.json) |
| metrics_at_tuned_threshold.f1_consistent | 0.900047122886015 | [P082](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/validation_metrics.json) |
| metrics_at_tuned_threshold.macro_f1 | 0.9002711507054522 | [P082](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/validation_metrics.json) |
| metrics_at_tuned_threshold.roc_auc | 0.9658074169699808 | [P082](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/validation_metrics.json) |
| metrics_at_tuned_threshold.average_precision | 0.9658258248803385 | [P082](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/validation_metrics.json) |
| metrics_at_tuned_threshold.confusion_matrix.tn | 25914 | [P082](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/validation_metrics.json) |
| metrics_at_tuned_threshold.confusion_matrix.fp | 2799 | [P082](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/validation_metrics.json) |
| metrics_at_tuned_threshold.confusion_matrix.fn | 2928 | [P082](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/validation_metrics.json) |
| metrics_at_tuned_threshold.confusion_matrix.tp | 25785 | [P082](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/validation_metrics.json) |
| note | Threshold tuned on validation only. Test remains untouched. | [P082](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/validation_metrics.json) |

Nguồn: `C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/validation_metrics.json` — [P082](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/validation_metrics.json). Field/CSV columns/source code: `Các field ghi ở cột trái; không tính lại metric`.

A: cặp title-description qua sequence classifier; B: title với beginning_of_anchor_article (article_char_limit theo actual config). C: initialization từ checkpoint B, binary chunk score = logit_consistent − logit_inconsistent; max score trên chunks, BCEWithLogitsLoss. Không coi A/B có chunk policy của C. Với trường không lưu actual config, chỉ báo giá trị script và không khẳng định CLI đã chạy y hệt.
## Chính sách cache chunk của MAX-MIL
| Config/call trong mã (không tự coi là actual run) | Biểu thức | Vị trí | Source |
| --- | --- | --- | --- |
| MODEL_NAME | 'bert-base-multilingual-cased' | line 71 | [P083](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/17_prepare_mil_chunk_cache.py) |
| CHUNK_TOKENS | 120 | line 75 | [P083](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/17_prepare_mil_chunk_cache.py) |
| CHUNK_OVERLAP | 24 | line 77 | [P083](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/17_prepare_mil_chunk_cache.py) |
| MAX_CHUNKS | 6 | line 79 | [P083](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/17_prepare_mil_chunk_cache.py) |

Nguồn: `C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/17_prepare_mil_chunk_cache.py` — [P083](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/17_prepare_mil_chunk_cache.py). Field/CSV columns/source code: `MODEL_NAME, CHUNK_TOKENS, CHUNK_OVERLAP, MAX_CHUNKS`.

## Cache metadata MAX-MIL
| Field JSON | Giá trị lưu trong file | Source |
| --- | --- | --- |
| stage | 17_prepare_mil_chunk_cache | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| model_name | bert-base-multilingual-cased | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| test_used | false | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| chunk_policy.chunk_tokens | 120 | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| chunk_policy.chunk_overlap | 24 | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| chunk_policy.max_chunks | 6 | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| chunk_policy.sampling | uniform coverage across full article | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| train.split | train | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| train.articles | 153002 | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| train.chunk_slots | 918012 | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| train.actual_chunks | 871184 | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| train.chunks_per_article.min | 1 | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| train.chunks_per_article.mean | 5.69393864132495 | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| train.chunks_per_article.median | 6.0 | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| train.chunks_per_article.p95 | 6.0 | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| train.chunks_per_article.max | 6 | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| train.article_tokens.min | 60 | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| train.article_tokens.median | 774.0 | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| train.article_tokens.mean | 874.4242820355289 | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| train.article_tokens.p95 | 1746.0 | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| train.article_tokens.max | 52002 | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| train.chunk_lengths.min | 1 | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| train.chunk_lengths.median | 120.0 | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| train.chunk_lengths.mean | 107.25349639111829 | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| train.chunk_lengths.max | 120 | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| train.files.chunk_ids | data\interim\consistency\mil_chunk_cache_mbert_v1\train_chunk_ids.npy | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| train.files.chunk_lengths | data\interim\consistency\mil_chunk_cache_mbert_v1\train_chunk_lengths.npy | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| train.files.chunk_starts | data\interim\consistency\mil_chunk_cache_mbert_v1\train_chunk_starts.npy | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| train.files.chunk_counts | data\interim\consistency\mil_chunk_cache_mbert_v1\train_chunk_counts.npy | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| train.files.article_index | data\interim\consistency\mil_chunk_cache_mbert_v1\train_article_index.parquet | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| validation.split | validation | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| validation.articles | 28713 | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| validation.chunk_slots | 172278 | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| validation.actual_chunks | 162793 | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| validation.chunks_per_article.min | 1 | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| validation.chunks_per_article.mean | 5.669661825653885 | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| validation.chunks_per_article.median | 6.0 | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| validation.chunks_per_article.p95 | 6.0 | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| validation.chunks_per_article.max | 6 | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| validation.article_tokens.min | 65 | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| validation.article_tokens.median | 756.0 | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| validation.article_tokens.mean | 859.4635182669871 | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| validation.article_tokens.p95 | 1713.3999999999978 | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| validation.article_tokens.max | 11648 | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| validation.chunk_lengths.min | 1 | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| validation.chunk_lengths.median | 120.0 | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| validation.chunk_lengths.mean | 107.1782509075943 | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| validation.chunk_lengths.max | 120 | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| validation.files.chunk_ids | data\interim\consistency\mil_chunk_cache_mbert_v1\validation_chunk_ids.npy | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| validation.files.chunk_lengths | data\interim\consistency\mil_chunk_cache_mbert_v1\validation_chunk_lengths.npy | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| validation.files.chunk_starts | data\interim\consistency\mil_chunk_cache_mbert_v1\validation_chunk_starts.npy | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| validation.files.chunk_counts | data\interim\consistency\mil_chunk_cache_mbert_v1\validation_chunk_counts.npy | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| validation.files.article_index | data\interim\consistency\mil_chunk_cache_mbert_v1\validation_article_index.parquet | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |
| cross_split_article_uid_overlap | 0 | [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json) |

Nguồn: `C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json` — [P084](/C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json). Field/CSV columns/source code: `Các field ghi ở cột trái; không tính lại metric`.

D — Fusion: không có encoder/checkpoint train thứ tư độc lập; kết hợp score từ A và C với trọng số/threshold trong final_model_lock. M2 clickbait có checkpoint khác các nhánh consistency. Runtime load đúng các file trong release manifest ở mục 19.
## Chẩn đoán A/B và oracle complementarity
| Field JSON | Giá trị lưu trong file | Source |
| --- | --- | --- |
| split | validation | [P085](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_a_b_comparison/comparison_summary.json) |
| test_used | false | [P085](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_a_b_comparison/comparison_summary.json) |
| baseline_a.input | headline + description | [P085](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_a_b_comparison/comparison_summary.json) |
| baseline_a.threshold | 0.43000000000000005 | [P085](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_a_b_comparison/comparison_summary.json) |
| baseline_a.macro_f1 | 0.8966718738837183 | [P085](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_a_b_comparison/comparison_summary.json) |
| baseline_b.input | headline + article head | [P085](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_a_b_comparison/comparison_summary.json) |
| baseline_b.threshold | 0.45000000000000007 | [P085](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_a_b_comparison/comparison_summary.json) |
| baseline_b.macro_f1 | 0.8614223379134691 | [P085](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_a_b_comparison/comparison_summary.json) |
| macro_f1_delta_b_minus_a | -0.03524953597024916 | [P085](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_a_b_comparison/comparison_summary.json) |
| error_overlap.BOTH_CORRECT.count | 45725 | [P085](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_a_b_comparison/comparison_summary.json) |
| error_overlap.BOTH_CORRECT.rate | 0.7962421202939435 | [P085](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_a_b_comparison/comparison_summary.json) |
| error_overlap.A_ONLY_CORRECT.count | 5768 | [P085](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_a_b_comparison/comparison_summary.json) |
| error_overlap.A_ONLY_CORRECT.rate | 0.10044230836206597 | [P085](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_a_b_comparison/comparison_summary.json) |
| error_overlap.B_ONLY_CORRECT.count | 3744 | [P085](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_a_b_comparison/comparison_summary.json) |
| error_overlap.B_ONLY_CORRECT.rate | 0.06519694911712465 | [P085](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_a_b_comparison/comparison_summary.json) |
| error_overlap.BOTH_WRONG.count | 2189 | [P085](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_a_b_comparison/comparison_summary.json) |
| error_overlap.BOTH_WRONG.rate | 0.03811862222686588 | [P085](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_a_b_comparison/comparison_summary.json) |
| a_only_correct | 5768 | [P085](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_a_b_comparison/comparison_summary.json) |
| b_only_correct | 3744 | [P085](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_a_b_comparison/comparison_summary.json) |
| both_wrong | 2189 | [P085](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_a_b_comparison/comparison_summary.json) |
| oracle_either_model_correct_rate | 0.9618813777731341 | [P085](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_a_b_comparison/comparison_summary.json) |
| oracle_note | This is not a deployable model score. It measures prediction complementarity only. | [P085](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_a_b_comparison/comparison_summary.json) |

Nguồn: `C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_a_b_comparison/comparison_summary.json` — [P085](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_a_b_comparison/comparison_summary.json). Field/CSV columns/source code: `Các field ghi ở cột trái; không tính lại metric`.

Oracle either-model-correct là diagnostic complementarity dùng nhãn thật, không phải deployed model score.
# 11. Module 3 — Fusion và threshold
## Search grid / selection config (đọc mã, không chạy search)
| Config/call trong mã (không tự coi là actual run) | Biểu thức | Vị trí | Source |
| --- | --- | --- | --- |
| SEED | 20260920 | line 101 | [P087](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/19_compare_and_fuse_description_mil.py) |
| ALPHA_VALUES | np.arange(0.0, 1.001, 0.05) | line 103 | [P087](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/19_compare_and_fuse_description_mil.py) |
| THRESHOLD_VALUES | np.arange(0.05, 0.951, 0.01) | line 109 | [P087](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/19_compare_and_fuse_description_mil.py) |

Nguồn: `C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/19_compare_and_fuse_description_mil.py` — [P087](/C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/19_compare_and_fuse_description_mil.py). Field/CSV columns/source code: `SEED, ALPHA_VALUES, THRESHOLD_VALUES`.

## Fusion: calibration và holdout riêng
| Field JSON | Giá trị lưu trong file | Source |
| --- | --- | --- |
| stage | 19_description_mil_fusion | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| dataset | headline_consistency_v1 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| analysis_split | validation | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| test_used | false | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| internal_protocol.split_unit | anchor_article_uid | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| internal_protocol.calibration_anchors | 14356 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| internal_protocol.holdout_anchors | 14357 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| internal_protocol.calibration_pairs | 28712 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| internal_protocol.holdout_pairs | 28714 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| calibration.description_threshold | 0.44000000000000006 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| calibration.mil_threshold | 0.37000000000000005 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| calibration.fusion_description_weight | 0.5 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| calibration.fusion_mil_weight | 0.5 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| calibration.fusion_threshold | 0.45000000000000007 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| calibration.fusion_macro_f1 | 0.9389798949018551 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| holdout.description.threshold | 0.44000000000000006 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| holdout.description.accuracy | 0.8971233544612384 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| holdout.description.precision_consistent | 0.9078034475359417 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| holdout.description.recall_consistent | 0.8840286968029533 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| holdout.description.f1_consistent | 0.8957583456842403 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| holdout.description.macro_f1 | 0.8971057111714429 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| holdout.description.roc_auc | 0.9661835296575112 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| holdout.description.average_precision | 0.9695071841193982 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| holdout.description.confusion_matrix.tn | 13068 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| holdout.description.confusion_matrix.fp | 1289 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| holdout.description.confusion_matrix.fn | 1665 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| holdout.description.confusion_matrix.tp | 12692 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| holdout.mil.threshold | 0.37000000000000005 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| holdout.mil.accuracy | 0.8986905342341714 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| holdout.mil.precision_consistent | 0.8912508544087492 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| holdout.mil.recall_consistent | 0.9081980915232988 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| holdout.mil.f1_consistent | 0.8996446682995826 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| holdout.mil.macro_f1 | 0.8986813756743509 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| holdout.mil.roc_auc | 0.9655854681531162 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| holdout.mil.average_precision | 0.9652584268704383 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| holdout.mil.confusion_matrix.tn | 12766 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| holdout.mil.confusion_matrix.fp | 1591 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| holdout.mil.confusion_matrix.fn | 1318 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| holdout.mil.confusion_matrix.tp | 13039 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| holdout.fusion.threshold | 0.45000000000000007 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| holdout.fusion.accuracy | 0.9378003761231455 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| holdout.fusion.precision_consistent | 0.9324389404884761 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| holdout.fusion.recall_consistent | 0.9439994427805252 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| holdout.fusion.f1_consistent | 0.9381835802298214 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| holdout.fusion.macro_f1 | 0.9377979857975571 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| holdout.fusion.roc_auc | 0.9839671443689069 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| holdout.fusion.average_precision | 0.9836726163067558 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| holdout.fusion.confusion_matrix.tn | 13375 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| holdout.fusion.confusion_matrix.fp | 982 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| holdout.fusion.confusion_matrix.fn | 804 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| holdout.fusion.confusion_matrix.tp | 13553 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| holdout.fusion_minus_description_macro_f1 | 0.040692274626114244 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| holdout.fusion_minus_mil_macro_f1 | 0.03911661012320622 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| error_overlap_description_vs_mil.BOTH_CORRECT.count | 23719 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| error_overlap_description_vs_mil.BOTH_CORRECT.rate | 0.82604304520443 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| error_overlap_description_vs_mil.A_ONLY_CORRECT.count | 2041 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| error_overlap_description_vs_mil.A_ONLY_CORRECT.rate | 0.07108030925680853 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| error_overlap_description_vs_mil.C_ONLY_CORRECT.count | 2086 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| error_overlap_description_vs_mil.C_ONLY_CORRECT.rate | 0.0726474890297416 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| error_overlap_description_vs_mil.BOTH_WRONG.count | 868 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| error_overlap_description_vs_mil.BOTH_WRONG.rate | 0.03022915650901999 | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |
| warning | Validation diagnostic only. No test examples or test predictions were used. | [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json) |

Nguồn: `C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json` — [P088](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json). Field/CSV columns/source code: `Các field ghi ở cột trái; không tính lại metric`.

## Final lock hybrid
| Field JSON | Giá trị lưu trong file | Source |
| --- | --- | --- |
| architecture | description_mbert + full_article_max_mil_mbert | [P089](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/final_model_lock.json) |
| status | FROZEN_BEFORE_TEST_EVALUATION | [P089](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/final_model_lock.json) |
| description_weight | 0.5 | [P089](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/final_model_lock.json) |
| mil_weight | 0.5 | [P089](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/final_model_lock.json) |
| decision_threshold | 0.45 | [P089](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/final_model_lock.json) |
| model_name | bert-base-multilingual-cased | [P089](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/final_model_lock.json) |
| chunk_policy.chunk_tokens | 120 | [P089](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/final_model_lock.json) |
| chunk_policy.chunk_overlap | 24 | [P089](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/final_model_lock.json) |
| chunk_policy.max_chunks | 6 | [P089](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/final_model_lock.json) |
| chunk_policy.sampling | uniform positions across full article | [P089](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/final_model_lock.json) |
| selection_source | validation calibration/holdout Steps 15-19 | [P089](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/final_model_lock.json) |
| test_tuning_allowed | false | [P089](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/final_model_lock.json) |

Nguồn: `C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/final_model_lock.json` — [P089](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/final_model_lock.json). Field/CSV columns/source code: `Các field ghi ở cột trái; không tính lại metric`.

validation selection != official test. Split nội bộ theo anchor_article_uid; trọng số đã chọn description=0.5, MIL=0.5 và threshold=0.45 theo final lock. Script 20 ghi lock trước tạo cache test; script 21 đọc và kiểm tra lock. Không gộp calibration/holdout với full-validation metrics hay official test.
Nguồn: `C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/final_model_lock.json` — [P089](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/final_model_lock.json). Field/CSV columns/source code: `description_weight, mil_weight, decision_threshold, selection_source, test_tuning_allowed`.

# 12. Module 3 — Official test
## Official hybrid và branch reference tại final threshold
| Field JSON | Giá trị lưu trong file | Source |
| --- | --- | --- |
| stage | 21_official_final_test_evaluation | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| dataset | headline_consistency_v1 | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| split | test | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| evaluation_status | OFFICIAL_TEST_EVALUATED | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| configuration_frozen_before_test | true | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| test_tuning_performed | false | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| frozen_configuration.description_weight | 0.5 | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| frozen_configuration.mil_weight | 0.5 | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| frozen_configuration.decision_threshold | 0.45 | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| frozen_configuration.description_model | baseline_title_description_mbert | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| frozen_configuration.mil_model | full_article_max_mil_mbert | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| test_pairs | 57376 | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| test_anchors | 28688 | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| official_final_metrics.threshold | 0.45 | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| official_final_metrics.accuracy | 0.9395914668153932 | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| official_final_metrics.precision_consistent | 0.9372442965120311 | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| official_final_metrics.recall_consistent | 0.9422755158951478 | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| official_final_metrics.f1_consistent | 0.9397531722579524 | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| official_final_metrics.macro_f1 | 0.9395910316219684 | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| official_final_metrics.roc_auc | 0.9845379930509566 | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| official_final_metrics.average_precision | 0.9845682749831874 | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| official_final_metrics.confusion_matrix.tn | 26878 | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| official_final_metrics.confusion_matrix.fp | 1810 | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| official_final_metrics.confusion_matrix.fn | 1656 | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| official_final_metrics.confusion_matrix.tp | 27032 | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| branch_reference_metrics_fixed_at_final_threshold.description.threshold | 0.45 | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| branch_reference_metrics_fixed_at_final_threshold.description.accuracy | 0.8975181260457334 | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| branch_reference_metrics_fixed_at_final_threshold.description.precision_consistent | 0.9140284635492303 | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| branch_reference_metrics_fixed_at_final_threshold.description.recall_consistent | 0.877579475738985 | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| branch_reference_metrics_fixed_at_final_threshold.description.f1_consistent | 0.8954332052923603 | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| branch_reference_metrics_fixed_at_final_threshold.description.macro_f1 | 0.8974773681964194 | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| branch_reference_metrics_fixed_at_final_threshold.description.roc_auc | 0.965732003105939 | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| branch_reference_metrics_fixed_at_final_threshold.description.average_precision | 0.9696157162985932 | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| branch_reference_metrics_fixed_at_final_threshold.description.confusion_matrix.tn | 26320 | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| branch_reference_metrics_fixed_at_final_threshold.description.confusion_matrix.fp | 2368 | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| branch_reference_metrics_fixed_at_final_threshold.description.confusion_matrix.fn | 3512 | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| branch_reference_metrics_fixed_at_final_threshold.description.confusion_matrix.tp | 25176 | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| branch_reference_metrics_fixed_at_final_threshold.mil.threshold | 0.45 | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| branch_reference_metrics_fixed_at_final_threshold.mil.accuracy | 0.9001673173452315 | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| branch_reference_metrics_fixed_at_final_threshold.mil.precision_consistent | 0.9127714655544369 | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| branch_reference_metrics_fixed_at_final_threshold.mil.recall_consistent | 0.8848996095928612 | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| branch_reference_metrics_fixed_at_final_threshold.mil.f1_consistent | 0.8986194690265487 | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| branch_reference_metrics_fixed_at_final_threshold.mil.macro_f1 | 0.9001440406315192 | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| branch_reference_metrics_fixed_at_final_threshold.mil.roc_auc | 0.9667867237407575 | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| branch_reference_metrics_fixed_at_final_threshold.mil.average_precision | 0.9667674678611565 | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| branch_reference_metrics_fixed_at_final_threshold.mil.confusion_matrix.tn | 26262 | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| branch_reference_metrics_fixed_at_final_threshold.mil.confusion_matrix.fp | 2426 | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| branch_reference_metrics_fixed_at_final_threshold.mil.confusion_matrix.fn | 3302 | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| branch_reference_metrics_fixed_at_final_threshold.mil.confusion_matrix.tp | 25386 | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |
| methodological_warning | This test set uses presumed-consistent original positives and synthetic hard negatives. It is not a human-gold real-world consistency benchmark. | [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json) |

Nguồn: `C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json` — [P092](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json). Field/CSV columns/source code: `Các field ghi ở cột trái; không tính lại metric`.

Positive class CONSISTENT (1), dataset headline_consistency_v1, split test. Branch reference metrics ở official artifact đều được tính tại final threshold 0.45, không tráo với ngưỡng tune riêng description/MAX-MIL trên validation. Macro precision và macro recall không có field official: [CHƯA ĐỦ DỮ LIỆU].
## M3 official_test_by_category.csv
| category | n | accuracy | macro_f1 | f1_consistent | Source |
| --- | --- | --- | --- | --- | --- |
| Giáo dục | 5138 | 0.9274036590112884 | 0.9274035242629501 | 0.9275024295432459 | [P093](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_by_category.csv) |
| Kinh doanh | 7618 | 0.9276713048044106 | 0.9276709446168588 | 0.9278323510150622 | [P093](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_by_category.csv) |
| Thể thao | 8606 | 0.93016500116198 | 0.9301578635433801 | 0.9308639134936155 | [P093](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_by_category.csv) |
| Sức khỏe | 3482 | 0.9336588167719702 | 0.9336587675262479 | 0.9337159253945481 | [P093](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_by_category.csv) |
| Thế giới | 2960 | 0.9344594594594594 | 0.9344589807084055 | 0.9342818428184282 | [P093](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_by_category.csv) |
| Bất động sản | 1498 | 0.9372496662216289 | 0.937246869741308 | 0.9368279569892473 | [P093](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_by_category.csv) |
| Xe | 4040 | 0.9408415841584158 | 0.940840768623967 | 0.9410604192355118 | [P093](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_by_category.csv) |
| Khoa học công nghệ | 4900 | 0.9420408163265306 | 0.9420261260704295 | 0.9429489754921655 | [P093](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_by_category.csv) |
| Đời sống | 2748 | 0.9479621542940321 | 0.9479555311452434 | 0.9485426412378554 | [P093](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_by_category.csv) |
| Du lịch | 1852 | 0.9519438444924406 | 0.9519437183937117 | 0.952021563342318 | [P093](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_by_category.csv) |
| Giải trí | 8004 | 0.9523988005997002 | 0.9523986750280544 | 0.9523213615317232 | [P093](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_by_category.csv) |
| Thời sự | 5808 | 0.9555785123966942 | 0.9555757257443565 | 0.9552238805970149 | [P093](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_by_category.csv) |
| Pháp luật | 722 | 0.961218836565097 | 0.9612113955915761 | 0.9606741573033708 | [P093](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_by_category.csv) |

Nguồn: `C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_by_category.csv` — [P093](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_by_category.csv). Field/CSV columns/source code: `category, n, accuracy, macro_f1, f1_consistent`.

## M3 official_test_negative_by_tier.csv
| tier | n | negative_recall | false_positives | Source |
| --- | --- | --- | --- | --- |
| FALLBACK | 14639 | 0.9571008948698682 | 628 | [P094](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_negative_by_tier.csv) |
| PRIMARY | 14049 | 0.9158658979286782 | 1182 | [P094](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_negative_by_tier.csv) |

Nguồn: `C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_negative_by_tier.csv` — [P094](/C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_negative_by_tier.csv). Field/CSV columns/source code: `tier, n, negative_recall, false_positives`.

Methodological warning: test sử dụng presumed-consistent original positives và synthetic hard negatives; không phải human-gold real-world consistency benchmark. Kích thước negative tier là số negative, không phải toàn bộ pairs.

<!-- EVIDENCE_STATE {"sources": {"C:/projectSamSung_v1/docs/thesis/00_outline_v2.md": {"id": "P001", "module": "SHARED", "type": ["Đề cương"], "sections": ["1"], "notes": ["Điểm xuất phát S01–S42, B01–B07, G01–G11; không lấy số liệu thay artifact"], "mode": "Đọc nội dung"}, "C:/projectSamSung_v1/README_V1.md": {"id": "P002", "module": "SHARED", "type": ["L"], "sections": ["1"], "notes": ["Phạm vi triển khai"], "mode": "Đọc nội dung"}, "C:/projectSamSung_v1/system/backend/app/services/model_registry.py": {"id": "P003", "module": "SHARED", "type": ["Triển khai"], "sections": ["2"], "notes": [], "mode": "Đọc nội dung"}, "C:/projectSamSung_v1/system/backend/app/adapters/spelling.py": {"id": "P004", "module": "M1", "type": ["Triển khai"], "sections": ["2"], "notes": [], "mode": "Đọc nội dung"}, "C:/projectSamSung_v1/system/backend/app/adapters/clickbait.py": {"id": "P005", "module": "M2", "type": ["Triển khai"], "sections": ["2"], "notes": [], "mode": "Đọc nội dung"}, "C:/projectSamSung_v1/system/backend/app/adapters/headline_consistency.py": {"id": "P006", "module": "M3", "type": ["Triển khai"], "sections": ["2"], "notes": [], "mode": "Đọc nội dung"}, "C:/projectSamSung_v1/system/backend/app/api/main.py": {"id": "P007", "module": "SHARED", "type": ["Triển khai"], "sections": ["2"], "notes": [], "mode": "Đọc nội dung"}, "C:/projectSamSung_v1/system/backend/app/api/schemas.py": {"id": "P008", "module": "SHARED", "type": ["Triển khai"], "sections": ["2"], "notes": [], "mode": "Đọc nội dung"}, "C:/projectSamSung_v1/system/backend/app/api/service.py": {"id": "P009", "module": "SHARED", "type": ["Triển khai"], "sections": ["2"], "notes": [], "mode": "Đọc nội dung"}, "C:/1_projectSamSung/scripts/01_download_raw.py": {"id": "P010", "module": "M1", "type": ["A"], "sections": ["3"], "notes": [], "mode": "Đọc nội dung"}, "C:/1_projectSamSung/scripts/06_mine_real_error_patterns.py": {"id": "P011", "module": "M1", "type": ["A"], "sections": ["3"], "notes": [], "mode": "Đọc nội dung"}, "C:/1_projectSamSung/scripts/07_filter_real_error_patterns.py": {"id": "P012", "module": "M1", "type": ["A"], "sections": ["3"], "notes": [], "mode": "Đọc nội dung"}, "C:/1_projectSamSung/scripts/08_build_trusted_error_lookup.py": {"id": "P013", "module": "M1", "type": ["A"], "sections": ["3"], "notes": [], "mode": "Đọc nội dung"}, "C:/1_projectSamSung/scripts/09_generate_hybrid_spelling_dataset.py": {"id": "P014", "module": "M1", "type": ["A"], "sections": ["3"], "notes": [], "mode": "Đọc nội dung"}, "C:/1_projectSamSung/scripts/05_validate_spelling_dataset.py": {"id": "P015", "module": "M1", "type": ["A"], "sections": ["3"], "notes": [], "mode": "Đọc nội dung"}, "C:/1_projectSamSung/scripts/12_split_spelling_dataset.py": {"id": "P016", "module": "M1", "type": ["A"], "sections": ["3"], "notes": [], "mode": "Đọc nội dung"}, "C:/1_projectSamSung/data/raw/vietnews/train/dataset_info.json": {"id": "P017", "module": "M1", "type": ["A"], "sections": ["3"], "notes": [], "mode": "Đọc nội dung"}, "C:/1_projectSamSung/data/interim/real_error_patterns_v03/trusted_error_summary.json": {"id": "P018", "module": "M1", "type": ["A"], "sections": ["3"], "notes": [], "mode": "Đọc nội dung"}, "C:/1_projectSamSung/data/interim/vn_newsspell_hybrid_v03_full/generation_summary.json": {"id": "P019", "module": "M1", "type": ["A"], "sections": ["3"], "notes": [], "mode": "Đọc nội dung"}, "C:/projectSamSung_v1/data/processed/vn_newsspell_v1/split_summary.json": {"id": "P020", "module": "M1", "type": ["A"], "sections": ["3"], "notes": [], "mode": "Đọc nội dung"}, "C:/1_projectSamSung/results/module1_baselines/baseline_comparison.json": {"id": "P021", "module": "M1", "type": ["D"], "sections": ["4"], "notes": [], "mode": "Đọc nội dung"}, "C:/1_projectSamSung/scripts/13_train_classical_baselines.py": {"id": "P022", "module": "M1", "type": ["B"], "sections": ["4"], "notes": [], "mode": "Đọc nội dung"}, "C:/projectSamSung_v1/results/module1_phobert_mil_full/training_config.json": {"id": "P023", "module": "M1", "type": ["B"], "sections": ["4"], "notes": [], "mode": "Đọc nội dung"}, "C:/1_projectSamSung/scripts/14_train_phobert_mil.py": {"id": "P024", "module": "M1", "type": ["B"], "sections": ["4"], "notes": [], "mode": "Đọc nội dung"}, "C:/projectSamSung_v1/results/module1_phobert_mil_full/training_history.json": {"id": "P025", "module": "M1", "type": ["C"], "sections": ["4"], "notes": [], "mode": "Đọc nội dung"}, "C:/projectSamSung_v1/results/module1_phobert_mil_full/best_validation_metrics.json": {"id": "P026", "module": "M1", "type": ["D"], "sections": ["4"], "notes": [], "mode": "Đọc nội dung"}, "C:/1_projectSamSung/scripts/15_tune_phobert_threshold.py": {"id": "P027", "module": "M1", "type": ["D"], "sections": ["4"], "notes": [], "mode": "Đọc nội dung"}, "C:/projectSamSung_v1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json": {"id": "P028", "module": "M1", "type": ["D"], "sections": ["4"], "notes": [], "mode": "Đọc nội dung"}, "C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/official_test_metrics.json": {"id": "P029", "module": "M1", "type": ["E"], "sections": ["5"], "notes": [], "mode": "Đọc nội dung"}, "C:/projectSamSung_v1/scripts/16_evaluate_phobert_test.py": {"id": "P030", "module": "M1", "type": ["E"], "sections": ["5"], "notes": [], "mode": "Đọc nội dung"}, "C:/projectSamSung_v1/results/module1_phobert_mil_full/test_evaluation/test_error_analysis.json": {"id": "P031", "module": "M1", "type": ["E"], "sections": ["5"], "notes": [], "mode": "Đọc nội dung"}, "C:/1_projectSamSung/module2_headline_integrity/data/raw/download_manifest.json": {"id": "P032", "module": "SHARED", "type": ["A"], "sections": ["6"], "notes": [], "mode": "Đọc nội dung"}, "C:/1_projectSamSung/module2_headline_integrity/data/raw/clickbait/viclickbait_2025/README.md": {"id": "P033", "module": "M2", "type": ["A"], "sections": ["6"], "notes": ["Nguồn dataset, chưa đủ metadata bibliography"], "mode": "Đọc nội dung"}, "C:/projectSamSung_v1/module2_headline_integrity/data/processed/clickbait/viclickbait_v1/prepare_report.json": {"id": "P034", "module": "M2", "type": ["A"], "sections": ["6"], "notes": [], "mode": "Đọc nội dung"}, "C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/03_prepare_clickbait_dataset.py": {"id": "P035", "module": "M2", "type": ["A"], "sections": ["6"], "notes": [], "mode": "Đọc nội dung"}, "C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/diagnostic_baselines_v1/diagnostic_report.json": {"id": "P036", "module": "M2", "type": ["D"], "sections": ["7"], "notes": [], "mode": "Đọc nội dung"}, "C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/05_train_diagnostic_baselines.py": {"id": "P037", "module": "M2", "type": ["B"], "sections": ["7"], "notes": [], "mode": "Đọc nội dung"}, "C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/run_config.json": {"id": "P038", "module": "M2", "type": ["B"], "sections": ["7"], "notes": [], "mode": "Đọc nội dung"}, "C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/06_train_mbert_title_only.py": {"id": "P039", "module": "M2", "type": ["B"], "sections": ["7"], "notes": [], "mode": "Đọc nội dung"}, "C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/training_history.csv": {"id": "P040", "module": "M2", "type": ["C"], "sections": ["7"], "notes": [], "mode": "Đọc nội dung"}, "C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_only_v1/validation_metrics.json": {"id": "P041", "module": "M2", "type": ["D"], "sections": ["7"], "notes": [], "mode": "Đọc nội dung"}, "C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/run_config.json": {"id": "P042", "module": "M2", "type": ["B"], "sections": ["7"], "notes": [], "mode": "Đọc nội dung"}, "C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/07_train_mbert_title_lead.py": {"id": "P043", "module": "M2", "type": ["B"], "sections": ["7"], "notes": [], "mode": "Đọc nội dung"}, "C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/training_history.csv": {"id": "P044", "module": "M2", "type": ["C"], "sections": ["7"], "notes": [], "mode": "Đọc nội dung"}, "C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/mbert_title_lead_v1/validation_metrics.json": {"id": "P045", "module": "M2", "type": ["D"], "sections": ["7"], "notes": [], "mode": "Đọc nội dung"}, "C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/compare_title_vs_title_lead_v1/comparison_report.json": {"id": "P046", "module": "M2", "type": ["D"], "sections": ["7"], "notes": [], "mode": "Đọc nội dung"}, "C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/08_compare_mbert_title_vs_title_lead.py": {"id": "P047", "module": "M2", "type": ["D"], "sections": ["7"], "notes": [], "mode": "Đọc nội dung"}, "C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/final_model_lock.json": {"id": "P048", "module": "M2", "type": ["L"], "sections": ["7"], "notes": [], "mode": "Đọc nội dung"}, "C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/09_lock_final_clickbait_model.py": {"id": "P049", "module": "M2", "type": ["L"], "sections": ["7"], "notes": [], "mode": "Đọc nội dung"}, "C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_metrics.json": {"id": "P050", "module": "M2", "type": ["E"], "sections": ["8"], "notes": [], "mode": "Đọc nội dung"}, "C:/1_projectSamSung/module2_headline_integrity/scripts/clickbait/10_evaluate_final_clickbait_test.py": {"id": "P051", "module": "M2", "type": ["E"], "sections": ["8"], "notes": [], "mode": "Đọc nội dung"}, "C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_by_source.csv": {"id": "P052", "module": "M2", "type": ["E"], "sections": ["8"], "notes": [], "mode": "Đọc nội dung"}, "C:/projectSamSung_v1/module2_headline_integrity/results/clickbait/final_title_only_v1/official_test_by_category.csv": {"id": "P053", "module": "M2", "type": ["E"], "sections": ["8"], "notes": [], "mode": "Đọc nội dung"}, "C:/1_projectSamSung/module2_headline_integrity/data/raw/consistency/viet_online_news/README.md": {"id": "P054", "module": "M3", "type": ["A"], "sections": ["9"], "notes": ["README corpus có số split cần đối chiếu prepare_report; không thay dataset consistency"], "mode": "Đọc nội dung"}, "C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/03_prepare_articles.py": {"id": "P055", "module": "M3", "type": ["A"], "sections": ["9"], "notes": [], "mode": "Đọc nội dung"}, "C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/04_build_positive_pairs.py": {"id": "P056", "module": "M3", "type": ["A"], "sections": ["9"], "notes": [], "mode": "Đọc nội dung"}, "C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/05_mine_hard_negative_candidates.py": {"id": "P057", "module": "M3", "type": ["A"], "sections": ["9"], "notes": [], "mode": "Đọc nội dung"}, "C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/06_analyze_candidate_quality.py": {"id": "P058", "module": "M3", "type": ["A"], "sections": ["9"], "notes": [], "mode": "Đọc nội dung"}, "C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/06b_prepare_manual_review_fulltext.py": {"id": "P059", "module": "M3", "type": ["A"], "sections": ["9"], "notes": [], "mode": "Đọc nội dung"}, "C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/07_analyze_manual_annotations.py": {"id": "P060", "module": "M3", "type": ["A"], "sections": ["9"], "notes": [], "mode": "Đọc nội dung"}, "C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/08_build_negative_pairs.py": {"id": "P061", "module": "M3", "type": ["A"], "sections": ["9"], "notes": [], "mode": "Đọc nội dung"}, "C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/09_build_consistency_dataset.py": {"id": "P062", "module": "M3", "type": ["A"], "sections": ["9"], "notes": [], "mode": "Đọc nội dung"}, "C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/10_validate_consistency_dataset.py": {"id": "P063", "module": "M3", "type": ["A"], "sections": ["9"], "notes": [], "mode": "Đọc nội dung"}, "C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/articles_v1/prepare_report.json": {"id": "P064", "module": "M3", "type": ["A"], "sections": ["9"], "notes": [], "mode": "Đọc nội dung"}, "C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/positive_pairs_v1/build_report.json": {"id": "P065", "module": "M3", "type": ["A"], "sections": ["9"], "notes": [], "mode": "Đọc nội dung"}, "C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/candidate_pairs_v1/mining_report.json": {"id": "P066", "module": "M3", "type": ["A"], "sections": ["9"], "notes": [], "mode": "Đọc nội dung"}, "C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/negative_pairs_v1/build_report.json": {"id": "P067", "module": "M3", "type": ["A"], "sections": ["9"], "notes": [], "mode": "Đọc nội dung"}, "C:/projectSamSung_v1/module2_headline_integrity/results/consistency/manual_annotation_analysis_v1/analysis_report.json": {"id": "P068", "module": "M3", "type": ["A"], "sections": ["9"], "notes": [], "mode": "Đọc nội dung"}, "C:/projectSamSung_v1/module2_headline_integrity/data/processed/consistency/headline_consistency_v1/dataset_manifest.json": {"id": "P069", "module": "M3", "type": ["A"], "sections": ["9"], "notes": [], "mode": "Đọc nội dung"}, "C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_dataset_validation_v1/validation_report.json": {"id": "P070", "module": "M3", "type": ["A"], "sections": ["9"], "notes": [], "mode": "Đọc nội dung"}, "C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/run_config.json": {"id": "P071", "module": "M3", "type": ["B"], "sections": ["10"], "notes": [], "mode": "Đọc nội dung"}, "C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/11_train_baseline_title_description.py": {"id": "P072", "module": "M3", "type": ["B"], "sections": ["10"], "notes": [], "mode": "Đọc nội dung"}, "C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/training_history.csv": {"id": "P073", "module": "M3", "type": ["C"], "sections": ["10"], "notes": [], "mode": "Đọc nội dung"}, "C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_description_mbert/validation_metrics.json": {"id": "P074", "module": "M3", "type": ["D"], "sections": ["10"], "notes": [], "mode": "Đọc nội dung"}, "C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/run_config.json": {"id": "P075", "module": "M3", "type": ["B"], "sections": ["10"], "notes": [], "mode": "Đọc nội dung"}, "C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/13_train_baseline_title_article_head.py": {"id": "P076", "module": "M3", "type": ["B"], "sections": ["10"], "notes": [], "mode": "Đọc nội dung"}, "C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/training_history.csv": {"id": "P077", "module": "M3", "type": ["C"], "sections": ["10"], "notes": [], "mode": "Đọc nội dung"}, "C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_title_article_head_mbert/validation_metrics.json": {"id": "P078", "module": "M3", "type": ["D"], "sections": ["10"], "notes": [], "mode": "Đọc nội dung"}, "C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/run_config.json": {"id": "P079", "module": "M3", "type": ["B"], "sections": ["10"], "notes": [], "mode": "Đọc nội dung"}, "C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/18_train_full_article_max_mil.py": {"id": "P080", "module": "M3", "type": ["B"], "sections": ["10"], "notes": [], "mode": "Đọc nội dung"}, "C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/training_history.csv": {"id": "P081", "module": "M3", "type": ["C"], "sections": ["10"], "notes": [], "mode": "Đọc nội dung"}, "C:/projectSamSung_v1/module2_headline_integrity/results/consistency/full_article_max_mil_mbert/validation_metrics.json": {"id": "P082", "module": "M3", "type": ["D"], "sections": ["10"], "notes": [], "mode": "Đọc nội dung"}, "C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/17_prepare_mil_chunk_cache.py": {"id": "P083", "module": "M3", "type": ["B"], "sections": ["10"], "notes": [], "mode": "Đọc nội dung"}, "C:/1_projectSamSung/module2_headline_integrity/data/interim/consistency/mil_chunk_cache_mbert_v1/cache_report.json": {"id": "P084", "module": "M3", "type": ["A"], "sections": ["10"], "notes": [], "mode": "Đọc nội dung"}, "C:/projectSamSung_v1/module2_headline_integrity/results/consistency/baseline_a_b_comparison/comparison_summary.json": {"id": "P085", "module": "M3", "type": ["D"], "sections": ["10"], "notes": [], "mode": "Đọc nội dung"}, "C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/14_compare_baselines_a_b.py": {"id": "P086", "module": "M3", "type": ["D"], "sections": ["10"], "notes": [], "mode": "Đọc nội dung"}, "C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/19_compare_and_fuse_description_mil.py": {"id": "P087", "module": "M3", "type": ["D"], "sections": ["11"], "notes": [], "mode": "Đọc nội dung"}, "C:/projectSamSung_v1/module2_headline_integrity/results/consistency/description_mil_fusion_v1/fusion_report.json": {"id": "P088", "module": "M3", "type": ["D"], "sections": ["11"], "notes": [], "mode": "Đọc nội dung"}, "C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/final_model_lock.json": {"id": "P089", "module": "M3", "type": ["L"], "sections": ["11"], "notes": [], "mode": "Đọc nội dung"}, "C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/20_prepare_test_mil_chunk_cache.py": {"id": "P090", "module": "M3", "type": ["L"], "sections": ["11"], "notes": ["Ghi final lock trước test preprocessing"], "mode": "Đọc nội dung"}, "C:/1_projectSamSung/module2_headline_integrity/scripts/consistency/21_evaluate_final_hybrid_test.py": {"id": "P091", "module": "M3", "type": ["E"], "sections": ["11"], "notes": ["load_and_validate_lock; không tune test"], "mode": "Đọc nội dung"}, "C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_metrics.json": {"id": "P092", "module": "M3", "type": ["E"], "sections": ["12"], "notes": [], "mode": "Đọc nội dung"}, "C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_by_category.csv": {"id": "P093", "module": "M3", "type": ["E"], "sections": ["12"], "notes": [], "mode": "Đọc nội dung"}, "C:/projectSamSung_v1/module2_headline_integrity/results/consistency/final_hybrid_v1/official_test_negative_by_tier.csv": {"id": "P094", "module": "M3", "type": ["E"], "sections": ["12"], "notes": [], "mode": "Đọc nội dung"}}, "tables": 62, "figures": 0, "conflicts": [], "section": 12, "outline_hash": "82ebe3b7fe765199c2a66e823b4f4a92fed7114d019e9604b6cba54032a9308b"} -->
