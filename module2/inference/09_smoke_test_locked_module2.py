from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import time


# ============================================================
# PATHS
# ============================================================

MODULE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = MODULE_ROOT.parent

PIPELINE_PATH = (
    MODULE_ROOT
    / "inference"
    / "headline_integrity_pipeline.py"
)

LOCK_PATH = (
    MODULE_ROOT
    / "results"
    / "integration"
    / "unified_module2_v1"
    / "unified_module2_lock.json"
)

MANIFEST_PATH = (
    MODULE_ROOT
    / "results"
    / "integration"
    / "unified_module2_v1"
    / "artifact_manifest.json"
)

OUTPUT_DIR = (
    MODULE_ROOT
    / "results"
    / "integration"
    / "unified_module2_v1"
    / "smoke_test"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "smoke_test_report.json"
)

OUTPUTS_PATH = (
    OUTPUT_DIR
    / "smoke_test_outputs.json"
)


# ============================================================
# EXPECTED CONTRACT
# ============================================================

EXPECTED_VERSION = (
    "unified_module2_v1"
)

EXPECTED_PIPELINE_STATUS = (
    "LOCKED"
)

EXPECTED_CLICKBAIT_THRESHOLD = 0.61
EXPECTED_CONSISTENCY_THRESHOLD = 0.45

EXPECTED_DESCRIPTION_WEIGHT = 0.50
EXPECTED_MIL_WEIGHT = 0.50

EXPECTED_MAX_CHUNKS = 6

VALID_CLICKBAIT_LABELS = {
    "NORMAL",
    "CLICKBAIT",
}

VALID_CONSISTENCY_LABELS = {
    "INCONSISTENT",
    "CONSISTENT",
}


# ============================================================
# HELPERS
# ============================================================

def require_file(path):

    if not path.exists():

        raise FileNotFoundError(
            path
        )


def load_json(path):

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


def save_json(data, path):

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
        )


def sha256_file(path):

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        while True:

            block = f.read(
                1024 * 1024
            )

            if not block:

                break

            digest.update(
                block
            )

    return digest.hexdigest()


def resolve_module_path(
    relative_path,
):

    path = (
        MODULE_ROOT
        /
        relative_path
    )

    return path


def assert_probability(
    value,
    name,
):

    value = float(
        value
    )

    if not math.isfinite(
        value
    ):

        raise RuntimeError(
            f"{name} is not finite: "
            f"{value}"
        )

    if (
        value < 0.0
        or
        value > 1.0
    ):

        raise RuntimeError(
            f"{name} outside [0,1]: "
            f"{value}"
        )

    return value


# ============================================================
# VERIFY LOCK BEFORE IMPORTING PIPELINE
# ============================================================

def verify_locked_artifacts():

    print()
    print("=" * 80)
    print(
        "VERIFY LOCKED ARTIFACTS"
    )
    print("=" * 80)

    require_file(
        PIPELINE_PATH
    )

    require_file(
        LOCK_PATH
    )

    require_file(
        MANIFEST_PATH
    )

    lock = load_json(
        LOCK_PATH
    )

    manifest = load_json(
        MANIFEST_PATH
    )

    if (
        lock.get(
            "version"
        )
        !=
        EXPECTED_VERSION
    ):

        raise RuntimeError(
            "Unexpected unified module version."
        )

    if (
        lock.get(
            "status"
        )
        !=
        EXPECTED_PIPELINE_STATUS
    ):

        raise RuntimeError(
            "Unified pipeline is not LOCKED."
        )

    # --------------------------------------------------------
    # Pipeline hash
    # --------------------------------------------------------

    expected_pipeline_hash = (
        lock[
            "pipeline"
        ][
            "sha256"
        ]
    )

    actual_pipeline_hash = (
        sha256_file(
            PIPELINE_PATH
        )
    )

    if (
        actual_pipeline_hash
        !=
        expected_pipeline_hash
    ):

        raise RuntimeError(
            "PIPELINE SHA256 MISMATCH.\n"
            "The locked pipeline file has changed."
        )

    manifest_pipeline_hash = (
        manifest[
            "pipeline"
        ][
            "sha256"
        ]
    )

    if (
        manifest_pipeline_hash
        !=
        expected_pipeline_hash
    ):

        raise RuntimeError(
            "Manifest pipeline hash differs "
            "from unified lock."
        )

    print()
    print(
        "Pipeline SHA256: VERIFIED"
    )

    print(
        f"  {actual_pipeline_hash}"
    )

    # --------------------------------------------------------
    # Clickbait checkpoint
    # --------------------------------------------------------

    clickbait_info = (
        lock[
            "task_2a_clickbait"
        ]
    )

    clickbait_path = resolve_module_path(
        clickbait_info[
            "checkpoint"
        ]
    )

    require_file(
        clickbait_path
    )

    clickbait_hash = sha256_file(
        clickbait_path
    )

    if (
        clickbait_hash
        !=
        clickbait_info[
            "checkpoint_sha256"
        ]
    ):

        raise RuntimeError(
            "Clickbait checkpoint SHA256 mismatch."
        )

    print()
    print(
        "Clickbait checkpoint: VERIFIED"
    )

    # --------------------------------------------------------
    # Description checkpoint
    # --------------------------------------------------------

    description_info = (
        lock[
            "task_2b_consistency"
        ][
            "description_branch"
        ]
    )

    description_path = resolve_module_path(
        description_info[
            "checkpoint"
        ]
    )

    require_file(
        description_path
    )

    description_hash = sha256_file(
        description_path
    )

    if (
        description_hash
        !=
        description_info[
            "checkpoint_sha256"
        ]
    ):

        raise RuntimeError(
            "Description checkpoint SHA256 mismatch."
        )

    print(
        "Description checkpoint: VERIFIED"
    )

    # --------------------------------------------------------
    # MAX-MIL checkpoint
    # --------------------------------------------------------

    mil_info = (
        lock[
            "task_2b_consistency"
        ][
            "max_mil_branch"
        ]
    )

    mil_path = resolve_module_path(
        mil_info[
            "checkpoint"
        ]
    )

    require_file(
        mil_path
    )

    mil_hash = sha256_file(
        mil_path
    )

    if (
        mil_hash
        !=
        mil_info[
            "checkpoint_sha256"
        ]
    ):

        raise RuntimeError(
            "MAX-MIL checkpoint SHA256 mismatch."
        )

    print(
        "MAX-MIL checkpoint: VERIFIED"
    )

    # --------------------------------------------------------
    # Frozen configuration
    # --------------------------------------------------------

    if (
        abs(
            float(
                clickbait_info[
                    "threshold"
                ]
            )
            -
            EXPECTED_CLICKBAIT_THRESHOLD
        )
        >
        1e-12
    ):

        raise RuntimeError(
            "Clickbait threshold mismatch."
        )

    consistency_info = (
        lock[
            "task_2b_consistency"
        ]
    )

    fusion_info = (
        consistency_info[
            "fusion"
        ]
    )

    if (
        abs(
            float(
                fusion_info[
                    "threshold"
                ]
            )
            -
            EXPECTED_CONSISTENCY_THRESHOLD
        )
        >
        1e-12
    ):

        raise RuntimeError(
            "Consistency threshold mismatch."
        )

    if (
        abs(
            float(
                fusion_info[
                    "description_weight"
                ]
            )
            -
            EXPECTED_DESCRIPTION_WEIGHT
        )
        >
        1e-12
    ):

        raise RuntimeError(
            "Description weight mismatch."
        )

    if (
        abs(
            float(
                fusion_info[
                    "mil_weight"
                ]
            )
            -
            EXPECTED_MIL_WEIGHT
        )
        >
        1e-12
    ):

        raise RuntimeError(
            "MIL weight mismatch."
        )

    if (
        int(
            mil_info[
                "max_chunks"
            ]
        )
        !=
        EXPECTED_MAX_CHUNKS
    ):

        raise RuntimeError(
            "MAX_CHUNKS mismatch."
        )

    print()
    print(
        "Frozen configuration: VERIFIED"
    )

    return {
        "lock":
            lock,

        "pipeline_sha256":
            actual_pipeline_hash,

        "clickbait_checkpoint_sha256":
            clickbait_hash,

        "description_checkpoint_sha256":
            description_hash,

        "mil_checkpoint_sha256":
            mil_hash,
    }


# ============================================================
# FIXED SMOKE CASES
# ============================================================

def build_smoke_cases():

    short_case = {
        "case_id":
            "short_consistent_article",

        "title":
            (
                "Giá vàng hôm nay tăng mạnh, "
                "chuyên gia lý giải nguyên nhân"
            ),

        "lead":
            (
                "Giá vàng trong nước tiếp tục tăng "
                "khi thị trường thế giới có nhiều "
                "biến động."
            ),

        "content":
            (
                "Giá vàng trong nước sáng nay tiếp tục "
                "tăng tại nhiều doanh nghiệp. "
                "Trên thị trường thế giới, giá vàng "
                "cũng đi lên khi nhà đầu tư theo dõi "
                "các dữ liệu kinh tế mới. "
                "Một số chuyên gia cho rằng diễn biến "
                "này liên quan đến kỳ vọng chính sách "
                "tiền tệ và nhu cầu trú ẩn an toàn."
            ),

        "expected_chunk_policy":
            "short",
    }

    no_lead_case = {
        "case_id":
            "missing_optional_lead",

        "title":
            (
                "Thành phố triển khai thêm nhiều "
                "tuyến xe buýt mới"
            ),

        "lead":
            "",

        "content":
            (
                "Cơ quan quản lý giao thông cho biết "
                "một số tuyến xe buýt mới sẽ được đưa "
                "vào hoạt động trong thời gian tới. "
                "Các tuyến được bố trí nhằm tăng khả "
                "năng kết nối giữa khu dân cư, trường "
                "học và các đầu mối giao thông công cộng."
            ),

        "expected_chunk_policy":
            "short",
    }

    long_paragraph = (
        "Theo báo cáo của cơ quan chuyên môn, "
        "kế hoạch được triển khai theo nhiều giai đoạn. "
        "Các đơn vị sẽ tiếp tục thu thập dữ liệu, "
        "đánh giá kết quả và điều chỉnh phương án "
        "theo tình hình thực tế. "
        "Những thay đổi quan trọng sẽ được công bố "
        "sau khi quá trình đánh giá hoàn tất. "
    )

    long_case = {
        "case_id":
            "very_long_article_six_chunk_sampling",

        "title":
            (
                "Kế hoạch mới được triển khai "
                "theo nhiều giai đoạn"
            ),

        "lead":
            (
                "Các đơn vị liên quan sẽ theo dõi "
                "kết quả và điều chỉnh phương án "
                "trong quá trình triển khai."
            ),

        "content":
            " ".join(
                [
                    long_paragraph
                    for _ in range(
                        160
                    )
                ]
            ),

        "expected_chunk_policy":
            "six_chunks",
    }

    return [
        short_case,
        no_lead_case,
        long_case,
    ]


# ============================================================
# OUTPUT CONTRACT VALIDATION
# ============================================================

def validate_result(
    case,
    result,
):

    failures = []

    # --------------------------------------------------------
    # Top level
    # --------------------------------------------------------

    if (
        result.get(
            "module"
        )
        !=
        "Module 2 - Headline Integrity"
    ):

        failures.append(
            "Unexpected module name."
        )

    if (
        result.get(
            "overall_integrity_score",
            "MISSING",
        )
        is not None
    ):

        failures.append(
            "overall_integrity_score must remain None."
        )

    # --------------------------------------------------------
    # Clickbait
    # --------------------------------------------------------

    clickbait = result.get(
        "clickbait"
    )

    if not isinstance(
        clickbait,
        dict,
    ):

        failures.append(
            "Missing clickbait object."
        )

    else:

        try:

            assert_probability(
                clickbait[
                    "probability_clickbait"
                ],
                "probability_clickbait",
            )

        except Exception as exc:

            failures.append(
                str(
                    exc
                )
            )

        if (
            clickbait.get(
                "label"
            )
            not in
            VALID_CLICKBAIT_LABELS
        ):

            failures.append(
                "Invalid clickbait label."
            )

        if (
            clickbait.get(
                "prediction"
            )
            not in
            {
                0,
                1,
            }
        ):

            failures.append(
                "Invalid clickbait prediction."
            )

        if (
            abs(
                float(
                    clickbait.get(
                        "threshold",
                        -1,
                    )
                )
                -
                EXPECTED_CLICKBAIT_THRESHOLD
            )
            >
            1e-12
        ):

            failures.append(
                "Clickbait threshold changed."
            )

    # --------------------------------------------------------
    # Consistency
    # --------------------------------------------------------

    consistency = result.get(
        "headline_consistency"
    )

    if not isinstance(
        consistency,
        dict,
    ):

        failures.append(
            "Missing headline_consistency object."
        )

        return failures

    try:

        assert_probability(
            consistency[
                "probability_consistent"
            ],
            "probability_consistent",
        )

    except Exception as exc:

        failures.append(
            str(
                exc
            )
        )

    if (
        consistency.get(
            "label"
        )
        not in
        VALID_CONSISTENCY_LABELS
    ):

        failures.append(
            "Invalid consistency label."
        )

    if (
        consistency.get(
            "prediction"
        )
        not in
        {
            0,
            1,
        }
    ):

        failures.append(
            "Invalid consistency prediction."
        )

    if (
        abs(
            float(
                consistency.get(
                    "threshold",
                    -1,
                )
            )
            -
            EXPECTED_CONSISTENCY_THRESHOLD
        )
        >
        1e-12
    ):

        failures.append(
            "Consistency threshold changed."
        )

    branches = consistency.get(
        "branches",
        {},
    )

    description = branches.get(
        "description",
        {},
    )

    mil = branches.get(
        "max_mil",
        {},
    )

    try:

        assert_probability(
            description[
                "probability_consistent"
            ],
            "description probability",
        )

    except Exception as exc:

        failures.append(
            str(
                exc
            )
        )

    try:

        assert_probability(
            mil[
                "probability_consistent"
            ],
            "MIL probability",
        )

    except Exception as exc:

        failures.append(
            str(
                exc
            )
        )

    if (
        abs(
            float(
                description.get(
                    "weight",
                    -1,
                )
            )
            -
            EXPECTED_DESCRIPTION_WEIGHT
        )
        >
        1e-12
    ):

        failures.append(
            "Description fusion weight changed."
        )

    if (
        abs(
            float(
                mil.get(
                    "weight",
                    -1,
                )
            )
            -
            EXPECTED_MIL_WEIGHT
        )
        >
        1e-12
    ):

        failures.append(
            "MIL fusion weight changed."
        )

    chunk_count = int(
        mil.get(
            "chunk_count",
            0,
        )
    )

    if not (
        1
        <=
        chunk_count
        <=
        EXPECTED_MAX_CHUNKS
    ):

        failures.append(
            f"Invalid chunk count: "
            f"{chunk_count}"
        )

    chunk_starts = mil.get(
        "chunk_starts",
        []
    )

    if (
        len(
            chunk_starts
        )
        !=
        chunk_count
    ):

        failures.append(
            "chunk_starts length does not "
            "match chunk_count."
        )

    chunk_scores = mil.get(
        "chunk_scores",
        []
    )

    if (
        len(
            chunk_scores
        )
        !=
        chunk_count
    ):

        failures.append(
            "chunk_scores length does not "
            "match chunk_count."
        )

    max_chunk_index = int(
        mil.get(
            "max_chunk_index",
            -1,
        )
    )

    if not (
        0
        <=
        max_chunk_index
        <
        chunk_count
    ):

        failures.append(
            "Invalid max_chunk_index."
        )

    # --------------------------------------------------------
    # Specific smoke-case expectations
    #
    # These check pipeline mechanics only.
    # They do NOT assert semantic model labels.
    # --------------------------------------------------------

    policy = case[
        "expected_chunk_policy"
    ]

    if (
        policy
        ==
        "short"
        and
        chunk_count
        !=
        1
    ):

        failures.append(
            "Short article should produce exactly 1 chunk."
        )

    if (
        policy
        ==
        "six_chunks"
        and
        chunk_count
        !=
        EXPECTED_MAX_CHUNKS
    ):

        failures.append(
            "Long article should exercise the "
            "6-chunk sampling path."
        )

    # --------------------------------------------------------
    # Fusion arithmetic
    # --------------------------------------------------------

    try:

        expected_fusion = (
            EXPECTED_DESCRIPTION_WEIGHT
            *
            float(
                description[
                    "probability_consistent"
                ]
            )
            +
            EXPECTED_MIL_WEIGHT
            *
            float(
                mil[
                    "probability_consistent"
                ]
            )
        )

        actual_fusion = float(
            consistency[
                "probability_consistent"
            ]
        )

        if (
            abs(
                expected_fusion
                -
                actual_fusion
            )
            >
            1e-9
        ):

            failures.append(
                "Fusion arithmetic mismatch."
            )

    except Exception as exc:

        failures.append(
            f"Fusion validation error: "
            f"{exc}"
        )

    return failures


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test locked Unified Module 2 v1"
        )
    )

    parser.add_argument(
        "--device",
        choices=[
            "cpu",
            "cuda",
        ],
        default=None,
    )

    args = parser.parse_args()

    print("=" * 80)
    print(
        "MODULE 2 - HEADLINE INTEGRITY"
    )
    print(
        "INTEGRATION STEP 09"
    )
    print(
        "SMOKE TEST LOCKED UNIFIED MODULE 2 V1"
    )
    print("=" * 80)

    print()
    print(
        "NO TRAINING."
    )

    print(
        "NO DATASET TEST EVALUATION."
    )

    print(
        "NO THRESHOLD SEARCH."
    )

    print(
        "NO MODEL SELECTION."
    )

    print(
        "FIXED SYNTHETIC SMOKE CASES ONLY."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # VERIFY LOCK BEFORE IMPORT
    # ========================================================

    verification = (
        verify_locked_artifacts()
    )

    # ========================================================
    # IMPORT ONLY AFTER HASH CHECK
    # ========================================================

    if str(
        PROJECT_ROOT
    ) not in sys.path:

        sys.path.insert(
            0,
            str(
                PROJECT_ROOT
            ),
        )

    from module2.inference.headline_integrity_pipeline import (
        HeadlineIntegrityPipeline,
    )

    # ========================================================
    # LOAD PIPELINE
    # ========================================================

    print()
    print("=" * 80)
    print(
        "LOAD LOCKED PIPELINE"
    )
    print("=" * 80)

    pipeline = (
        HeadlineIntegrityPipeline(
            device=args.device
        )
    )

    # ========================================================
    # RUN FIXED CASES
    # ========================================================

    cases = build_smoke_cases()

    outputs = []

    total_start = time.time()

    print()
    print("=" * 80)
    print(
        "RUN SMOKE CASES"
    )
    print("=" * 80)

    for index, case in enumerate(
        cases,
        start=1,
    ):

        print()
        print(
            f"[{index}/{len(cases)}] "
            f"{case['case_id']}"
        )

        result = pipeline.predict(
            title=(
                case[
                    "title"
                ]
            ),
            lead=(
                case[
                    "lead"
                ]
            ),
            content=(
                case[
                    "content"
                ]
            ),
        )

        failures = validate_result(
            case,
            result,
        )

        passed = (
            len(
                failures
            )
            ==
            0
        )

        clickbait = (
            result[
                "clickbait"
            ]
        )

        consistency = (
            result[
                "headline_consistency"
            ]
        )

        mil = (
            consistency[
                "branches"
            ][
                "max_mil"
            ]
        )

        print(
            f"  clickbait   : "
            f"{clickbait['label']} "
            f"(p={clickbait['probability_clickbait']:.6f})"
        )

        print(
            f"  consistency : "
            f"{consistency['label']} "
            f"(p={consistency['probability_consistent']:.6f})"
        )

        print(
            f"  article tokens : "
            f"{mil['article_token_count']:,}"
        )

        print(
            f"  chunks         : "
            f"{mil['chunk_count']}"
        )

        print(
            f"  chunk starts   : "
            f"{mil['chunk_starts']}"
        )

        print(
            f"  elapsed        : "
            f"{result['elapsed_seconds']:.4f}s"
        )

        print(
            f"  RESULT         : "
            f"{'PASS' if passed else 'FAIL'}"
        )

        if failures:

            for failure in failures:

                print(
                    f"    - {failure}"
                )

        outputs.append(
            {
                "case_id":
                    case[
                        "case_id"
                    ],

                "expected_chunk_policy":
                    case[
                        "expected_chunk_policy"
                    ],

                "passed":
                    passed,

                "failures":
                    failures,

                "result":
                    result,
            }
        )

    total_elapsed = (
        time.time()
        -
        total_start
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    passed_cases = sum(
        1
        for item in outputs
        if item[
            "passed"
        ]
    )

    failed_cases = (
        len(
            outputs
        )
        -
        passed_cases
    )

    overall_pass = (
        failed_cases
        ==
        0
    )

    report = {
        "stage":
            "integration_step_09_smoke_test",

        "module":
            "Module 2 - Headline Integrity",

        "version":
            EXPECTED_VERSION,

        "fixed_synthetic_cases":
            True,

        "training_performed":
            False,

        "dataset_test_evaluation_performed":
            False,

        "threshold_search_performed":
            False,

        "model_selection_performed":
            False,

        "locked_artifact_verification": {
            "pipeline_sha256":
                verification[
                    "pipeline_sha256"
                ],

            "clickbait_checkpoint_sha256":
                verification[
                    "clickbait_checkpoint_sha256"
                ],

            "description_checkpoint_sha256":
                verification[
                    "description_checkpoint_sha256"
                ],

            "mil_checkpoint_sha256":
                verification[
                    "mil_checkpoint_sha256"
                ],
        },

        "cases":
            [
                {
                    "case_id":
                        item[
                            "case_id"
                        ],

                    "expected_chunk_policy":
                        item[
                            "expected_chunk_policy"
                        ],

                    "passed":
                        item[
                            "passed"
                        ],

                    "failures":
                        item[
                            "failures"
                        ],
                }

                for item in outputs
            ],

        "summary": {
            "total_cases":
                int(
                    len(
                        outputs
                    )
                ),

            "passed_cases":
                int(
                    passed_cases
                ),

            "failed_cases":
                int(
                    failed_cases
                ),

            "overall_pass":
                bool(
                    overall_pass
                ),

            "elapsed_seconds":
                float(
                    total_elapsed
                ),
        },
    }

    save_json(
        outputs,
        OUTPUTS_PATH,
    )

    save_json(
        report,
        REPORT_PATH,
    )

    print()
    print("=" * 80)
    print(
        "SMOKE TEST SUMMARY"
    )
    print("=" * 80)

    print()
    print(
        f"Total cases : "
        f"{len(outputs)}"
    )

    print(
        f"Passed      : "
        f"{passed_cases}"
    )

    print(
        f"Failed      : "
        f"{failed_cases}"
    )

    print()

    print(
        f"LOCKED PIPELINE SMOKE TEST: "
        f"{'PASS' if overall_pass else 'FAIL'}"
    )

    print()
    print(
        f"Report:\n"
        f"  {REPORT_PATH}"
    )

    print(
        f"Outputs:\n"
        f"  {OUTPUTS_PATH}"
    )

    if not overall_pass:

        raise RuntimeError(
            "Locked pipeline smoke test FAILED."
        )

    print()
    print(
        "UNIFIED MODULE 2 V1 IS READY."
    )

    print()
    print(
        "INTEGRATION STEP 09 FINISHED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()