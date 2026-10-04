from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone


# ============================================================
# PATHS
# ============================================================

MODULE_ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------
# UNIFIED PIPELINE
# ------------------------------------------------------------

PIPELINE_PATH = (
    MODULE_ROOT
    / "inference"
    / "headline_integrity_pipeline.py"
)


# ------------------------------------------------------------
# REGRESSION EVIDENCE
# ------------------------------------------------------------

REGRESSION_REPORT_PATH = (
    MODULE_ROOT
    / "results"
    / "integration"
    / "regression_v1"
    / "regression_report.json"
)


# ------------------------------------------------------------
# TASK 2A
# ------------------------------------------------------------

CLICKBAIT_LOCK_PATH = (
    MODULE_ROOT
    / "results"
    / "clickbait"
    / "final_title_only_v1"
    / "final_model_lock.json"
)


# ------------------------------------------------------------
# TASK 2B
# ------------------------------------------------------------

CONSISTENCY_LOCK_PATH = (
    MODULE_ROOT
    / "results"
    / "consistency"
    / "final_hybrid_v1"
    / "final_model_lock.json"
)


# ------------------------------------------------------------
# CONSISTENCY CHECKPOINTS
# ------------------------------------------------------------

DESCRIPTION_CHECKPOINT_PATH = (
    MODULE_ROOT
    / "models"
    / "consistency"
    / "baseline_title_description_mbert"
    / "best_model.pt"
)

MIL_CHECKPOINT_PATH = (
    MODULE_ROOT
    / "models"
    / "consistency"
    / "full_article_max_mil_mbert"
    / "best_model.pt"
)


# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

OUTPUT_DIR = (
    MODULE_ROOT
    / "results"
    / "integration"
    / "unified_module2_v1"
)

LOCK_PATH = (
    OUTPUT_DIR
    / "unified_module2_lock.json"
)

MANIFEST_PATH = (
    OUTPUT_DIR
    / "artifact_manifest.json"
)

CARD_PATH = (
    OUTPUT_DIR
    / "UNIFIED_MODULE2_CARD.md"
)


# ============================================================
# EXPECTED FROZEN CONFIG
# ============================================================

MODEL_NAME = (
    "bert-base-multilingual-cased"
)

CLICKBAIT_INPUT = (
    "title_only"
)

CLICKBAIT_MAX_LENGTH = 160
CLICKBAIT_THRESHOLD = 0.61

CONSISTENCY_MAX_LENGTH = 160

DESCRIPTION_WEIGHT = 0.50
MIL_WEIGHT = 0.50
CONSISTENCY_THRESHOLD = 0.45

CHUNK_TOKENS = 120
CHUNK_OVERLAP = 24
CHUNK_STEP = 96
MAX_CHUNKS = 6

CHUNK_SAMPLING = (
    "np.round(np.linspace(0, len(starts)-1, MAX_CHUNKS))"
)

PROBABILITY_TOLERANCE = 0.005


# ============================================================
# HELPERS
# ============================================================

def require_file(
    path,
):

    if not path.exists():

        raise FileNotFoundError(
            path
        )


def load_json(
    path,
):

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(
            f
        )


def save_json(
    data,
    path,
):

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


def sha256_file(
    path,
):

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


def relative(
    path,
):

    return str(
        path.relative_to(
            MODULE_ROOT
        )
    )


def deep_find(
    obj,
    key,
):

    if isinstance(
        obj,
        dict,
    ):

        if key in obj:

            return obj[
                key
            ]

        for value in obj.values():

            result = deep_find(
                value,
                key,
            )

            if result is not None:

                return result

    elif isinstance(
        obj,
        list,
    ):

        for value in obj:

            result = deep_find(
                value,
                key,
            )

            if result is not None:

                return result

    return None


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "MODULE 2 - HEADLINE INTEGRITY"
    )
    print(
        "INTEGRATION STEP 08"
    )
    print(
        "LOCK UNIFIED MODULE 2 PIPELINE V1"
    )
    print("=" * 80)

    print()
    print(
        "NO TRAINING."
    )

    print(
        "NO INFERENCE."
    )

    print(
        "NO MODEL SELECTION."
    )

    print(
        "NO THRESHOLD SEARCH."
    )

    print()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # REQUIRED FILES
    # ========================================================

    required_files = [
        PIPELINE_PATH,
        REGRESSION_REPORT_PATH,
        CLICKBAIT_LOCK_PATH,
        CONSISTENCY_LOCK_PATH,
        DESCRIPTION_CHECKPOINT_PATH,
        MIL_CHECKPOINT_PATH,
    ]

    for path in required_files:

        require_file(
            path
        )

    # ========================================================
    # LOAD REGRESSION
    # ========================================================

    print("=" * 80)
    print(
        "VERIFY REGRESSION"
    )
    print("=" * 80)

    regression = load_json(
        REGRESSION_REPORT_PATH
    )

    if (
        regression.get(
            "overall_pass"
        )
        is not True
    ):

        raise RuntimeError(
            "Unified regression did not PASS."
        )

    if (
        regression.get(
            "validation_only"
        )
        is not True
    ):

        raise RuntimeError(
            "Regression is not marked validation_only."
        )

    if (
        regression.get(
            "training_performed"
        )
        is not False
    ):

        raise RuntimeError(
            "Regression report indicates training."
        )

    if (
        regression.get(
            "threshold_search_performed"
        )
        is not False
    ):

        raise RuntimeError(
            "Regression report indicates threshold search."
        )

    if (
        regression.get(
            "model_selection_performed"
        )
        is not False
    ):

        raise RuntimeError(
            "Regression report indicates model selection."
        )

    tolerance = float(
        regression[
            "probability_tolerance"
        ]
    )

    if (
        abs(
            tolerance
            -
            PROBABILITY_TOLERANCE
        )
        >
        1e-12
    ):

        raise RuntimeError(
            "Unexpected regression tolerance."
        )

    clickbait_regression = (
        regression[
            "clickbait"
        ]
    )

    consistency_regression = (
        regression[
            "consistency"
        ]
    )

    if (
        clickbait_regression.get(
            "passed"
        )
        is not True
    ):

        raise RuntimeError(
            "Clickbait regression failed."
        )

    if (
        consistency_regression.get(
            "passed"
        )
        is not True
    ):

        raise RuntimeError(
            "Consistency regression failed."
        )

    if (
        consistency_regression[
            "description"
        ].get(
            "passed"
        )
        is not True
    ):

        raise RuntimeError(
            "Description regression failed."
        )

    if (
        consistency_regression[
            "max_mil"
        ].get(
            "passed"
        )
        is not True
    ):

        raise RuntimeError(
            "MAX-MIL regression failed."
        )

    if (
        consistency_regression[
            "fusion"
        ].get(
            "passed"
        )
        is not True
    ):

        raise RuntimeError(
            "Fusion regression failed."
        )

    clickbait_max_error = float(
        clickbait_regression[
            "probability_error"
        ][
            "max_abs_error"
        ]
    )

    description_max_error = float(
        consistency_regression[
            "description"
        ][
            "probability_error"
        ][
            "max_abs_error"
        ]
    )

    mil_max_error = float(
        consistency_regression[
            "max_mil"
        ][
            "probability_error"
        ][
            "max_abs_error"
        ]
    )

    fusion_max_error = float(
        consistency_regression[
            "fusion"
        ][
            "probability_error"
        ][
            "max_abs_error"
        ]
    )

    clickbait_prediction_mismatches = int(
        clickbait_regression[
            "prediction_mismatches"
        ]
    )

    fusion_prediction_mismatches = int(
        consistency_regression[
            "fusion"
        ][
            "prediction_mismatches"
        ]
    )

    token_count_mismatches = int(
        consistency_regression[
            "max_mil"
        ][
            "token_count_mismatches"
        ]
    )

    chunk_count_mismatches = int(
        consistency_regression[
            "max_mil"
        ][
            "chunk_count_mismatches"
        ]
    )

    if clickbait_max_error > tolerance:

        raise RuntimeError(
            "Clickbait max error exceeds tolerance."
        )

    if description_max_error > tolerance:

        raise RuntimeError(
            "Description max error exceeds tolerance."
        )

    if mil_max_error > tolerance:

        raise RuntimeError(
            "MIL max error exceeds tolerance."
        )

    if fusion_max_error > tolerance:

        raise RuntimeError(
            "Fusion max error exceeds tolerance."
        )

    if clickbait_prediction_mismatches != 0:

        raise RuntimeError(
            "Clickbait prediction mismatch exists."
        )

    if fusion_prediction_mismatches != 0:

        raise RuntimeError(
            "Fusion prediction mismatch exists."
        )

    if token_count_mismatches != 0:

        raise RuntimeError(
            "MIL token-count mismatch exists."
        )

    if chunk_count_mismatches != 0:

        raise RuntimeError(
            "MIL chunk-count mismatch exists."
        )

    print()
    print(
        "Regression: VERIFIED"
    )

    print(
        f"  Clickbait max error   : "
        f"{clickbait_max_error:.9f}"
    )

    print(
        f"  Description max error : "
        f"{description_max_error:.9f}"
    )

    print(
        f"  MAX-MIL max error     : "
        f"{mil_max_error:.9f}"
    )

    print(
        f"  Fusion max error      : "
        f"{fusion_max_error:.9f}"
    )

    print(
        "  Prediction mismatches : 0"
    )

    print(
        "  Token mismatches      : 0"
    )

    print(
        "  Chunk mismatches      : 0"
    )

    # ========================================================
    # CLICKBAIT LOCK
    # ========================================================

    print()
    print("=" * 80)
    print(
        "VERIFY TASK 2A LOCK"
    )
    print("=" * 80)

    clickbait_lock = load_json(
        CLICKBAIT_LOCK_PATH
    )

    if (
        clickbait_lock.get(
            "status"
        )
        !=
        "LOCKED_BEFORE_TEST"
    ):

        raise RuntimeError(
            "Clickbait model was not locked before TEST."
        )

    clickbait_model = (
        clickbait_lock[
            "final_model"
        ]
    )

    if (
        clickbait_model[
            "architecture"
        ]
        !=
        MODEL_NAME
    ):

        raise RuntimeError(
            "Clickbait architecture mismatch."
        )

    if (
        clickbait_model[
            "input"
        ]
        !=
        CLICKBAIT_INPUT
    ):

        raise RuntimeError(
            "Clickbait input mismatch."
        )

    if (
        int(
            clickbait_model[
                "max_length"
            ]
        )
        !=
        CLICKBAIT_MAX_LENGTH
    ):

        raise RuntimeError(
            "Clickbait max length mismatch."
        )

    if (
        abs(
            float(
                clickbait_model[
                    "decision_threshold"
                ]
            )
            -
            CLICKBAIT_THRESHOLD
        )
        >
        1e-12
    ):

        raise RuntimeError(
            "Clickbait threshold mismatch."
        )

    clickbait_checkpoint_path = (
        MODULE_ROOT
        /
        clickbait_model[
            "checkpoint"
        ]
    )

    require_file(
        clickbait_checkpoint_path
    )

    clickbait_hash = sha256_file(
        clickbait_checkpoint_path
    )

    expected_clickbait_hash = (
        clickbait_model[
            "checkpoint_sha256"
        ]
    )

    if (
        clickbait_hash
        !=
        expected_clickbait_hash
    ):

        raise RuntimeError(
            "Clickbait checkpoint SHA256 mismatch."
        )

    print()
    print(
        "Task 2A lock: VERIFIED"
    )

    print(
        f"  checkpoint SHA256: "
        f"{clickbait_hash}"
    )

    # ========================================================
    # CONSISTENCY LOCK
    # ========================================================

    print()
    print("=" * 80)
    print(
        "VERIFY TASK 2B LOCK"
    )
    print("=" * 80)

    consistency_lock = load_json(
        CONSISTENCY_LOCK_PATH
    )

    description_weight = deep_find(
        consistency_lock,
        "description_weight",
    )

    mil_weight = deep_find(
        consistency_lock,
        "mil_weight",
    )

    consistency_threshold = deep_find(
        consistency_lock,
        "decision_threshold",
    )

    if description_weight is None:

        raise RuntimeError(
            "Cannot find description_weight "
            "in consistency lock."
        )

    if mil_weight is None:

        raise RuntimeError(
            "Cannot find mil_weight "
            "in consistency lock."
        )

    if consistency_threshold is None:

        raise RuntimeError(
            "Cannot find decision_threshold "
            "in consistency lock."
        )

    if (
        abs(
            float(
                description_weight
            )
            -
            DESCRIPTION_WEIGHT
        )
        >
        1e-12
    ):

        raise RuntimeError(
            "Description fusion weight mismatch."
        )

    if (
        abs(
            float(
                mil_weight
            )
            -
            MIL_WEIGHT
        )
        >
        1e-12
    ):

        raise RuntimeError(
            "MIL fusion weight mismatch."
        )

    if (
        abs(
            float(
                consistency_threshold
            )
            -
            CONSISTENCY_THRESHOLD
        )
        >
        1e-12
    ):

        raise RuntimeError(
            "Consistency threshold mismatch."
        )

    description_hash = sha256_file(
        DESCRIPTION_CHECKPOINT_PATH
    )

    mil_hash = sha256_file(
        MIL_CHECKPOINT_PATH
    )

    print()
    print(
        "Task 2B lock: VERIFIED"
    )

    print(
        f"  description checkpoint SHA256: "
        f"{description_hash}"
    )

    print(
        f"  MAX-MIL checkpoint SHA256    : "
        f"{mil_hash}"
    )

    # ========================================================
    # PIPELINE HASH
    # ========================================================

    print()
    print("=" * 80)
    print(
        "HASH UNIFIED PIPELINE"
    )
    print("=" * 80)

    pipeline_hash = sha256_file(
        PIPELINE_PATH
    )

    regression_mtime = (
        REGRESSION_REPORT_PATH
        .stat()
        .st_mtime
    )

    pipeline_mtime = (
        PIPELINE_PATH
        .stat()
        .st_mtime
    )

    if regression_mtime < pipeline_mtime:

        raise RuntimeError(
            "Regression report is older than the "
            "current pipeline file.\n"
            "Run 04_regression_check_unified_pipeline.py "
            "again before locking."
        )

    print()
    print(
        f"Pipeline SHA256:\n"
        f"  {pipeline_hash}"
    )

    print()
    print(
        "Regression timestamp check: VERIFIED"
    )

    # ========================================================
    # ARTIFACT MANIFEST
    # ========================================================

    manifest = {
        "version":
            "unified_module2_v1",

        "pipeline": {
            "path":
                relative(
                    PIPELINE_PATH
                ),

            "sha256":
                pipeline_hash,
        },

        "models": {
            "clickbait": {
                "path":
                    relative(
                        clickbait_checkpoint_path
                    ),

                "sha256":
                    clickbait_hash,
            },

            "description_consistency": {
                "path":
                    relative(
                        DESCRIPTION_CHECKPOINT_PATH
                    ),

                "sha256":
                    description_hash,
            },

            "max_mil_consistency": {
                "path":
                    relative(
                        MIL_CHECKPOINT_PATH
                    ),

                "sha256":
                    mil_hash,
            },
        },

        "source_locks": {
            "clickbait":
                relative(
                    CLICKBAIT_LOCK_PATH
                ),

            "consistency":
                relative(
                    CONSISTENCY_LOCK_PATH
                ),
        },

        "regression_evidence":
            relative(
                REGRESSION_REPORT_PATH
            ),
    }

    save_json(
        manifest,
        MANIFEST_PATH,
    )

    # ========================================================
    # FINAL LOCK
    # ========================================================

    lock = {
        "module":
            "Module 2 - Headline Integrity",

        "version":
            "unified_module2_v1",

        "status":
            "LOCKED",

        "locked_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "pipeline": {
            "file":
                relative(
                    PIPELINE_PATH
                ),

            "sha256":
                pipeline_hash,
        },

        "task_2a_clickbait": {
            "architecture":
                MODEL_NAME,

            "input":
                CLICKBAIT_INPUT,

            "max_length":
                CLICKBAIT_MAX_LENGTH,

            "threshold":
                CLICKBAIT_THRESHOLD,

            "checkpoint":
                relative(
                    clickbait_checkpoint_path
                ),

            "checkpoint_sha256":
                clickbait_hash,
        },

        "task_2b_consistency": {
            "architecture":
                MODEL_NAME,

            "description_branch": {
                "input":
                    "headline + description",

                "weight":
                    DESCRIPTION_WEIGHT,

                "max_length":
                    CONSISTENCY_MAX_LENGTH,

                "checkpoint":
                    relative(
                        DESCRIPTION_CHECKPOINT_PATH
                    ),

                "checkpoint_sha256":
                    description_hash,
            },

            "max_mil_branch": {
                "input":
                    "headline + full article chunks",

                "weight":
                    MIL_WEIGHT,

                "max_length":
                    CONSISTENCY_MAX_LENGTH,

                "chunk_tokens":
                    CHUNK_TOKENS,

                "chunk_overlap":
                    CHUNK_OVERLAP,

                "chunk_step":
                    CHUNK_STEP,

                "max_chunks":
                    MAX_CHUNKS,

                "sampling":
                    CHUNK_SAMPLING,

                "pooling":
                    "max(logit_consistent - logit_inconsistent)",

                "checkpoint":
                    relative(
                        MIL_CHECKPOINT_PATH
                    ),

                "checkpoint_sha256":
                    mil_hash,
            },

            "fusion": {
                "description_weight":
                    DESCRIPTION_WEIGHT,

                "mil_weight":
                    MIL_WEIGHT,

                "threshold":
                    CONSISTENCY_THRESHOLD,
            },
        },

        "regression": {
            "status":
                "PASS",

            "validation_only":
                True,

            "probability_tolerance":
                tolerance,

            "clickbait": {
                "samples":
                    int(
                        clickbait_regression[
                            "samples"
                        ]
                    ),

                "max_abs_error":
                    clickbait_max_error,

                "prediction_mismatches":
                    clickbait_prediction_mismatches,
            },

            "description": {
                "samples":
                    int(
                        consistency_regression[
                            "samples"
                        ]
                    ),

                "max_abs_error":
                    description_max_error,
            },

            "max_mil": {
                "samples":
                    int(
                        consistency_regression[
                            "samples"
                        ]
                    ),

                "max_abs_error":
                    mil_max_error,

                "token_count_mismatches":
                    token_count_mismatches,

                "chunk_count_mismatches":
                    chunk_count_mismatches,
            },

            "fusion": {
                "samples":
                    int(
                        consistency_regression[
                            "samples"
                        ]
                    ),

                "max_abs_error":
                    fusion_max_error,

                "prediction_mismatches":
                    fusion_prediction_mismatches,
            },

            "report":
                relative(
                    REGRESSION_REPORT_PATH
                ),
        },

        "output_contract": {
            "clickbait_labels": [
                "NORMAL",
                "CLICKBAIT",
            ],

            "consistency_labels": [
                "INCONSISTENT",
                "CONSISTENT",
            ],

            "overall_integrity_score":
                None,

            "note":
                (
                    "Task 2A and Task 2B remain separate "
                    "predictions. No empirically validated "
                    "single overall integrity score exists."
                ),
        },

        "integrity": {
            "task_2a_final_model_frozen":
                True,

            "task_2b_final_model_frozen":
                True,

            "unified_regression_passed":
                True,

            "test_based_tuning":
                False,

            "pipeline_locked_after_regression":
                True,
        },
    }

    save_json(
        lock,
        LOCK_PATH,
    )

    # ========================================================
    # EXPERIMENT CARD
    # ========================================================

    card_lines = [
        "# Unified Module 2 — Headline Integrity",
        "",
        "## Status",
        "",
        "**LOCKED — unified_module2_v1**",
        "",
        "The unified inference pipeline passed regression checks "
        "against saved validation artifacts before being locked.",
        "",
        "---",
        "",
        "## Task 2A — Clickbait",
        "",
        f"- Architecture: `{MODEL_NAME}`",
        f"- Input: `{CLICKBAIT_INPUT}`",
        f"- Max length: `{CLICKBAIT_MAX_LENGTH}`",
        f"- Decision threshold: `{CLICKBAIT_THRESHOLD:.2f}`",
        (
            f"- Regression max absolute error: "
            f"`{clickbait_max_error:.9f}`"
        ),
        (
            f"- Prediction mismatches: "
            f"`{clickbait_prediction_mismatches}`"
        ),
        "",
        "---",
        "",
        "## Task 2B — Headline/Article Consistency",
        "",
        "Description branch:",
        "",
        f"- Weight: `{DESCRIPTION_WEIGHT:.2f}`",
        f"- Max length: `{CONSISTENCY_MAX_LENGTH}`",
        (
            f"- Regression max absolute error: "
            f"`{description_max_error:.9f}`"
        ),
        "",
        "MAX-MIL branch:",
        "",
        f"- Weight: `{MIL_WEIGHT:.2f}`",
        f"- Chunk tokens: `{CHUNK_TOKENS}`",
        f"- Chunk overlap: `{CHUNK_OVERLAP}`",
        f"- Chunk step: `{CHUNK_STEP}`",
        f"- Maximum chunks: `{MAX_CHUNKS}`",
        (
            "- Sampling: "
            "`np.round(np.linspace(0, len(starts)-1, 6))`"
        ),
        (
            f"- Regression max absolute error: "
            f"`{mil_max_error:.9f}`"
        ),
        (
            f"- Token-count mismatches: "
            f"`{token_count_mismatches}`"
        ),
        (
            f"- Chunk-count mismatches: "
            f"`{chunk_count_mismatches}`"
        ),
        "",
        "Fusion:",
        "",
        (
            f"- Description weight: "
            f"`{DESCRIPTION_WEIGHT:.2f}`"
        ),
        (
            f"- MAX-MIL weight: "
            f"`{MIL_WEIGHT:.2f}`"
        ),
        (
            f"- Decision threshold: "
            f"`{CONSISTENCY_THRESHOLD:.2f}`"
        ),
        (
            f"- Regression max absolute error: "
            f"`{fusion_max_error:.9f}`"
        ),
        (
            f"- Prediction mismatches: "
            f"`{fusion_prediction_mismatches}`"
        ),
        "",
        "---",
        "",
        "## Unified regression",
        "",
        "| Branch | Result | Max abs error |",
        "|---|---:|---:|",
        (
            f"| Clickbait | PASS | "
            f"{clickbait_max_error:.9f} |"
        ),
        (
            f"| Description | PASS | "
            f"{description_max_error:.9f} |"
        ),
        (
            f"| MAX-MIL | PASS | "
            f"{mil_max_error:.9f} |"
        ),
        (
            f"| Fusion | PASS | "
            f"{fusion_max_error:.9f} |"
        ),
        "",
        f"Tolerance: `{tolerance:.6f}`",
        "",
        "**Unified pipeline regression: PASS**",
        "",
        "---",
        "",
        "## Pipeline SHA256",
        "",
        f"`{pipeline_hash}`",
        "",
        "## Important design decision",
        "",
        (
            "The module does not produce a single overall "
            "integrity score. Clickbait detection and "
            "headline/article consistency remain separate "
            "model outputs."
        ),
        "",
    ]

    with CARD_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "\n".join(
                card_lines
            )
        )

    # ========================================================
    # FINAL PRINT
    # ========================================================

    print()
    print("=" * 80)
    print(
        "UNIFIED MODULE 2 LOCK"
    )
    print("=" * 80)

    print()
    print(
        "Version:"
    )

    print(
        "  unified_module2_v1"
    )

    print()
    print(
        "Regression:"
    )

    print(
        "  Clickbait   : PASS"
    )

    print(
        "  Description : PASS"
    )

    print(
        "  MAX-MIL     : PASS"
    )

    print(
        "  Fusion      : PASS"
    )

    print()
    print(
        "Pipeline:"
    )

    print(
        f"  {relative(PIPELINE_PATH)}"
    )

    print(
        f"  SHA256: "
        f"{pipeline_hash}"
    )

    print()
    print(
        "Outputs:"
    )

    print(
        f"  Lock:\n"
        f"    {LOCK_PATH}"
    )

    print(
        f"  Manifest:\n"
        f"    {MANIFEST_PATH}"
    )

    print(
        f"  Card:\n"
        f"    {CARD_PATH}"
    )

    print()
    print(
        "UNIFIED MODULE 2 STATUS: LOCKED"
    )

    print()
    print(
        "INTEGRATION STEP 08 FINISHED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()