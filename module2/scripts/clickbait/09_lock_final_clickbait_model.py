from pathlib import Path
import hashlib
import json


# ============================================================
# PATHS
# ============================================================

MODULE_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = (
    MODULE_ROOT
    / "models"
    / "clickbait"
    / "mbert_title_only_v1"
)

RESULT_DIR = (
    MODULE_ROOT
    / "results"
    / "clickbait"
    / "mbert_title_only_v1"
)

COMPARE_DIR = (
    MODULE_ROOT
    / "results"
    / "clickbait"
    / "compare_title_vs_title_lead_v1"
)

FINAL_DIR = (
    MODULE_ROOT
    / "results"
    / "clickbait"
    / "final_title_only_v1"
)

CHECKPOINT_PATH = (
    MODEL_DIR
    / "best_model.pt"
)

METRICS_PATH = (
    RESULT_DIR
    / "validation_metrics.json"
)

CONFIG_PATH = (
    RESULT_DIR
    / "run_config.json"
)

COMPARISON_PATH = (
    COMPARE_DIR
    / "comparison_report.json"
)

LOCK_PATH = (
    FINAL_DIR
    / "final_model_lock.json"
)


# ============================================================
# EXPECTED FROZEN CONFIG
# ============================================================

EXPECTED_MODEL = (
    "bert-base-multilingual-cased"
)

EXPECTED_INPUT = (
    "title_only"
)

EXPECTED_MAX_LENGTH = 160

EXPECTED_THRESHOLD = 0.61

EXPECTED_BEST_EPOCH = 4


# ============================================================
# HELPERS
# ============================================================

def load_json(path):

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


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


def sha256_file(path):

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as f:

        while True:

            chunk = f.read(
                1024 * 1024
            )

            if not chunk:

                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def relative(path):

    return str(
        path.relative_to(
            MODULE_ROOT
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "MODULE 2 - HEADLINE INTEGRITY"
    )
    print(
        "TASK 2A - CLICKBAIT DETECTION"
    )
    print(
        "STEP 09 - FREEZE FINAL MODEL"
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
        "TEST IS NOT LOADED."
    )

    print(
        "THIS STEP FREEZES THE FINAL "
        "CONFIGURATION BEFORE TEST."
    )

    FINAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # REQUIRED ARTIFACTS
    # ========================================================

    required = [
        CHECKPOINT_PATH,
        METRICS_PATH,
        CONFIG_PATH,
        COMPARISON_PATH,
    ]

    for path in required:

        if not path.exists():

            raise FileNotFoundError(
                path
            )

    # ========================================================
    # LOAD STEP 06
    # ========================================================

    metrics = load_json(
        METRICS_PATH
    )

    config = load_json(
        CONFIG_PATH
    )

    comparison = load_json(
        COMPARISON_PATH
    )

    # ========================================================
    # VALIDATE STEP 06 CONFIG
    # ========================================================

    if (
        metrics.get(
            "test_used"
        )
        is not False
    ):

        raise RuntimeError(
            "Step 06 does not confirm test_used=False."
        )

    if (
        config.get(
            "test_loaded"
        )
        is not False
    ):

        raise RuntimeError(
            "Step 06 config does not confirm "
            "test_loaded=False."
        )

    if (
        config.get(
            "model"
        )
        !=
        EXPECTED_MODEL
    ):

        raise RuntimeError(
            "Unexpected model architecture."
        )

    if (
        config.get(
            "input"
        )
        !=
        EXPECTED_INPUT
    ):

        raise RuntimeError(
            "Unexpected model input."
        )

    if (
        int(
            config.get(
                "max_length"
            )
        )
        !=
        EXPECTED_MAX_LENGTH
    ):

        raise RuntimeError(
            "Unexpected max_length."
        )

    # ========================================================
    # THRESHOLD
    # ========================================================

    threshold = float(
        metrics[
            "best_validation_threshold"
        ]
    )

    if (
        abs(
            threshold
            -
            EXPECTED_THRESHOLD
        )
        >
        1e-12
    ):

        raise RuntimeError(
            f"Unexpected threshold: "
            f"{threshold}"
        )

    best_epoch = int(
        metrics[
            "best_epoch"
        ]
    )

    if (
        best_epoch
        !=
        EXPECTED_BEST_EPOCH
    ):

        raise RuntimeError(
            f"Unexpected best epoch: "
            f"{best_epoch}"
        )

    # ========================================================
    # VALIDATION METRICS
    # ========================================================

    tuned = (
        metrics[
            "metrics_at_tuned_threshold"
        ]
    )

    metrics_05 = (
        metrics[
            "metrics_at_0_5"
        ]
    )

    # ========================================================
    # STEP 08 COMPARISON
    # ========================================================

    if (
        comparison.get(
            "test_used"
        )
        is not False
    ):

        raise RuntimeError(
            "Step 08 does not confirm test_used=False."
        )

    delta = float(
        comparison[
            "deltas_b_minus_a"
        ][
            "macro_f1_tuned"
        ]
    )

    mcnemar_p = float(
        comparison[
            "mcnemar_exact"
        ][
            "two_sided_p_value"
        ]
    )

    bootstrap = (
        comparison[
            "paired_bootstrap"
        ]
    )

    bootstrap_low = float(
        bootstrap[
            "ci95_low"
        ]
    )

    bootstrap_high = float(
        bootstrap[
            "ci95_high"
        ]
    )

    # ========================================================
    # HASH CHECKPOINT
    # ========================================================

    print()
    print(
        "Hashing final checkpoint..."
    )

    checkpoint_sha256 = (
        sha256_file(
            CHECKPOINT_PATH
        )
    )

    # ========================================================
    # FINAL LOCK
    # ========================================================

    lock = {
        "stage":
            "09_lock_final_clickbait_model",

        "task":
            "2A_clickbait_detection",

        "dataset":
            "viclickbait_v1",

        "status":
            "LOCKED_BEFORE_TEST",

        "test_opened":
            False,

        "test_metrics_computed":
            False,

        "final_model": {
            "architecture":
                EXPECTED_MODEL,

            "input":
                EXPECTED_INPUT,

            "input_description":
                "headline/title only",

            "max_length":
                EXPECTED_MAX_LENGTH,

            "checkpoint":
                relative(
                    CHECKPOINT_PATH
                ),

            "checkpoint_sha256":
                checkpoint_sha256,

            "best_epoch":
                best_epoch,

            "decision_threshold":
                threshold,

            "label_mapping": {
                "0":
                    "NORMAL",

                "1":
                    "CLICKBAIT",
            },
        },

        "validation_reference": {
            "threshold_0_5": {
                "accuracy":
                    float(
                        metrics_05[
                            "accuracy"
                        ]
                    ),

                "macro_f1":
                    float(
                        metrics_05[
                            "macro_f1"
                        ]
                    ),

                "precision_clickbait":
                    float(
                        metrics_05[
                            "precision_clickbait"
                        ]
                    ),

                "recall_clickbait":
                    float(
                        metrics_05[
                            "recall_clickbait"
                        ]
                    ),

                "f1_clickbait":
                    float(
                        metrics_05[
                            "f1_clickbait"
                        ]
                    ),

                "roc_auc":
                    float(
                        metrics_05[
                            "roc_auc"
                        ]
                    ),

                "average_precision":
                    float(
                        metrics_05[
                            "average_precision"
                        ]
                    ),
            },

            "frozen_threshold": {
                "threshold":
                    threshold,

                "accuracy":
                    float(
                        tuned[
                            "accuracy"
                        ]
                    ),

                "macro_f1":
                    float(
                        tuned[
                            "macro_f1"
                        ]
                    ),

                "precision_clickbait":
                    float(
                        tuned[
                            "precision_clickbait"
                        ]
                    ),

                "recall_clickbait":
                    float(
                        tuned[
                            "recall_clickbait"
                        ]
                    ),

                "f1_clickbait":
                    float(
                        tuned[
                            "f1_clickbait"
                        ]
                    ),

                "roc_auc":
                    float(
                        tuned[
                            "roc_auc"
                        ]
                    ),

                "average_precision":
                    float(
                        tuned[
                            "average_precision"
                        ]
                    ),
            },
        },

        "model_selection_evidence": {
            "compared_against":
                "mBERT title + lead",

            "title_lead_minus_title_macro_f1":
                delta,

            "mcnemar_exact_p_value":
                mcnemar_p,

            "paired_bootstrap_macro_f1_delta_ci95": [
                bootstrap_low,
                bootstrap_high,
            ],

            "reason_for_title_only_selection":
                (
                    "Title+lead produced essentially "
                    "the same validation Macro-F1. "
                    "Paired McNemar and bootstrap analyses "
                    "did not show reliable superiority. "
                    "Title-only is therefore retained as "
                    "the simpler final architecture."
                ),
        },

        "frozen_test_protocol": {
            "checkpoint_selection":
                (
                    "No checkpoint may be selected "
                    "using TEST."
                ),

            "threshold_selection":
                (
                    "Decision threshold is frozen "
                    "at 0.61 before TEST."
                ),

            "architecture_selection":
                (
                    "Architecture is frozen as "
                    "mBERT title-only before TEST."
                ),

            "test_evaluation":
                (
                    "Run once using the frozen "
                    "checkpoint and threshold."
                ),
        },
    }

    save_json(
        lock,
        LOCK_PATH,
    )

    # ========================================================
    # PRINT
    # ========================================================

    print()
    print("=" * 80)
    print(
        "FINAL CLICKBAIT MODEL LOCK"
    )
    print("=" * 80)

    print()
    print(
        f"Architecture : "
        f"{EXPECTED_MODEL}"
    )

    print(
        f"Input        : "
        f"{EXPECTED_INPUT}"
    )

    print(
        f"Max length   : "
        f"{EXPECTED_MAX_LENGTH}"
    )

    print(
        f"Best epoch   : "
        f"{best_epoch}"
    )

    print(
        f"Threshold    : "
        f"{threshold:.2f}"
    )

    print()
    print(
        "Validation reference:"
    )

    print(
        f"  Macro-F1 @0.5 : "
        f"{metrics_05['macro_f1']:.6f}"
    )

    print(
        f"  Macro-F1 @0.61: "
        f"{tuned['macro_f1']:.6f}"
    )

    print(
        f"  ROC-AUC       : "
        f"{tuned['roc_auc']:.6f}"
    )

    print()
    print(
        "Title+lead comparison:"
    )

    print(
        f"  Macro-F1 delta B-A : "
        f"{delta:+.6f}"
    )

    print(
        f"  McNemar p-value    : "
        f"{mcnemar_p:.6f}"
    )

    print(
        f"  Bootstrap 95% CI   : "
        f"[{bootstrap_low:+.6f}, "
        f"{bootstrap_high:+.6f}]"
    )

    print()
    print(
        "Checkpoint SHA256:"
    )

    print(
        f"  {checkpoint_sha256}"
    )

    print()
    print(
        "TEST OPENED          : NO"
    )

    print(
        "TEST METRICS COMPUTED: NO"
    )

    print()
    print(
        f"Lock file:\n"
        f"{LOCK_PATH}"
    )

    print()
    print(
        "FINAL CONFIGURATION IS NOW FROZEN."
    )

    print()
    print(
        "STEP 09 FINISHED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()