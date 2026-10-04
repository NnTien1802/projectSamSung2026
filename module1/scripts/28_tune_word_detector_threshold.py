from pathlib import Path
import csv
import json

import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULT_DIR = (
    PROJECT_ROOT
    / "results"
    / "module1_phobert_word_detector_v1"
)

PROBABILITIES_PATH = (
    RESULT_DIR
    / "best_validation_word_probabilities.npz"
)

BEST_METRICS_PATH = (
    RESULT_DIR
    / "best_validation_metrics.json"
)

OUTPUT_DIR = (
    RESULT_DIR
    / "threshold_tuning"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

SWEEP_PATH = (
    OUTPUT_DIR
    / "threshold_sweep.csv"
)

BEST_THRESHOLD_PATH = (
    OUTPUT_DIR
    / "best_threshold.json"
)


# ============================================================
# CONFIG
# ============================================================

THRESHOLD_MIN = 0.01
THRESHOLD_MAX = 0.99
THRESHOLD_STEP = 0.01

RECALL_TARGETS = [
    0.94,
    0.95,
    0.96,
]

PRECISION_TARGETS = [
    0.90,
    0.92,
    0.95,
]


# ============================================================
# HELPERS
# ============================================================

def safe_divide(
    numerator,
    denominator,
):

    if denominator == 0:
        return 0.0

    return (
        numerator
        / denominator
    )


def calculate_metrics(
    labels,
    probabilities,
    covered,
    threshold,
):

    predictions = (
        probabilities
        >= threshold
    )

    positive = (
        labels == 1
    )

    negative = (
        labels == 0
    )

    tp = int(
        np.sum(
            predictions
            &
            positive
        )
    )

    fp = int(
        np.sum(
            predictions
            &
            negative
        )
    )

    fn = int(
        np.sum(
            (~predictions)
            &
            positive
        )
    )

    tn = int(
        np.sum(
            (~predictions)
            &
            negative
        )
    )

    precision = safe_divide(
        tp,
        tp + fp,
    )

    recall = safe_divide(
        tp,
        tp + fn,
    )

    f1 = safe_divide(
        2
        * precision
        * recall,

        precision
        + recall,
    )

    specificity = safe_divide(
        tn,
        tn + fp,
    )

    covered_positive = (
        positive
        &
        (
            covered == 1
        )
    )

    covered_positive_count = int(
        np.sum(
            covered_positive
        )
    )

    covered_tp = int(
        np.sum(
            predictions
            &
            covered_positive
        )
    )

    covered_recall = safe_divide(
        covered_tp,
        covered_positive_count,
    )

    predicted_positive = (
        tp + fp
    )

    return {
        "threshold":
            float(
                threshold
            ),

        "precision":
            float(
                precision
            ),

        "recall":
            float(
                recall
            ),

        "f1":
            float(
                f1
            ),

        "specificity":
            float(
                specificity
            ),

        "covered_error_recall":
            float(
                covered_recall
            ),

        "tp":
            tp,

        "fp":
            fp,

        "fn":
            fn,

        "tn":
            tn,

        "predicted_error_words":
            int(
                predicted_positive
            ),
    }


# ============================================================
# PRINT
# ============================================================

def print_metrics(
    title,
    metrics,
):

    print(
        "\n" + "-" * 80
    )

    print(
        title
    )

    print(
        "-" * 80
    )

    if metrics is None:

        print(
            "No threshold satisfies "
            "this condition."
        )

        return

    print(
        f"Threshold             : "
        f"{metrics['threshold']:.4f}"
    )

    print(
        f"Word Precision        : "
        f"{metrics['precision']:.6f}"
    )

    print(
        f"Word Recall           : "
        f"{metrics['recall']:.6f}"
    )

    print(
        f"Word F1               : "
        f"{metrics['f1']:.6f}"
    )

    print(
        f"Specificity           : "
        f"{metrics['specificity']:.6f}"
    )

    print(
        f"Covered-error Recall  : "
        f"{metrics['covered_error_recall']:.6f}"
    )

    print(
        f"\nTP: "
        f"{metrics['tp']:,}"
    )

    print(
        f"FP: "
        f"{metrics['fp']:,}"
    )

    print(
        f"FN: "
        f"{metrics['fn']:,}"
    )

    print(
        f"TN: "
        f"{metrics['tn']:,}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)

    print(
        "VN-NEWSSPELL - WORD DETECTOR "
        "THRESHOLD TUNING"
    )

    print("=" * 80)

    print(
        "\nValidation probabilities ONLY."
    )

    print(
        "Test set is NOT read or used."
    )

    # ========================================================
    # LOAD
    # ========================================================

    print(
        f"\nLoading:\n"
        f"{PROBABILITIES_PATH}"
    )

    data = np.load(
        PROBABILITIES_PATH
    )

    probabilities = (
        data[
            "probabilities"
        ]
        .astype(
            np.float32
        )
    )

    labels = (
        data[
            "labels"
        ]
        .astype(
            np.uint8
        )
    )

    covered = (
        data[
            "covered"
        ]
        .astype(
            np.uint8
        )
    )

    if not (
        len(
            probabilities
        )
        ==
        len(
            labels
        )
        ==
        len(
            covered
        )
    ):

        raise RuntimeError(
            "Array length mismatch."
        )

    word_count = len(
        labels
    )

    positive_count = int(
        np.sum(
            labels == 1
        )
    )

    negative_count = (
        word_count
        - positive_count
    )

    covered_positive_count = int(
        np.sum(
            (labels == 1)
            &
            (covered == 1)
        )
    )

    uncovered_positive_count = (
        positive_count
        - covered_positive_count
    )

    print(
        f"\nWord positions       : "
        f"{word_count:,}"
    )

    print(
        f"Error words          : "
        f"{positive_count:,}"
    )

    print(
        f"Clean words          : "
        f"{negative_count:,}"
    )

    print(
        f"Covered error words  : "
        f"{covered_positive_count:,}"
    )

    print(
        f"Uncovered error words: "
        f"{uncovered_positive_count:,}"
    )

    print(
        f"Maximum recall from current "
        f"4-chunk coverage: "
        f"{covered_positive_count / positive_count:.6f}"
    )

    # ========================================================
    # ORIGINAL BEST-EPOCH METRICS
    # ========================================================

    if BEST_METRICS_PATH.exists():

        with open(
            BEST_METRICS_PATH,
            "r",
            encoding="utf-8",
        ) as f:

            original_metrics = (
                json.load(f)
            )

        print(
            f"\nCheckpoint epoch: "
            f"{original_metrics.get('epoch')}"
        )

        print(
            f"Validation AP  : "
            f"{original_metrics.get('average_precision'):.6f}"
        )

        print(
            f"Validation AUC : "
            f"{original_metrics.get('roc_auc'):.6f}"
        )

    # ========================================================
    # THRESHOLDS
    # ========================================================

    thresholds = np.arange(
        THRESHOLD_MIN,
        THRESHOLD_MAX
        + THRESHOLD_STEP / 2,
        THRESHOLD_STEP,
    )

    thresholds = np.round(
        thresholds,
        4,
    )

    thresholds = np.unique(
        np.append(
            thresholds,
            0.5,
        )
    )

    print(
        f"\nThresholds: "
        f"{len(thresholds)}"
    )

    print(
        f"Range     : "
        f"{thresholds[0]:.2f} "
        f"-> "
        f"{thresholds[-1]:.2f}"
    )

    # ========================================================
    # SWEEP
    # ========================================================

    rows = []

    for index, threshold in enumerate(
        thresholds,
        start=1,
    ):

        metrics = (
            calculate_metrics(
                labels=
                    labels,

                probabilities=
                    probabilities,

                covered=
                    covered,

                threshold=
                    threshold,
            )
        )

        rows.append(
            metrics
        )

        if (
            index
            % 10
            == 0
        ):

            print(
                f"\rProcessed "
                f"{index}/{len(thresholds)}",
                end="",
                flush=True,
            )

    print()

    # ========================================================
    # DEFAULT 0.5
    # ========================================================

    default_metrics = min(
        rows,
        key=lambda row:
            abs(
                row[
                    "threshold"
                ]
                - 0.5
            ),
    )

    # ========================================================
    # BEST WORD F1
    # ========================================================

    best_f1 = max(
        rows,
        key=lambda row: (
            row[
                "f1"
            ],

            row[
                "precision"
            ],

            row[
                "recall"
            ],
        ),
    )

    # ========================================================
    # RECALL TARGETS
    # ========================================================

    recall_points = {}

    for target in RECALL_TARGETS:

        candidates = [
            row
            for row in rows
            if (
                row[
                    "recall"
                ]
                >= target
            )
        ]

        key = (
            f"recall_at_least_"
            f"{int(target * 100)}"
        )

        if not candidates:

            recall_points[
                key
            ] = None

        else:

            recall_points[
                key
            ] = max(
                candidates,
                key=lambda row: (
                    row[
                        "precision"
                    ],

                    row[
                        "f1"
                    ],

                    row[
                        "threshold"
                    ],
                ),
            )

    # ========================================================
    # PRECISION TARGETS
    # ========================================================

    precision_points = {}

    for target in PRECISION_TARGETS:

        candidates = [
            row
            for row in rows
            if (
                row[
                    "precision"
                ]
                >= target
            )
        ]

        key = (
            f"precision_at_least_"
            f"{int(target * 100)}"
        )

        if not candidates:

            precision_points[
                key
            ] = None

        else:

            precision_points[
                key
            ] = max(
                candidates,
                key=lambda row: (
                    row[
                        "recall"
                    ],

                    row[
                        "f1"
                    ],

                    -row[
                        "threshold"
                    ],
                ),
            )

    # ========================================================
    # PRINT
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "SELECTED OPERATING POINTS"
    )

    print(
        "=" * 80
    )

    print_metrics(
        "DEFAULT THRESHOLD = 0.50",
        default_metrics,
    )

    print_metrics(
        "BEST VALIDATION WORD-F1",
        best_f1,
    )

    for target in RECALL_TARGETS:

        key = (
            f"recall_at_least_"
            f"{int(target * 100)}"
        )

        print_metrics(
            (
                f"RECALL >= "
                f"{target:.2f} "
                f"WITH HIGHEST PRECISION"
            ),

            recall_points[
                key
            ],
        )

    for target in PRECISION_TARGETS:

        key = (
            f"precision_at_least_"
            f"{int(target * 100)}"
        )

        print_metrics(
            (
                f"PRECISION >= "
                f"{target:.2f} "
                f"WITH HIGHEST RECALL"
            ),

            precision_points[
                key
            ],
        )

    # ========================================================
    # SAVE SWEEP
    # ========================================================

    with open(
        SWEEP_PATH,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        fieldnames = [
            "threshold",
            "precision",
            "recall",
            "f1",
            "specificity",
            "covered_error_recall",
            "tp",
            "fp",
            "fn",
            "tn",
            "predicted_error_words",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in rows:

            writer.writerow(
                row
            )

    # ========================================================
    # SAVE PRIMARY THRESHOLD
    # ========================================================

    output = {
        "dataset":
            "validation",

        "test_used":
            False,

        "checkpoint_selection_metric":
            "validation_average_precision",

        "threshold_selection_metric":
            "maximum_validation_word_f1",

        "selected_threshold":
            float(
                best_f1[
                    "threshold"
                ]
            ),

        "selected_metrics":
            best_f1,

        "default_threshold_metrics":
            default_metrics,

        "recall_operating_points":
            recall_points,

        "precision_operating_points":
            precision_points,

        "word_positions":
            word_count,

        "error_words":
            positive_count,

        "covered_error_words":
            covered_positive_count,

        "uncovered_error_words":
            uncovered_positive_count,

        "maximum_recall_from_current_coverage":
            float(
                covered_positive_count
                / positive_count
            ),
    }

    with open(
        BEST_THRESHOLD_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # FINISH
    # ========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        "WORD THRESHOLD TUNING FINISHED"
    )

    print(
        "=" * 80
    )

    print(
        f"\nPRIMARY THRESHOLD: "
        f"{best_f1['threshold']:.4f}"
    )

    print(
        f"Word F1: "
        f"{best_f1['f1']:.6f}"
    )

    print(
        f"Precision: "
        f"{best_f1['precision']:.6f}"
    )

    print(
        f"Recall: "
        f"{best_f1['recall']:.6f}"
    )

    print(
        f"\nSweep:\n"
        f"{SWEEP_PATH}"
    )

    print(
        f"\nBest threshold:\n"
        f"{BEST_THRESHOLD_PATH}"
    )

    print(
        "\nTest set was NOT read or used."
    )


if __name__ == "__main__":

    main()