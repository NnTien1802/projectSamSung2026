from pathlib import Path
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)

from scipy.stats import binomtest


# ============================================================
# PATHS
# ============================================================

MODULE_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    MODULE_ROOT
    / "data"
    / "processed"
    / "clickbait"
    / "viclickbait_v1"
)

VALIDATION_PATH = (
    DATA_DIR
    / "validation.parquet"
)


# ------------------------------------------------------------
# MODEL A — TITLE ONLY
# ------------------------------------------------------------

A_DIR = (
    MODULE_ROOT
    / "results"
    / "clickbait"
    / "mbert_title_only_v1"
)

A_PRED_PATH = (
    A_DIR
    / "validation_predictions.parquet"
)

A_METRICS_PATH = (
    A_DIR
    / "validation_metrics.json"
)


# ------------------------------------------------------------
# MODEL B — TITLE + LEAD
# ------------------------------------------------------------

B_DIR = (
    MODULE_ROOT
    / "results"
    / "clickbait"
    / "mbert_title_lead_v1"
)

B_PRED_PATH = (
    B_DIR
    / "validation_predictions.parquet"
)

B_METRICS_PATH = (
    B_DIR
    / "validation_metrics.json"
)


# ------------------------------------------------------------
# OUTPUTS
# ------------------------------------------------------------

OUTPUT_DIR = (
    MODULE_ROOT
    / "results"
    / "clickbait"
    / "compare_title_vs_title_lead_v1"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "comparison_report.json"
)

SAMPLE_PATH = (
    OUTPUT_DIR
    / "validation_sample_comparison.parquet"
)

SOURCE_PATH = (
    OUTPUT_DIR
    / "comparison_by_source.csv"
)

CATEGORY_PATH = (
    OUTPUT_DIR
    / "comparison_by_category.csv"
)

BOOTSTRAP_PATH = (
    OUTPUT_DIR
    / "paired_bootstrap_macro_f1.csv"
)

OVERLAP_PATH = (
    OUTPUT_DIR
    / "error_overlap.csv"
)


# ============================================================
# CONFIG
# ============================================================

SEED = 20260921

BOOTSTRAP_ITERATIONS = 10000

MIN_GROUP_SIZE = 15


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


# ============================================================
# METRICS
# ============================================================

def compute_metrics(
    labels,
    probabilities,
    threshold,
):

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    predictions = (
        probabilities
        >=
        threshold
    ).astype(
        np.int64
    )

    tn, fp, fn, tp = (
        confusion_matrix(
            labels,
            predictions,
            labels=[
                0,
                1,
            ],
        )
        .ravel()
    )

    return {
        "threshold":
            float(
                threshold
            ),

        "accuracy":
            float(
                accuracy_score(
                    labels,
                    predictions,
                )
            ),

        "precision_clickbait":
            float(
                precision_score(
                    labels,
                    predictions,
                    zero_division=0,
                )
            ),

        "recall_clickbait":
            float(
                recall_score(
                    labels,
                    predictions,
                    zero_division=0,
                )
            ),

        "f1_clickbait":
            float(
                f1_score(
                    labels,
                    predictions,
                    zero_division=0,
                )
            ),

        "macro_f1":
            float(
                f1_score(
                    labels,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            ),

        "roc_auc":
            float(
                roc_auc_score(
                    labels,
                    probabilities,
                )
            ),

        "average_precision":
            float(
                average_precision_score(
                    labels,
                    probabilities,
                )
            ),

        "confusion_matrix": {
            "tn":
                int(
                    tn
                ),

            "fp":
                int(
                    fp
                ),

            "fn":
                int(
                    fn
                ),

            "tp":
                int(
                    tp
                ),
        },
    }


# ============================================================
# SUBGROUP COMPARISON
# ============================================================

def compare_groups(
    frame,
    group_column,
    threshold_a,
    threshold_b,
):

    rows = []

    for group_name, group in (
        frame.groupby(
            group_column,
            dropna=False,
        )
    ):

        n = len(
            group
        )

        if (
            n
            <
            MIN_GROUP_SIZE
        ):

            continue

        labels = (
            group[
                "label"
            ]
            .to_numpy(
                dtype=np.int64
            )
        )

        pred_a = (
            group[
                "prob_a"
            ]
            .to_numpy()
            >=
            threshold_a
        ).astype(
            np.int64
        )

        pred_b = (
            group[
                "prob_b"
            ]
            .to_numpy()
            >=
            threshold_b
        ).astype(
            np.int64
        )

        macro_a = float(
            f1_score(
                labels,
                pred_a,
                average="macro",
                zero_division=0,
            )
        )

        macro_b = float(
            f1_score(
                labels,
                pred_b,
                average="macro",
                zero_division=0,
            )
        )

        click_f1_a = float(
            f1_score(
                labels,
                pred_a,
                zero_division=0,
            )
        )

        click_f1_b = float(
            f1_score(
                labels,
                pred_b,
                zero_division=0,
            )
        )

        rows.append(
            {
                group_column:
                    str(
                        group_name
                    ),

                "n":
                    int(
                        n
                    ),

                "clickbait_n":
                    int(
                        (
                            labels
                            ==
                            1
                        ).sum()
                    ),

                "clickbait_rate":
                    float(
                        labels.mean()
                    ),

                "macro_f1_title":
                    macro_a,

                "macro_f1_title_lead":
                    macro_b,

                "macro_f1_delta_b_minus_a":
                    float(
                        macro_b
                        -
                        macro_a
                    ),

                "clickbait_f1_title":
                    click_f1_a,

                "clickbait_f1_title_lead":
                    click_f1_b,

                "clickbait_f1_delta_b_minus_a":
                    float(
                        click_f1_b
                        -
                        click_f1_a
                    ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    if not result.empty:

        result.sort_values(
            [
                "macro_f1_delta_b_minus_a",
                "n",
            ],
            ascending=[
                False,
                False,
            ],
            inplace=True,
        )

    return result


# ============================================================
# PAIRED BOOTSTRAP
# ============================================================

def paired_bootstrap(
    labels,
    prob_a,
    prob_b,
    threshold_a,
    threshold_b,
):

    labels = np.asarray(
        labels,
        dtype=np.int64,
    )

    prob_a = np.asarray(
        prob_a,
        dtype=np.float64,
    )

    prob_b = np.asarray(
        prob_b,
        dtype=np.float64,
    )

    n = len(
        labels
    )

    rng = np.random.default_rng(
        SEED
    )

    rows = []

    for iteration in range(
        BOOTSTRAP_ITERATIONS
    ):

        indices = rng.integers(
            low=0,
            high=n,
            size=n,
        )

        boot_labels = (
            labels[
                indices
            ]
        )

        # Very unlikely, but skip a resample
        # if it contains only one class.
        if (
            len(
                np.unique(
                    boot_labels
                )
            )
            <
            2
        ):

            continue

        pred_a = (
            prob_a[
                indices
            ]
            >=
            threshold_a
        ).astype(
            np.int64
        )

        pred_b = (
            prob_b[
                indices
            ]
            >=
            threshold_b
        ).astype(
            np.int64
        )

        macro_a = float(
            f1_score(
                boot_labels,
                pred_a,
                average="macro",
                zero_division=0,
            )
        )

        macro_b = float(
            f1_score(
                boot_labels,
                pred_b,
                average="macro",
                zero_division=0,
            )
        )

        rows.append(
            {
                "iteration":
                    int(
                        iteration
                    ),

                "macro_f1_title":
                    macro_a,

                "macro_f1_title_lead":
                    macro_b,

                "delta_b_minus_a":
                    float(
                        macro_b
                        -
                        macro_a
                    ),
            }
        )

    return pd.DataFrame(
        rows
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
        "STEP 08 - COMPARE MBERT TITLE VS TITLE+LEAD"
    )
    print("=" * 80)

    print()
    print(
        "VALIDATION ONLY."
    )

    print(
        "NO TRAINING."
    )

    print(
        "TEST IS NOT LOADED."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # REQUIRED FILES
    # ========================================================

    for path in [
        VALIDATION_PATH,
        A_PRED_PATH,
        B_PRED_PATH,
        A_METRICS_PATH,
        B_METRICS_PATH,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                path
            )

    # ========================================================
    # LOAD VALIDATION METADATA
    # ========================================================

    validation = (
        pq.read_table(
            VALIDATION_PATH,
            columns=[
                "sample_id",
                "title",
                "lead",
                "source",
                "category",
                "label",
            ],
        )
        .to_pandas()
    )

    validation[
        "sample_id"
    ] = (
        validation[
            "sample_id"
        ]
        .astype(str)
    )

    if (
        validation[
            "sample_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate validation sample_id."
        )

    # ========================================================
    # LOAD PREDICTIONS
    # ========================================================

    pred_a = (
        pq.read_table(
            A_PRED_PATH,
            columns=[
                "sample_id",
                "label",
                "prob_clickbait",
            ],
        )
        .to_pandas()
        .rename(
            columns={
                "label":
                    "label_a",

                "prob_clickbait":
                    "prob_a",
            }
        )
    )

    pred_b = (
        pq.read_table(
            B_PRED_PATH,
            columns=[
                "sample_id",
                "label",
                "prob_clickbait",
            ],
        )
        .to_pandas()
        .rename(
            columns={
                "label":
                    "label_b",

                "prob_clickbait":
                    "prob_b",
            }
        )
    )

    # ========================================================
    # JOIN
    # ========================================================

    frame = (
        validation
        .merge(
            pred_a,
            on="sample_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            pred_b,
            on="sample_id",
            how="inner",
            validate="one_to_one",
        )
    )

    if (
        len(
            frame
        )
        !=
        len(
            validation
        )
    ):

        raise RuntimeError(
            "Prediction join lost validation rows."
        )

    if not (
        frame[
            "label"
        ].to_numpy()
        ==
        frame[
            "label_a"
        ].to_numpy()
    ).all():

        raise RuntimeError(
            "Title-only label mismatch."
        )

    if not (
        frame[
            "label"
        ].to_numpy()
        ==
        frame[
            "label_b"
        ].to_numpy()
    ).all():

        raise RuntimeError(
            "Title+lead label mismatch."
        )

    # ========================================================
    # LOAD FROZEN VALIDATION THRESHOLDS
    # ========================================================

    metrics_a = load_json(
        A_METRICS_PATH
    )

    metrics_b = load_json(
        B_METRICS_PATH
    )

    threshold_a = float(
        metrics_a[
            "best_validation_threshold"
        ]
    )

    threshold_b = float(
        metrics_b[
            "best_validation_threshold"
        ]
    )

    print()
    print(
        f"Validation samples: "
        f"{len(frame):,}"
    )

    print()
    print(
        "Saved thresholds:"
    )

    print(
        f"  Title-only : "
        f"{threshold_a:.2f}"
    )

    print(
        f"  Title+lead : "
        f"{threshold_b:.2f}"
    )

    # ========================================================
    # OVERALL METRICS
    # ========================================================

    labels = (
        frame[
            "label"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    prob_a = (
        frame[
            "prob_a"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    prob_b = (
        frame[
            "prob_b"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    # --------------------------------------------------------
    # Fixed 0.50
    # --------------------------------------------------------

    a_05 = compute_metrics(
        labels,
        prob_a,
        0.5,
    )

    b_05 = compute_metrics(
        labels,
        prob_b,
        0.5,
    )

    # --------------------------------------------------------
    # Saved tuned thresholds
    # --------------------------------------------------------

    a_tuned = compute_metrics(
        labels,
        prob_a,
        threshold_a,
    )

    b_tuned = compute_metrics(
        labels,
        prob_b,
        threshold_b,
    )

    # ========================================================
    # PER-SAMPLE PREDICTIONS
    # ========================================================

    frame[
        "pred_a"
    ] = (
        frame[
            "prob_a"
        ]
        >=
        threshold_a
    ).astype(
        np.int8
    )

    frame[
        "pred_b"
    ] = (
        frame[
            "prob_b"
        ]
        >=
        threshold_b
    ).astype(
        np.int8
    )

    frame[
        "correct_a"
    ] = (
        frame[
            "pred_a"
        ]
        ==
        frame[
            "label"
        ]
    )

    frame[
        "correct_b"
    ] = (
        frame[
            "pred_b"
        ]
        ==
        frame[
            "label"
        ]
    )

    # ========================================================
    # ERROR OVERLAP
    # ========================================================

    frame[
        "error_group"
    ] = "BOTH_WRONG"

    frame.loc[
        (
            frame[
                "correct_a"
            ]
            &
            frame[
                "correct_b"
            ]
        ),
        "error_group",
    ] = "BOTH_CORRECT"

    frame.loc[
        (
            frame[
                "correct_a"
            ]
            &
            ~frame[
                "correct_b"
            ]
        ),
        "error_group",
    ] = "TITLE_ONLY_CORRECT"

    frame.loc[
        (
            ~frame[
                "correct_a"
            ]
            &
            frame[
                "correct_b"
            ]
        ),
        "error_group",
    ] = "TITLE_LEAD_CORRECT"

    overlap_order = [
        "BOTH_CORRECT",
        "TITLE_ONLY_CORRECT",
        "TITLE_LEAD_CORRECT",
        "BOTH_WRONG",
    ]

    overlap_counts = (
        frame[
            "error_group"
        ]
        .value_counts()
        .reindex(
            overlap_order,
            fill_value=0,
        )
    )

    overlap_rows = []

    for group_name, count in (
        overlap_counts.items()
    ):

        overlap_rows.append(
            {
                "group":
                    group_name,

                "count":
                    int(
                        count
                    ),

                "rate":
                    float(
                        count
                        /
                        len(
                            frame
                        )
                    ),
            }
        )

    overlap_df = pd.DataFrame(
        overlap_rows
    )

    overlap_df.to_csv(
        OVERLAP_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # MCNEMAR EXACT TEST
    # ========================================================

    a_only_correct = int(
        overlap_counts[
            "TITLE_ONLY_CORRECT"
        ]
    )

    b_only_correct = int(
        overlap_counts[
            "TITLE_LEAD_CORRECT"
        ]
    )

    discordant = (
        a_only_correct
        +
        b_only_correct
    )

    if discordant > 0:

        mcnemar_result = binomtest(
            k=min(
                a_only_correct,
                b_only_correct,
            ),
            n=discordant,
            p=0.5,
            alternative="two-sided",
        )

        mcnemar_p = float(
            mcnemar_result.pvalue
        )

    else:

        mcnemar_p = 1.0

    # ========================================================
    # PAIRED BOOTSTRAP
    # ========================================================

    print()
    print(
        f"Running paired bootstrap "
        f"({BOOTSTRAP_ITERATIONS:,} iterations)..."
    )

    bootstrap_df = paired_bootstrap(
        labels=labels,
        prob_a=prob_a,
        prob_b=prob_b,
        threshold_a=threshold_a,
        threshold_b=threshold_b,
    )

    bootstrap_df.to_csv(
        BOOTSTRAP_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    deltas = (
        bootstrap_df[
            "delta_b_minus_a"
        ]
        .to_numpy()
    )

    bootstrap_mean_delta = float(
        np.mean(
            deltas
        )
    )

    bootstrap_median_delta = float(
        np.median(
            deltas
        )
    )

    bootstrap_ci_low = float(
        np.percentile(
            deltas,
            2.5,
        )
    )

    bootstrap_ci_high = float(
        np.percentile(
            deltas,
            97.5,
        )
    )

    probability_b_better = float(
        (
            deltas
            >
            0
        ).mean()
    )

    # ========================================================
    # SOURCE / CATEGORY
    # ========================================================

    source_df = compare_groups(
        frame=frame,
        group_column="source",
        threshold_a=threshold_a,
        threshold_b=threshold_b,
    )

    source_df.to_csv(
        SOURCE_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    category_df = compare_groups(
        frame=frame,
        group_column="category",
        threshold_a=threshold_a,
        threshold_b=threshold_b,
    )

    category_df.to_csv(
        CATEGORY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # SAVE SAMPLE COMPARISON
    # ========================================================

    frame[
        [
            "sample_id",
            "source",
            "category",
            "label",
            "prob_a",
            "prob_b",
            "pred_a",
            "pred_b",
            "correct_a",
            "correct_b",
            "error_group",
            "title",
            "lead",
        ]
    ].to_parquet(
        SAMPLE_PATH,
        index=False,
    )

    # ========================================================
    # REPORT
    # ========================================================

    report = {
        "stage":
            "08_compare_mbert_title_vs_title_lead",

        "task":
            "2A_clickbait_detection",

        "dataset":
            "viclickbait_v1",

        "split":
            "validation",

        "test_used":
            False,

        "models": {
            "A_title_only": {
                "threshold":
                    threshold_a,

                "metrics_at_0_5":
                    a_05,

                "metrics_at_saved_tuned_threshold":
                    a_tuned,
            },

            "B_title_plus_lead": {
                "threshold":
                    threshold_b,

                "metrics_at_0_5":
                    b_05,

                "metrics_at_saved_tuned_threshold":
                    b_tuned,
            },
        },

        "deltas_b_minus_a": {
            "macro_f1_at_0_5":
                float(
                    b_05[
                        "macro_f1"
                    ]
                    -
                    a_05[
                        "macro_f1"
                    ]
                ),

            "macro_f1_tuned":
                float(
                    b_tuned[
                        "macro_f1"
                    ]
                    -
                    a_tuned[
                        "macro_f1"
                    ]
                ),

            "accuracy_tuned":
                float(
                    b_tuned[
                        "accuracy"
                    ]
                    -
                    a_tuned[
                        "accuracy"
                    ]
                ),

            "precision_clickbait_tuned":
                float(
                    b_tuned[
                        "precision_clickbait"
                    ]
                    -
                    a_tuned[
                        "precision_clickbait"
                    ]
                ),

            "recall_clickbait_tuned":
                float(
                    b_tuned[
                        "recall_clickbait"
                    ]
                    -
                    a_tuned[
                        "recall_clickbait"
                    ]
                ),

            "f1_clickbait_tuned":
                float(
                    b_tuned[
                        "f1_clickbait"
                    ]
                    -
                    a_tuned[
                        "f1_clickbait"
                    ]
                ),

            "roc_auc":
                float(
                    b_tuned[
                        "roc_auc"
                    ]
                    -
                    a_tuned[
                        "roc_auc"
                    ]
                ),

            "average_precision":
                float(
                    b_tuned[
                        "average_precision"
                    ]
                    -
                    a_tuned[
                        "average_precision"
                    ]
                ),
        },

        "paired_error_overlap": {
            row[
                "group"
            ]: {
                "count":
                    row[
                        "count"
                    ],

                "rate":
                    row[
                        "rate"
                    ],
            }

            for row in overlap_rows
        },

        "mcnemar_exact": {
            "title_only_correct_title_lead_wrong":
                a_only_correct,

            "title_lead_correct_title_only_wrong":
                b_only_correct,

            "discordant_samples":
                int(
                    discordant
                ),

            "two_sided_p_value":
                mcnemar_p,
        },

        "paired_bootstrap": {
            "iterations":
                int(
                    len(
                        bootstrap_df
                    )
                ),

            "metric":
                "Macro-F1",

            "thresholds_reoptimized_per_bootstrap":
                False,

            "mean_delta_b_minus_a":
                bootstrap_mean_delta,

            "median_delta_b_minus_a":
                bootstrap_median_delta,

            "ci95_low":
                bootstrap_ci_low,

            "ci95_high":
                bootstrap_ci_high,

            "fraction_delta_greater_than_zero":
                probability_b_better,

            "warning":
                (
                    "Thresholds were already tuned on this "
                    "validation set. Bootstrap therefore "
                    "measures sample stability, not performance "
                    "on an untouched external population."
                ),
        },

        "interpretation_note":
            (
                "Primary comparison remains Macro-F1. "
                "Subgroup and bootstrap analyses are "
                "diagnostic only. TEST remains untouched."
            ),
    }

    save_json(
        report,
        REPORT_PATH,
    )

    # ========================================================
    # PRINT
    # ========================================================

    print()
    print("=" * 80)
    print(
        "OVERALL COMPARISON"
    )
    print("=" * 80)

    print()
    print(
        "Threshold 0.50:"
    )

    print(
        f"  Title-only Macro-F1 : "
        f"{a_05['macro_f1']:.6f}"
    )

    print(
        f"  Title+lead Macro-F1 : "
        f"{b_05['macro_f1']:.6f}"
    )

    print(
        f"  Delta B-A           : "
        f"{b_05['macro_f1'] - a_05['macro_f1']:+.6f}"
    )

    print()
    print(
        "Saved tuned thresholds:"
    )

    print(
        f"  Title-only threshold: "
        f"{threshold_a:.2f}"
    )

    print(
        f"  Title+lead threshold: "
        f"{threshold_b:.2f}"
    )

    print()
    print(
        f"  Title-only Macro-F1 : "
        f"{a_tuned['macro_f1']:.6f}"
    )

    print(
        f"  Title+lead Macro-F1 : "
        f"{b_tuned['macro_f1']:.6f}"
    )

    print(
        f"  Delta B-A           : "
        f"{b_tuned['macro_f1'] - a_tuned['macro_f1']:+.6f}"
    )

    print()
    print(
        "CLICKBAIT class:"
    )

    print(
        f"  Precision A/B : "
        f"{a_tuned['precision_clickbait']:.4f} / "
        f"{b_tuned['precision_clickbait']:.4f}"
    )

    print(
        f"  Recall A/B    : "
        f"{a_tuned['recall_clickbait']:.4f} / "
        f"{b_tuned['recall_clickbait']:.4f}"
    )

    print(
        f"  F1 A/B        : "
        f"{a_tuned['f1_clickbait']:.4f} / "
        f"{b_tuned['f1_clickbait']:.4f}"
    )

    print()
    print(
        f"ROC-AUC A/B : "
        f"{a_tuned['roc_auc']:.6f} / "
        f"{b_tuned['roc_auc']:.6f}"
    )

    print(
        f"AP A/B      : "
        f"{a_tuned['average_precision']:.6f} / "
        f"{b_tuned['average_precision']:.6f}"
    )

    # ========================================================
    # ERROR OVERLAP
    # ========================================================

    print()
    print("=" * 80)
    print(
        "PAIRED ERROR OVERLAP"
    )
    print("=" * 80)

    print()

    for row in overlap_rows:

        print(
            f"{row['group']:24s}: "
            f"{row['count']:3d} "
            f"({row['rate']:.2%})"
        )

    print()
    print(
        "McNemar exact test:"
    )

    print(
        f"  A-only correct : "
        f"{a_only_correct}"
    )

    print(
        f"  B-only correct : "
        f"{b_only_correct}"
    )

    print(
        f"  p-value        : "
        f"{mcnemar_p:.6f}"
    )

    # ========================================================
    # BOOTSTRAP
    # ========================================================

    print()
    print("=" * 80)
    print(
        "PAIRED BOOTSTRAP — TUNED THRESHOLDS"
    )
    print("=" * 80)

    print()

    print(
        f"Iterations       : "
        f"{len(bootstrap_df):,}"
    )

    print(
        f"Observed delta   : "
        f"{b_tuned['macro_f1'] - a_tuned['macro_f1']:+.6f}"
    )

    print(
        f"Mean delta       : "
        f"{bootstrap_mean_delta:+.6f}"
    )

    print(
        f"Median delta     : "
        f"{bootstrap_median_delta:+.6f}"
    )

    print(
        f"95% CI           : "
        f"[{bootstrap_ci_low:+.6f}, "
        f"{bootstrap_ci_high:+.6f}]"
    )

    print(
        f"P(delta > 0)     : "
        f"{probability_b_better:.4f}"
    )

    # ========================================================
    # SOURCES
    # ========================================================

    if not source_df.empty:

        print()
        print("=" * 80)
        print(
            "LARGEST TITLE+LEAD GAINS BY SOURCE"
        )
        print("=" * 80)

        print()

        for row in (
            source_df
            .head(5)
            .itertuples(
                index=False
            )
        ):

            print(
                f"{row.source:20s} "
                f"n={row.n:3d} "
                f"A={row.macro_f1_title:.4f} "
                f"B={row.macro_f1_title_lead:.4f} "
                f"delta="
                f"{row.macro_f1_delta_b_minus_a:+.4f}"
            )

        print()
        print(
            "Largest losses by source:"
        )

        for row in (
            source_df
            .tail(5)
            .sort_values(
                "macro_f1_delta_b_minus_a"
            )
            .itertuples(
                index=False
            )
        ):

            print(
                f"{row.source:20s} "
                f"n={row.n:3d} "
                f"A={row.macro_f1_title:.4f} "
                f"B={row.macro_f1_title_lead:.4f} "
                f"delta="
                f"{row.macro_f1_delta_b_minus_a:+.4f}"
            )

    # ========================================================
    # CATEGORY
    # ========================================================

    if not category_df.empty:

        print()
        print("=" * 80)
        print(
            "LARGEST TITLE+LEAD GAINS BY CATEGORY"
        )
        print("=" * 80)

        print()

        for row in (
            category_df
            .head(5)
            .itertuples(
                index=False
            )
        ):

            print(
                f"{row.category:24s} "
                f"n={row.n:3d} "
                f"A={row.macro_f1_title:.4f} "
                f"B={row.macro_f1_title_lead:.4f} "
                f"delta="
                f"{row.macro_f1_delta_b_minus_a:+.4f}"
            )

    print()
    print(
        "TEST WAS NOT LOADED."
    )

    print()
    print(
        f"Output directory:\n"
        f"{OUTPUT_DIR}"
    )

    print()
    print(
        "STEP 08 FINISHED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()