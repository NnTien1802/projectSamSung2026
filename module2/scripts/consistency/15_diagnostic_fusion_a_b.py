from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# PATHS
# ============================================================

MODULE_ROOT = Path(__file__).resolve().parents[2]

PAIR_PATH = (
    MODULE_ROOT
    / "data"
    / "processed"
    / "consistency"
    / "headline_consistency_v1"
    / "validation.parquet"
)

A_PRED_PATH = (
    MODULE_ROOT
    / "results"
    / "consistency"
    / "baseline_title_description_mbert"
    / "validation_predictions.parquet"
)

B_PRED_PATH = (
    MODULE_ROOT
    / "results"
    / "consistency"
    / "baseline_title_article_head_mbert"
    / "validation_predictions.parquet"
)

OUTPUT_DIR = (
    MODULE_ROOT
    / "results"
    / "consistency"
    / "fusion_a_b_diagnostic"
)

SEARCH_PATH = (
    OUTPUT_DIR
    / "fusion_search.csv"
)

METRICS_PATH = (
    OUTPUT_DIR
    / "holdout_metrics.json"
)

PREDICTION_PATH = (
    OUTPUT_DIR
    / "holdout_predictions.parquet"
)

CATEGORY_PATH = (
    OUTPUT_DIR
    / "holdout_by_category.csv"
)


# ============================================================
# CONFIG
# ============================================================

SEED = 20260920

ALPHA_VALUES = np.arange(
    0.0,
    1.001,
    0.05,
)

THRESHOLD_VALUES = np.arange(
    0.05,
    0.951,
    0.01,
)


# ============================================================
# HELPERS
# ============================================================

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


def stable_key(value):

    raw = (
        f"{value}|{SEED}"
    ).encode("utf-8")

    digest = hashlib.blake2b(
        raw,
        digest_size=8,
    ).digest()

    return int.from_bytes(
        digest,
        byteorder="big",
        signed=False,
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

    cm = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    )

    tn, fp, fn, tp = (
        cm.ravel()
    )

    return {
        "threshold":
            float(threshold),

        "accuracy":
            float(
                accuracy_score(
                    labels,
                    predictions,
                )
            ),

        "precision":
            float(
                precision_score(
                    labels,
                    predictions,
                    zero_division=0,
                )
            ),

        "recall":
            float(
                recall_score(
                    labels,
                    predictions,
                    zero_division=0,
                )
            ),

        "f1":
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
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }


# ============================================================
# THRESHOLD SEARCH FOR ONE MODEL
# ============================================================

def tune_threshold(
    labels,
    probabilities,
):

    best = None

    for threshold in (
        THRESHOLD_VALUES
    ):

        metrics = compute_metrics(
            labels,
            probabilities,
            float(threshold),
        )

        if best is None:

            best = metrics
            continue

        current_key = (
            metrics[
                "macro_f1"
            ],
            metrics[
                "accuracy"
            ],
            -abs(
                threshold
                -
                0.5
            ),
        )

        best_key = (
            best[
                "macro_f1"
            ],
            best[
                "accuracy"
            ],
            -abs(
                best[
                    "threshold"
                ]
                -
                0.5
            ),
        )

        if current_key > best_key:

            best = metrics

    return best


# ============================================================
# FUSION SEARCH
# ============================================================

def tune_fusion(
    frame,
):

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

    best = None

    rows = []

    for alpha in ALPHA_VALUES:

        # alpha = weight of Baseline A
        #
        # alpha = 1.0:
        #   only description model
        #
        # alpha = 0.0:
        #   only article-head model

        fused = (
            alpha
            *
            prob_a
            +
            (
                1.0
                -
                alpha
            )
            *
            prob_b
        )

        for threshold in (
            THRESHOLD_VALUES
        ):

            predictions = (
                fused
                >=
                threshold
            ).astype(
                np.int64
            )

            macro_f1 = float(
                f1_score(
                    labels,
                    predictions,
                    average="macro",
                    zero_division=0,
                )
            )

            accuracy = float(
                accuracy_score(
                    labels,
                    predictions,
                )
            )

            row = {
                "alpha_a":
                    float(alpha),

                "alpha_b":
                    float(
                        1.0
                        -
                        alpha
                    ),

                "threshold":
                    float(
                        threshold
                    ),

                "macro_f1":
                    macro_f1,

                "accuracy":
                    accuracy,
            }

            rows.append(
                row
            )

            if best is None:

                best = row
                continue

            current_key = (
                macro_f1,
                accuracy,
                -abs(
                    threshold
                    -
                    0.5
                ),
                -abs(
                    alpha
                    -
                    0.5
                ),
            )

            best_key = (
                best[
                    "macro_f1"
                ],
                best[
                    "accuracy"
                ],
                -abs(
                    best[
                        "threshold"
                    ]
                    -
                    0.5
                ),
                -abs(
                    best[
                        "alpha_a"
                    ]
                    -
                    0.5
                ),
            )

            if current_key > best_key:

                best = row

    return (
        best,
        pd.DataFrame(
            rows
        ),
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
        "TASK 2B - HEADLINE CONSISTENCY"
    )
    print(
        "STEP 15 - DIAGNOSTIC FUSION A + B"
    )
    print("=" * 80)

    print()
    print(
        "VALIDATION is divided by "
        "anchor_article_uid:"
    )

    print(
        "  calibration -> tune"
    )

    print(
        "  holdout     -> evaluate"
    )

    print()
    print(
        "TEST IS NOT USED."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in [
        PAIR_PATH,
        A_PRED_PATH,
        B_PRED_PATH,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                path
            )

    # ========================================================
    # LOAD
    # ========================================================

    pairs = (
        pq.read_table(
            PAIR_PATH,
            columns=[
                "pair_id",
                "anchor_article_uid",
                "anchor_category",
                "label",
            ],
        )
        .to_pandas()
    )

    a = (
        pq.read_table(
            A_PRED_PATH,
            columns=[
                "pair_id",
                "prob_consistent",
            ],
        )
        .to_pandas()
        .rename(
            columns={
                "prob_consistent":
                    "prob_a"
            }
        )
    )

    b = (
        pq.read_table(
            B_PRED_PATH,
            columns=[
                "pair_id",
                "prob_consistent",
            ],
        )
        .to_pandas()
        .rename(
            columns={
                "prob_consistent":
                    "prob_b"
            }
        )
    )

    frame = (
        pairs
        .merge(
            a,
            on="pair_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            b,
            on="pair_id",
            how="inner",
            validate="one_to_one",
        )
    )

    if len(frame) != len(
        pairs
    ):

        raise RuntimeError(
            "Prediction join lost rows."
        )

    print()
    print(
        f"Validation pairs: "
        f"{len(frame):,}"
    )

    print(
        f"Validation anchors: "
        f"{frame['anchor_article_uid'].nunique():,}"
    )

    # ========================================================
    # GROUPED CALIBRATION / HOLDOUT SPLIT
    # ========================================================

    anchors = (
        frame[
            [
                "anchor_article_uid"
            ]
        ]
        .drop_duplicates()
        .copy()
    )

    anchors[
        "__key"
    ] = (
        anchors[
            "anchor_article_uid"
        ]
        .map(
            stable_key
        )
    )

    anchors.sort_values(
        "__key",
        inplace=True,
    )

    midpoint = (
        len(
            anchors
        )
        //
        2
    )

    calibration_anchors = set(
        anchors.iloc[
            :midpoint
        ][
            "anchor_article_uid"
        ]
    )

    holdout_anchors = set(
        anchors.iloc[
            midpoint:
        ][
            "anchor_article_uid"
        ]
    )

    if (
        calibration_anchors
        &
        holdout_anchors
    ):

        raise RuntimeError(
            "Anchor leakage between "
            "calibration and holdout."
        )

    calibration = (
        frame[
            frame[
                "anchor_article_uid"
            ]
            .isin(
                calibration_anchors
            )
        ]
        .copy()
    )

    holdout = (
        frame[
            frame[
                "anchor_article_uid"
            ]
            .isin(
                holdout_anchors
            )
        ]
        .copy()
    )

    print()
    print(
        "Internal validation split:"
    )

    print(
        f"  calibration anchors: "
        f"{len(calibration_anchors):,}"
    )

    print(
        f"  calibration pairs  : "
        f"{len(calibration):,}"
    )

    print(
        f"  holdout anchors    : "
        f"{len(holdout_anchors):,}"
    )

    print(
        f"  holdout pairs      : "
        f"{len(holdout):,}"
    )

    # Because each anchor contributes
    # one positive + one negative,
    # both subsets should remain balanced.

    for name, subset in [
        (
            "calibration",
            calibration,
        ),
        (
            "holdout",
            holdout,
        ),
    ]:

        counts = (
            subset[
                "label"
            ]
            .value_counts()
        )

        n0 = int(
            counts.get(
                0,
                0,
            )
        )

        n1 = int(
            counts.get(
                1,
                0,
            )
        )

        if n0 != n1:

            raise RuntimeError(
                f"{name}: label imbalance "
                f"{n0} vs {n1}"
            )

    # ========================================================
    # TUNE A AND B ON CALIBRATION
    # ========================================================

    a_calibration = (
        tune_threshold(
            calibration[
                "label"
            ],
            calibration[
                "prob_a"
            ],
        )
    )

    b_calibration = (
        tune_threshold(
            calibration[
                "label"
            ],
            calibration[
                "prob_b"
            ],
        )
    )

    # ========================================================
    # TUNE FUSION ON CALIBRATION
    # ========================================================

    (
        fusion_best,
        search_frame,
    ) = tune_fusion(
        calibration
    )

    search_frame.to_csv(
        SEARCH_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    alpha_a = float(
        fusion_best[
            "alpha_a"
        ]
    )

    alpha_b = float(
        fusion_best[
            "alpha_b"
        ]
    )

    fusion_threshold = float(
        fusion_best[
            "threshold"
        ]
    )

    # ========================================================
    # HOLDOUT EVALUATION
    # ========================================================

    holdout_labels = (
        holdout[
            "label"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    holdout_prob_a = (
        holdout[
            "prob_a"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    holdout_prob_b = (
        holdout[
            "prob_b"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    holdout_prob_fusion = (
        alpha_a
        *
        holdout_prob_a
        +
        alpha_b
        *
        holdout_prob_b
    )

    a_holdout = compute_metrics(
        holdout_labels,
        holdout_prob_a,
        a_calibration[
            "threshold"
        ],
    )

    b_holdout = compute_metrics(
        holdout_labels,
        holdout_prob_b,
        b_calibration[
            "threshold"
        ],
    )

    fusion_holdout = compute_metrics(
        holdout_labels,
        holdout_prob_fusion,
        fusion_threshold,
    )

    # ========================================================
    # SAVE HOLDOUT PREDICTIONS
    # ========================================================

    result = (
        holdout[
            [
                "pair_id",
                "anchor_article_uid",
                "anchor_category",
                "label",
            ]
        ]
        .copy()
    )

    result[
        "prob_a"
    ] = holdout_prob_a

    result[
        "prob_b"
    ] = holdout_prob_b

    result[
        "prob_fusion"
    ] = holdout_prob_fusion

    result[
        "pred_a"
    ] = (
        holdout_prob_a
        >=
        a_calibration[
            "threshold"
        ]
    ).astype(
        np.int8
    )

    result[
        "pred_b"
    ] = (
        holdout_prob_b
        >=
        b_calibration[
            "threshold"
        ]
    ).astype(
        np.int8
    )

    result[
        "pred_fusion"
    ] = (
        holdout_prob_fusion
        >=
        fusion_threshold
    ).astype(
        np.int8
    )

    result.to_parquet(
        PREDICTION_PATH,
        index=False,
    )

    # ========================================================
    # CATEGORY DIAGNOSTIC
    # ========================================================

    category_rows = []

    for category, group in (
        result.groupby(
            "anchor_category"
        )
    ):

        labels = (
            group[
                "label"
            ]
            .to_numpy()
        )

        a_macro = float(
            f1_score(
                labels,
                group[
                    "pred_a"
                ],
                average="macro",
                zero_division=0,
            )
        )

        fusion_macro = float(
            f1_score(
                labels,
                group[
                    "pred_fusion"
                ],
                average="macro",
                zero_division=0,
            )
        )

        category_rows.append(
            {
                "category":
                    category,

                "n":
                    int(
                        len(group)
                    ),

                "macro_f1_a":
                    a_macro,

                "macro_f1_fusion":
                    fusion_macro,

                "delta_fusion_minus_a":
                    float(
                        fusion_macro
                        -
                        a_macro
                    ),
            }
        )

    category_df = pd.DataFrame(
        category_rows
    )

    category_df.sort_values(
        "delta_fusion_minus_a",
        ascending=False,
        inplace=True,
    )

    category_df.to_csv(
        CATEGORY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # REPORT
    # ========================================================

    report = {
        "stage":
            "15_diagnostic_fusion_a_b",

        "test_used":
            False,

        "internal_protocol": {
            "split_unit":
                "anchor_article_uid",

            "calibration_anchors":
                int(
                    len(
                        calibration_anchors
                    )
                ),

            "holdout_anchors":
                int(
                    len(
                        holdout_anchors
                    )
                ),

            "calibration_pairs":
                int(
                    len(
                        calibration
                    )
                ),

            "holdout_pairs":
                int(
                    len(
                        holdout
                    )
                ),
        },

        "calibration": {
            "baseline_a_threshold":
                float(
                    a_calibration[
                        "threshold"
                    ]
                ),

            "baseline_b_threshold":
                float(
                    b_calibration[
                        "threshold"
                    ]
                ),

            "fusion_alpha_a":
                alpha_a,

            "fusion_alpha_b":
                alpha_b,

            "fusion_threshold":
                fusion_threshold,

            "fusion_macro_f1":
                float(
                    fusion_best[
                        "macro_f1"
                    ]
                ),
        },

        "holdout": {
            "baseline_a":
                a_holdout,

            "baseline_b":
                b_holdout,

            "fusion":
                fusion_holdout,

            "fusion_minus_a_macro_f1":
                float(
                    fusion_holdout[
                        "macro_f1"
                    ]
                    -
                    a_holdout[
                        "macro_f1"
                    ]
                ),
        },

        "warning":
            (
                "This is a validation diagnostic. "
                "It is not a final test result."
            ),
    }

    save_json(
        report,
        METRICS_PATH,
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print(
        "CALIBRATION RESULT"
    )
    print("=" * 80)

    print()
    print(
        f"Baseline A threshold : "
        f"{a_calibration['threshold']:.2f}"
    )

    print(
        f"Baseline B threshold : "
        f"{b_calibration['threshold']:.2f}"
    )

    print()
    print(
        "Best fusion:"
    )

    print(
        f"  A weight   : "
        f"{alpha_a:.2f}"
    )

    print(
        f"  B weight   : "
        f"{alpha_b:.2f}"
    )

    print(
        f"  threshold  : "
        f"{fusion_threshold:.2f}"
    )

    print(
        f"  Macro-F1   : "
        f"{fusion_best['macro_f1']:.6f}"
    )

    print()
    print("=" * 80)
    print(
        "HOLDOUT RESULT"
    )
    print("=" * 80)

    print()
    print(
        f"Baseline A Macro-F1 : "
        f"{a_holdout['macro_f1']:.6f}"
    )

    print(
        f"Baseline B Macro-F1 : "
        f"{b_holdout['macro_f1']:.6f}"
    )

    print(
        f"Fusion Macro-F1     : "
        f"{fusion_holdout['macro_f1']:.6f}"
    )

    delta = (
        fusion_holdout[
            "macro_f1"
        ]
        -
        a_holdout[
            "macro_f1"
        ]
    )

    print(
        f"Fusion - A          : "
        f"{delta:+.6f}"
    )

    print()
    print(
        "Fusion confusion:"
    )

    cm = (
        fusion_holdout[
            "confusion_matrix"
        ]
    )

    print(
        f"  TN={cm['tn']:,} "
        f"FP={cm['fp']:,}"
    )

    print(
        f"  FN={cm['fn']:,} "
        f"TP={cm['tp']:,}"
    )

    print()
    print(
        "Interpretation:"
    )

    if delta >= 0.005:

        print(
            "  Fusion provides a meaningful "
            "gain over description-only."
        )

        print(
            "  Hybrid architecture is strongly "
            "supported."
        )

    elif delta > 0:

        print(
            "  Fusion improves slightly."
        )

        print(
            "  Hybrid remains promising, but "
            "article evidence needs better "
            "extraction than article-head."
        )

    else:

        print(
            "  Simple probability fusion does "
            "not beat description-only."
        )

        print(
            "  Do not discard article evidence; "
            "move to chunk/evidence retrieval "
            "instead of naive fusion."
        )

    print()
    print(
        "TEST WAS NOT USED."
    )

    print(
        "STEP 15 FINISHED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()