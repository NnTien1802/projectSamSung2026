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

A_DIR = (
    MODULE_ROOT
    / "results"
    / "consistency"
    / "baseline_title_description_mbert"
)

C_DIR = (
    MODULE_ROOT
    / "results"
    / "consistency"
    / "full_article_max_mil_mbert"
)

A_PRED_PATH = (
    A_DIR
    / "validation_predictions.parquet"
)

C_PRED_PATH = (
    C_DIR
    / "validation_predictions.parquet"
)

OUTPUT_DIR = (
    MODULE_ROOT
    / "results"
    / "consistency"
    / "description_mil_fusion_v1"
)

SEARCH_PATH = (
    OUTPUT_DIR
    / "fusion_search.csv"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "fusion_report.json"
)

HOLDOUT_PRED_PATH = (
    OUTPUT_DIR
    / "holdout_predictions.parquet"
)

OVERLAP_PATH = (
    OUTPUT_DIR
    / "error_overlap.csv"
)

CATEGORY_PATH = (
    OUTPUT_DIR
    / "holdout_by_category.csv"
)

TIER_PATH = (
    OUTPUT_DIR
    / "holdout_negative_by_tier.csv"
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


def stable_key(
    value,
):
    """
    Same split logic as Steps 15 and 16.
    """

    raw = (
        f"{value}|{SEED}"
    ).encode(
        "utf-8"
    )

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
        labels=[
            0,
            1,
        ],
    )

    tn, fp, fn, tp = (
        cm.ravel()
    )

    try:

        roc_auc = roc_auc_score(
            labels,
            probabilities,
        )

    except ValueError:

        roc_auc = None

    try:

        ap = average_precision_score(
            labels,
            probabilities,
        )

    except ValueError:

        ap = None

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

        "precision_consistent":
            float(
                precision_score(
                    labels,
                    predictions,
                    zero_division=0,
                )
            ),

        "recall_consistent":
            float(
                recall_score(
                    labels,
                    predictions,
                    zero_division=0,
                )
            ),

        "f1_consistent":
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
            (
                float(
                    roc_auc
                )
                if roc_auc is not None
                else None
            ),

        "average_precision":
            (
                float(
                    ap
                )
                if ap is not None
                else None
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
# THRESHOLD SEARCH
# ============================================================

def tune_single_model(
    labels,
    probabilities,
):

    best = None

    for threshold in (
        THRESHOLD_VALUES
    ):

        result = compute_metrics(
            labels=labels,
            probabilities=probabilities,
            threshold=float(
                threshold
            ),
        )

        key = (
            result[
                "macro_f1"
            ],
            result[
                "accuracy"
            ],
            -abs(
                threshold
                -
                0.5
            ),
        )

        if best is None:

            best = (
                result,
                key,
            )

        elif (
            key
            >
            best[
                1
            ]
        ):

            best = (
                result,
                key,
            )

    return best[
        0
    ]


# ============================================================
# FUSION SEARCH
# ============================================================

def tune_fusion(
    calibration,
):

    labels = (
        calibration[
            "label"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    prob_a = (
        calibration[
            "prob_a"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    prob_c = (
        calibration[
            "prob_c"
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    rows = []

    best = None

    for alpha_a in (
        ALPHA_VALUES
    ):

        alpha_c = (
            1.0
            -
            alpha_a
        )

        fused = (
            alpha_a
            *
            prob_a
            +
            alpha_c
            *
            prob_c
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
                "alpha_description":
                    float(
                        alpha_a
                    ),

                "alpha_mil":
                    float(
                        alpha_c
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

            key = (
                macro_f1,
                accuracy,
                -abs(
                    alpha_a
                    -
                    0.5
                ),
                -abs(
                    threshold
                    -
                    0.5
                ),
            )

            if best is None:

                best = (
                    row,
                    key,
                )

            elif (
                key
                >
                best[
                    1
                ]
            ):

                best = (
                    row,
                    key,
                )

    return (
        best[
            0
        ],
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
        "STEP 19 - DESCRIPTION + TRAINED MIL FUSION"
    )
    print("=" * 80)

    print()
    print(
        "VALIDATION ONLY."
    )

    print(
        "Calibration and holdout are separated "
        "by anchor_article_uid."
    )

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
        C_PRED_PATH,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                path
            )

    # ========================================================
    # LOAD VALIDATION PAIRS
    # ========================================================

    pairs = (
        pq.read_table(
            PAIR_PATH,
            columns=[
                "pair_id",
                "anchor_article_uid",
                "headline_article_uid",
                "anchor_category",
                "selection_tier",
                "lexical_similarity",
                "candidate_rank",
                "label",
            ],
        )
        .to_pandas()
    )

    if (
        pairs[
            "pair_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate validation pair_id."
        )

    # ========================================================
    # LOAD A
    # ========================================================

    pred_a = (
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

    # ========================================================
    # LOAD C
    # ========================================================

    pred_c = (
        pq.read_table(
            C_PRED_PATH,
            columns=[
                "pair_id",
                "prob_consistent",
            ],
        )
        .to_pandas()
        .rename(
            columns={
                "prob_consistent":
                    "prob_c"
            }
        )
    )

    if (
        pred_a[
            "pair_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate Baseline A pair_id."
        )

    if (
        pred_c[
            "pair_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate MIL pair_id."
        )

    # ========================================================
    # JOIN
    # ========================================================

    frame = (
        pairs
        .merge(
            pred_a,
            on="pair_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            pred_c,
            on="pair_id",
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
            pairs
        )
    ):

        raise RuntimeError(
            "Prediction join lost rows."
        )

    print()
    print(
        f"Validation pairs  : "
        f"{len(frame):,}"
    )

    print(
        f"Validation anchors: "
        f"{frame['anchor_article_uid'].nunique():,}"
    )

    # ========================================================
    # SAME INTERNAL SPLIT AS STEPS 15 / 16
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

    # ========================================================
    # BALANCE CHECK
    # ========================================================

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
                f"{n0:,} vs {n1:,}"
            )

    # ========================================================
    # CALIBRATION
    # ========================================================

    a_cal = tune_single_model(
        labels=(
            calibration[
                "label"
            ]
        ),
        probabilities=(
            calibration[
                "prob_a"
            ]
        ),
    )

    c_cal = tune_single_model(
        labels=(
            calibration[
                "label"
            ]
        ),
        probabilities=(
            calibration[
                "prob_c"
            ]
        ),
    )

    (
        fusion_best,
        fusion_search,
    ) = tune_fusion(
        calibration
    )

    fusion_search.to_csv(
        SEARCH_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    alpha_a = float(
        fusion_best[
            "alpha_description"
        ]
    )

    alpha_c = float(
        fusion_best[
            "alpha_mil"
        ]
    )

    fusion_threshold = float(
        fusion_best[
            "threshold"
        ]
    )

    # ========================================================
    # HOLDOUT PROBABILITIES
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

    holdout_prob_c = (
        holdout[
            "prob_c"
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
        alpha_c
        *
        holdout_prob_c
    )

    # ========================================================
    # HOLDOUT METRICS
    # ========================================================

    a_holdout = compute_metrics(
        labels=holdout_labels,
        probabilities=holdout_prob_a,
        threshold=(
            a_cal[
                "threshold"
            ]
        ),
    )

    c_holdout = compute_metrics(
        labels=holdout_labels,
        probabilities=holdout_prob_c,
        threshold=(
            c_cal[
                "threshold"
            ]
        ),
    )

    fusion_holdout = compute_metrics(
        labels=holdout_labels,
        probabilities=(
            holdout_prob_fusion
        ),
        threshold=(
            fusion_threshold
        ),
    )

    # ========================================================
    # PREDICTIONS
    # ========================================================

    result = (
        holdout[
            [
                "pair_id",
                "anchor_article_uid",
                "headline_article_uid",
                "anchor_category",
                "selection_tier",
                "lexical_similarity",
                "candidate_rank",
                "label",
            ]
        ]
        .copy()
    )

    result[
        "prob_description"
    ] = (
        holdout_prob_a
    )

    result[
        "prob_mil"
    ] = (
        holdout_prob_c
    )

    result[
        "prob_fusion"
    ] = (
        holdout_prob_fusion
    )

    result[
        "pred_description"
    ] = (
        holdout_prob_a
        >=
        a_cal[
            "threshold"
        ]
    ).astype(
        np.int8
    )

    result[
        "pred_mil"
    ] = (
        holdout_prob_c
        >=
        c_cal[
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

    result[
        "correct_description"
    ] = (
        result[
            "pred_description"
        ]
        ==
        result[
            "label"
        ]
    )

    result[
        "correct_mil"
    ] = (
        result[
            "pred_mil"
        ]
        ==
        result[
            "label"
        ]
    )

    result[
        "correct_fusion"
    ] = (
        result[
            "pred_fusion"
        ]
        ==
        result[
            "label"
        ]
    )

    result.to_parquet(
        HOLDOUT_PRED_PATH,
        index=False,
    )

    # ========================================================
    # ERROR OVERLAP A vs C
    # ========================================================

    result[
        "a_c_group"
    ] = "BOTH_WRONG"

    result.loc[
        (
            result[
                "correct_description"
            ]
            &
            result[
                "correct_mil"
            ]
        ),
        "a_c_group",
    ] = "BOTH_CORRECT"

    result.loc[
        (
            result[
                "correct_description"
            ]
            &
            ~result[
                "correct_mil"
            ]
        ),
        "a_c_group",
    ] = "A_ONLY_CORRECT"

    result.loc[
        (
            ~result[
                "correct_description"
            ]
            &
            result[
                "correct_mil"
            ]
        ),
        "a_c_group",
    ] = "C_ONLY_CORRECT"

    overlap = (
        result[
            "a_c_group"
        ]
        .value_counts()
        .reindex(
            [
                "BOTH_CORRECT",
                "A_ONLY_CORRECT",
                "C_ONLY_CORRECT",
                "BOTH_WRONG",
            ],
            fill_value=0,
        )
    )

    overlap_rows = []

    for name, count in (
        overlap.items()
    ):

        overlap_rows.append(
            {
                "group":
                    name,

                "count":
                    int(
                        count
                    ),

                "rate":
                    float(
                        count
                        /
                        len(
                            result
                        )
                    ),
            }
        )

    pd.DataFrame(
        overlap_rows
    ).to_csv(
        OVERLAP_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # CATEGORY ANALYSIS
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

        macro_a = float(
            f1_score(
                labels,
                group[
                    "pred_description"
                ],
                average="macro",
                zero_division=0,
            )
        )

        macro_c = float(
            f1_score(
                labels,
                group[
                    "pred_mil"
                ],
                average="macro",
                zero_division=0,
            )
        )

        macro_fusion = float(
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
                        len(
                            group
                        )
                    ),

                "macro_f1_description":
                    macro_a,

                "macro_f1_mil":
                    macro_c,

                "macro_f1_fusion":
                    macro_fusion,

                "fusion_minus_description":
                    float(
                        macro_fusion
                        -
                        macro_a
                    ),

                "fusion_minus_mil":
                    float(
                        macro_fusion
                        -
                        macro_c
                    ),
            }
        )

    category_df = pd.DataFrame(
        category_rows
    )

    category_df.sort_values(
        "fusion_minus_description",
        ascending=False,
        inplace=True,
    )

    category_df.to_csv(
        CATEGORY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # NEGATIVE TIER ANALYSIS
    # ========================================================

    negative = (
        result[
            result[
                "label"
            ]
            ==
            0
        ]
        .copy()
    )

    tier_rows = []

    for tier, group in (
        negative.groupby(
            "selection_tier"
        )
    ):

        total = len(
            group
        )

        recall_a = float(
            (
                group[
                    "pred_description"
                ]
                ==
                0
            ).mean()
        )

        recall_c = float(
            (
                group[
                    "pred_mil"
                ]
                ==
                0
            ).mean()
        )

        recall_fusion = float(
            (
                group[
                    "pred_fusion"
                ]
                ==
                0
            ).mean()
        )

        tier_rows.append(
            {
                "tier":
                    tier,

                "n":
                    int(
                        total
                    ),

                "negative_recall_description":
                    recall_a,

                "negative_recall_mil":
                    recall_c,

                "negative_recall_fusion":
                    recall_fusion,
            }
        )

    tier_df = pd.DataFrame(
        tier_rows
    )

    tier_df.to_csv(
        TIER_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # REPORT
    # ========================================================

    delta_fusion_a = (
        fusion_holdout[
            "macro_f1"
        ]
        -
        a_holdout[
            "macro_f1"
        ]
    )

    delta_fusion_c = (
        fusion_holdout[
            "macro_f1"
        ]
        -
        c_holdout[
            "macro_f1"
        ]
    )

    report = {
        "stage":
            "19_description_mil_fusion",

        "dataset":
            "headline_consistency_v1",

        "analysis_split":
            "validation",

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
            "description_threshold":
                float(
                    a_cal[
                        "threshold"
                    ]
                ),

            "mil_threshold":
                float(
                    c_cal[
                        "threshold"
                    ]
                ),

            "fusion_description_weight":
                alpha_a,

            "fusion_mil_weight":
                alpha_c,

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
            "description":
                a_holdout,

            "mil":
                c_holdout,

            "fusion":
                fusion_holdout,

            "fusion_minus_description_macro_f1":
                float(
                    delta_fusion_a
                ),

            "fusion_minus_mil_macro_f1":
                float(
                    delta_fusion_c
                ),
        },

        "error_overlap_description_vs_mil": {
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

        "warning":
            (
                "Validation diagnostic only. "
                "No test examples or test "
                "predictions were used."
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
        "CALIBRATION RESULT"
    )
    print("=" * 80)

    print()
    print(
        f"Description threshold : "
        f"{a_cal['threshold']:.2f}"
    )

    print(
        f"MIL threshold         : "
        f"{c_cal['threshold']:.2f}"
    )

    print()
    print(
        "Best A + C fusion:"
    )

    print(
        f"  description weight : "
        f"{alpha_a:.2f}"
    )

    print(
        f"  MIL weight         : "
        f"{alpha_c:.2f}"
    )

    print(
        f"  threshold          : "
        f"{fusion_threshold:.2f}"
    )

    print(
        f"  calibration F1     : "
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
        f"Description Macro-F1 : "
        f"{a_holdout['macro_f1']:.6f}"
    )

    print(
        f"MIL Macro-F1         : "
        f"{c_holdout['macro_f1']:.6f}"
    )

    print(
        f"Fusion Macro-F1      : "
        f"{fusion_holdout['macro_f1']:.6f}"
    )

    print(
        f"Fusion - Description : "
        f"{delta_fusion_a:+.6f}"
    )

    print(
        f"Fusion - MIL         : "
        f"{delta_fusion_c:+.6f}"
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
        "Description vs MIL error overlap:"
    )

    for row in overlap_rows:

        print(
            f"  {row['group']:15s}: "
            f"{row['count']:,} "
            f"({row['rate']:.2%})"
        )

    print()
    print(
        "Negative tier performance:"
    )

    for row in tier_df.itertuples(
        index=False
    ):

        print(
            f"  {row.tier:10s}: "
            f"A={row.negative_recall_description:.4f}, "
            f"C={row.negative_recall_mil:.4f}, "
            f"Fusion={row.negative_recall_fusion:.4f}"
        )

    print()
    print(
        "Largest fusion gains by category:"
    )

    for row in (
        category_df
        .head(5)
        .itertuples(
            index=False
        )
    ):

        print(
            f"  {row.category}: "
            f"A={row.macro_f1_description:.4f}, "
            f"C={row.macro_f1_mil:.4f}, "
            f"Fusion={row.macro_f1_fusion:.4f}, "
            f"delta="
            f"{row.fusion_minus_description:+.4f}"
        )

    print()
    print(
        f"Output directory:\n"
        f"{OUTPUT_DIR}"
    )

    print()
    print(
        "TEST WAS NOT USED."
    )

    print(
        "STEP 19 FINISHED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()