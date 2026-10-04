from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


# ============================================================
# PATHS
# ============================================================

MODULE_ROOT = Path(__file__).resolve().parents[1]

PROJECT_ROOT = MODULE_ROOT.parent


# Allow importing:
# module2.inference.headline_integrity_pipeline
if str(PROJECT_ROOT) not in sys.path:

    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from module2.inference.headline_integrity_pipeline import (
    HeadlineIntegrityPipeline,
)


# ============================================================
# TASK 2A PATHS
# ============================================================

CLICKBAIT_VALIDATION_PATH = (
    MODULE_ROOT
    / "data"
    / "processed"
    / "clickbait"
    / "viclickbait_v1"
    / "validation.parquet"
)

CLICKBAIT_PREDICTIONS_PATH = (
    MODULE_ROOT
    / "results"
    / "clickbait"
    / "mbert_title_only_v1"
    / "validation_predictions.parquet"
)


# ============================================================
# TASK 2B PATHS
# ============================================================

CONSISTENCY_ARTICLES_PATH = (
    MODULE_ROOT
    / "data"
    / "interim"
    / "consistency"
    / "articles_v1"
    / "validation.parquet"
)

CONSISTENCY_HOLDOUT_PATH = (
    MODULE_ROOT
    / "results"
    / "consistency"
    / "description_mil_fusion_v1"
    / "holdout_predictions.parquet"
)

CONSISTENCY_DESCRIPTION_PATH = (
    MODULE_ROOT
    / "results"
    / "consistency"
    / "baseline_title_description_mbert"
    / "validation_predictions.parquet"
)

CONSISTENCY_MIL_PATH = (
    MODULE_ROOT
    / "results"
    / "consistency"
    / "full_article_max_mil_mbert"
    / "validation_predictions.parquet"
)

CONSISTENCY_CHUNK_INDEX_PATH = (
    MODULE_ROOT
    / "data"
    / "interim"
    / "consistency"
    / "mil_chunk_cache_mbert_v1"
    / "validation_article_index.parquet"
)


# ============================================================
# OUTPUTS
# ============================================================

OUTPUT_DIR = (
    MODULE_ROOT
    / "results"
    / "integration"
    / "regression_v1"
)

CLICKBAIT_DETAIL_PATH = (
    OUTPUT_DIR
    / "clickbait_regression_samples.csv"
)

CONSISTENCY_DETAIL_PATH = (
    OUTPUT_DIR
    / "consistency_regression_samples.csv"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "regression_report.json"
)


# ============================================================
# CONFIG
# ============================================================

SEED = 20260921

CLICKBAIT_SAMPLE_SIZE = 32

CONSISTENCY_SAMPLE_SIZE = 24


# Numerical tolerance.
#
# Previous validation inference and new integration inference
# can differ slightly because of AMP / batch shape / GPU
# kernel differences.
#
# Do not increase this automatically if regression fails.
PROBABILITY_TOLERANCE = 0.005


# Frozen thresholds
CLICKBAIT_THRESHOLD = 0.61
CONSISTENCY_THRESHOLD = 0.45


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


def select_evenly(
    frame,
    n,
    sort_columns,
):
    """
    Deterministically select rows spread across a sorted frame.
    """

    if len(frame) <= n:

        return (
            frame
            .copy()
            .reset_index(
                drop=True
            )
        )

    frame = (
        frame
        .sort_values(
            sort_columns
        )
        .reset_index(
            drop=True
        )
    )

    indices = np.linspace(
        0,
        len(frame) - 1,
        num=n,
        dtype=int,
    )

    return (
        frame
        .iloc[
            indices
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


def summarize_errors(
    frame,
    error_column,
):

    values = (
        frame[
            error_column
        ]
        .to_numpy(
            dtype=np.float64
        )
    )

    return {
        "mean_abs_error":
            float(
                np.mean(
                    values
                )
            ),

        "median_abs_error":
            float(
                np.median(
                    values
                )
            ),

        "p95_abs_error":
            float(
                np.percentile(
                    values,
                    95,
                )
            ),

        "max_abs_error":
            float(
                np.max(
                    values
                )
            ),
    }


# ============================================================
# TASK 2A SAMPLE PREPARATION
# ============================================================

def prepare_clickbait_samples():

    validation = (
        pq.read_table(
            CLICKBAIT_VALIDATION_PATH,
            columns=[
                "sample_id",
                "title",
                "label",
            ],
        )
        .to_pandas()
    )

    predictions = (
        pq.read_table(
            CLICKBAIT_PREDICTIONS_PATH,
            columns=[
                "sample_id",
                "label",
                "prob_clickbait",
                "prediction_tuned",
            ],
        )
        .to_pandas()
        .rename(
            columns={
                "label":
                    "saved_label",

                "prob_clickbait":
                    "saved_probability",

                "prediction_tuned":
                    "saved_prediction",
            }
        )
    )

    frame = (
        validation
        .merge(
            predictions,
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
            "Clickbait prediction join lost rows."
        )

    if not (
        frame[
            "label"
        ]
        .to_numpy()
        ==
        frame[
            "saved_label"
        ]
        .to_numpy()
    ).all():

        raise RuntimeError(
            "Clickbait label mismatch."
        )

    # Balanced sample:
    # 16 NORMAL + 16 CLICKBAIT
    per_class = (
        CLICKBAIT_SAMPLE_SIZE
        //
        2
    )

    parts = []

    for label in [
        0,
        1,
    ]:

        subset = (
            frame[
                frame[
                    "label"
                ]
                ==
                label
            ]
            .copy()
        )

        selected = select_evenly(
            subset,
            n=per_class,
            sort_columns=[
                "sample_id",
            ],
        )

        parts.append(
            selected
        )

    sample = (
        pd.concat(
            parts,
            ignore_index=True,
        )
        .sort_values(
            [
                "label",
                "sample_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return sample


# ============================================================
# TASK 2B SAMPLE PREPARATION
# ============================================================

def prepare_consistency_samples():

    # --------------------------------------------------------
    # Fusion holdout
    # --------------------------------------------------------

    holdout = (
        pq.read_table(
            CONSISTENCY_HOLDOUT_PATH
        )
        .to_pandas()
    )

    required_holdout = {
        "pair_id",
        "anchor_article_uid",
        "headline_article_uid",
        "label",
        "prob_description",
        "prob_mil",
        "prob_fusion",
        "pred_fusion",
    }

    missing = (
        required_holdout
        -
        set(
            holdout.columns
        )
    )

    if missing:

        raise RuntimeError(
            "Missing holdout columns: "
            f"{sorted(missing)}"
        )

    # --------------------------------------------------------
    # Full validation prediction artifacts
    # --------------------------------------------------------

    description_predictions = (
        pq.read_table(
            CONSISTENCY_DESCRIPTION_PATH,
            columns=[
                "pair_id",
                "label",
                "prob_consistent",
            ],
        )
        .to_pandas()
        .rename(
            columns={
                "label":
                    "description_label",

                "prob_consistent":
                    "description_validation_probability",
            }
        )
    )

    mil_predictions = (
        pq.read_table(
            CONSISTENCY_MIL_PATH,
            columns=[
                "pair_id",
                "label",
                "prob_consistent",
            ],
        )
        .to_pandas()
        .rename(
            columns={
                "label":
                    "mil_label",

                "prob_consistent":
                    "mil_validation_probability",
            }
        )
    )

    frame = (
        holdout
        .merge(
            description_predictions,
            on="pair_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            mil_predictions,
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
            holdout
        )
    ):

        raise RuntimeError(
            "Consistency prediction joins lost holdout rows."
        )

    # --------------------------------------------------------
    # Check Step 19 holdout branch probabilities are the
    # same saved branch probabilities produced by the original
    # validation models.
    # --------------------------------------------------------

    description_internal_error = np.abs(
        frame[
            "prob_description"
        ].to_numpy(
            dtype=np.float64
        )
        -
        frame[
            "description_validation_probability"
        ].to_numpy(
            dtype=np.float64
        )
    )

    mil_internal_error = np.abs(
        frame[
            "prob_mil"
        ].to_numpy(
            dtype=np.float64
        )
        -
        frame[
            "mil_validation_probability"
        ].to_numpy(
            dtype=np.float64
        )
    )

    if (
        float(
            description_internal_error.max()
        )
        >
        1e-6
    ):

        raise RuntimeError(
            "Step 19 description probabilities do not "
            "match original validation predictions.\n"
            f"Max error: "
            f"{description_internal_error.max():.9f}"
        )

    if (
        float(
            mil_internal_error.max()
        )
        >
        1e-6
    ):

        raise RuntimeError(
            "Step 19 MIL probabilities do not "
            "match original validation predictions.\n"
            f"Max error: "
            f"{mil_internal_error.max():.9f}"
        )

    if not (
        frame[
            "label"
        ]
        .to_numpy()
        ==
        frame[
            "description_label"
        ]
        .to_numpy()
    ).all():

        raise RuntimeError(
            "Description validation label mismatch."
        )

    if not (
        frame[
            "label"
        ]
        .to_numpy()
        ==
        frame[
            "mil_label"
        ]
        .to_numpy()
    ).all():

        raise RuntimeError(
            "MIL validation label mismatch."
        )

    # --------------------------------------------------------
    # Article text store
    # --------------------------------------------------------

    articles = (
        pq.read_table(
            CONSISTENCY_ARTICLES_PATH,
            columns=[
                "article_uid",
                "title",
                "description",
                "content",
            ],
        )
        .to_pandas()
    )

    if (
        articles[
            "article_uid"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate article_uid in validation article store."
        )

    # Anchor provides:
    # description + full article content
    anchor_articles = (
        articles[
            [
                "article_uid",
                "description",
                "content",
            ]
        ]
        .rename(
            columns={
                "article_uid":
                    "anchor_article_uid",

                "description":
                    "anchor_description",

                "content":
                    "anchor_content",
            }
        )
    )

    # Headline article provides:
    # headline/title used in pair
    headline_articles = (
        articles[
            [
                "article_uid",
                "title",
            ]
        ]
        .rename(
            columns={
                "article_uid":
                    "headline_article_uid",

                "title":
                    "headline_title",
            }
        )
    )

    frame = (
        frame
        .merge(
            anchor_articles,
            on="anchor_article_uid",
            how="inner",
            validate="many_to_one",
        )
        .merge(
            headline_articles,
            on="headline_article_uid",
            how="inner",
            validate="many_to_one",
        )
    )

    if (
        len(
            frame
        )
        !=
        len(
            holdout
        )
    ):

        raise RuntimeError(
            "Article text join lost consistency rows."
        )

    # --------------------------------------------------------
    # Original MIL cache article statistics
    # --------------------------------------------------------

    chunk_index = (
        pq.read_table(
            CONSISTENCY_CHUNK_INDEX_PATH,
            columns=[
                "article_uid",
                "token_count",
                "chunk_count",
            ],
        )
        .to_pandas()
        .rename(
            columns={
                "article_uid":
                    "anchor_article_uid",

                "token_count":
                    "original_article_token_count",

                "chunk_count":
                    "original_chunk_count",
            }
        )
    )

    if (
        chunk_index[
            "anchor_article_uid"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            "Duplicate article_uid in MIL validation index."
        )

    frame = (
        frame
        .merge(
            chunk_index,
            on="anchor_article_uid",
            how="inner",
            validate="many_to_one",
        )
    )

    if (
        len(
            frame
        )
        !=
        len(
            holdout
        )
    ):

        raise RuntimeError(
            "MIL chunk-index join lost rows."
        )

    # --------------------------------------------------------
    # Input validation
    # --------------------------------------------------------

    frame[
        "headline_title"
    ] = (
        frame[
            "headline_title"
        ]
        .fillna("")
        .astype(str)
    )

    frame[
        "anchor_description"
    ] = (
        frame[
            "anchor_description"
        ]
        .fillna("")
        .astype(str)
    )

    frame[
        "anchor_content"
    ] = (
        frame[
            "anchor_content"
        ]
        .fillna("")
        .astype(str)
    )

    if (
        frame[
            "headline_title"
        ]
        .eq("")
        .any()
    ):

        raise RuntimeError(
            "Empty headline title found."
        )

    if (
        frame[
            "anchor_content"
        ]
        .eq("")
        .any()
    ):

        raise RuntimeError(
            "Empty anchor content found."
        )

    # --------------------------------------------------------
    # Regression sample
    #
    # Sort by original chunk count + token count so evenly
    # selected rows cover short, medium and long articles.
    #
    # Balanced by label.
    # --------------------------------------------------------

    per_class = (
        CONSISTENCY_SAMPLE_SIZE
        //
        2
    )

    parts = []

    for label in [
        0,
        1,
    ]:

        subset = (
            frame[
                frame[
                    "label"
                ]
                ==
                label
            ]
            .copy()
        )

        selected = select_evenly(
            subset,
            n=per_class,
            sort_columns=[
                "original_chunk_count",
                "original_article_token_count",
                "pair_id",
            ],
        )

        parts.append(
            selected
        )

    sample = (
        pd.concat(
            parts,
            ignore_index=True,
        )
        .sort_values(
            [
                "label",
                "original_chunk_count",
                "original_article_token_count",
                "pair_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return sample


# ============================================================
# TASK 2A REGRESSION
# ============================================================

def run_clickbait_regression(
    pipeline,
):

    print()
    print("=" * 80)
    print(
        "TASK 2A REGRESSION"
    )
    print("=" * 80)

    sample = (
        prepare_clickbait_samples()
    )

    print()
    print(
        f"Samples: "
        f"{len(sample):,}"
    )

    new_probabilities = []
    new_predictions = []

    start = time.time()

    for index, row in enumerate(
        sample.itertuples(
            index=False
        ),
        start=1,
    ):

        result = (
            pipeline.predict_clickbait(
                title=row.title
            )
        )

        new_probabilities.append(
            float(
                result[
                    "probability_clickbait"
                ]
            )
        )

        new_predictions.append(
            int(
                result[
                    "prediction"
                ]
            )
        )

        print(
            f"  [{index:02d}/{len(sample):02d}] "
            f"{row.sample_id}"
        )

    elapsed = (
        time.time()
        -
        start
    )

    sample[
        "new_probability"
    ] = (
        new_probabilities
    )

    sample[
        "new_prediction"
    ] = (
        new_predictions
    )

    sample[
        "abs_error"
    ] = np.abs(
        sample[
            "new_probability"
        ]
        -
        sample[
            "saved_probability"
        ]
    )

    sample[
        "prediction_match"
    ] = (
        sample[
            "new_prediction"
        ]
        ==
        sample[
            "saved_prediction"
        ]
    )

    probability_stats = (
        summarize_errors(
            sample,
            "abs_error",
        )
    )

    prediction_mismatches = int(
        (
            ~sample[
                "prediction_match"
            ]
        ).sum()
    )

    passed = (
        probability_stats[
            "max_abs_error"
        ]
        <=
        PROBABILITY_TOLERANCE
        and
        prediction_mismatches
        ==
        0
    )

    sample.to_csv(
        CLICKBAIT_DETAIL_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        f"Mean abs error : "
        f"{probability_stats['mean_abs_error']:.9f}"
    )

    print(
        f"P95 abs error  : "
        f"{probability_stats['p95_abs_error']:.9f}"
    )

    print(
        f"Max abs error  : "
        f"{probability_stats['max_abs_error']:.9f}"
    )

    print(
        f"Prediction mismatch: "
        f"{prediction_mismatches}"
    )

    print(
        f"Tolerance      : "
        f"{PROBABILITY_TOLERANCE:.6f}"
    )

    print(
        f"RESULT         : "
        f"{'PASS' if passed else 'FAIL'}"
    )

    return {
        "passed":
            bool(
                passed
            ),

        "samples":
            int(
                len(
                    sample
                )
            ),

        "probability_error":
            probability_stats,

        "prediction_mismatches":
            prediction_mismatches,

        "elapsed_seconds":
            float(
                elapsed
            ),
    }


# ============================================================
# TASK 2B REGRESSION
# ============================================================

def run_consistency_regression(
    pipeline,
):

    print()
    print("=" * 80)
    print(
        "TASK 2B REGRESSION"
    )
    print("=" * 80)

    sample = (
        prepare_consistency_samples()
    )

    print()
    print(
        f"Samples: "
        f"{len(sample):,}"
    )

    print()
    print(
        "Original chunk-count coverage:"
    )

    print(
        sample[
            "original_chunk_count"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    new_description = []
    new_mil = []
    new_fusion = []
    new_predictions = []

    new_token_counts = []
    new_chunk_counts = []

    start = time.time()

    for index, row in enumerate(
        sample.itertuples(
            index=False
        ),
        start=1,
    ):

        result = (
            pipeline.predict_consistency(
                title=(
                    row.headline_title
                ),
                lead=(
                    row.anchor_description
                ),
                content=(
                    row.anchor_content
                ),
            )
        )

        description_probability = float(
            result[
                "branches"
            ][
                "description"
            ][
                "probability_consistent"
            ]
        )

        mil_probability = float(
            result[
                "branches"
            ][
                "max_mil"
            ][
                "probability_consistent"
            ]
        )

        fusion_probability = float(
            result[
                "probability_consistent"
            ]
        )

        prediction = int(
            result[
                "prediction"
            ]
        )

        token_count = int(
            result[
                "branches"
            ][
                "max_mil"
            ][
                "article_token_count"
            ]
        )

        chunk_count = int(
            result[
                "branches"
            ][
                "max_mil"
            ][
                "chunk_count"
            ]
        )

        new_description.append(
            description_probability
        )

        new_mil.append(
            mil_probability
        )

        new_fusion.append(
            fusion_probability
        )

        new_predictions.append(
            prediction
        )

        new_token_counts.append(
            token_count
        )

        new_chunk_counts.append(
            chunk_count
        )

        print(
            f"  [{index:02d}/{len(sample):02d}] "
            f"{row.pair_id} | "
            f"label={row.label} | "
            f"original_chunks="
            f"{row.original_chunk_count}"
        )

    elapsed = (
        time.time()
        -
        start
    )

    sample[
        "new_prob_description"
    ] = new_description

    sample[
        "new_prob_mil"
    ] = new_mil

    sample[
        "new_prob_fusion"
    ] = new_fusion

    sample[
        "new_prediction_fusion"
    ] = new_predictions

    sample[
        "new_article_token_count"
    ] = new_token_counts

    sample[
        "new_chunk_count"
    ] = new_chunk_counts

    # --------------------------------------------------------
    # Probability errors
    # --------------------------------------------------------

    sample[
        "description_abs_error"
    ] = np.abs(
        sample[
            "new_prob_description"
        ]
        -
        sample[
            "prob_description"
        ]
    )

    sample[
        "mil_abs_error"
    ] = np.abs(
        sample[
            "new_prob_mil"
        ]
        -
        sample[
            "prob_mil"
        ]
    )

    sample[
        "fusion_abs_error"
    ] = np.abs(
        sample[
            "new_prob_fusion"
        ]
        -
        sample[
            "prob_fusion"
        ]
    )

    # --------------------------------------------------------
    # Prediction equivalence
    # --------------------------------------------------------

    sample[
        "prediction_match"
    ] = (
        sample[
            "new_prediction_fusion"
        ]
        ==
        sample[
            "pred_fusion"
        ]
    )

    # --------------------------------------------------------
    # Cache / chunking equivalence diagnostics
    # --------------------------------------------------------

    sample[
        "token_count_delta"
    ] = (
        sample[
            "new_article_token_count"
        ]
        -
        sample[
            "original_article_token_count"
        ]
    )

    sample[
        "chunk_count_delta"
    ] = (
        sample[
            "new_chunk_count"
        ]
        -
        sample[
            "original_chunk_count"
        ]
    )

    description_stats = (
        summarize_errors(
            sample,
            "description_abs_error",
        )
    )

    mil_stats = (
        summarize_errors(
            sample,
            "mil_abs_error",
        )
    )

    fusion_stats = (
        summarize_errors(
            sample,
            "fusion_abs_error",
        )
    )

    prediction_mismatches = int(
        (
            ~sample[
                "prediction_match"
            ]
        ).sum()
    )

    token_count_mismatches = int(
        (
            sample[
                "token_count_delta"
            ]
            !=
            0
        ).sum()
    )

    chunk_count_mismatches = int(
        (
            sample[
                "chunk_count_delta"
            ]
            !=
            0
        ).sum()
    )

    description_pass = (
        description_stats[
            "max_abs_error"
        ]
        <=
        PROBABILITY_TOLERANCE
    )

    mil_pass = (
        mil_stats[
            "max_abs_error"
        ]
        <=
        PROBABILITY_TOLERANCE
    )

    fusion_pass = (
        fusion_stats[
            "max_abs_error"
        ]
        <=
        PROBABILITY_TOLERANCE
    )

    prediction_pass = (
        prediction_mismatches
        ==
        0
    )

    chunking_pass = (
        token_count_mismatches
        ==
        0
        and
        chunk_count_mismatches
        ==
        0
    )

    overall_pass = (
        description_pass
        and
        mil_pass
        and
        fusion_pass
        and
        prediction_pass
        and
        chunking_pass
    )

    sample.to_csv(
        CONSISTENCY_DETAIL_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 80)
    print(
        "DESCRIPTION BRANCH"
    )
    print("=" * 80)

    print()
    print(
        f"Mean abs error : "
        f"{description_stats['mean_abs_error']:.9f}"
    )

    print(
        f"P95 abs error  : "
        f"{description_stats['p95_abs_error']:.9f}"
    )

    print(
        f"Max abs error  : "
        f"{description_stats['max_abs_error']:.9f}"
    )

    print(
        f"RESULT         : "
        f"{'PASS' if description_pass else 'FAIL'}"
    )

    print()
    print("=" * 80)
    print(
        "MAX-MIL BRANCH"
    )
    print("=" * 80)

    print()
    print(
        f"Mean abs error : "
        f"{mil_stats['mean_abs_error']:.9f}"
    )

    print(
        f"P95 abs error  : "
        f"{mil_stats['p95_abs_error']:.9f}"
    )

    print(
        f"Max abs error  : "
        f"{mil_stats['max_abs_error']:.9f}"
    )

    print()
    print(
        f"Token-count mismatches : "
        f"{token_count_mismatches}"
    )

    print(
        f"Chunk-count mismatches : "
        f"{chunk_count_mismatches}"
    )

    print(
        f"RESULT                 : "
        f"{'PASS' if (mil_pass and chunking_pass) else 'FAIL'}"
    )

    print()
    print("=" * 80)
    print(
        "FINAL FUSION"
    )
    print("=" * 80)

    print()
    print(
        f"Mean abs error : "
        f"{fusion_stats['mean_abs_error']:.9f}"
    )

    print(
        f"P95 abs error  : "
        f"{fusion_stats['p95_abs_error']:.9f}"
    )

    print(
        f"Max abs error  : "
        f"{fusion_stats['max_abs_error']:.9f}"
    )

    print(
        f"Prediction mismatches: "
        f"{prediction_mismatches}"
    )

    print(
        f"RESULT               : "
        f"{'PASS' if (fusion_pass and prediction_pass) else 'FAIL'}"
    )

    return {
        "passed":
            bool(
                overall_pass
            ),

        "samples":
            int(
                len(
                    sample
                )
            ),

        "description":
            {
                "passed":
                    bool(
                        description_pass
                    ),

                "probability_error":
                    description_stats,
            },

        "max_mil":
            {
                "passed":
                    bool(
                        mil_pass
                        and
                        chunking_pass
                    ),

                "probability_error":
                    mil_stats,

                "token_count_mismatches":
                    token_count_mismatches,

                "chunk_count_mismatches":
                    chunk_count_mismatches,
            },

        "fusion":
            {
                "passed":
                    bool(
                        fusion_pass
                        and
                        prediction_pass
                    ),

                "probability_error":
                    fusion_stats,

                "prediction_mismatches":
                    prediction_mismatches,
            },

        "elapsed_seconds":
            float(
                elapsed
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "MODULE 2 - HEADLINE INTEGRITY"
    )
    print(
        "INTEGRATION STEP 03"
    )
    print(
        "UNIFIED PIPELINE REGRESSION CHECK"
    )
    print("=" * 80)

    print()
    print(
        "NO TRAINING."
    )

    print(
        "NO THRESHOLD SEARCH."
    )

    print(
        "NO MODEL SELECTION."
    )

    print(
        "VALIDATION ARTIFACTS ONLY."
    )

    print()
    print(
        f"Probability tolerance: "
        f"{PROBABILITY_TOLERANCE:.6f}"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # REQUIRED FILES
    # ========================================================

    required = [
        CLICKBAIT_VALIDATION_PATH,
        CLICKBAIT_PREDICTIONS_PATH,
        CONSISTENCY_ARTICLES_PATH,
        CONSISTENCY_HOLDOUT_PATH,
        CONSISTENCY_DESCRIPTION_PATH,
        CONSISTENCY_MIL_PATH,
        CONSISTENCY_CHUNK_INDEX_PATH,
    ]

    for path in required:

        require_file(
            path
        )

    # ========================================================
    # LOAD UNIFIED PIPELINE
    # ========================================================

    print()
    print("=" * 80)
    print(
        "LOAD UNIFIED PIPELINE"
    )
    print("=" * 80)

    pipeline = (
        HeadlineIntegrityPipeline()
    )

    # ========================================================
    # TASK 2A
    # ========================================================

    clickbait_result = (
        run_clickbait_regression(
            pipeline
        )
    )

    # ========================================================
    # TASK 2B
    # ========================================================

    consistency_result = (
        run_consistency_regression(
            pipeline
        )
    )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    overall_pass = (
        clickbait_result[
            "passed"
        ]
        and
        consistency_result[
            "passed"
        ]
    )

    report = {
        "stage":
            "integration_step_03_regression",

        "module":
            "Module 2 - Headline Integrity",

        "validation_only":
            True,

        "training_performed":
            False,

        "threshold_search_performed":
            False,

        "model_selection_performed":
            False,

        "probability_tolerance":
            PROBABILITY_TOLERANCE,

        "clickbait":
            clickbait_result,

        "consistency":
            consistency_result,

        "overall_pass":
            bool(
                overall_pass
            ),

        "interpretation": (
            "PASS means the unified inference pipeline "
            "reproduces saved development-time probabilities "
            "and decisions within the fixed numerical tolerance. "
            "FAIL must be investigated; tolerance should not "
            "be loosened automatically."
        ),
    }

    save_json(
        report,
        REPORT_PATH,
    )

    print()
    print("=" * 80)
    print(
        "REGRESSION SUMMARY"
    )
    print("=" * 80)

    print()
    print(
        f"Task 2A Clickbait    : "
        f"{'PASS' if clickbait_result['passed'] else 'FAIL'}"
    )

    print(
        f"Task 2B Description  : "
        f"{'PASS' if consistency_result['description']['passed'] else 'FAIL'}"
    )

    print(
        f"Task 2B MAX-MIL      : "
        f"{'PASS' if consistency_result['max_mil']['passed'] else 'FAIL'}"
    )

    print(
        f"Task 2B Fusion       : "
        f"{'PASS' if consistency_result['fusion']['passed'] else 'FAIL'}"
    )

    print()
    print(
        "UNIFIED PIPELINE:"
    )

    print(
        f"  {'PASS' if overall_pass else 'FAIL'}"
    )

    print()
    print(
        "Outputs:"
    )

    print(
        f"  Clickbait detail:\n"
        f"    {CLICKBAIT_DETAIL_PATH}"
    )

    print(
        f"  Consistency detail:\n"
        f"    {CONSISTENCY_DETAIL_PATH}"
    )

    print(
        f"  Report:\n"
        f"    {REPORT_PATH}"
    )

    print()

    if overall_pass:

        print(
            "INTEGRATION REGRESSION PASSED."
        )

        print(
            "Unified Module 2 inference is behaviorally "
            "consistent with saved validation artifacts."
        )

    else:

        print(
            "INTEGRATION REGRESSION FAILED."
        )

        print(
            "DO NOT loosen the tolerance."
        )

        print(
            "Inspect the failing branch before "
            "using the unified pipeline in production."
        )

    print()
    print(
        "INTEGRATION STEP 03 FINISHED"
    )


if __name__ == "__main__":
    main()