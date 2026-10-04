from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


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

TRAIN_PATH = (
    DATA_DIR
    / "train.parquet"
)

VALIDATION_PATH = (
    DATA_DIR
    / "validation.parquet"
)

OUTPUT_DIR = (
    MODULE_ROOT
    / "results"
    / "clickbait"
    / "eda_v1"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "eda_report.json"
)

SPLIT_SUMMARY_PATH = (
    OUTPUT_DIR
    / "split_summary.csv"
)

LENGTH_STATS_PATH = (
    OUTPUT_DIR
    / "length_stats.csv"
)

FEATURE_STATS_PATH = (
    OUTPUT_DIR
    / "surface_feature_stats.csv"
)

CATEGORY_STATS_PATH = (
    OUTPUT_DIR
    / "category_label_stats.csv"
)

SOURCE_STATS_PATH = (
    OUTPUT_DIR
    / "source_label_stats.csv"
)

TRAIN_CUE_STATS_PATH = (
    OUTPUT_DIR
    / "train_lexical_cue_stats.csv"
)

EXTREME_EXAMPLES_PATH = (
    OUTPUT_DIR
    / "train_extreme_length_examples.csv"
)


# ============================================================
# CONFIG
# ============================================================

# Fixed before looking at label associations.
#
# These are only diagnostic lexical cues.
# They will NOT automatically become model features.
CLICKBAIT_CUES = [
    "bất ngờ",
    "gây sốc",
    "sốc",
    "không ngờ",
    "không thể tin",
    "bí mật",
    "hé lộ",
    "tiết lộ",
    "sự thật",
    "kinh ngạc",
    "choáng",
    "ngỡ ngàng",
    "giật mình",
    "bạn có biết",
    "ai cũng",
    "điều gì",
    "vì sao",
    "tại sao",
    "lý do",
    "đây là",
    "cái kết",
    "kết quả bất ngờ",
    "chuyện gì xảy ra",
    "khiến dân mạng",
    "gây tranh cãi",
    "gây bão",
]


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


def safe_text(
    value,
):

    if pd.isna(
        value
    ):

        return ""

    return str(
        value
    )


def word_count(
    text,
):

    text = safe_text(
        text
    ).strip()

    if not text:

        return 0

    return len(
        text.split()
    )


def uppercase_ratio(
    text,
):

    text = safe_text(
        text
    )

    letters = [
        char
        for char in text
        if char.isalpha()
    ]

    if not letters:

        return 0.0

    uppercase = sum(
        char.isupper()
        for char in letters
    )

    return float(
        uppercase
        /
        len(
            letters
        )
    )


def contains_number(
    text,
):

    return int(
        bool(
            re.search(
                r"\d",
                safe_text(
                    text
                ),
            )
        )
    )


def contains_question(
    text,
):

    return int(
        "?"
        in
        safe_text(
            text
        )
    )


def contains_exclamation(
    text,
):

    return int(
        "!"
        in
        safe_text(
            text
        )
    )


def contains_ellipsis(
    text,
):

    text = safe_text(
        text
    )

    return int(
        (
            "..."
            in
            text
        )
        or
        (
            "…"
            in
            text
        )
    )


def contains_quote(
    text,
):

    text = safe_text(
        text
    )

    quote_chars = [
        '"',
        "'",
        "“",
        "”",
        "‘",
        "’",
    ]

    return int(
        any(
            char
            in
            text
            for char in quote_chars
        )
    )


def contains_colon(
    text,
):

    return int(
        ":"
        in
        safe_text(
            text
        )
    )


def contains_dash(
    text,
):

    text = safe_text(
        text
    )

    return int(
        (
            "-"
            in
            text
        )
        or
        (
            "–"
            in
            text
        )
        or
        (
            "—"
            in
            text
        )
    )


# ============================================================
# FEATURE ENGINEERING FOR EDA ONLY
# ============================================================

def add_diagnostic_features(
    frame,
):

    frame = (
        frame
        .copy()
    )

    frame[
        "title"
    ] = (
        frame[
            "title"
        ]
        .fillna("")
        .astype(str)
    )

    frame[
        "lead"
    ] = (
        frame[
            "lead"
        ]
        .fillna("")
        .astype(str)
    )

    frame[
        "title_chars"
    ] = (
        frame[
            "title"
        ]
        .str.len()
        .astype(int)
    )

    frame[
        "title_words"
    ] = (
        frame[
            "title"
        ]
        .map(
            word_count
        )
        .astype(int)
    )

    frame[
        "lead_chars"
    ] = (
        frame[
            "lead"
        ]
        .str.len()
        .astype(int)
    )

    frame[
        "lead_words"
    ] = (
        frame[
            "lead"
        ]
        .map(
            word_count
        )
        .astype(int)
    )

    frame[
        "lead_missing"
    ] = (
        frame[
            "lead"
        ]
        .eq("")
        .astype(int)
    )

    frame[
        "title_has_question"
    ] = (
        frame[
            "title"
        ]
        .map(
            contains_question
        )
    )

    frame[
        "title_has_exclamation"
    ] = (
        frame[
            "title"
        ]
        .map(
            contains_exclamation
        )
    )

    frame[
        "title_has_number"
    ] = (
        frame[
            "title"
        ]
        .map(
            contains_number
        )
    )

    frame[
        "title_has_ellipsis"
    ] = (
        frame[
            "title"
        ]
        .map(
            contains_ellipsis
        )
    )

    frame[
        "title_has_quote"
    ] = (
        frame[
            "title"
        ]
        .map(
            contains_quote
        )
    )

    frame[
        "title_has_colon"
    ] = (
        frame[
            "title"
        ]
        .map(
            contains_colon
        )
    )

    frame[
        "title_has_dash"
    ] = (
        frame[
            "title"
        ]
        .map(
            contains_dash
        )
    )

    frame[
        "title_uppercase_ratio"
    ] = (
        frame[
            "title"
        ]
        .map(
            uppercase_ratio
        )
    )

    frame[
        "title_all_caps"
    ] = (
        frame[
            "title_uppercase_ratio"
        ]
        .ge(
            0.80
        )
        .astype(int)
    )

    frame[
        "title_lower"
    ] = (
        frame[
            "title"
        ]
        .str.lower()
    )

    return frame


# ============================================================
# LOAD SPLIT
# ============================================================

def load_split(
    split_name,
    path,
):

    if not path.exists():

        raise FileNotFoundError(
            path
        )

    frame = (
        pq.read_table(
            path
        )
        .to_pandas()
    )

    required = {
        "sample_id",
        "title",
        "lead",
        "category",
        "source",
        "label",
        "label_name",
    }

    missing = (
        required
        -
        set(
            frame.columns
        )
    )

    if missing:

        raise RuntimeError(
            f"{split_name}: "
            f"missing columns "
            f"{sorted(missing)}"
        )

    if (
        frame[
            "sample_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            f"{split_name}: "
            "duplicate sample_id."
        )

    frame[
        "split"
    ] = split_name

    return add_diagnostic_features(
        frame
    )


# ============================================================
# SPLIT SUMMARY
# ============================================================

def build_split_summary(
    frame,
):

    rows = []

    for split, group in (
        frame.groupby(
            "split",
            sort=False,
        )
    ):

        counts = (
            group[
                "label"
            ]
            .value_counts()
        )

        rows.append(
            {
                "split":
                    split,

                "rows":
                    int(
                        len(
                            group
                        )
                    ),

                "normal":
                    int(
                        counts.get(
                            0,
                            0,
                        )
                    ),

                "clickbait":
                    int(
                        counts.get(
                            1,
                            0,
                        )
                    ),

                "clickbait_rate":
                    float(
                        group[
                            "label"
                        ].mean()
                    ),

                "empty_title":
                    int(
                        group[
                            "title"
                        ]
                        .eq("")
                        .sum()
                    ),

                "empty_lead":
                    int(
                        group[
                            "lead_missing"
                        ]
                        .sum()
                    ),

                "unique_categories":
                    int(
                        group[
                            "category"
                        ]
                        .nunique()
                    ),

                "unique_sources":
                    int(
                        group[
                            "source"
                        ]
                        .nunique()
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# LENGTH STATISTICS
# ============================================================

def describe_numeric(
    values,
):

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "mean":
            float(
                np.mean(
                    values
                )
            ),

        "median":
            float(
                np.median(
                    values
                )
            ),

        "p05":
            float(
                np.percentile(
                    values,
                    5,
                )
            ),

        "p25":
            float(
                np.percentile(
                    values,
                    25,
                )
            ),

        "p75":
            float(
                np.percentile(
                    values,
                    75,
                )
            ),

        "p95":
            float(
                np.percentile(
                    values,
                    95,
                )
            ),

        "min":
            float(
                np.min(
                    values
                )
            ),

        "max":
            float(
                np.max(
                    values
                )
            ),
    }


def build_length_stats(
    frame,
):

    rows = []

    numeric_columns = [
        "title_chars",
        "title_words",
        "lead_chars",
        "lead_words",
    ]

    for split, split_group in (
        frame.groupby(
            "split",
            sort=False,
        )
    ):

        for label in [
            0,
            1,
        ]:

            label_group = (
                split_group[
                    split_group[
                        "label"
                    ]
                    ==
                    label
                ]
            )

            for column in (
                numeric_columns
            ):

                stats = (
                    describe_numeric(
                        label_group[
                            column
                        ]
                    )
                )

                rows.append(
                    {
                        "split":
                            split,

                        "label":
                            label,

                        "label_name":
                            (
                                "CLICKBAIT"
                                if label == 1
                                else "NORMAL"
                            ),

                        "feature":
                            column,

                        "n":
                            int(
                                len(
                                    label_group
                                )
                            ),

                        **stats,
                    }
                )

    return pd.DataFrame(
        rows
    )


# ============================================================
# SURFACE SHORTCUT STATISTICS
# ============================================================

def build_surface_feature_stats(
    frame,
):

    features = [
        "title_has_question",
        "title_has_exclamation",
        "title_has_number",
        "title_has_ellipsis",
        "title_has_quote",
        "title_has_colon",
        "title_has_dash",
        "title_all_caps",
        "lead_missing",
    ]

    rows = []

    for split, split_group in (
        frame.groupby(
            "split",
            sort=False,
        )
    ):

        for feature in features:

            normal = (
                split_group[
                    split_group[
                        "label"
                    ]
                    ==
                    0
                ][
                    feature
                ]
            )

            clickbait = (
                split_group[
                    split_group[
                        "label"
                    ]
                    ==
                    1
                ][
                    feature
                ]
            )

            normal_rate = float(
                normal.mean()
            )

            clickbait_rate = float(
                clickbait.mean()
            )

            rows.append(
                {
                    "split":
                        split,

                    "feature":
                        feature,

                    "normal_rate":
                        normal_rate,

                    "clickbait_rate":
                        clickbait_rate,

                    "absolute_rate_gap":
                        float(
                            abs(
                                clickbait_rate
                                -
                                normal_rate
                            )
                        ),

                    "signed_rate_gap_clickbait_minus_normal":
                        float(
                            clickbait_rate
                            -
                            normal_rate
                        ),
                }
            )

    result = pd.DataFrame(
        rows
    )

    result.sort_values(
        [
            "split",
            "absolute_rate_gap",
        ],
        ascending=[
            True,
            False,
        ],
        inplace=True,
    )

    return result


# ============================================================
# CATEGORY / SOURCE LABEL ASSOCIATIONS
# ============================================================

def categorical_stats(
    frame,
    column,
):

    rows = []

    for split, split_group in (
        frame.groupby(
            "split",
            sort=False,
        )
    ):

        grouped = (
            split_group
            .groupby(
                column,
                dropna=False,
            )
        )

        for value, group in grouped:

            value = safe_text(
                value
            )

            rows.append(
                {
                    "split":
                        split,

                    column:
                        value,

                    "n":
                        int(
                            len(
                                group
                            )
                        ),

                    "normal":
                        int(
                            (
                                group[
                                    "label"
                                ]
                                ==
                                0
                            )
                            .sum()
                        ),

                    "clickbait":
                        int(
                            (
                                group[
                                    "label"
                                ]
                                ==
                                1
                            )
                            .sum()
                        ),

                    "clickbait_rate":
                        float(
                            group[
                                "label"
                            ].mean()
                        ),
                }
            )

    result = pd.DataFrame(
        rows
    )

    result.sort_values(
        [
            "split",
            "n",
        ],
        ascending=[
            True,
            False,
        ],
        inplace=True,
    )

    return result


# ============================================================
# FIXED LEXICAL CUE ANALYSIS — TRAIN ONLY
# ============================================================

def build_train_cue_stats(
    train,
):

    rows = []

    normal = (
        train[
            train[
                "label"
            ]
            ==
            0
        ]
    )

    clickbait = (
        train[
            train[
                "label"
            ]
            ==
            1
        ]
    )

    for cue in CLICKBAIT_CUES:

        escaped = re.escape(
            cue.lower()
        )

        normal_hits = (
            normal[
                "title_lower"
            ]
            .str.contains(
                escaped,
                regex=True,
                na=False,
            )
        )

        clickbait_hits = (
            clickbait[
                "title_lower"
            ]
            .str.contains(
                escaped,
                regex=True,
                na=False,
            )
        )

        normal_count = int(
            normal_hits.sum()
        )

        clickbait_count = int(
            clickbait_hits.sum()
        )

        normal_rate = float(
            normal_hits.mean()
        )

        clickbait_rate = float(
            clickbait_hits.mean()
        )

        rows.append(
            {
                "cue":
                    cue,

                "normal_count":
                    normal_count,

                "clickbait_count":
                    clickbait_count,

                "normal_rate":
                    normal_rate,

                "clickbait_rate":
                    clickbait_rate,

                "absolute_rate_gap":
                    float(
                        abs(
                            clickbait_rate
                            -
                            normal_rate
                        )
                    ),

                "signed_rate_gap_clickbait_minus_normal":
                    float(
                        clickbait_rate
                        -
                        normal_rate
                    ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    result.sort_values(
        "absolute_rate_gap",
        ascending=False,
        inplace=True,
    )

    return result


# ============================================================
# EXTREME TRAIN EXAMPLES
# ============================================================

def build_extreme_examples(
    train,
):

    rows = []

    for label in [
        0,
        1,
    ]:

        group = (
            train[
                train[
                    "label"
                ]
                ==
                label
            ]
        )

        longest_title = (
            group
            .nlargest(
                5,
                "title_words",
            )
        )

        longest_lead = (
            group
            .nlargest(
                5,
                "lead_words",
            )
        )

        for _, row in (
            longest_title
            .iterrows()
        ):

            rows.append(
                {
                    "kind":
                        "longest_title",

                    "label":
                        int(
                            label
                        ),

                    "label_name":
                        (
                            "CLICKBAIT"
                            if label == 1
                            else "NORMAL"
                        ),

                    "sample_id":
                        row[
                            "sample_id"
                        ],

                    "title_words":
                        int(
                            row[
                                "title_words"
                            ]
                        ),

                    "lead_words":
                        int(
                            row[
                                "lead_words"
                            ]
                        ),

                    "title":
                        row[
                            "title"
                        ],

                    "lead":
                        row[
                            "lead"
                        ],
                }
            )

        for _, row in (
            longest_lead
            .iterrows()
        ):

            rows.append(
                {
                    "kind":
                        "longest_lead",

                    "label":
                        int(
                            label
                        ),

                    "label_name":
                        (
                            "CLICKBAIT"
                            if label == 1
                            else "NORMAL"
                        ),

                    "sample_id":
                        row[
                            "sample_id"
                        ],

                    "title_words":
                        int(
                            row[
                                "title_words"
                            ]
                        ),

                    "lead_words":
                        int(
                            row[
                                "lead_words"
                            ]
                        ),

                    "title":
                        row[
                            "title"
                        ],

                    "lead":
                        row[
                            "lead"
                        ],
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
        "STEP 04 - EDA PROCESSED CLICKBAIT DATA"
    )
    print("=" * 80)

    print()
    print(
        "TRAIN + VALIDATION ONLY."
    )

    print(
        "TEST IS NOT LOADED."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # LOAD
    # ========================================================

    train = load_split(
        "train",
        TRAIN_PATH,
    )

    validation = load_split(
        "validation",
        VALIDATION_PATH,
    )

    combined = pd.concat(
        [
            train,
            validation,
        ],
        ignore_index=True,
    )

    print()
    print(
        f"TRAIN rows      : "
        f"{len(train):,}"
    )

    print(
        f"VALIDATION rows : "
        f"{len(validation):,}"
    )

    print(
        f"Development rows: "
        f"{len(combined):,}"
    )

    # ========================================================
    # SPLIT SUMMARY
    # ========================================================

    split_summary = (
        build_split_summary(
            combined
        )
    )

    split_summary.to_csv(
        SPLIT_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # LENGTHS
    # ========================================================

    length_stats = (
        build_length_stats(
            combined
        )
    )

    length_stats.to_csv(
        LENGTH_STATS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # SURFACE FEATURES
    # ========================================================

    surface_stats = (
        build_surface_feature_stats(
            combined
        )
    )

    surface_stats.to_csv(
        FEATURE_STATS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # CATEGORY
    # ========================================================

    category_stats = (
        categorical_stats(
            combined,
            "category",
        )
    )

    category_stats.to_csv(
        CATEGORY_STATS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # SOURCE
    # ========================================================

    source_stats = (
        categorical_stats(
            combined,
            "source",
        )
    )

    source_stats.to_csv(
        SOURCE_STATS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # TRAIN-ONLY FIXED LEXICAL CUES
    # ========================================================

    cue_stats = (
        build_train_cue_stats(
            train
        )
    )

    cue_stats.to_csv(
        TRAIN_CUE_STATS_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # EXTREME EXAMPLES — TRAIN ONLY
    # ========================================================

    extreme_examples = (
        build_extreme_examples(
            train
        )
    )

    extreme_examples.to_csv(
        EXTREME_EXAMPLES_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # DRIFT CHECK TRAIN vs VALIDATION
    # ========================================================

    train_clickbait_rate = float(
        train[
            "label"
        ].mean()
    )

    validation_clickbait_rate = float(
        validation[
            "label"
        ].mean()
    )

    train_title_median = float(
        train[
            "title_words"
        ].median()
    )

    validation_title_median = float(
        validation[
            "title_words"
        ].median()
    )

    train_lead_median = float(
        train[
            "lead_words"
        ].median()
    )

    validation_lead_median = float(
        validation[
            "lead_words"
        ].median()
    )

    # ========================================================
    # STRONGEST SURFACE GAPS
    # ========================================================

    train_surface = (
        surface_stats[
            surface_stats[
                "split"
            ]
            ==
            "train"
        ]
        .sort_values(
            "absolute_rate_gap",
            ascending=False,
        )
        .head(5)
    )

    strongest_surface = (
        train_surface
        .to_dict(
            orient="records"
        )
    )

    strongest_cues = (
        cue_stats
        .head(10)
        .to_dict(
            orient="records"
        )
    )

    # ========================================================
    # REPORT
    # ========================================================

    report = {
        "stage":
            "04_eda_clickbait_processed",

        "task":
            "2A_clickbait_detection",

        "dataset":
            "viclickbait_v1",

        "test_loaded":
            False,

        "development_rows":
            int(
                len(
                    combined
                )
            ),

        "train_rows":
            int(
                len(
                    train
                )
            ),

        "validation_rows":
            int(
                len(
                    validation
                )
            ),

        "train": {
            "normal":
                int(
                    (
                        train[
                            "label"
                        ]
                        ==
                        0
                    )
                    .sum()
                ),

            "clickbait":
                int(
                    (
                        train[
                            "label"
                        ]
                        ==
                        1
                    )
                    .sum()
                ),

            "clickbait_rate":
                train_clickbait_rate,

            "empty_lead":
                int(
                    train[
                        "lead_missing"
                    ]
                    .sum()
                ),

            "title_words_median":
                train_title_median,

            "lead_words_median":
                train_lead_median,
        },

        "validation": {
            "normal":
                int(
                    (
                        validation[
                            "label"
                        ]
                        ==
                        0
                    )
                    .sum()
                ),

            "clickbait":
                int(
                    (
                        validation[
                            "label"
                        ]
                        ==
                        1
                    )
                    .sum()
                ),

            "clickbait_rate":
                validation_clickbait_rate,

            "empty_lead":
                int(
                    validation[
                        "lead_missing"
                    ]
                    .sum()
                ),

            "title_words_median":
                validation_title_median,

            "lead_words_median":
                validation_lead_median,
        },

        "train_validation_drift": {
            "clickbait_rate_delta":
                float(
                    validation_clickbait_rate
                    -
                    train_clickbait_rate
                ),

            "title_words_median_delta":
                float(
                    validation_title_median
                    -
                    train_title_median
                ),

            "lead_words_median_delta":
                float(
                    validation_lead_median
                    -
                    train_lead_median
                ),
        },

        "strongest_train_surface_feature_gaps":
            strongest_surface,

        "strongest_train_fixed_lexical_cues":
            strongest_cues,

        "notes": [
            (
                "Surface and lexical features are "
                "diagnostics only and are not automatically "
                "used as classifier inputs."
            ),

            (
                "Lexical cue analysis is restricted to TRAIN "
                "to reduce adaptive use of VALIDATION."
            ),

            (
                "TEST remains locked and was not loaded."
            ),
        ],
    }

    save_json(
        report,
        REPORT_PATH,
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print(
        "CLASS DISTRIBUTION"
    )
    print("=" * 80)

    for row in (
        split_summary
        .itertuples(
            index=False
        )
    ):

        print()
        print(
            f"{row.split.upper()}:"
        )

        print(
            f"  rows          : "
            f"{row.rows:,}"
        )

        print(
            f"  NORMAL        : "
            f"{row.normal:,}"
        )

        print(
            f"  CLICKBAIT     : "
            f"{row.clickbait:,}"
        )

        print(
            f"  clickbait rate: "
            f"{row.clickbait_rate:.4f}"
        )

        print(
            f"  empty lead    : "
            f"{row.empty_lead:,}"
        )

    # ========================================================
    # LENGTH SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print(
        "TRAIN LENGTH SUMMARY"
    )
    print("=" * 80)

    for label in [
        0,
        1,
    ]:

        label_name = (
            "CLICKBAIT"
            if label == 1
            else "NORMAL"
        )

        print()
        print(
            f"{label_name}:"
        )

        rows = (
            length_stats[
                (
                    length_stats[
                        "split"
                    ]
                    ==
                    "train"
                )
                &
                (
                    length_stats[
                        "label"
                    ]
                    ==
                    label
                )
            ]
        )

        for row in rows.itertuples(
            index=False
        ):

            print(
                f"  {row.feature:12s}: "
                f"median={row.median:.1f}, "
                f"p95={row.p95:.1f}, "
                f"max={row.max:.1f}"
            )

    # ========================================================
    # SURFACE FEATURES
    # ========================================================

    print()
    print("=" * 80)
    print(
        "LARGEST TRAIN SURFACE FEATURE GAPS"
    )
    print("=" * 80)

    print()

    for row in (
        train_surface
        .itertuples(
            index=False
        )
    ):

        print(
            f"{row.feature:24s} "
            f"NORMAL={row.normal_rate:.4f} "
            f"CLICKBAIT={row.clickbait_rate:.4f} "
            f"delta="
            f"{row.signed_rate_gap_clickbait_minus_normal:+.4f}"
        )

    # ========================================================
    # LEXICAL CUES
    # ========================================================

    print()
    print("=" * 80)
    print(
        "TOP FIXED LEXICAL CUES — TRAIN ONLY"
    )
    print("=" * 80)

    print()

    for row in (
        cue_stats
        .head(10)
        .itertuples(
            index=False
        )
    ):

        print(
            f"{row.cue:24s} "
            f"NORMAL={row.normal_count:4d} "
            f"CLICKBAIT={row.clickbait_count:4d} "
            f"delta="
            f"{row.signed_rate_gap_clickbait_minus_normal:+.4f}"
        )

    # ========================================================
    # SOURCE/CATEGORY SIGNAL
    # ========================================================

    print()
    print("=" * 80)
    print(
        "TRAIN SOURCE DISTRIBUTION - TOP 10"
    )
    print("=" * 80)

    print()

    train_source = (
        source_stats[
            source_stats[
                "split"
            ]
            ==
            "train"
        ]
        .head(10)
    )

    for row in (
        train_source
        .itertuples(
            index=False
        )
    ):

        print(
            f"{row.source:24s} "
            f"n={row.n:4d} "
            f"clickbait_rate="
            f"{row.clickbait_rate:.4f}"
        )

    print()
    print("=" * 80)
    print(
        "TRAIN/VALIDATION DRIFT"
    )
    print("=" * 80)

    print()

    print(
        f"Clickbait-rate delta : "
        f"{validation_clickbait_rate - train_clickbait_rate:+.6f}"
    )

    print(
        f"Title median delta   : "
        f"{validation_title_median - train_title_median:+.1f} words"
    )

    print(
        f"Lead median delta    : "
        f"{validation_lead_median - train_lead_median:+.1f} words"
    )

    print()
    print(
        "TEST WAS NOT LOADED."
    )

    print()
    print(
        "Outputs:"
    )

    print(
        f"  {REPORT_PATH}"
    )

    print(
        f"  {SPLIT_SUMMARY_PATH}"
    )

    print(
        f"  {LENGTH_STATS_PATH}"
    )

    print(
        f"  {FEATURE_STATS_PATH}"
    )

    print(
        f"  {CATEGORY_STATS_PATH}"
    )

    print(
        f"  {SOURCE_STATS_PATH}"
    )

    print(
        f"  {TRAIN_CUE_STATS_PATH}"
    )

    print(
        f"  {EXTREME_EXAMPLES_PATH}"
    )

    print()
    print(
        "STEP 04 FINISHED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()