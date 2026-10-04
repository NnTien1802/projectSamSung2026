from pathlib import Path
import json
import re
import unicodedata

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

MODULE_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = (
    MODULE_ROOT
    / "data"
    / "raw"
    / "clickbait"
    / "viclickbait_2025"
)

OUTPUT_DIR = (
    MODULE_ROOT
    / "data"
    / "processed"
    / "clickbait"
    / "viclickbait_v1"
)

TRAIN_PATH = (
    OUTPUT_DIR
    / "train.parquet"
)

VALIDATION_PATH = (
    OUTPUT_DIR
    / "validation.parquet"
)

TEST_PATH = (
    OUTPUT_DIR
    / "test.parquet"
)

REPORT_PATH = (
    OUTPUT_DIR
    / "prepare_report.json"
)

DUPLICATE_REVIEW_PATH = (
    OUTPUT_DIR
    / "duplicate_title_groups.csv"
)


# ============================================================
# CONFIG
# ============================================================

SEED = 20260921

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


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


def normalize_text(
    value,
):

    if pd.isna(
        value
    ):

        return ""

    value = str(
        value
    )

    value = unicodedata.normalize(
        "NFC",
        value,
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def normalize_title_for_group(
    value,
):

    value = normalize_text(
        value
    )

    value = value.lower()

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


# ============================================================
# LOCATE RAW SOURCE
# ============================================================

def find_source_file():

    if not RAW_DIR.exists():

        raise FileNotFoundError(
            RAW_DIR
        )

    candidates = []

    for pattern in [
        "*.csv",
        "*.parquet",
        "*.jsonl",
        "*.json",
    ]:

        candidates.extend(
            RAW_DIR.rglob(
                pattern
            )
        )

    candidates = [
        path
        for path in candidates
        if path.is_file()
    ]

    if not candidates:

        raise FileNotFoundError(
            "No supported dataset file "
            f"found under:\n{RAW_DIR}"
        )

    print()
    print(
        "Candidate source files:"
    )

    for path in candidates:

        print(
            "  "
            + str(
                path.relative_to(
                    MODULE_ROOT
                )
            )
        )

    # --------------------------------------------------------
    # Prefer CSV because the raw ViClickbait repository
    # currently provides both CSV and JSONL versions
    # containing the same dataset.
    # --------------------------------------------------------

    suffix_priority = {
        ".csv":
            0,

        ".parquet":
            1,

        ".jsonl":
            2,

        ".json":
            3,
    }

    candidates.sort(
        key=lambda path: (
            suffix_priority.get(
                path.suffix.lower(),
                99,
            ),
            str(path),
        )
    )

    return candidates[
        0
    ]


# ============================================================
# LOAD RAW FILE
# ============================================================

def load_frame(
    path,
):

    suffix = (
        path.suffix
        .lower()
    )

    if suffix == ".csv":

        return pd.read_csv(
            path
        )

    if suffix == ".parquet":

        return pd.read_parquet(
            path
        )

    if suffix == ".jsonl":

        return pd.read_json(
            path,
            lines=True,
        )

    if suffix == ".json":

        try:

            return pd.read_json(
                path,
                lines=True,
            )

        except ValueError:

            return pd.read_json(
                path
            )

    raise RuntimeError(
        f"Unsupported source format: "
        f"{path}"
    )


# ============================================================
# COLUMN DETECTION
# ============================================================

def find_column(
    columns,
    candidates,
):

    lower_map = {
        str(column).strip().lower():
            column

        for column in columns
    }

    for candidate in candidates:

        key = (
            candidate
            .strip()
            .lower()
        )

        if key in lower_map:

            return lower_map[
                key
            ]

    return None


def detect_columns(
    frame,
):

    title_col = find_column(
        frame.columns,
        [
            "title",
            "headline",
            "news_title",
        ],
    )

    lead_col = find_column(
        frame.columns,
        [
            # ViClickbait-2025
            "lead_paragraph",

            "lead",
            "description",
            "summary",
            "abstract",
            "sapo",
            "lead_text",
        ],
    )

    label_col = find_column(
        frame.columns,
        [
            "label",
            "class",
            "target",
            "clickbait",
        ],
    )

    url_col = find_column(
        frame.columns,
        [
            "url",
            "link",
            "source_url",
        ],
    )

    category_col = find_column(
        frame.columns,
        [
            "category",
            "topic",
        ],
    )

    source_col = find_column(
        frame.columns,
        [
            "source",
            "publisher",
        ],
    )

    publish_col = find_column(
        frame.columns,
        [
            "publish_datetime",
            "published_at",
            "publish_time",
            "date",
        ],
    )

    raw_id_col = find_column(
        frame.columns,
        [
            "id",
            "guid",
            "article_id",
        ],
    )

    if title_col is None:

        raise RuntimeError(
            "Could not detect title column. "
            f"Available columns: "
            f"{list(frame.columns)}"
        )

    if label_col is None:

        raise RuntimeError(
            "Could not detect label column. "
            f"Available columns: "
            f"{list(frame.columns)}"
        )

    return {
        "raw_id":
            raw_id_col,

        "title":
            title_col,

        "lead":
            lead_col,

        "label":
            label_col,

        "url":
            url_col,

        "category":
            category_col,

        "source":
            source_col,

        "publish_datetime":
            publish_col,
    }


# ============================================================
# LABEL NORMALIZATION
# ============================================================

def normalize_label(
    value,
):

    if pd.isna(
        value
    ):

        raise ValueError(
            "Missing clickbait label."
        )

    # --------------------------------------------------------
    # Numeric label
    # --------------------------------------------------------

    if isinstance(
        value,
        (
            int,
            np.integer,
            float,
            np.floating,
        ),
    ):

        numeric = int(
            value
        )

        if numeric in {
            0,
            1,
        }:

            return numeric

    # --------------------------------------------------------
    # Text label
    # --------------------------------------------------------

    text = (
        str(value)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    normal_values = {
        "0",
        "normal",
        "non clickbait",
        "nonclickbait",
        "not clickbait",
        "no clickbait",
        "negative",
    }

    clickbait_values = {
        "1",
        "clickbait",
        "click bait",
        "positive",
    }

    if text in normal_values:

        return 0

    if text in clickbait_values:

        return 1

    raise ValueError(
        f"Unknown label value: "
        f"{value!r}"
    )


# ============================================================
# DUPLICATE TITLE ANALYSIS
# ============================================================

def inspect_duplicate_title_groups(
    frame,
):

    duplicated_mask = (
        frame[
            "_normalized_title"
        ]
        .duplicated(
            keep=False
        )
    )

    duplicate_rows = (
        frame[
            duplicated_mask
        ]
        .copy()
    )

    if duplicate_rows.empty:

        print()
        print(
            "Duplicate title groups: 0"
        )

        return {
            "duplicate_rows":
                0,

            "duplicate_groups":
                0,

            "conflicting_groups":
                0,

            "max_group_size":
                1,
        }

    duplicate_rows.sort_values(
        [
            "_normalized_title",
            "sample_id",
        ],
        inplace=True,
    )

    # Save for later inspection.
    review_columns = [
        "sample_id",
        "raw_id",
        "title",
        "lead",
        "url",
        "category",
        "source",
        "label",
        "label_name",
        "_normalized_title",
    ]

    review_columns = [
        column
        for column in review_columns
        if column
        in
        duplicate_rows.columns
    ]

    duplicate_rows[
        review_columns
    ].to_csv(
        DUPLICATE_REVIEW_PATH,
        index=False,
        encoding="utf-8-sig",
    )

    group_stats = (
        duplicate_rows
        .groupby(
            "_normalized_title"
        )
        .agg(
            group_size=(
                "sample_id",
                "size",
            ),

            label_nunique=(
                "label",
                "nunique",
            ),
        )
        .reset_index()
    )

    conflicting = (
        group_stats[
            group_stats[
                "label_nunique"
            ]
            >
            1
        ]
    )

    print()
    print(
        "Duplicate normalized titles:"
    )

    print(
        f"  duplicate rows  : "
        f"{len(duplicate_rows):,}"
    )

    print(
        f"  duplicate groups: "
        f"{len(group_stats):,}"
    )

    print(
        f"  largest group   : "
        f"{int(group_stats['group_size'].max()):,}"
    )

    print(
        f"  conflicting label groups: "
        f"{len(conflicting):,}"
    )

    print()
    print(
        "Duplicate records:"
    )

    print(
        duplicate_rows[
            review_columns
        ].to_string(
            index=False
        )
    )

    if not conflicting.empty:

        print()
        print(
            "Conflicting duplicate groups:"
        )

        print(
            conflicting.to_string(
                index=False
            )
        )

        raise RuntimeError(
            "At least one normalized-title "
            "group contains conflicting labels. "
            "Manual review is required."
        )

    print()
    print(
        "Duplicate title groups have "
        "consistent labels."
    )

    print(
        "They will be kept in the same split."
    )

    return {
        "duplicate_rows":
            int(
                len(
                    duplicate_rows
                )
            ),

        "duplicate_groups":
            int(
                len(
                    group_stats
                )
            ),

        "conflicting_groups":
            int(
                len(
                    conflicting
                )
            ),

        "max_group_size":
            int(
                group_stats[
                    "group_size"
                ].max()
            ),
    }


# ============================================================
# GROUPED STRATIFIED SPLIT
# ============================================================

def create_splits(
    frame,
):
    """
    Split at normalized-title group level.

    All rows with the same normalized title are forced into
    exactly one split.

    Stratification is performed using the label assigned to
    each title group.
    """

    # ========================================================
    # BUILD GROUP TABLE
    # ========================================================

    group_frame = (
        frame
        .groupby(
            "_normalized_title",
            sort=False,
        )
        .agg(
            group_size=(
                "sample_id",
                "size",
            ),

            label_nunique=(
                "label",
                "nunique",
            ),

            label=(
                "label",
                "first",
            ),
        )
        .reset_index()
    )

    conflicting = (
        group_frame[
            group_frame[
                "label_nunique"
            ]
            >
            1
        ]
    )

    if not conflicting.empty:

        raise RuntimeError(
            "Cannot split dataset because "
            "one or more title groups contain "
            "conflicting labels."
        )

    print()
    print("=" * 80)
    print(
        "TITLE GROUPS"
    )
    print("=" * 80)

    print()
    print(
        f"Rows             : "
        f"{len(frame):,}"
    )

    print(
        f"Normalized titles: "
        f"{len(group_frame):,}"
    )

    print(
        f"Multi-row groups : "
        f"{int((group_frame['group_size'] > 1).sum()):,}"
    )

    print(
        f"Largest group    : "
        f"{int(group_frame['group_size'].max()):,}"
    )

    group_label_counts = (
        group_frame[
            "label"
        ]
        .value_counts()
    )

    print()
    print(
        "Group-level labels:"
    )

    print(
        f"  NORMAL   : "
        f"{int(group_label_counts.get(0, 0)):,}"
    )

    print(
        f"  CLICKBAIT: "
        f"{int(group_label_counts.get(1, 0)):,}"
    )

    # ========================================================
    # TEST = 15% OF GROUPS
    # ========================================================

    (
        train_validation_groups,
        test_groups,
    ) = train_test_split(
        group_frame,
        test_size=TEST_RATIO,
        random_state=SEED,
        stratify=(
            group_frame[
                "label"
            ]
        ),
    )

    # ========================================================
    # VALIDATION = 15% TOTAL
    #
    # Remaining proportion = 85%.
    # validation / remaining = 15 / 85
    # ========================================================

    validation_fraction = (
        VALIDATION_RATIO
        /
        (
            TRAIN_RATIO
            +
            VALIDATION_RATIO
        )
    )

    (
        train_groups,
        validation_groups,
    ) = train_test_split(
        train_validation_groups,
        test_size=(
            validation_fraction
        ),
        random_state=SEED,
        stratify=(
            train_validation_groups[
                "label"
            ]
        ),
    )

    # ========================================================
    # GROUP SETS
    # ========================================================

    train_group_set = set(
        train_groups[
            "_normalized_title"
        ]
    )

    validation_group_set = set(
        validation_groups[
            "_normalized_title"
        ]
    )

    test_group_set = set(
        test_groups[
            "_normalized_title"
        ]
    )

    if (
        train_group_set
        &
        validation_group_set
    ):

        raise RuntimeError(
            "TRAIN/VALIDATION title-group leakage."
        )

    if (
        train_group_set
        &
        test_group_set
    ):

        raise RuntimeError(
            "TRAIN/TEST title-group leakage."
        )

    if (
        validation_group_set
        &
        test_group_set
    ):

        raise RuntimeError(
            "VALIDATION/TEST title-group leakage."
        )

    # ========================================================
    # MAP GROUPS BACK TO ROWS
    # ========================================================

    train = (
        frame[
            frame[
                "_normalized_title"
            ]
            .isin(
                train_group_set
            )
        ]
        .copy()
    )

    validation = (
        frame[
            frame[
                "_normalized_title"
            ]
            .isin(
                validation_group_set
            )
        ]
        .copy()
    )

    test = (
        frame[
            frame[
                "_normalized_title"
            ]
            .isin(
                test_group_set
            )
        ]
        .copy()
    )

    # ========================================================
    # REPRODUCIBLE ROW ORDER
    # ========================================================

    train = (
        train
        .sample(
            frac=1.0,
            random_state=SEED,
        )
        .reset_index(
            drop=True
        )
    )

    validation = (
        validation
        .sample(
            frac=1.0,
            random_state=(
                SEED
                +
                1
            ),
        )
        .reset_index(
            drop=True
        )
    )

    test = (
        test
        .sample(
            frac=1.0,
            random_state=(
                SEED
                +
                2
            ),
        )
        .reset_index(
            drop=True
        )
    )

    # ========================================================
    # TOTAL ROW COUNT
    # ========================================================

    if (
        len(
            train
        )
        +
        len(
            validation
        )
        +
        len(
            test
        )
        !=
        len(
            frame
        )
    ):

        raise RuntimeError(
            "Split row count mismatch."
        )

    return (
        train,
        validation,
        test,
    )


# ============================================================
# SPLIT VALIDATION
# ============================================================

def validate_split(
    name,
    frame,
):

    if (
        frame[
            "sample_id"
        ]
        .duplicated()
        .any()
    ):

        raise RuntimeError(
            f"{name}: duplicate sample_id."
        )

    labels = set(
        frame[
            "label"
        ].unique()
    )

    if not labels.issubset(
        {
            0,
            1,
        }
    ):

        raise RuntimeError(
            f"{name}: invalid labels: "
            f"{labels}"
        )

    if (
        frame[
            "title"
        ]
        .eq("")
        .any()
    ):

        raise RuntimeError(
            f"{name}: empty title."
        )


# ============================================================
# CROSS-SPLIT VALIDATION
# ============================================================

def validate_cross_split(
    train,
    validation,
    test,
):

    # --------------------------------------------------------
    # Sample IDs
    # --------------------------------------------------------

    train_ids = set(
        train[
            "sample_id"
        ]
    )

    validation_ids = set(
        validation[
            "sample_id"
        ]
    )

    test_ids = set(
        test[
            "sample_id"
        ]
    )

    sample_overlap = {
        "train_validation":
            len(
                train_ids
                &
                validation_ids
            ),

        "train_test":
            len(
                train_ids
                &
                test_ids
            ),

        "validation_test":
            len(
                validation_ids
                &
                test_ids
            ),
    }

    # --------------------------------------------------------
    # Normalized titles
    # --------------------------------------------------------

    train_titles = set(
        train[
            "_normalized_title"
        ]
    )

    validation_titles = set(
        validation[
            "_normalized_title"
        ]
    )

    test_titles = set(
        test[
            "_normalized_title"
        ]
    )

    title_overlap = {
        "train_validation":
            len(
                train_titles
                &
                validation_titles
            ),

        "train_test":
            len(
                train_titles
                &
                test_titles
            ),

        "validation_test":
            len(
                validation_titles
                &
                test_titles
            ),
    }

    # --------------------------------------------------------
    # URLs
    # --------------------------------------------------------

    def nonempty_urls(
        split_frame,
    ):

        return set(
            split_frame.loc[
                split_frame[
                    "url"
                ]
                .ne(""),
                "url",
            ]
        )

    train_urls = nonempty_urls(
        train
    )

    validation_urls = nonempty_urls(
        validation
    )

    test_urls = nonempty_urls(
        test
    )

    url_overlap = {
        "train_validation":
            len(
                train_urls
                &
                validation_urls
            ),

        "train_test":
            len(
                train_urls
                &
                test_urls
            ),

        "validation_test":
            len(
                validation_urls
                &
                test_urls
            ),
    }

    if any(
        value > 0
        for value in (
            sample_overlap.values()
        )
    ):

        raise RuntimeError(
            "Cross-split sample_id leakage."
        )

    if any(
        value > 0
        for value in (
            title_overlap.values()
        )
    ):

        raise RuntimeError(
            "Cross-split normalized-title leakage."
        )

    # URL overlap is expected to be zero according to
    # previous EDA. Abort if raw data has changed.
    if any(
        value > 0
        for value in (
            url_overlap.values()
        )
    ):

        raise RuntimeError(
            "Cross-split URL leakage."
        )

    return (
        sample_overlap,
        title_overlap,
        url_overlap,
    )


# ============================================================
# SPLIT STATISTICS
# ============================================================

def summarize_split(
    name,
    frame,
):

    counts = (
        frame[
            "label"
        ]
        .value_counts()
    )

    normal = int(
        counts.get(
            0,
            0,
        )
    )

    clickbait = int(
        counts.get(
            1,
            0,
        )
    )

    total = int(
        len(
            frame
        )
    )

    return {
        "split":
            name,

        "rows":
            total,

        "title_groups":
            int(
                frame[
                    "_normalized_title"
                ]
                .nunique()
            ),

        "normal":
            normal,

        "clickbait":
            clickbait,

        "normal_rate":
            float(
                normal
                /
                total
            ),

        "clickbait_rate":
            float(
                clickbait
                /
                total
            ),

        "empty_lead":
            int(
                frame[
                    "lead"
                ]
                .eq("")
                .sum()
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
        "TASK 2A - CLICKBAIT DETECTION"
    )
    print(
        "STEP 03 - PREPARE CLICKBAIT DATASET"
    )
    print("=" * 80)

    print()
    print(
        "Raw data remains unchanged."
    )

    print(
        "Splitting unit: normalized title group."
    )

    print(
        "Split policy: 70% train / "
        "15% validation / 15% test."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # LOAD SOURCE
    # ========================================================

    source_path = (
        find_source_file()
    )

    print()
    print(
        f"Selected source:\n"
        f"{source_path}"
    )

    raw = load_frame(
        source_path
    )

    print()
    print(
        f"Raw rows: "
        f"{len(raw):,}"
    )

    print(
        f"Columns : "
        f"{list(raw.columns)}"
    )

    # ========================================================
    # DETECT COLUMNS
    # ========================================================

    columns = detect_columns(
        raw
    )

    print()
    print(
        "Detected columns:"
    )

    for key, value in (
        columns.items()
    ):

        print(
            f"  {key:16s}: "
            f"{value}"
        )

    # ========================================================
    # STANDARDIZE
    # ========================================================

    frame = pd.DataFrame(
        index=raw.index
    )

    # Stable internal ID based on immutable raw row position.
    frame[
        "sample_id"
    ] = [
        f"vcb_{index:05d}"
        for index in range(
            len(
                raw
            )
        )
    ]

    # --------------------------------------------------------
    # Preserve original ID if available.
    # --------------------------------------------------------

    if (
        columns[
            "raw_id"
        ]
        is not None
    ):

        frame[
            "raw_id"
        ] = (
            raw[
                columns[
                    "raw_id"
                ]
            ]
            .map(
                normalize_text
            )
        )

    else:

        frame[
            "raw_id"
        ] = ""

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    frame[
        "title"
    ] = (
        raw[
            columns[
                "title"
            ]
        ]
        .map(
            normalize_text
        )
    )

    # --------------------------------------------------------
    # Lead
    # --------------------------------------------------------

    if (
        columns[
            "lead"
        ]
        is not None
    ):

        frame[
            "lead"
        ] = (
            raw[
                columns[
                    "lead"
                ]
            ]
            .map(
                normalize_text
            )
        )

    else:

        frame[
            "lead"
        ] = ""

    # --------------------------------------------------------
    # URL
    # --------------------------------------------------------

    if (
        columns[
            "url"
        ]
        is not None
    ):

        frame[
            "url"
        ] = (
            raw[
                columns[
                    "url"
                ]
            ]
            .map(
                normalize_text
            )
        )

    else:

        frame[
            "url"
        ] = ""

    # --------------------------------------------------------
    # Category
    # --------------------------------------------------------

    if (
        columns[
            "category"
        ]
        is not None
    ):

        frame[
            "category"
        ] = (
            raw[
                columns[
                    "category"
                ]
            ]
            .map(
                normalize_text
            )
        )

    else:

        frame[
            "category"
        ] = ""

    # --------------------------------------------------------
    # Source
    # --------------------------------------------------------

    if (
        columns[
            "source"
        ]
        is not None
    ):

        frame[
            "source"
        ] = (
            raw[
                columns[
                    "source"
                ]
            ]
            .map(
                normalize_text
            )
        )

    else:

        frame[
            "source"
        ] = ""

    # --------------------------------------------------------
    # Publish datetime
    # --------------------------------------------------------

    if (
        columns[
            "publish_datetime"
        ]
        is not None
    ):

        frame[
            "publish_datetime"
        ] = (
            raw[
                columns[
                    "publish_datetime"
                ]
            ]
            .map(
                normalize_text
            )
        )

    else:

        frame[
            "publish_datetime"
        ] = ""

    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    frame[
        "label"
    ] = (
        raw[
            columns[
                "label"
            ]
        ]
        .map(
            normalize_label
        )
        .astype(
            np.int8
        )
    )

    frame[
        "label_name"
    ] = np.where(
        frame[
            "label"
        ]
        ==
        1,
        "CLICKBAIT",
        "NORMAL",
    )

    # --------------------------------------------------------
    # Group key — never used as model input.
    # --------------------------------------------------------

    frame[
        "_normalized_title"
    ] = (
        frame[
            "title"
        ]
        .map(
            normalize_title_for_group
        )
    )

    # ========================================================
    # BASIC INTEGRITY
    # ========================================================

    empty_title = int(
        frame[
            "title"
        ]
        .eq("")
        .sum()
    )

    empty_lead = int(
        frame[
            "lead"
        ]
        .eq("")
        .sum()
    )

    duplicate_url_rows = int(
        (
            frame[
                "url"
            ]
            .ne("")
            &
            frame[
                "url"
            ]
            .duplicated(
                keep=False
            )
        )
        .sum()
    )

    print()
    print("=" * 80)
    print(
        "RAW INTEGRITY"
    )
    print("=" * 80)

    print()
    print(
        f"Empty title             : "
        f"{empty_title:,}"
    )

    print(
        f"Empty lead              : "
        f"{empty_lead:,}"
    )

    print(
        f"Duplicate non-empty URLs: "
        f"{duplicate_url_rows:,}"
    )

    if empty_title > 0:

        raise RuntimeError(
            "Dataset contains empty titles."
        )

    # ========================================================
    # DUPLICATE TITLE ANALYSIS
    # ========================================================

    duplicate_stats = (
        inspect_duplicate_title_groups(
            frame
        )
    )

    # ========================================================
    # LABEL DISTRIBUTION
    # ========================================================

    label_counts = (
        frame[
            "label"
        ]
        .value_counts()
    )

    normal_total = int(
        label_counts.get(
            0,
            0,
        )
    )

    clickbait_total = int(
        label_counts.get(
            1,
            0,
        )
    )

    print()
    print("=" * 80)
    print(
        "LABEL DISTRIBUTION"
    )
    print("=" * 80)

    print()
    print(
        f"NORMAL   : "
        f"{normal_total:,}"
    )

    print(
        f"CLICKBAIT: "
        f"{clickbait_total:,}"
    )

    print(
        f"Clickbait rate: "
        f"{frame['label'].mean():.4f}"
    )

    # Dataset version integrity.
    if (
        len(
            frame
        )
        !=
        3414
    ):

        print()
        print(
            "WARNING:"
        )

        print(
            "  Expected 3,414 rows from "
            "previous EDA, but current raw "
            f"dataset has {len(frame):,}."
        )

    # ========================================================
    # CREATE SPLITS
    # ========================================================

    (
        train,
        validation,
        test,
    ) = create_splits(
        frame
    )

    # ========================================================
    # VALIDATE INDIVIDUAL SPLITS
    # ========================================================

    validate_split(
        "train",
        train,
    )

    validate_split(
        "validation",
        validation,
    )

    validate_split(
        "test",
        test,
    )

    # ========================================================
    # CROSS-SPLIT VALIDATION
    # ========================================================

    (
        sample_overlap,
        title_overlap,
        url_overlap,
    ) = validate_cross_split(
        train=train,
        validation=validation,
        test=test,
    )

    # ========================================================
    # STATS
    # ========================================================

    train_stats = summarize_split(
        "train",
        train,
    )

    validation_stats = summarize_split(
        "validation",
        validation,
    )

    test_stats = summarize_split(
        "test",
        test,
    )

    split_rows = [
        train_stats,
        validation_stats,
        test_stats,
    ]

    # ========================================================
    # SAVE PROCESSED DATA
    # ========================================================

    output_columns = [
        "sample_id",
        "raw_id",
        "title",
        "lead",
        "url",
        "category",
        "source",
        "publish_datetime",
        "label",
        "label_name",
    ]

    train[
        output_columns
    ].to_parquet(
        TRAIN_PATH,
        index=False,
    )

    validation[
        output_columns
    ].to_parquet(
        VALIDATION_PATH,
        index=False,
    )

    test[
        output_columns
    ].to_parquet(
        TEST_PATH,
        index=False,
    )

    # ========================================================
    # REPORT
    # ========================================================

    report = {
        "stage":
            "03_prepare_clickbait_dataset",

        "task":
            "2A_clickbait_detection",

        "dataset":
            "viclickbait_v1",

        "source":
            str(
                source_path.relative_to(
                    MODULE_ROOT
                )
            ),

        "raw_data_modified":
            False,

        "seed":
            SEED,

        "raw_rows":
            int(
                len(
                    frame
                )
            ),

        "raw_label_distribution": {
            "normal":
                normal_total,

            "clickbait":
                clickbait_total,

            "clickbait_rate":
                float(
                    frame[
                        "label"
                    ].mean()
                ),
        },

        "missing": {
            "empty_title":
                empty_title,

            "empty_lead":
                empty_lead,
        },

        "duplicates": {
            "normalized_title":
                duplicate_stats,

            "duplicate_nonempty_url_rows":
                duplicate_url_rows,
        },

        "split_policy": {
            "train_target_ratio":
                TRAIN_RATIO,

            "validation_target_ratio":
                VALIDATION_RATIO,

            "test_target_ratio":
                TEST_RATIO,

            "split_unit":
                "normalized_title",

            "stratification":
                "clickbait label",

            "method":
                (
                    "stratified group split "
                    "by normalized title"
                ),
        },

        "splits": {
            row[
                "split"
            ]:
                row

            for row in split_rows
        },

        "cross_split_sample_overlap":
            sample_overlap,

        "cross_split_title_overlap":
            title_overlap,

        "cross_split_url_overlap":
            url_overlap,

        "test_policy": {
            "status":
                "LOCKED",

            "allowed_during_model_development":
                False,

            "usage":
                (
                    "Official final evaluation "
                    "after architecture and "
                    "threshold are frozen."
                ),
        },
    }

    save_json(
        report,
        REPORT_PATH,
    )

    # ========================================================
    # PRINT FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print(
        "FINAL SPLITS"
    )
    print("=" * 80)

    for row in split_rows:

        print()
        print(
            f"{row['split'].upper()}:"
        )

        print(
            f"  rows          : "
            f"{row['rows']:,}"
        )

        print(
            f"  title groups  : "
            f"{row['title_groups']:,}"
        )

        print(
            f"  NORMAL        : "
            f"{row['normal']:,}"
        )

        print(
            f"  CLICKBAIT     : "
            f"{row['clickbait']:,}"
        )

        print(
            f"  clickbait rate: "
            f"{row['clickbait_rate']:.4f}"
        )

        print(
            f"  empty lead    : "
            f"{row['empty_lead']:,}"
        )

    print()
    print("=" * 80)
    print(
        "CROSS-SPLIT VALIDATION"
    )
    print("=" * 80)

    print()
    print(
        "Sample ID overlap:"
    )

    print(
        f"  train/validation : "
        f"{sample_overlap['train_validation']:,}"
    )

    print(
        f"  train/test       : "
        f"{sample_overlap['train_test']:,}"
    )

    print(
        f"  validation/test  : "
        f"{sample_overlap['validation_test']:,}"
    )

    print()
    print(
        "Normalized title overlap:"
    )

    print(
        f"  train/validation : "
        f"{title_overlap['train_validation']:,}"
    )

    print(
        f"  train/test       : "
        f"{title_overlap['train_test']:,}"
    )

    print(
        f"  validation/test  : "
        f"{title_overlap['validation_test']:,}"
    )

    print()
    print(
        "URL overlap:"
    )

    print(
        f"  train/validation : "
        f"{url_overlap['train_validation']:,}"
    )

    print(
        f"  train/test       : "
        f"{url_overlap['train_test']:,}"
    )

    print(
        f"  validation/test  : "
        f"{url_overlap['validation_test']:,}"
    )

    print()
    print(
        "VALIDATION: PASSED"
    )

    print()
    print(
        f"Output directory:\n"
        f"{OUTPUT_DIR}"
    )

    print()
    print(
        f"Report:\n"
        f"{REPORT_PATH}"
    )

    if (
        duplicate_stats[
            "duplicate_rows"
        ]
        >
        0
    ):

        print()
        print(
            f"Duplicate review:\n"
            f"{DUPLICATE_REVIEW_PATH}"
        )

    print()
    print(
        "TEST IS NOW LOCKED."
    )

    print(
        "DO NOT USE TEST DURING "
        "MODEL DEVELOPMENT."
    )

    print()
    print(
        "STEP 03 FINISHED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()