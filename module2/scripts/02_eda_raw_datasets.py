from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import json
import re
import unicodedata

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

MODULE_ROOT = Path(__file__).resolve().parents[1]

RAW_ROOT = (
    MODULE_ROOT
    / "data"
    / "raw"
)

VICLICKBAIT_DIR = (
    RAW_ROOT
    / "clickbait"
    / "viclickbait_2025"
)

VIANLI_DIR = (
    RAW_ROOT
    / "consistency"
    / "vianli"
)

VIET_ONLINE_NEWS_DIR = (
    RAW_ROOT
    / "consistency"
    / "viet_online_news"
)

EDA_RESULT_DIR = (
    MODULE_ROOT
    / "results"
    / "eda"
)

EDA_RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_PATH = (
    EDA_RESULT_DIR
    / "raw_dataset_eda.json"
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def is_missing(value):
    """
    Check whether a scalar value is missing / blank.
    """

    if value is None:
        return True

    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass

    if isinstance(value, str):
        return not value.strip()

    return False


def normalize_text(value):
    """
    Normalize text ONLY for EDA / duplicate analysis.

    Raw files are never modified.
    """

    if is_missing(value):
        return ""

    if not isinstance(value, str):
        value = str(value)

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


def text_hash(value):
    """
    Deterministic hash for normalized text.
    """

    text = normalize_text(
        value
    )

    if not text:
        return None

    return hashlib.blake2b(
        text.encode("utf-8"),
        digest_size=16,
    ).hexdigest()


def pair_hash(a, b):
    """
    Hash two normalized text fields together.
    """

    a = normalize_text(a)
    b = normalize_text(b)

    if not a and not b:
        return None

    combined = (
        a
        + "\u241f"
        + b
    )

    return hashlib.blake2b(
        combined.encode("utf-8"),
        digest_size=16,
    ).hexdigest()


def find_column(
    columns,
    candidates,
):
    """
    Find a column using case-insensitive matching.
    """

    lookup = {
        str(column).lower(): column
        for column in columns
    }

    for candidate in candidates:

        candidate_lower = (
            candidate.lower()
        )

        if candidate_lower in lookup:

            return lookup[
                candidate_lower
            ]

    return None


def length_statistics(values):
    """
    Basic descriptive statistics.
    """

    if not values:

        return {
            "count": 0,
        }

    arr = np.asarray(
        values,
        dtype=np.int64,
    )

    return {
        "count":
            int(len(arr)),

        "min":
            int(arr.min()),

        "p25":
            float(
                np.percentile(
                    arr,
                    25,
                )
            ),

        "median":
            float(
                np.percentile(
                    arr,
                    50,
                )
            ),

        "mean":
            float(
                arr.mean()
            ),

        "p75":
            float(
                np.percentile(
                    arr,
                    75,
                )
            ),

        "p90":
            float(
                np.percentile(
                    arr,
                    90,
                )
            ),

        "p95":
            float(
                np.percentile(
                    arr,
                    95,
                )
            ),

        "p99":
            float(
                np.percentile(
                    arr,
                    99,
                )
            ),

        "max":
            int(arr.max()),
    }


def print_length_stats(
    name,
    stats,
):
    """
    Print compact length statistics.
    """

    if stats.get(
        "count",
        0,
    ) == 0:

        print(
            f"{name}: no data"
        )

        return

    print(
        f"{name}: "
        f"median={stats['median']:.0f}, "
        f"mean={stats['mean']:.1f}, "
        f"p95={stats['p95']:.0f}, "
        f"max={stats['max']}"
    )


def safe_counter_dict(counter):
    """
    Make Counter JSON serializable.
    """

    return {
        str(key): int(value)
        for key, value
        in counter.items()
    }


def save_json(
    data,
    path,
):
    """
    Save JSON report.
    """

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


def infer_split_from_path(path):
    """
    Infer train / validation / test from filename/path.
    """

    text = str(path).lower()

    if "train" in text:
        return "train"

    if (
        "validation" in text
        or "valid" in text
        or "dev" in text
    ):
        return "validation"

    if "test" in text:
        return "test"

    return "unknown"


# ============================================================
# 1. VICLICKBAIT-2025
# ============================================================

def analyze_viclickbait():

    print()
    print("=" * 80)
    print(
        "EDA 1/3 - ViClickbait-2025"
    )
    print("=" * 80)

    csv_path = (
        VICLICKBAIT_DIR
        / "clickbait_dataset_vietnamese.csv"
    )

    jsonl_path = (
        VICLICKBAIT_DIR
        / "clickbait_dataset_vietnamese.jsonl"
    )

    if not csv_path.exists():

        raise FileNotFoundError(
            csv_path
        )

    # ========================================================
    # LOAD CSV
    # ========================================================

    df = pd.read_csv(
        csv_path,
        encoding="utf-8-sig",
        low_memory=False,
    )

    print(
        f"CSV records : "
        f"{len(df):,}"
    )

    print(
        f"Columns     : "
        f"{list(df.columns)}"
    )

    # ========================================================
    # CHECK JSONL COUNT
    # ========================================================

    jsonl_count = None

    if jsonl_path.exists():

        with jsonl_path.open(
            "r",
            encoding="utf-8",
        ) as f:

            jsonl_count = sum(
                1
                for line in f
                if line.strip()
            )

        print(
            f"JSONL records: "
            f"{jsonl_count:,}"
        )

    # ========================================================
    # DETECT COLUMNS
    # ========================================================

    title_col = find_column(
        df.columns,
        [
            "title",
            "headline",
        ],
    )

    lead_col = find_column(
        df.columns,
        [
            "lead_paragraph",
            "lead",
            "sapo",
            "description",
            "abstract",
        ],
    )

    label_col = find_column(
        df.columns,
        [
            "label",
            "final_label",
            "clickbait",
            "is_clickbait",
        ],
    )

    url_col = find_column(
        df.columns,
        [
            "url",
            "link",
        ],
    )

    source_col = find_column(
        df.columns,
        [
            "source",
            "publisher",
        ],
    )

    category_col = find_column(
        df.columns,
        [
            "category",
            "topic",
        ],
    )

    print()
    print(
        "Detected columns:"
    )

    print(
        f"  title    : {title_col}"
    )

    print(
        f"  lead     : {lead_col}"
    )

    print(
        f"  label    : {label_col}"
    )

    print(
        f"  url      : {url_col}"
    )

    print(
        f"  source   : {source_col}"
    )

    print(
        f"  category : {category_col}"
    )

    # ========================================================
    # MISSING VALUES
    # ========================================================

    missing = {}

    for column in df.columns:

        count = sum(
            is_missing(value)
            for value in df[column]
        )

        missing[
            str(column)
        ] = int(count)

    print()
    print(
        "Missing / blank:"
    )

    found_missing = False

    for column, count in (
        missing.items()
    ):

        if count > 0:

            found_missing = True

            print(
                f"  {column}: "
                f"{count:,} "
                f"({count / len(df):.2%})"
            )

    if not found_missing:

        print(
            "  None"
        )

    # ========================================================
    # LABEL DISTRIBUTION
    # ========================================================

    label_distribution = {}

    if label_col is not None:

        counts = (
            df[label_col]
            .value_counts(
                dropna=False
            )
        )

        print()
        print(
            "Label distribution:"
        )

        for label, count in (
            counts.items()
        ):

            print(
                f"  {label}: "
                f"{count:,} "
                f"({count / len(df):.2%})"
            )

            label_distribution[
                str(label)
            ] = int(count)

        counts.rename(
            "count"
        ).to_csv(
            EDA_RESULT_DIR
            / (
                "viclickbait_"
                "label_distribution.csv"
            ),
            encoding="utf-8-sig",
        )

    # ========================================================
    # SOURCE DISTRIBUTION
    # ========================================================

    source_distribution = {}

    if source_col is not None:

        counts = (
            df[source_col]
            .value_counts(
                dropna=False
            )
        )

        print()
        print(
            "Top sources:"
        )

        for source, count in (
            counts.head(20).items()
        ):

            print(
                f"  {source}: "
                f"{count:,}"
            )

        source_distribution = {
            str(key): int(value)
            for key, value
            in counts.items()
        }

        counts.rename(
            "count"
        ).to_csv(
            EDA_RESULT_DIR
            / (
                "viclickbait_"
                "source_distribution.csv"
            ),
            encoding="utf-8-sig",
        )

    # ========================================================
    # CATEGORY DISTRIBUTION
    # ========================================================

    category_distribution = {}

    if category_col is not None:

        counts = (
            df[category_col]
            .value_counts(
                dropna=False
            )
        )

        category_distribution = {
            str(key): int(value)
            for key, value
            in counts.items()
        }

        counts.rename(
            "count"
        ).to_csv(
            EDA_RESULT_DIR
            / (
                "viclickbait_"
                "category_distribution.csv"
            ),
            encoding="utf-8-sig",
        )

    # ========================================================
    # LENGTHS
    # ========================================================

    title_char_lengths = []
    title_word_lengths = []

    lead_char_lengths = []
    lead_word_lengths = []

    if title_col is not None:

        for value in df[
            title_col
        ]:

            text = normalize_text(
                value
            )

            if not text:
                continue

            title_char_lengths.append(
                len(text)
            )

            title_word_lengths.append(
                len(
                    text.split()
                )
            )

    if lead_col is not None:

        for value in df[
            lead_col
        ]:

            text = normalize_text(
                value
            )

            if not text:
                continue

            lead_char_lengths.append(
                len(text)
            )

            lead_word_lengths.append(
                len(
                    text.split()
                )
            )

    title_char_stats = (
        length_statistics(
            title_char_lengths
        )
    )

    title_word_stats = (
        length_statistics(
            title_word_lengths
        )
    )

    lead_char_stats = (
        length_statistics(
            lead_char_lengths
        )
    )

    lead_word_stats = (
        length_statistics(
            lead_word_lengths
        )
    )

    print()

    print_length_stats(
        "Title chars",
        title_char_stats,
    )

    print_length_stats(
        "Title words",
        title_word_stats,
    )

    print_length_stats(
        "Lead chars",
        lead_char_stats,
    )

    print_length_stats(
        "Lead words",
        lead_word_stats,
    )

    # ========================================================
    # DUPLICATE ANALYSIS
    # ========================================================

    duplicate_info = {}

    if title_col is not None:

        normalized_titles = (
            df[title_col]
            .map(
                normalize_text
            )
        )

        valid = (
            normalized_titles
            != ""
        )

        duplicate_titles = int(
            normalized_titles[
                valid
            ]
            .duplicated()
            .sum()
        )

        duplicate_info[
            "normalized_title_duplicates_beyond_first"
        ] = duplicate_titles

        print()
        print(
            "Normalized title duplicates "
            f"(beyond first): "
            f"{duplicate_titles:,}"
        )

    if url_col is not None:

        normalized_urls = (
            df[url_col]
            .map(
                normalize_text
            )
        )

        valid = (
            normalized_urls
            != ""
        )

        duplicate_urls = int(
            normalized_urls[
                valid
            ]
            .duplicated()
            .sum()
        )

        duplicate_info[
            "url_duplicates_beyond_first"
        ] = duplicate_urls

        print(
            "URL duplicates "
            f"(beyond first): "
            f"{duplicate_urls:,}"
        )

    if (
        title_col is not None
        and lead_col is not None
    ):

        hashes = []

        for title, lead in zip(
            df[title_col],
            df[lead_col],
        ):

            hashes.append(
                pair_hash(
                    title,
                    lead,
                )
            )

        hash_series = pd.Series(
            hashes,
            dtype="object",
        )

        valid = (
            hash_series.notna()
        )

        duplicate_pairs = int(
            hash_series[
                valid
            ]
            .duplicated()
            .sum()
        )

        duplicate_info[
            "title_lead_duplicates_beyond_first"
        ] = duplicate_pairs

        print(
            "Title+lead duplicates "
            f"(beyond first): "
            f"{duplicate_pairs:,}"
        )

    # ========================================================
    # BUILD REPORT
    # ========================================================

    report = {
        "records_csv":
            int(len(df)),

        "records_jsonl":
            (
                int(jsonl_count)
                if jsonl_count is not None
                else None
            ),

        "columns":
            [
                str(column)
                for column
                in df.columns
            ],

        "detected_columns": {
            "title":
                title_col,

            "lead":
                lead_col,

            "label":
                label_col,

            "url":
                url_col,

            "source":
                source_col,

            "category":
                category_col,
        },

        "missing":
            missing,

        "label_distribution":
            label_distribution,

        "source_distribution":
            source_distribution,

        "category_distribution":
            category_distribution,

        "lengths": {
            "title_chars":
                title_char_stats,

            "title_words":
                title_word_stats,

            "lead_chars":
                lead_char_stats,

            "lead_words":
                lead_word_stats,
        },

        "duplicates":
            duplicate_info,
    }

    save_json(
        report,
        EDA_RESULT_DIR
        / "viclickbait_eda.json",
    )

    return report


# ============================================================
# 2. ViANLI
# ============================================================

def analyze_vianli():

    print()
    print("=" * 80)
    print(
        "EDA 2/3 - ViANLI"
    )
    print("=" * 80)

    jsonl_files = sorted(
        VIANLI_DIR.rglob(
            "*.jsonl"
        )
    )

    if not jsonl_files:

        raise FileNotFoundError(
            "No ViANLI JSONL files found."
        )

    frames = []
    split_summary = []

    print(
        f"JSONL files found: "
        f"{len(jsonl_files)}"
    )

    # ========================================================
    # LOAD SPLITS
    # ========================================================

    for path in jsonl_files:

        df = pd.read_json(
            path,
            lines=True,
        )

        split = (
            infer_split_from_path(
                path
            )
        )

        df[
            "__split"
        ] = split

        frames.append(
            df
        )

        split_summary.append(
            {
                "split":
                    split,

                "file":
                    path.name,

                "records":
                    int(len(df)),
            }
        )

        print(
            f"  {split:12s}: "
            f"{len(df):,} "
            f"({path.name})"
        )

    combined = pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )

    display_columns = [
        column
        for column in combined.columns
        if column != "__split"
    ]

    print()
    print(
        f"Total records: "
        f"{len(combined):,}"
    )

    print(
        f"Columns: "
        f"{display_columns}"
    )

    # ========================================================
    # DETECT COLUMNS
    # ========================================================

    premise_col = find_column(
        combined.columns,
        [
            "premise",
            "context",
            "sentence1",
            "text_a",
        ],
    )

    hypothesis_col = find_column(
        combined.columns,
        [
            "hypothesis",
            "sentence2",
            "text_b",
        ],
    )

    label_col = find_column(
        combined.columns,
        [
            "label",
            "gold_label",
            "relation",
        ],
    )

    print()
    print(
        "Detected columns:"
    )

    print(
        f"  premise    : "
        f"{premise_col}"
    )

    print(
        f"  hypothesis : "
        f"{hypothesis_col}"
    )

    print(
        f"  label      : "
        f"{label_col}"
    )

    # ========================================================
    # LABEL DISTRIBUTION
    # ========================================================

    label_distribution = {}

    split_label_distribution = {}

    if label_col is not None:

        counts = (
            combined[label_col]
            .value_counts(
                dropna=False
            )
        )

        print()
        print(
            "Overall label distribution:"
        )

        for label, count in (
            counts.items()
        ):

            print(
                f"  {label}: "
                f"{count:,} "
                f"({count / len(combined):.2%})"
            )

            label_distribution[
                str(label)
            ] = int(count)

        cross = pd.crosstab(
            combined[
                "__split"
            ],
            combined[
                label_col
            ],
            dropna=False,
        )

        print()
        print(
            "Label distribution by split:"
        )

        print(
            cross
        )

        cross.to_csv(
            EDA_RESULT_DIR
            / "vianli_label_by_split.csv",
            encoding="utf-8-sig",
        )

        for split_name, row in (
            cross.iterrows()
        ):

            split_label_distribution[
                str(split_name)
            ] = {
                str(column):
                    int(value)

                for column, value
                in row.items()
            }

    # ========================================================
    # MISSING
    # ========================================================

    missing = {}

    for column in combined.columns:

        if column == "__split":
            continue

        count = sum(
            is_missing(value)
            for value
            in combined[column]
        )

        missing[
            str(column)
        ] = int(count)

    # ========================================================
    # LENGTHS
    # ========================================================

    premise_chars = []
    premise_words = []

    hypothesis_chars = []
    hypothesis_words = []

    if premise_col is not None:

        for value in combined[
            premise_col
        ]:

            text = normalize_text(
                value
            )

            if not text:
                continue

            premise_chars.append(
                len(text)
            )

            premise_words.append(
                len(
                    text.split()
                )
            )

    if hypothesis_col is not None:

        for value in combined[
            hypothesis_col
        ]:

            text = normalize_text(
                value
            )

            if not text:
                continue

            hypothesis_chars.append(
                len(text)
            )

            hypothesis_words.append(
                len(
                    text.split()
                )
            )

    premise_char_stats = (
        length_statistics(
            premise_chars
        )
    )

    premise_word_stats = (
        length_statistics(
            premise_words
        )
    )

    hypothesis_char_stats = (
        length_statistics(
            hypothesis_chars
        )
    )

    hypothesis_word_stats = (
        length_statistics(
            hypothesis_words
        )
    )

    print()

    print_length_stats(
        "Premise chars",
        premise_char_stats,
    )

    print_length_stats(
        "Premise words",
        premise_word_stats,
    )

    print_length_stats(
        "Hypothesis chars",
        hypothesis_char_stats,
    )

    print_length_stats(
        "Hypothesis words",
        hypothesis_word_stats,
    )

        # ========================================================
    # DUPLICATE + SPLIT LEAKAGE
    # ========================================================

    pair_sets = defaultdict(
        set
    )

    all_pair_hashes = set()

    duplicate_pairs = 0

    if (
        premise_col is not None
        and hypothesis_col is not None
    ):

        # ----------------------------------------------------
        # Use positional tuples instead of namedtuples.
        #
        # Reason:
        # pandas.itertuples() renames invalid namedtuple fields.
        # "__split" therefore may no longer be called
        # "__split" inside row._asdict().
        # ----------------------------------------------------

        premise_index = (
            combined.columns.get_loc(
                premise_col
            )
        )

        hypothesis_index = (
            combined.columns.get_loc(
                hypothesis_col
            )
        )

        split_index = (
            combined.columns.get_loc(
                "__split"
            )
        )

        for row in combined.itertuples(
            index=False,
            name=None,
        ):

            premise_value = row[
                premise_index
            ]

            hypothesis_value = row[
                hypothesis_index
            ]

            split_value = row[
                split_index
            ]

            h = pair_hash(
                premise_value,
                hypothesis_value,
            )

            if h is None:
                continue

            if h in all_pair_hashes:

                duplicate_pairs += 1

            else:

                all_pair_hashes.add(
                    h
                )

            pair_sets[
                split_value
            ].add(
                h
            )

    print()
    print(
        "Duplicate premise+hypothesis "
        f"pairs beyond first: "
        f"{duplicate_pairs:,}"
    )

    leakage = {}

    split_names = sorted(
        pair_sets.keys()
    )

    print()
    print(
        "Cross-split pair overlap:"
    )

    for i in range(
        len(split_names)
    ):

        for j in range(
            i + 1,
            len(split_names),
        ):

            split_a = (
                split_names[i]
            )

            split_b = (
                split_names[j]
            )

            overlap = len(
                pair_sets[
                    split_a
                ]
                &
                pair_sets[
                    split_b
                ]
            )

            key = (
                f"{split_a}__"
                f"{split_b}"
            )

            leakage[
                key
            ] = int(
                overlap
            )

            print(
                f"  {split_a} ∩ "
                f"{split_b}: "
                f"{overlap:,}"
            )

    # ========================================================
    # BUILD REPORT
    # ========================================================

    report = {
        "records":
            int(
                len(combined)
            ),

        "splits":
            split_summary,

        "columns":
            [
                str(column)
                for column
                in display_columns
            ],

        "detected_columns": {
            "premise":
                premise_col,

            "hypothesis":
                hypothesis_col,

            "label":
                label_col,
        },

        "missing":
            missing,

        "label_distribution":
            label_distribution,

        "label_by_split":
            split_label_distribution,

        "lengths": {
            "premise_chars":
                premise_char_stats,

            "premise_words":
                premise_word_stats,

            "hypothesis_chars":
                hypothesis_char_stats,

            "hypothesis_words":
                hypothesis_word_stats,
        },

        "duplicates": {
            "pair_duplicates_beyond_first":
                int(
                    duplicate_pairs
                ),

            "cross_split_overlap":
                leakage,
        },
    }

    save_json(
        report,
        EDA_RESULT_DIR
        / "vianli_eda.json",
    )

    return report


# ============================================================
# 3. VIET ONLINE NEWS
# ============================================================

def analyze_viet_online_news():

    print()
    print("=" * 80)
    print(
        "EDA 3/3 - VietOnlineNews"
    )
    print("=" * 80)

    # ========================================================
    # FIND CSV FILES
    # ========================================================

    all_csv_files = sorted(
        [
            path
            for path in (
                VIET_ONLINE_NEWS_DIR
                .rglob("*")
            )
            if (
                path.is_file()
                and
                path.suffix.lower()
                == ".csv"
            )
        ]
    )

    if not all_csv_files:

        print()
        print(
            "Files currently present:"
        )

        for path in (
            VIET_ONLINE_NEWS_DIR
            .rglob("*")
        ):

            if path.is_file():

                print(
                    "  "
                    + str(
                        path.relative_to(
                            VIET_ONLINE_NEWS_DIR
                        )
                    )
                )

        raise FileNotFoundError(
            "No CSV files found in "
            "VietOnlineNews repository."
        )

    print(
        f"CSV files found: "
        f"{len(all_csv_files)}"
    )

    print()

    for path in all_csv_files:

        print(
            "  "
            + str(
                path.relative_to(
                    VIET_ONLINE_NEWS_DIR
                )
            )
        )

    # ========================================================
    # SELECT SPLIT FILES
    # ========================================================

    split_files = []

    for path in all_csv_files:

        split = infer_split_from_path(
            path
        )

        if split != "unknown":

            split_files.append(
                (
                    path,
                    split,
                )
            )

    # --------------------------------------------------------
    # If filenames do not contain train/dev/test,
    # analyze all CSV files.
    # --------------------------------------------------------

    if not split_files:

        print()
        print(
            "WARNING: No CSV filename "
            "was recognized as a split."
        )

        print(
            "All CSV files will be analyzed."
        )

        split_files = [
            (
                path,
                "unknown",
            )
            for path
            in all_csv_files
        ]

    print()
    print(
        "Dataset files selected:"
    )

    for path, split in split_files:

        print(
            f"  {split:12s} "
            f"{path.name}"
        )

    # ========================================================
    # READ HEADER
    # ========================================================

    first_path = (
        split_files[0][0]
    )

    header_df = pd.read_csv(
        first_path,
        nrows=0,
        encoding="utf-8-sig",
    )

    columns = list(
        header_df.columns
    )

    print()
    print(
        f"Columns: "
        f"{columns}"
    )

    # ========================================================
    # DETECT COLUMNS
    # ========================================================

    id_col = find_column(
        columns,
        [
            "id",
            "article_id",
        ],
    )

    title_col = find_column(
        columns,
        [
            "title",
            "headline",
        ],
    )

    description_col = find_column(
        columns,
        [
            "description",
            "abstract",
            "summary",
            "sapo",
        ],
    )

    content_col = find_column(
        columns,
        [
            "content",
            "article",
            "text",
            "body",
        ],
    )

    category_col = find_column(
        columns,
        [
            "category",
            "topic",
            "label",
        ],
    )

    print()
    print(
        "Detected columns:"
    )

    print(
        f"  id          : "
        f"{id_col}"
    )

    print(
        f"  title       : "
        f"{title_col}"
    )

    print(
        f"  description : "
        f"{description_col}"
    )

    print(
        f"  content     : "
        f"{content_col}"
    )

    print(
        f"  category    : "
        f"{category_col}"
    )

    if title_col is None:

        raise ValueError(
            "VietOnlineNews has no "
            "detected title column."
        )

    if content_col is None:

        raise ValueError(
            "VietOnlineNews has no "
            "detected content column."
        )

    # ========================================================
    # SELECT REQUIRED COLUMNS
    # ========================================================

    selected_columns = []

    for column in [
        id_col,
        title_col,
        description_col,
        content_col,
        category_col,
    ]:

        if (
            column is not None
            and
            column not in selected_columns
        ):

            selected_columns.append(
                column
            )

    # ========================================================
    # COUNTERS
    # ========================================================

    total_records = 0

    split_counts = Counter()

    category_counts = Counter()

    missing_counts = Counter()

    # ========================================================
    # LENGTH ARRAYS
    # ========================================================

    title_char_lengths = []
    title_word_lengths = []

    description_char_lengths = []
    description_word_lengths = []

    content_char_lengths = []
    content_word_lengths = []

    # ========================================================
    # GLOBAL DUPLICATE SETS
    # ========================================================

    seen_ids = set()
    seen_titles = set()
    seen_contents = set()
    seen_pairs = set()

    duplicate_ids = 0
    duplicate_titles = 0
    duplicate_contents = 0
    duplicate_pairs = 0

    # ========================================================
    # SPLIT-SPECIFIC SETS FOR LEAKAGE
    # ========================================================

    split_title_hashes = defaultdict(
        set
    )

    split_content_hashes = defaultdict(
        set
    )

    split_pair_hashes = defaultdict(
        set
    )

    # ========================================================
    # STREAM EACH CSV FILE
    # ========================================================

    for file_index, (
        path,
        split,
    ) in enumerate(
        split_files,
        start=1,
    ):

        print()
        print(
            f"[{file_index}/"
            f"{len(split_files)}] "
            f"{path.name}"
        )

        # ----------------------------------------------------
        # Validate schema.
        # ----------------------------------------------------

        file_header = pd.read_csv(
            path,
            nrows=0,
            encoding="utf-8-sig",
        )

        file_columns = list(
            file_header.columns
        )

        missing_required = [
            column
            for column
            in selected_columns
            if column
            not in file_columns
        ]

        if missing_required:

            raise ValueError(
                f"{path.name} "
                f"is missing columns: "
                f"{missing_required}"
            )

        processed_this_file = 0

        # ----------------------------------------------------
        # Read only ~2048 records at a time.
        # ----------------------------------------------------

        reader = pd.read_csv(
            path,
            usecols=selected_columns,
            chunksize=2048,
            encoding="utf-8-sig",
            low_memory=False,
        )

        for chunk in reader:

            # ------------------------------------------------
            # Force exact column order so tuple indices below
            # are deterministic.
            # ------------------------------------------------

            chunk = chunk[
                selected_columns
            ]

            column_index = {
                column: index
                for index, column
                in enumerate(
                    chunk.columns
                )
            }

            batch_size = len(
                chunk
            )

            total_records += (
                batch_size
            )

            split_counts[
                split
            ] += (
                batch_size
            )

            processed_this_file += (
                batch_size
            )

            # ------------------------------------------------
            # Faster than iterrows().
            # ------------------------------------------------

            for row in (
                chunk.itertuples(
                    index=False,
                    name=None,
                )
            ):

                # ============================================
                # ID
                # ============================================

                if id_col is not None:

                    value = row[
                        column_index[
                            id_col
                        ]
                    ]

                    if is_missing(
                        value
                    ):

                        missing_counts[
                            id_col
                        ] += 1

                    else:

                        h = text_hash(
                            value
                        )

                        if h in seen_ids:

                            duplicate_ids += 1

                        else:

                            seen_ids.add(
                                h
                            )

                # ============================================
                # TITLE
                # ============================================

                title_value = row[
                    column_index[
                        title_col
                    ]
                ]

                title_norm = (
                    normalize_text(
                        title_value
                    )
                )

                title_h = None

                if not title_norm:

                    missing_counts[
                        title_col
                    ] += 1

                else:

                    title_char_lengths.append(
                        len(
                            title_norm
                        )
                    )

                    title_word_lengths.append(
                        len(
                            title_norm.split()
                        )
                    )

                    title_h = text_hash(
                        title_norm
                    )

                    if (
                        title_h
                        in seen_titles
                    ):

                        duplicate_titles += 1

                    else:

                        seen_titles.add(
                            title_h
                        )

                    split_title_hashes[
                        split
                    ].add(
                        title_h
                    )

                # ============================================
                # DESCRIPTION
                # ============================================

                if (
                    description_col
                    is not None
                ):

                    description_value = row[
                        column_index[
                            description_col
                        ]
                    ]

                    description_norm = (
                        normalize_text(
                            description_value
                        )
                    )

                    if not description_norm:

                        missing_counts[
                            description_col
                        ] += 1

                    else:

                        description_char_lengths.append(
                            len(
                                description_norm
                            )
                        )

                        description_word_lengths.append(
                            len(
                                description_norm
                                .split()
                            )
                        )

                # ============================================
                # CONTENT
                # ============================================

                content_value = row[
                    column_index[
                        content_col
                    ]
                ]

                content_norm = (
                    normalize_text(
                        content_value
                    )
                )

                content_h = None

                if not content_norm:

                    missing_counts[
                        content_col
                    ] += 1

                else:

                    content_char_lengths.append(
                        len(
                            content_norm
                        )
                    )

                    content_word_lengths.append(
                        len(
                            content_norm.split()
                        )
                    )

                    content_h = text_hash(
                        content_norm
                    )

                    if (
                        content_h
                        in seen_contents
                    ):

                        duplicate_contents += 1

                    else:

                        seen_contents.add(
                            content_h
                        )

                    split_content_hashes[
                        split
                    ].add(
                        content_h
                    )

                # ============================================
                # TITLE + CONTENT PAIR
                # ============================================

                if (
                    title_norm
                    and
                    content_norm
                ):

                    pair_h = pair_hash(
                        title_norm,
                        content_norm,
                    )

                    if (
                        pair_h
                        in seen_pairs
                    ):

                        duplicate_pairs += 1

                    else:

                        seen_pairs.add(
                            pair_h
                        )

                    split_pair_hashes[
                        split
                    ].add(
                        pair_h
                    )

                # ============================================
                # CATEGORY
                # ============================================

                if (
                    category_col
                    is not None
                ):

                    category_value = row[
                        column_index[
                            category_col
                        ]
                    ]

                    if is_missing(
                        category_value
                    ):

                        missing_counts[
                            category_col
                        ] += 1

                    else:

                        category_norm = (
                            normalize_text(
                                category_value
                            )
                        )

                        category_counts[
                            category_norm
                        ] += 1

        print(
            f"  Records: "
            f"{processed_this_file:,}"
        )

    # ========================================================
    # LENGTH STATISTICS
    # ========================================================

    title_char_stats = (
        length_statistics(
            title_char_lengths
        )
    )

    title_word_stats = (
        length_statistics(
            title_word_lengths
        )
    )

    description_char_stats = (
        length_statistics(
            description_char_lengths
        )
    )

    description_word_stats = (
        length_statistics(
            description_word_lengths
        )
    )

    content_char_stats = (
        length_statistics(
            content_char_lengths
        )
    )

    content_word_stats = (
        length_statistics(
            content_word_lengths
        )
    )

    # ========================================================
    # CROSS-SPLIT OVERLAP
    # ========================================================

    split_names = sorted(
        split_counts.keys()
    )

    title_overlap = {}
    content_overlap = {}
    pair_overlap = {}

    for i in range(
        len(split_names)
    ):

        for j in range(
            i + 1,
            len(split_names),
        ):

            split_a = (
                split_names[i]
            )

            split_b = (
                split_names[j]
            )

            key = (
                f"{split_a}__"
                f"{split_b}"
            )

            title_overlap[
                key
            ] = len(
                split_title_hashes[
                    split_a
                ]
                &
                split_title_hashes[
                    split_b
                ]
            )

            content_overlap[
                key
            ] = len(
                split_content_hashes[
                    split_a
                ]
                &
                split_content_hashes[
                    split_b
                ]
            )

            pair_overlap[
                key
            ] = len(
                split_pair_hashes[
                    split_a
                ]
                &
                split_pair_hashes[
                    split_b
                ]
            )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print()
    print("-" * 80)
    print(
        "VietOnlineNews summary"
    )
    print("-" * 80)

    print()
    print(
        f"Total records: "
        f"{total_records:,}"
    )

    print()
    print(
        "Split counts:"
    )

    for split, count in sorted(
        split_counts.items()
    ):

        print(
            f"  {split}: "
            f"{count:,}"
        )

    print()
    print(
        "Missing / blank:"
    )

    for column in (
        selected_columns
    ):

        count = int(
            missing_counts[
                column
            ]
        )

        percentage = (
            count / total_records
            if total_records > 0
            else 0
        )

        print(
            f"  {column}: "
            f"{count:,} "
            f"({percentage:.2%})"
        )

    print()
    print(
        "Duplicates beyond first:"
    )

    if id_col is not None:

        print(
            f"  ID: "
            f"{duplicate_ids:,}"
        )

    print(
        "  normalized title: "
        f"{duplicate_titles:,}"
    )

    print(
        "  normalized content: "
        f"{duplicate_contents:,}"
    )

    print(
        "  title+content pair: "
        f"{duplicate_pairs:,}"
    )

    print()
    print(
        "Cross-split overlap:"
    )

    if not title_overlap:

        print(
            "  No multiple recognized "
            "splits to compare."
        )

    else:

        for key in sorted(
            title_overlap.keys()
        ):

            print(
                f"  {key}"
            )

            print(
                "    title   : "
                f"{title_overlap[key]:,}"
            )

            print(
                "    content : "
                f"{content_overlap[key]:,}"
            )

            print(
                "    pair    : "
                f"{pair_overlap[key]:,}"
            )

    print()

    print_length_stats(
        "Title chars",
        title_char_stats,
    )

    print_length_stats(
        "Title words",
        title_word_stats,
    )

    print_length_stats(
        "Description chars",
        description_char_stats,
    )

    print_length_stats(
        "Description words",
        description_word_stats,
    )

    print_length_stats(
        "Content chars",
        content_char_stats,
    )

    print_length_stats(
        "Content words",
        content_word_stats,
    )

    print()
    print(
        "Categories:"
    )

    for category, count in (
        category_counts
        .most_common()
    ):

        percentage = (
            count / total_records
            if total_records > 0
            else 0
        )

        print(
            f"  {category}: "
            f"{count:,} "
            f"({percentage:.2%})"
        )

    # ========================================================
    # SAVE CATEGORY CSV
    # ========================================================

    if category_counts:

        category_df = pd.DataFrame(
            category_counts
            .most_common(),
            columns=[
                "category",
                "count",
            ],
        )

        category_df[
            "percentage"
        ] = (
            category_df[
                "count"
            ]
            / total_records
        )

        category_df.to_csv(
            EDA_RESULT_DIR
            / (
                "viet_online_news_"
                "category_distribution.csv"
            ),
            index=False,
            encoding="utf-8-sig",
        )

    # ========================================================
    # REPORT
    # ========================================================

    report = {
        "csv_files_found":
            int(
                len(
                    all_csv_files
                )
            ),

        "dataset_files_analyzed":
            [
                {
                    "file":
                        str(
                            path.relative_to(
                                VIET_ONLINE_NEWS_DIR
                            )
                        ),

                    "split":
                        split,
                }

                for path, split
                in split_files
            ],

        "records":
            int(
                total_records
            ),

        "columns":
            [
                str(column)
                for column
                in columns
            ],

        "detected_columns": {
            "id":
                id_col,

            "title":
                title_col,

            "description":
                description_col,

            "content":
                content_col,

            "category":
                category_col,
        },

        "split_counts":
            safe_counter_dict(
                split_counts
            ),

        "category_distribution":
            safe_counter_dict(
                category_counts
            ),

        "missing": {
            str(column):
                int(
                    missing_counts[
                        column
                    ]
                )

            for column
            in selected_columns
        },

        "duplicates": {
            "id_duplicates_beyond_first":
                int(
                    duplicate_ids
                ),

            "normalized_title_duplicates_beyond_first":
                int(
                    duplicate_titles
                ),

            "normalized_content_duplicates_beyond_first":
                int(
                    duplicate_contents
                ),

            "title_content_duplicates_beyond_first":
                int(
                    duplicate_pairs
                ),
        },

        "cross_split_overlap": {
            "title": {
                str(key):
                    int(value)

                for key, value
                in title_overlap.items()
            },

            "content": {
                str(key):
                    int(value)

                for key, value
                in content_overlap.items()
            },

            "title_content_pair": {
                str(key):
                    int(value)

                for key, value
                in pair_overlap.items()
            },
        },

        "lengths": {
            "title_chars":
                title_char_stats,

            "title_words":
                title_word_stats,

            "description_chars":
                description_char_stats,

            "description_words":
                description_word_stats,

            "content_chars":
                content_char_stats,

            "content_words":
                content_word_stats,
        },
    }

    save_json(
        report,
        EDA_RESULT_DIR
        / "viet_online_news_eda.json",
    )

    return report


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print(
        "MODULE 2 - HEADLINE INTEGRITY"
    )
    print(
        "STEP 02 - RAW DATA EDA"
    )
    print("=" * 80)

    print()
    print(
        "Module root:"
    )

    print(
        MODULE_ROOT
    )

    # ========================================================
    # 1. ViClickbait
    # ========================================================

    viclickbait_report = (
        analyze_viclickbait()
    )

    # ========================================================
    # 2. ViANLI
    # ========================================================

    vianli_report = (
        analyze_vianli()
    )

    # ========================================================
    # 3. VietOnlineNews
    # ========================================================

    viet_online_news_report = (
        analyze_viet_online_news()
    )

    # ========================================================
    # FINAL COMBINED REPORT
    # ========================================================

    final_report = {
        "module":
            "module2_headline_integrity",

        "stage":
            "raw_eda",

        "viclickbait_2025":
            viclickbait_report,

        "vianli":
            vianli_report,

        "viet_online_news":
            viet_online_news_report,
    }

    save_json(
        final_report,
        REPORT_PATH,
    )

    print()
    print("=" * 80)
    print(
        "EDA FINISHED"
    )
    print("=" * 80)

    print()
    print(
        "Main report:"
    )

    print(
        REPORT_PATH
    )

    print()
    print(
        "EDA output directory:"
    )

    print(
        EDA_RESULT_DIR
    )

    print()
    print(
        "Raw datasets were NOT modified."
    )


if __name__ == "__main__":
    main()