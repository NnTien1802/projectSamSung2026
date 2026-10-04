from pathlib import Path
from collections import Counter, defaultdict
import argparse
import hashlib
import json
import re
import unicodedata

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================
# PATHS
# ============================================================

# File:
# module2_headline_integrity/scripts/consistency/03_prepare_articles.py
#
# parents[0] = consistency
# parents[1] = scripts
# parents[2] = module2_headline_integrity

MODULE_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = (
    MODULE_ROOT
    / "data"
    / "raw"
    / "consistency"
    / "viet_online_news"
    / "processed"
)

INTERIM_ROOT = (
    MODULE_ROOT
    / "data"
    / "interim"
    / "consistency"
)

OUTPUT_DIR = (
    INTERIM_ROOT
    / "articles_v1"
)

TRAIN_CSV = (
    RAW_DATA_DIR
    / "train.csv"
)

VALIDATION_CSV = (
    RAW_DATA_DIR
    / "dev.csv"
)

TEST_CSV = (
    RAW_DATA_DIR
    / "test.csv"
)

OUTPUT_TRAIN = (
    OUTPUT_DIR
    / "train.parquet"
)

OUTPUT_VALIDATION = (
    OUTPUT_DIR
    / "validation.parquet"
)

OUTPUT_TEST = (
    OUTPUT_DIR
    / "test.parquet"
)

OUTPUT_INDEX = (
    OUTPUT_DIR
    / "article_index.parquet"
)

OUTPUT_EXCLUDED = (
    OUTPUT_DIR
    / "excluded_cross_split_titles.csv"
)

OUTPUT_REPORT = (
    OUTPUT_DIR
    / "prepare_report.json"
)


# ============================================================
# CONFIG
# ============================================================

DATASET_NAME = "viet_online_news"

CHUNK_SIZE = 2048

SOURCE_COLUMNS = [
    "id",
    "title",
    "description",
    "content",
    "category",
]

SPLIT_FILES = {
    "train": TRAIN_CSV,
    "validation": VALIDATION_CSV,
    "test": TEST_CSV,
}

OUTPUT_SPLIT_FILES = {
    "train": OUTPUT_TRAIN,
    "validation": OUTPUT_VALIDATION,
    "test": OUTPUT_TEST,
}

# Bit masks used during first pass.
SPLIT_BITS = {
    "train": 1,
    "validation": 2,
    "test": 4,
}


# ============================================================
# PARQUET SCHEMAS
# ============================================================

ARTICLE_SCHEMA = pa.schema(
    [
        (
            "article_uid",
            pa.string(),
        ),
        (
            "dataset",
            pa.string(),
        ),
        (
            "original_id",
            pa.string(),
        ),
        (
            "split",
            pa.string(),
        ),
        (
            "title",
            pa.string(),
        ),
        (
            "description",
            pa.string(),
        ),
        (
            "content",
            pa.string(),
        ),
        (
            "category",
            pa.string(),
        ),
        (
            "title_hash",
            pa.string(),
        ),
        (
            "content_hash",
            pa.string(),
        ),
        (
            "title_char_count",
            pa.int32(),
        ),
        (
            "title_word_count",
            pa.int32(),
        ),
        (
            "description_char_count",
            pa.int32(),
        ),
        (
            "description_word_count",
            pa.int32(),
        ),
        (
            "content_char_count",
            pa.int32(),
        ),
        (
            "content_word_count",
            pa.int32(),
        ),
    ]
)


INDEX_SCHEMA = pa.schema(
    [
        (
            "article_uid",
            pa.string(),
        ),
        (
            "dataset",
            pa.string(),
        ),
        (
            "original_id",
            pa.string(),
        ),
        (
            "split",
            pa.string(),
        ),
        (
            "title",
            pa.string(),
        ),
        (
            "description",
            pa.string(),
        ),
        (
            "category",
            pa.string(),
        ),
        (
            "title_hash",
            pa.string(),
        ),
        (
            "content_hash",
            pa.string(),
        ),
        (
            "title_word_count",
            pa.int32(),
        ),
        (
            "description_word_count",
            pa.int32(),
        ),
        (
            "content_word_count",
            pa.int32(),
        ),
    ]
)


# ============================================================
# GENERAL HELPERS
# ============================================================

def is_missing(value):
    """
    Check whether a scalar value is missing.
    """

    if value is None:
        return True

    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass

    if isinstance(
        value,
        str,
    ):
        return not value.strip()

    return False


def normalize_text(value):
    """
    Normalize text for storage.

    - NFC Unicode
    - normalize line endings
    - collapse horizontal whitespace
    - preserve paragraph / line boundaries
    - trim excessive blank lines

    Raw data is never modified.
    """

    if is_missing(value):
        return ""

    if not isinstance(
        value,
        str,
    ):
        value = str(value)

    value = unicodedata.normalize(
        "NFC",
        value,
    )

    value = value.replace(
        "\r\n",
        "\n",
    )

    value = value.replace(
        "\r",
        "\n",
    )

    cleaned_lines = []

    for line in value.split(
        "\n"
    ):

        line = re.sub(
            r"[ \t\f\v]+",
            " ",
            line,
        )

        line = line.strip()

        cleaned_lines.append(
            line
        )

    value = "\n".join(
        cleaned_lines
    )

    value = re.sub(
        r"\n{3,}",
        "\n\n",
        value,
    )

    return value.strip()


def canonicalize_for_hash(value):
    """
    Canonical text representation used only for hashing.

    All whitespace differences become a single space.

    This matches the logic used during EDA.
    """

    value = normalize_text(
        value
    )

    if not value:
        return ""

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def hash_text(value):
    """
    BLAKE2b hash of canonical text.
    """

    canonical = (
        canonicalize_for_hash(
            value
        )
    )

    if not canonical:
        return None

    return hashlib.blake2b(
        canonical.encode(
            "utf-8"
        ),
        digest_size=16,
    ).hexdigest()


def word_count(value):
    """
    Count whitespace-separated tokens.
    """

    value = normalize_text(
        value
    )

    if not value:
        return 0

    return len(
        re.findall(
            r"\S+",
            value,
        )
    )


def safe_counter(counter):
    """
    Convert Counter to JSON-safe dict.
    """

    return {
        str(key):
            int(value)

        for key, value
        in counter.items()
    }


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


def mask_to_splits(mask):
    """
    Decode split bit mask.
    """

    result = []

    for split, bit in (
        SPLIT_BITS.items()
    ):

        if mask & bit:

            result.append(
                split
            )

    return result


def is_multi_split_mask(mask):
    """
    True when a title occurs in more than one split.
    """

    return (
        mask != 0
        and
        (mask & (mask - 1))
        != 0
    )


# ============================================================
# VALIDATE INPUT
# ============================================================

def validate_input_files():

    print()
    print("=" * 80)
    print("VALIDATE INPUT FILES")
    print("=" * 80)

    for split, path in (
        SPLIT_FILES.items()
    ):

        if not path.exists():

            raise FileNotFoundError(
                f"Missing {split} file: "
                f"{path}"
            )

        header = pd.read_csv(
            path,
            nrows=0,
            encoding="utf-8-sig",
        )

        missing_columns = [
            column
            for column
            in SOURCE_COLUMNS
            if column
            not in header.columns
        ]

        if missing_columns:

            raise ValueError(
                f"{path.name} missing columns: "
                f"{missing_columns}"
            )

        print(
            f"{split:12s}: "
            f"{path}"
        )

    print()
    print(
        "All input files are valid."
    )


# ============================================================
# OUTPUT SAFETY
# ============================================================

def prepare_output_directory(
    overwrite=False,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated_files = [
        OUTPUT_TRAIN,
        OUTPUT_VALIDATION,
        OUTPUT_TEST,
        OUTPUT_INDEX,
        OUTPUT_EXCLUDED,
        OUTPUT_REPORT,
    ]

    existing = [
        path
        for path in generated_files
        if path.exists()
    ]

    if (
        existing
        and
        not overwrite
    ):

        existing_text = "\n".join(
            str(path)
            for path in existing
        )

        raise RuntimeError(
            "Output files already exist:\n"
            f"{existing_text}\n\n"
            "Re-run with --overwrite "
            "if you intentionally want "
            "to regenerate Step 03."
        )

    if overwrite:

        for path in existing:

            path.unlink()

    print()
    print(
        f"Output directory:\n"
        f"{OUTPUT_DIR}"
    )


# ============================================================
# FIRST PASS
# DETECT TITLES THAT CROSS SPLITS
# ============================================================

def detect_cross_split_titles():

    print()
    print("=" * 80)
    print(
        "PASS 1 - DETECT CROSS-SPLIT TITLES"
    )
    print("=" * 80)

    # title_hash -> bit mask of splits
    title_split_mask = {}

    records_scanned = Counter()

    missing_titles = Counter()

    for split, path in (
        SPLIT_FILES.items()
    ):

        print()
        print(
            f"Scanning {split}: "
            f"{path.name}"
        )

        reader = pd.read_csv(
            path,
            usecols=[
                "title"
            ],
            chunksize=CHUNK_SIZE,
            encoding="utf-8-sig",
            low_memory=False,
        )

        for chunk in reader:

            records_scanned[
                split
            ] += len(chunk)

            for title in chunk[
                "title"
            ]:

                title_norm = (
                    normalize_text(
                        title
                    )
                )

                if not title_norm:

                    missing_titles[
                        split
                    ] += 1

                    continue

                title_hash = hash_text(
                    title_norm
                )

                old_mask = (
                    title_split_mask.get(
                        title_hash,
                        0,
                    )
                )

                title_split_mask[
                    title_hash
                ] = (
                    old_mask
                    |
                    SPLIT_BITS[
                        split
                    ]
                )

        print(
            f"  Records scanned: "
            f"{records_scanned[split]:,}"
        )

    overlap_hashes = {
        title_hash
        for title_hash, mask
        in title_split_mask.items()
        if is_multi_split_mask(
            mask
        )
    }

    print()
    print(
        f"Unique title hashes: "
        f"{len(title_split_mask):,}"
    )

    print(
        "Cross-split title groups: "
        f"{len(overlap_hashes):,}"
    )

    if any(
        missing_titles.values()
    ):

        print()
        print(
            "Missing titles:"
        )

        for split, count in (
            missing_titles.items()
        ):

            print(
                f"  {split}: "
                f"{count:,}"
            )

    return (
        title_split_mask,
        overlap_hashes,
        records_scanned,
        missing_titles,
    )


# ============================================================
# PARQUET WRITE HELPER
# ============================================================

def write_records(
    writer,
    records,
    schema,
):

    if not records:
        return

    table = pa.Table.from_pylist(
        records,
        schema=schema,
    )

    writer.write_table(
        table
    )


# ============================================================
# SECOND PASS
# CLEAN, HASH, FILTER, WRITE
# ============================================================

def prepare_articles(
    title_split_mask,
    overlap_hashes,
):

    print()
    print("=" * 80)
    print(
        "PASS 2 - PREPARE ARTICLE CORPUS"
    )
    print("=" * 80)

    retained_counts = Counter()

    excluded_counts = Counter()

    invalid_required_counts = (
        Counter()
    )

    category_counts = defaultdict(
        Counter
    )

    # --------------------------------------------------------
    # For validation.
    # --------------------------------------------------------

    title_hash_sets = defaultdict(
        set
    )

    content_hash_sets = defaultdict(
        set
    )

    global_content_hashes = set()

    duplicate_content_rows = 0

    # --------------------------------------------------------
    # Track excluded title groups.
    # Only a few hundred are expected.
    # --------------------------------------------------------

    excluded_group_counts = (
        defaultdict(
            Counter
        )
    )

    excluded_title_samples = {}

    # --------------------------------------------------------
    # Writers.
    # --------------------------------------------------------

    split_writers = {}

    index_writer = None

    try:

        for split, output_path in (
            OUTPUT_SPLIT_FILES.items()
        ):

            split_writers[
                split
            ] = pq.ParquetWriter(
                output_path,
                ARTICLE_SCHEMA,
                compression="zstd",
                use_dictionary=True,
            )

        index_writer = (
            pq.ParquetWriter(
                OUTPUT_INDEX,
                INDEX_SCHEMA,
                compression="zstd",
                use_dictionary=True,
            )
        )

        # ====================================================
        # PROCESS EACH OFFICIAL SPLIT
        # ====================================================

        for split, path in (
            SPLIT_FILES.items()
        ):

            print()
            print(
                f"Preparing {split}: "
                f"{path.name}"
            )

            reader = pd.read_csv(
                path,
                usecols=SOURCE_COLUMNS,
                chunksize=CHUNK_SIZE,
                encoding="utf-8-sig",
                low_memory=False,
            )

            scanned = 0

            for chunk in reader:

                chunk = chunk[
                    SOURCE_COLUMNS
                ]

                article_records = []

                index_records = []

                for row in chunk.itertuples(
                    index=False,
                    name=None,
                ):

                    scanned += 1

                    (
                        original_id,
                        title_raw,
                        description_raw,
                        content_raw,
                        category_raw,
                    ) = row

                    # ========================================
                    # NORMALIZE
                    # ========================================

                    title = normalize_text(
                        title_raw
                    )

                    description = (
                        normalize_text(
                            description_raw
                        )
                    )

                    content = (
                        normalize_text(
                            content_raw
                        )
                    )

                    category = (
                        normalize_text(
                            category_raw
                        )
                    )

                    original_id = (
                        normalize_text(
                            original_id
                        )
                    )

                    # ========================================
                    # REQUIRED FIELDS
                    # ========================================

                    if not title:

                        invalid_required_counts[
                            f"{split}:missing_title"
                        ] += 1

                        continue

                    if not content:

                        invalid_required_counts[
                            f"{split}:missing_content"
                        ] += 1

                        continue

                    # ========================================
                    # HASHES
                    # ========================================

                    title_hash = hash_text(
                        title
                    )

                    content_hash = hash_text(
                        content
                    )

                    # ========================================
                    # REMOVE CROSS-SPLIT TITLE GROUP
                    # ========================================

                    if (
                        title_hash
                        in overlap_hashes
                    ):

                        excluded_counts[
                            split
                        ] += 1

                        excluded_group_counts[
                            title_hash
                        ][
                            split
                        ] += 1

                        if (
                            title_hash
                            not in
                            excluded_title_samples
                        ):

                            excluded_title_samples[
                                title_hash
                            ] = title

                        continue

                    # ========================================
                    # CONTENT DUPLICATE CHECK
                    # ========================================

                    if (
                        content_hash
                        in
                        global_content_hashes
                    ):

                        duplicate_content_rows += 1

                    else:

                        global_content_hashes.add(
                            content_hash
                        )

                    # ========================================
                    # STABLE ARTICLE UID
                    # ========================================

                    article_uid = (
                        f"von_{content_hash}"
                    )

                    # ========================================
                    # LENGTH METADATA
                    # ========================================

                    title_chars = len(
                        title
                    )

                    title_words = word_count(
                        title
                    )

                    description_chars = len(
                        description
                    )

                    description_words = (
                        word_count(
                            description
                        )
                    )

                    content_chars = len(
                        content
                    )

                    content_words = word_count(
                        content
                    )

                    # ========================================
                    # ARTICLE RECORD
                    # ========================================

                    article_record = {
                        "article_uid":
                            article_uid,

                        "dataset":
                            DATASET_NAME,

                        "original_id":
                            original_id,

                        "split":
                            split,

                        "title":
                            title,

                        "description":
                            description,

                        "content":
                            content,

                        "category":
                            category,

                        "title_hash":
                            title_hash,

                        "content_hash":
                            content_hash,

                        "title_char_count":
                            title_chars,

                        "title_word_count":
                            title_words,

                        "description_char_count":
                            description_chars,

                        "description_word_count":
                            description_words,

                        "content_char_count":
                            content_chars,

                        "content_word_count":
                            content_words,
                    }

                    article_records.append(
                        article_record
                    )

                    # ========================================
                    # LIGHTWEIGHT INDEX
                    # No full article content.
                    # ========================================

                    index_records.append(
                        {
                            "article_uid":
                                article_uid,

                            "dataset":
                                DATASET_NAME,

                            "original_id":
                                original_id,

                            "split":
                                split,

                            "title":
                                title,

                            "description":
                                description,

                            "category":
                                category,

                            "title_hash":
                                title_hash,

                            "content_hash":
                                content_hash,

                            "title_word_count":
                                title_words,

                            "description_word_count":
                                description_words,

                            "content_word_count":
                                content_words,
                        }
                    )

                    # ========================================
                    # VALIDATION SETS
                    # ========================================

                    title_hash_sets[
                        split
                    ].add(
                        title_hash
                    )

                    content_hash_sets[
                        split
                    ].add(
                        content_hash
                    )

                    retained_counts[
                        split
                    ] += 1

                    category_counts[
                        split
                    ][
                        category
                    ] += 1

                # ============================================
                # WRITE CHUNK
                # ============================================

                write_records(
                    split_writers[
                        split
                    ],
                    article_records,
                    ARTICLE_SCHEMA,
                )

                write_records(
                    index_writer,
                    index_records,
                    INDEX_SCHEMA,
                )

            print(
                f"  Scanned : "
                f"{scanned:,}"
            )

            print(
                f"  Retained: "
                f"{retained_counts[split]:,}"
            )

            print(
                f"  Excluded: "
                f"{excluded_counts[split]:,}"
            )

    finally:

        for writer in (
            split_writers.values()
        ):

            writer.close()

        if index_writer is not None:

            index_writer.close()

    return {
        "retained_counts":
            retained_counts,

        "excluded_counts":
            excluded_counts,

        "invalid_required_counts":
            invalid_required_counts,

        "category_counts":
            category_counts,

        "title_hash_sets":
            title_hash_sets,

        "content_hash_sets":
            content_hash_sets,

        "duplicate_content_rows":
            duplicate_content_rows,

        "excluded_group_counts":
            excluded_group_counts,

        "excluded_title_samples":
            excluded_title_samples,
    }


# ============================================================
# SAVE EXCLUDED TITLE GROUPS
# ============================================================

def save_excluded_title_groups(
    title_split_mask,
    overlap_hashes,
    excluded_group_counts,
    excluded_title_samples,
):

    rows = []

    for title_hash in sorted(
        overlap_hashes
    ):

        mask = (
            title_split_mask[
                title_hash
            ]
        )

        splits = mask_to_splits(
            mask
        )

        counts = (
            excluded_group_counts[
                title_hash
            ]
        )

        rows.append(
            {
                "title_hash":
                    title_hash,

                "title":
                    excluded_title_samples.get(
                        title_hash,
                        "",
                    ),

                "splits":
                    "|".join(
                        splits
                    ),

                "train_count":
                    int(
                        counts.get(
                            "train",
                            0,
                        )
                    ),

                "validation_count":
                    int(
                        counts.get(
                            "validation",
                            0,
                        )
                    ),

                "test_count":
                    int(
                        counts.get(
                            "test",
                            0,
                        )
                    ),

                "total_rows":
                    int(
                        sum(
                            counts.values()
                        )
                    ),
            }
        )

    excluded_df = pd.DataFrame(
        rows
    )

    excluded_df.to_csv(
        OUTPUT_EXCLUDED,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print(
        "Excluded title groups saved:"
    )

    print(
        OUTPUT_EXCLUDED
    )


# ============================================================
# FINAL VALIDATION
# ============================================================

def validate_prepared_corpus(
    title_hash_sets,
    content_hash_sets,
    duplicate_content_rows,
):

    print()
    print("=" * 80)
    print(
        "FINAL STRUCTURAL VALIDATION"
    )
    print("=" * 80)

    split_names = [
        "train",
        "validation",
        "test",
    ]

    title_overlaps = {}

    content_overlaps = {}

    validation_passed = True

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

            title_overlap = len(
                title_hash_sets[
                    split_a
                ]
                &
                title_hash_sets[
                    split_b
                ]
            )

            content_overlap = len(
                content_hash_sets[
                    split_a
                ]
                &
                content_hash_sets[
                    split_b
                ]
            )

            title_overlaps[
                key
            ] = (
                title_overlap
            )

            content_overlaps[
                key
            ] = (
                content_overlap
            )

            print()
            print(
                f"{split_a} ∩ "
                f"{split_b}"
            )

            print(
                f"  title overlap   : "
                f"{title_overlap:,}"
            )

            print(
                f"  content overlap : "
                f"{content_overlap:,}"
            )

            if title_overlap != 0:

                validation_passed = (
                    False
                )

            if content_overlap != 0:

                validation_passed = (
                    False
                )

    print()
    print(
        "Duplicate content rows "
        f"inside prepared corpus: "
        f"{duplicate_content_rows:,}"
    )

    if duplicate_content_rows != 0:

        validation_passed = False

    print()

    if validation_passed:

        print(
            "STRUCTURAL VALIDATION: PASSED"
        )

    else:

        print(
            "STRUCTURAL VALIDATION: FAILED"
        )

    return (
        validation_passed,
        title_overlaps,
        content_overlaps,
    )


# ============================================================
# FILE SIZE HELPER
# ============================================================

def human_size(
    num_bytes,
):

    value = float(
        num_bytes
    )

    for unit in [
        "B",
        "KB",
        "MB",
        "GB",
        "TB",
    ]:

        if value < 1024:

            return (
                f"{value:.2f} "
                f"{unit}"
            )

        value /= 1024

    return (
        f"{value:.2f} PB"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Prepare VietOnlineNews "
            "for Module 2 headline consistency."
        )
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Overwrite existing Step 03 "
            "outputs."
        ),
    )

    args = parser.parse_args()

    print("=" * 80)
    print(
        "MODULE 2 - HEADLINE INTEGRITY"
    )
    print(
        "TASK 2B - HEADLINE CONSISTENCY"
    )
    print(
        "STEP 03 - PREPARE ARTICLES"
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
    # INPUT VALIDATION
    # ========================================================

    validate_input_files()

    # ========================================================
    # OUTPUT
    # ========================================================

    prepare_output_directory(
        overwrite=args.overwrite
    )

    # ========================================================
    # PASS 1
    # ========================================================

    (
        title_split_mask,
        overlap_hashes,
        records_scanned,
        missing_titles,
    ) = detect_cross_split_titles()

    # ========================================================
    # PASS 2
    # ========================================================

    preparation = (
        prepare_articles(
            title_split_mask=(
                title_split_mask
            ),
            overlap_hashes=(
                overlap_hashes
            ),
        )
    )

    # ========================================================
    # EXCLUSION REPORT
    # ========================================================

    save_excluded_title_groups(
        title_split_mask=(
            title_split_mask
        ),
        overlap_hashes=(
            overlap_hashes
        ),
        excluded_group_counts=(
            preparation[
                "excluded_group_counts"
            ]
        ),
        excluded_title_samples=(
            preparation[
                "excluded_title_samples"
            ]
        ),
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    (
        validation_passed,
        title_overlaps,
        content_overlaps,
    ) = validate_prepared_corpus(
        title_hash_sets=(
            preparation[
                "title_hash_sets"
            ]
        ),
        content_hash_sets=(
            preparation[
                "content_hash_sets"
            ]
        ),
        duplicate_content_rows=(
            preparation[
                "duplicate_content_rows"
            ]
        ),
    )

    # ========================================================
    # REPORT
    # ========================================================

    retained_counts = (
        preparation[
            "retained_counts"
        ]
    )

    excluded_counts = (
        preparation[
            "excluded_counts"
        ]
    )

    final_total = sum(
        retained_counts.values()
    )

    excluded_total = sum(
        excluded_counts.values()
    )

    report = {
        "module":
            "module2_headline_integrity",

        "task":
            "headline_consistency",

        "stage":
            "03_prepare_articles",

        "source_dataset":
            DATASET_NAME,

        "policy": {
            "preserve_official_splits":
                True,

            "normalize_unicode":
                "NFC",

            "cross_split_title_policy":
                (
                    "remove_all_rows_for_any_"
                    "normalized_title_hash_"
                    "appearing_in_multiple_splits"
                ),

            "within_split_duplicate_titles":
                (
                    "retained; later negative "
                    "mining must forbid same "
                    "title_hash pairs"
                ),

            "raw_data_modified":
                False,
        },

        "records_scanned":
            safe_counter(
                records_scanned
            ),

        "missing_titles_first_pass":
            safe_counter(
                missing_titles
            ),

        "cross_split_title_groups":
            int(
                len(
                    overlap_hashes
                )
            ),

        "excluded_records":
            safe_counter(
                excluded_counts
            ),

        "excluded_total":
            int(
                excluded_total
            ),

        "retained_records":
            safe_counter(
                retained_counts
            ),

        "retained_total":
            int(
                final_total
            ),

        "invalid_required_records":
            safe_counter(
                preparation[
                    "invalid_required_counts"
                ]
            ),

        "duplicate_content_rows":
            int(
                preparation[
                    "duplicate_content_rows"
                ]
            ),

        "final_cross_split_overlap": {
            "title":
                {
                    key:
                        int(value)

                    for key, value
                    in title_overlaps.items()
                },

            "content":
                {
                    key:
                        int(value)

                    for key, value
                    in content_overlaps.items()
                },
        },

        "category_distribution": {
            split:
                safe_counter(
                    counter
                )

            for split, counter
            in preparation[
                "category_counts"
            ].items()
        },

        "outputs": {
            "train":
                str(
                    OUTPUT_TRAIN
                    .relative_to(
                        MODULE_ROOT
                    )
                ),

            "validation":
                str(
                    OUTPUT_VALIDATION
                    .relative_to(
                        MODULE_ROOT
                    )
                ),

            "test":
                str(
                    OUTPUT_TEST
                    .relative_to(
                        MODULE_ROOT
                    )
                ),

            "article_index":
                str(
                    OUTPUT_INDEX
                    .relative_to(
                        MODULE_ROOT
                    )
                ),

            "excluded_titles":
                str(
                    OUTPUT_EXCLUDED
                    .relative_to(
                        MODULE_ROOT
                    )
                ),
        },

        "validation_passed":
            bool(
                validation_passed
            ),
    }

    save_json(
        report,
        OUTPUT_REPORT,
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print(
        "STEP 03 SUMMARY"
    )
    print("=" * 80)

    print()
    print(
        "Original records:"
    )

    for split in [
        "train",
        "validation",
        "test",
    ]:

        print(
            f"  {split:12s}: "
            f"{records_scanned[split]:,}"
        )

    print()
    print(
        "Excluded due to cross-split title:"
    )

    for split in [
        "train",
        "validation",
        "test",
    ]:

        print(
            f"  {split:12s}: "
            f"{excluded_counts[split]:,}"
        )

    print(
        f"  total       : "
        f"{excluded_total:,}"
    )

    print()
    print(
        "Final retained records:"
    )

    for split in [
        "train",
        "validation",
        "test",
    ]:

        print(
            f"  {split:12s}: "
            f"{retained_counts[split]:,}"
        )

    print(
        f"  total       : "
        f"{final_total:,}"
    )

    print()
    print(
        "Output files:"
    )

    for path in [
        OUTPUT_TRAIN,
        OUTPUT_VALIDATION,
        OUTPUT_TEST,
        OUTPUT_INDEX,
        OUTPUT_EXCLUDED,
        OUTPUT_REPORT,
    ]:

        if path.exists():

            print(
                f"  {path.name:35s} "
                f"{human_size(path.stat().st_size)}"
            )

    print()
    print(
        f"Report:\n"
        f"{OUTPUT_REPORT}"
    )

    print()

    if validation_passed:

        print(
            "STEP 03 FINISHED SUCCESSFULLY"
        )

    else:

        raise RuntimeError(
            "Prepared corpus failed "
            "structural validation."
        )


if __name__ == "__main__":
    main()