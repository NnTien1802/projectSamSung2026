from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ============================================================
# PATHS
# ============================================================

MODULE_ROOT = Path(__file__).resolve().parents[2]

ARTICLES_DIR = (
    MODULE_ROOT
    / "data"
    / "interim"
    / "consistency"
    / "articles_v1"
)

ARTICLE_INDEX_PATH = (
    ARTICLES_DIR
    / "article_index.parquet"
)

OUTPUT_DIR = (
    MODULE_ROOT
    / "data"
    / "interim"
    / "consistency"
    / "positive_pairs_v1"
)

OUTPUT_FILES = {
    "train":
        OUTPUT_DIR
        / "train.parquet",

    "validation":
        OUTPUT_DIR
        / "validation.parquet",

    "test":
        OUTPUT_DIR
        / "test.parquet",
}

REPORT_PATH = (
    OUTPUT_DIR
    / "build_report.json"
)


# ============================================================
# CONFIG
# ============================================================

VALID_SPLITS = [
    "train",
    "validation",
    "test",
]

LABEL_CONSISTENT = 1

PAIR_TYPE = "original_positive"

DIFFICULTY = "positive"


# ============================================================
# OUTPUT SCHEMA
# ============================================================

PAIR_SCHEMA = pa.schema(
    [
        (
            "pair_id",
            pa.string(),
        ),

        (
            "split",
            pa.string(),
        ),

        (
            "anchor_article_uid",
            pa.string(),
        ),

        (
            "headline_article_uid",
            pa.string(),
        ),

        (
            "label",
            pa.int8(),
        ),

        (
            "pair_type",
            pa.string(),
        ),

        (
            "difficulty",
            pa.string(),
        ),

        (
            "anchor_category",
            pa.string(),
        ),

        (
            "headline_category",
            pa.string(),
        ),

        (
            "same_category",
            pa.bool_(),
        ),

        (
            "anchor_content_hash",
            pa.string(),
        ),

        (
            "headline_title_hash",
            pa.string(),
        ),

        (
            "semantic_similarity",
            pa.float32(),
        ),

        (
            "source_dataset",
            pa.string(),
        ),
    ]
)


# ============================================================
# HELPERS
# ============================================================

def stable_pair_id(
    anchor_article_uid,
    headline_article_uid,
    label,
):
    """
    Create deterministic pair ID.
    """

    raw = (
        f"{anchor_article_uid}"
        f"\u241f"
        f"{headline_article_uid}"
        f"\u241f"
        f"{label}"
    )

    digest = hashlib.blake2b(
        raw.encode("utf-8"),
        digest_size=12,
    ).hexdigest()

    return (
        f"pair_{digest}"
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
# OUTPUT SAFETY
# ============================================================

def prepare_output_directory(
    overwrite=False,
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated_files = (
        list(
            OUTPUT_FILES.values()
        )
        +
        [
            REPORT_PATH
        ]
    )

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

        text = "\n".join(
            str(path)
            for path in existing
        )

        raise RuntimeError(
            "Output files already exist:\n"
            f"{text}\n\n"
            "Use --overwrite if you "
            "intentionally want to "
            "regenerate Step 04."
        )

    if overwrite:

        for path in existing:
            path.unlink()


# ============================================================
# VALIDATE INPUT
# ============================================================

def validate_input():

    print()
    print("=" * 80)
    print(
        "VALIDATE ARTICLE INDEX"
    )
    print("=" * 80)

    if not ARTICLE_INDEX_PATH.exists():

        raise FileNotFoundError(
            ARTICLE_INDEX_PATH
        )

    parquet_file = pq.ParquetFile(
        ARTICLE_INDEX_PATH
    )

    columns = (
        parquet_file
        .schema_arrow
        .names
    )

    required = [
        "article_uid",
        "dataset",
        "split",
        "category",
        "title_hash",
        "content_hash",
    ]

    missing = [
        column
        for column in required
        if column not in columns
    ]

    if missing:

        raise ValueError(
            "Article index missing "
            f"required columns: "
            f"{missing}"
        )

    print(
        f"Input:\n"
        f"{ARTICLE_INDEX_PATH}"
    )

    print()
    print(
        f"Rows: "
        f"{parquet_file.metadata.num_rows:,}"
    )

    print(
        "Input validation: PASSED"
    )


# ============================================================
# BUILD POSITIVE PAIRS
# ============================================================

def build_positive_pairs():

    print()
    print("=" * 80)
    print(
        "BUILD ORIGINAL POSITIVE PAIRS"
    )
    print("=" * 80)

    # --------------------------------------------------------
    # Only read metadata needed to construct pair definitions.
    # We do NOT read full article content here.
    # --------------------------------------------------------

    columns = [
        "article_uid",
        "dataset",
        "split",
        "category",
        "title_hash",
        "content_hash",
    ]

    table = pq.read_table(
        ARTICLE_INDEX_PATH,
        columns=columns,
    )

    df = table.to_pandas()

    print()
    print(
        f"Article index rows loaded: "
        f"{len(df):,}"
    )

    # ========================================================
    # BASIC INPUT VALIDATION
    # ========================================================

    if df[
        "article_uid"
    ].isna().any():

        raise ValueError(
            "Missing article_uid found."
        )

    if df[
        "article_uid"
    ].duplicated().any():

        duplicate_count = int(
            df[
                "article_uid"
            ]
            .duplicated()
            .sum()
        )

        raise ValueError(
            "Duplicate article_uid found: "
            f"{duplicate_count:,}"
        )

    unknown_splits = sorted(
        set(
            df[
                "split"
            ].unique()
        )
        -
        set(
            VALID_SPLITS
        )
    )

    if unknown_splits:

        raise ValueError(
            "Unknown split values: "
            f"{unknown_splits}"
        )

    # ========================================================
    # OUTPUT COUNTS
    # ========================================================

    counts = Counter()

    pair_ids_global = set()

    split_pair_ids = {
        split: set()
        for split
        in VALID_SPLITS
    }

    # ========================================================
    # PROCESS EACH SPLIT INDEPENDENTLY
    # ========================================================

    for split in VALID_SPLITS:

        print()
        print(
            f"Building {split}..."
        )

        split_df = (
            df[
                df[
                    "split"
                ]
                == split
            ]
            .copy()
        )

        records = []

        for row in split_df.itertuples(
            index=False,
        ):

            article_uid = (
                row.article_uid
            )

            pair_id = stable_pair_id(
                anchor_article_uid=(
                    article_uid
                ),
                headline_article_uid=(
                    article_uid
                ),
                label=(
                    LABEL_CONSISTENT
                ),
            )

            if pair_id in pair_ids_global:

                raise RuntimeError(
                    "Duplicate pair_id "
                    f"generated: "
                    f"{pair_id}"
                )

            pair_ids_global.add(
                pair_id
            )

            split_pair_ids[
                split
            ].add(
                pair_id
            )

            records.append(
                {
                    "pair_id":
                        pair_id,

                    "split":
                        split,

                    "anchor_article_uid":
                        article_uid,

                    "headline_article_uid":
                        article_uid,

                    "label":
                        LABEL_CONSISTENT,

                    "pair_type":
                        PAIR_TYPE,

                    "difficulty":
                        DIFFICULTY,

                    "anchor_category":
                        row.category,

                    "headline_category":
                        row.category,

                    "same_category":
                        True,

                    "anchor_content_hash":
                        row.content_hash,

                    "headline_title_hash":
                        row.title_hash,

                    "semantic_similarity":
                        None,

                    "source_dataset":
                        row.dataset,
                }
            )

        output_table = (
            pa.Table.from_pylist(
                records,
                schema=PAIR_SCHEMA,
            )
        )

        pq.write_table(
            output_table,
            OUTPUT_FILES[
                split
            ],
            compression="zstd",
            use_dictionary=True,
        )

        counts[
            split
        ] = len(
            records
        )

        print(
            f"  Positive pairs: "
            f"{len(records):,}"
        )

    return (
        counts,
        split_pair_ids,
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_outputs(
    counts,
    split_pair_ids,
):

    print()
    print("=" * 80)
    print(
        "FINAL STRUCTURAL VALIDATION"
    )
    print("=" * 80)

    passed = True

    # ========================================================
    # EXPECTED ROW COUNTS
    # ========================================================

    index_table = pq.read_table(
        ARTICLE_INDEX_PATH,
        columns=[
            "split"
        ],
    )

    index_df = (
        index_table
        .to_pandas()
    )

    expected_counts = (
        index_df[
            "split"
        ]
        .value_counts()
        .to_dict()
    )

    print()
    print(
        "Pair counts:"
    )

    for split in VALID_SPLITS:

        expected = int(
            expected_counts.get(
                split,
                0,
            )
        )

        actual = int(
            counts[
                split
            ]
        )

        print(
            f"  {split:12s}: "
            f"{actual:,} "
            f"(expected {expected:,})"
        )

        if actual != expected:
            passed = False

    # ========================================================
    # CROSS-SPLIT PAIR IDs
    # ========================================================

    print()
    print(
        "Cross-split pair overlap:"
    )

    for i in range(
        len(
            VALID_SPLITS
        )
    ):

        for j in range(
            i + 1,
            len(
                VALID_SPLITS
            ),
        ):

            split_a = (
                VALID_SPLITS[i]
            )

            split_b = (
                VALID_SPLITS[j]
            )

            overlap = len(
                split_pair_ids[
                    split_a
                ]
                &
                split_pair_ids[
                    split_b
                ]
            )

            print(
                f"  {split_a} ∩ "
                f"{split_b}: "
                f"{overlap:,}"
            )

            if overlap != 0:
                passed = False

    # ========================================================
    # READ BACK EACH OUTPUT
    # ========================================================

    print()
    print(
        "Read-back validation:"
    )

    for split in VALID_SPLITS:

        path = (
            OUTPUT_FILES[
                split
            ]
        )

        table = pq.read_table(
            path
        )

        frame = table.to_pandas()

        # ----------------------------------------------------
        # All positives.
        # ----------------------------------------------------

        labels_ok = bool(
            (
                frame[
                    "label"
                ]
                == LABEL_CONSISTENT
            ).all()
        )

        # ----------------------------------------------------
        # Positive pair must link article to itself.
        # ----------------------------------------------------

        self_pair_ok = bool(
            (
                frame[
                    "anchor_article_uid"
                ]
                ==
                frame[
                    "headline_article_uid"
                ]
            ).all()
        )

        split_ok = bool(
            (
                frame[
                    "split"
                ]
                == split
            ).all()
        )

        unique_pair_ids = bool(
            ~frame[
                "pair_id"
            ]
            .duplicated()
            .any()
        )

        print(
            f"  {split:12s}: "
            f"rows={len(frame):,}, "
            f"label_ok={labels_ok}, "
            f"self_pair_ok={self_pair_ok}, "
            f"split_ok={split_ok}, "
            f"unique_ids={unique_pair_ids}"
        )

        if not all(
            [
                labels_ok,
                self_pair_ok,
                split_ok,
                unique_pair_ids,
            ]
        ):
            passed = False

    print()

    if passed:

        print(
            "STRUCTURAL VALIDATION: PASSED"
        )

    else:

        print(
            "STRUCTURAL VALIDATION: FAILED"
        )

    return (
        passed,
        expected_counts,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Build positive headline-article "
            "pairs from prepared VietOnlineNews."
        )
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Overwrite existing Step 04 "
            "output files."
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
        "STEP 04 - BUILD POSITIVE PAIRS"
    )
    print("=" * 80)

    print()
    print(
        f"Module root:\n"
        f"{MODULE_ROOT}"
    )

    # ========================================================
    # PREPARE
    # ========================================================

    validate_input()

    prepare_output_directory(
        overwrite=args.overwrite
    )

    # ========================================================
    # BUILD
    # ========================================================

    (
        counts,
        split_pair_ids,
    ) = build_positive_pairs()

    # ========================================================
    # VALIDATE
    # ========================================================

    (
        validation_passed,
        expected_counts,
    ) = validate_outputs(
        counts=counts,
        split_pair_ids=(
            split_pair_ids
        ),
    )

    # ========================================================
    # REPORT
    # ========================================================

    report = {
        "module":
            "module2_headline_integrity",

        "task":
            "headline_consistency",

        "stage":
            "04_build_positive_pairs",

        "definition": {
            "positive_label":
                LABEL_CONSISTENT,

            "positive_meaning":
                (
                    "original headline "
                    "paired with its own "
                    "original article"
                ),

            "pair_type":
                PAIR_TYPE,

            "full_article_content_copied":
                False,

            "pair_storage":
                (
                    "reference-based via "
                    "article_uid"
                ),
        },

        "counts": {
            split:
                int(
                    counts[
                        split
                    ]
                )

            for split
            in VALID_SPLITS
        },

        "expected_counts": {
            str(key):
                int(value)

            for key, value
            in expected_counts.items()
        },

        "total_positive_pairs":
            int(
                sum(
                    counts.values()
                )
            ),

        "outputs": {
            split:
                str(
                    path.relative_to(
                        MODULE_ROOT
                    )
                )

            for split, path
            in OUTPUT_FILES.items()
        },

        "validation_passed":
            bool(
                validation_passed
            ),
    }

    save_json(
        report,
        REPORT_PATH,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print(
        "STEP 04 SUMMARY"
    )
    print("=" * 80)

    print()
    print(
        "Positive pairs:"
    )

    for split in VALID_SPLITS:

        print(
            f"  {split:12s}: "
            f"{counts[split]:,}"
        )

    print(
        f"  total       : "
        f"{sum(counts.values()):,}"
    )

    print()
    print(
        "Output files:"
    )

    for split in VALID_SPLITS:

        path = (
            OUTPUT_FILES[
                split
            ]
        )

        print(
            f"  {path.name:25s} "
            f"{human_size(path.stat().st_size)}"
        )

    print()

    print(
        f"Report:\n"
        f"{REPORT_PATH}"
    )

    print()

    if validation_passed:

        print(
            "STEP 04 FINISHED SUCCESSFULLY"
        )

    else:

        raise RuntimeError(
            "Positive pair dataset "
            "failed structural validation."
        )


if __name__ == "__main__":
    main()