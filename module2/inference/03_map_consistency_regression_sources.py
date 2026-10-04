from pathlib import Path
import json

import pandas as pd
import pyarrow.parquet as pq


# ============================================================
# PATHS
# ============================================================

MODULE_ROOT = Path(__file__).resolve().parents[1]

RESULTS_ROOT = (
    MODULE_ROOT
    / "results"
    / "consistency"
)

INTERIM_ROOT = (
    MODULE_ROOT
    / "data"
    / "interim"
    / "consistency"
)

PROCESSED_ROOT = (
    MODULE_ROOT
    / "data"
    / "processed"
    / "consistency"
)


# ============================================================
# KNOWN PREDICTION ARTIFACTS
# ============================================================

PREDICTION_FILES = [
    (
        RESULTS_ROOT
        / "baseline_title_description_mbert"
        / "validation_predictions.parquet"
    ),
    (
        RESULTS_ROOT
        / "full_article_max_mil_mbert"
        / "validation_predictions.parquet"
    ),
    (
        RESULTS_ROOT
        / "description_mil_fusion_v1"
        / "holdout_predictions.parquet"
    ),
    (
        RESULTS_ROOT
        / "final_hybrid_v1"
        / "official_test_predictions.parquet"
    ),
]


# ============================================================
# CONFIG
# ============================================================

MAX_COLUMNS_PRINT = 80

TEXT_COLUMN_HINTS = [
    "title",
    "headline",
    "description",
    "lead",
    "abstract",
    "summary",
    "content",
    "article",
    "body",
    "text",
]

UID_COLUMN_HINTS = [
    "article_uid",
    "anchor_article_uid",
    "headline_article_uid",
    "pair_id",
]

PROBABILITY_HINTS = [
    "prob",
    "score",
    "logit",
    "prediction",
    "pred",
]


# ============================================================
# HELPERS
# ============================================================

def relative(path):

    try:

        return str(
            path.relative_to(
                MODULE_ROOT
            )
        )

    except ValueError:

        return str(
            path
        )


def parquet_info(path):

    parquet_file = pq.ParquetFile(
        path
    )

    metadata = parquet_file.metadata

    return {
        "rows":
            int(
                metadata.num_rows
            ),

        "columns":
            list(
                parquet_file.schema.names
            ),
    }


def csv_info(path):

    frame = pd.read_csv(
        path,
        nrows=3,
    )

    return {
        "columns":
            list(
                frame.columns
            ),
    }


def json_info(path):

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(
            f
        )

    if isinstance(
        data,
        dict,
    ):

        keys = list(
            data.keys()
        )

    elif isinstance(
        data,
        list,
    ):

        keys = [
            f"<list length={len(data)}>"
        ]

    else:

        keys = [
            type(
                data
            ).__name__
        ]

    return {
        "keys":
            keys,
    }


def find_matching_columns(
    columns,
    hints,
):

    matches = []

    for column in columns:

        lowered = str(
            column
        ).lower()

        if any(
            hint in lowered
            for hint in hints
        ):

            matches.append(
                column
            )

    return matches


def exact_text_candidates(
    columns,
):
    """
    Stronger check than Step 02A.

    We do NOT treat hashes or UIDs as text fields.
    """

    candidates = []

    for column in columns:

        lowered = str(
            column
        ).lower()

        if (
            "hash"
            in lowered
            or
            "uid"
            in lowered
            or
            lowered.endswith(
                "_id"
            )
        ):

            continue

        if lowered in {
            "title",
            "headline",
            "description",
            "lead",
            "abstract",
            "summary",
            "content",
            "article",
            "body",
            "text",
            "article_text",
            "full_text",
            "full_article",
        }:

            candidates.append(
                column
            )

            continue

        if (
            lowered.endswith(
                "_title"
            )
            or
            lowered.endswith(
                "_description"
            )
            or
            lowered.endswith(
                "_content"
            )
            or
            lowered.endswith(
                "_text"
            )
        ):

            candidates.append(
                column
            )

    return candidates


def inspect_parquet_preview(
    path,
    columns,
):

    selected = []

    for column in columns:

        if column not in selected:

            selected.append(
                column
            )

    if not selected:

        return None

    selected = selected[
        :8
    ]

    try:

        frame = (
            pq.read_table(
                path,
                columns=selected,
            )
            .slice(
                0,
                2,
            )
            .to_pandas()
        )

    except Exception:

        return None

    preview = []

    for _, row in frame.iterrows():

        item = {}

        for column in selected:

            value = row[
                column
            ]

            if pd.isna(
                value
            ):

                item[
                    column
                ] = None

                continue

            value = str(
                value
            )

            if (
                len(
                    value
                )
                >
                180
            ):

                value = (
                    value[
                        :180
                    ]
                    +
                    "..."
                )

            item[
                column
            ] = value

        preview.append(
            item
        )

    return preview


# ============================================================
# PREDICTION ARTIFACTS
# ============================================================

def inspect_prediction_artifacts():

    print()
    print("=" * 80)
    print(
        "A. KNOWN CONSISTENCY PREDICTION ARTIFACTS"
    )
    print("=" * 80)

    found = []

    for path in PREDICTION_FILES:

        print()
        print(
            relative(
                path
            )
        )

        if not path.exists():

            print(
                "  exists: NO"
            )

            continue

        info = parquet_info(
            path
        )

        columns = info[
            "columns"
        ]

        probability_columns = (
            find_matching_columns(
                columns,
                PROBABILITY_HINTS,
            )
        )

        uid_columns = (
            find_matching_columns(
                columns,
                UID_COLUMN_HINTS,
            )
        )

        print(
            "  exists: YES"
        )

        print(
            f"  rows  : "
            f"{info['rows']:,}"
        )

        print(
            "  columns:"
        )

        for column in (
            columns[
                :MAX_COLUMNS_PRINT
            ]
        ):

            print(
                f"    - {column}"
            )

        print(
            "  probability-like:"
        )

        if probability_columns:

            for column in probability_columns:

                print(
                    f"    - {column}"
                )

        else:

            print(
                "    NONE"
            )

        print(
            "  identifier-like:"
        )

        if uid_columns:

            for column in uid_columns:

                print(
                    f"    - {column}"
                )

        else:

            print(
                "    NONE"
            )

        found.append(
            {
                "path":
                    path,

                "rows":
                    info[
                        "rows"
                    ],

                "columns":
                    columns,

                "probability_columns":
                    probability_columns,

                "uid_columns":
                    uid_columns,
            }
        )

    return found


# ============================================================
# INTERIM ARTIFACT DISCOVERY
# ============================================================

def inspect_interim_artifacts():

    print()
    print("=" * 80)
    print(
        "B. INTERIM CONSISTENCY ARTIFACTS"
    )
    print("=" * 80)

    print()
    print(
        f"Root: "
        f"{INTERIM_ROOT}"
    )

    if not INTERIM_ROOT.exists():

        print(
            "INTERIM ROOT DOES NOT EXIST."
        )

        return []

    parquet_files = sorted(
        INTERIM_ROOT.rglob(
            "*.parquet"
        )
    )

    print()
    print(
        f"Parquet files found: "
        f"{len(parquet_files):,}"
    )

    text_candidates = []

    for index, path in enumerate(
        parquet_files,
        start=1,
    ):

        try:

            info = parquet_info(
                path
            )

        except Exception as exc:

            print()
            print(
                f"[{index}] "
                f"{relative(path)}"
            )

            print(
                f"  ERROR: "
                f"{repr(exc)}"
            )

            continue

        columns = info[
            "columns"
        ]

        exact_text = (
            exact_text_candidates(
                columns
            )
        )

        uid_columns = (
            find_matching_columns(
                columns,
                UID_COLUMN_HINTS,
            )
        )

        print()
        print(
            f"[{index}] "
            f"{relative(path)}"
        )

        print(
            f"  rows: "
            f"{info['rows']:,}"
        )

        print(
            "  identifier columns:"
        )

        if uid_columns:

            for column in uid_columns:

                print(
                    f"    - {column}"
                )

        else:

            print(
                "    NONE"
            )

        print(
            "  real text candidates:"
        )

        if exact_text:

            for column in exact_text:

                print(
                    f"    - {column}"
                )

        else:

            print(
                "    NONE"
            )

        print(
            "  all columns:"
        )

        for column in (
            columns[
                :MAX_COLUMNS_PRINT
            ]
        ):

            print(
                f"    - {column}"
            )

        if exact_text:

            preview_columns = (
                uid_columns
                +
                exact_text
            )

            preview = (
                inspect_parquet_preview(
                    path,
                    preview_columns,
                )
            )

            if preview:

                print(
                    "  preview:"
                )

                for row_index, row in enumerate(
                    preview,
                    start=1,
                ):

                    print(
                        f"    row {row_index}:"
                    )

                    for key, value in row.items():

                        print(
                            f"      {key}: "
                            f"{value}"
                        )

            text_candidates.append(
                {
                    "path":
                        path,

                    "rows":
                        info[
                            "rows"
                        ],

                    "uid_columns":
                        uid_columns,

                    "text_columns":
                        exact_text,
                }
            )

    return text_candidates


# ============================================================
# OTHER FILES WHICH MAY CONTAIN ARTICLE TEXT
# ============================================================

def inspect_other_interim_files():

    print()
    print("=" * 80)
    print(
        "C. OTHER INTERIM FILES"
    )
    print("=" * 80)

    if not INTERIM_ROOT.exists():

        return

    csv_files = sorted(
        INTERIM_ROOT.rglob(
            "*.csv"
        )
    )

    json_files = sorted(
        INTERIM_ROOT.rglob(
            "*.json"
        )
    )

    jsonl_files = sorted(
        INTERIM_ROOT.rglob(
            "*.jsonl"
        )
    )

    print()
    print(
        f"CSV files   : "
        f"{len(csv_files):,}"
    )

    for path in csv_files:

        print()
        print(
            relative(
                path
            )
        )

        try:

            info = csv_info(
                path
            )

            print(
                "  columns:"
            )

            for column in info[
                "columns"
            ]:

                print(
                    f"    - {column}"
                )

        except Exception as exc:

            print(
                f"  ERROR: "
                f"{repr(exc)}"
            )

    print()
    print(
        f"JSON files  : "
        f"{len(json_files):,}"
    )

    for path in json_files:

        print()
        print(
            relative(
                path
            )
        )

        try:

            info = json_info(
                path
            )

            print(
                "  top-level keys:"
            )

            for key in info[
                "keys"
            ][
                :MAX_COLUMNS_PRINT
            ]:

                print(
                    f"    - {key}"
                )

        except Exception as exc:

            print(
                f"  ERROR: "
                f"{repr(exc)}"
            )

    print()
    print(
        f"JSONL files : "
        f"{len(jsonl_files):,}"
    )

    for path in jsonl_files:

        print(
            f"  {relative(path)}"
        )


# ============================================================
# PROCESSED PAIR SCHEMA
# ============================================================

def inspect_pair_schema():

    print()
    print("=" * 80)
    print(
        "D. PROCESSED VALIDATION PAIR SCHEMA"
    )
    print("=" * 80)

    path = (
        PROCESSED_ROOT
        / "headline_consistency_v1"
        / "validation.parquet"
    )

    if not path.exists():

        print(
            f"Missing: "
            f"{path}"
        )

        return

    info = parquet_info(
        path
    )

    print()
    print(
        f"File: "
        f"{relative(path)}"
    )

    print(
        f"Rows: "
        f"{info['rows']:,}"
    )

    print()
    print(
        "Columns:"
    )

    for column in info[
        "columns"
    ]:

        print(
            f"  - {column}"
        )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "This pair table contains IDs/hashes/metadata, "
        "not the original article text."
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
        "INTEGRATION STEP 02B"
    )
    print(
        "MAP CONSISTENCY REGRESSION SOURCES"
    )
    print("=" * 80)

    print()
    print(
        "READ-ONLY."
    )

    print(
        "NO TRAINING."
    )

    print(
        "NO INFERENCE."
    )

    print(
        "NO TEST TUNING."
    )

    prediction_info = (
        inspect_prediction_artifacts()
    )

    text_candidates = (
        inspect_interim_artifacts()
    )

    inspect_other_interim_files()

    inspect_pair_schema()

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print(
        "STEP 02B SUMMARY"
    )
    print("=" * 80)

    print()
    print(
        "Prediction artifacts found:"
    )

    for item in prediction_info:

        print(
            f"  {relative(item['path'])}"
        )

        print(
            "    probability columns: "
            +
            (
                ", ".join(
                    item[
                        "probability_columns"
                    ]
                )
                if item[
                    "probability_columns"
                ]
                else
                "NONE"
            )
        )

        print(
            "    identifier columns : "
            +
            (
                ", ".join(
                    item[
                        "uid_columns"
                    ]
                )
                if item[
                    "uid_columns"
                ]
                else
                "NONE"
            )
        )

    print()
    print(
        "Article-text candidate files:"
    )

    if text_candidates:

        for item in text_candidates:

            print(
                f"  {relative(item['path'])}"
            )

            print(
                "    UID columns : "
                +
                (
                    ", ".join(
                        item[
                            "uid_columns"
                        ]
                    )
                    if item[
                        "uid_columns"
                    ]
                    else
                    "NONE"
                )
            )

            print(
                "    text columns: "
                +
                ", ".join(
                    item[
                        "text_columns"
                    ]
                )
            )

    else:

        print(
            "  NONE FOUND"
        )

    print()
    print(
        "NEXT REQUIREMENT:"
    )

    print(
        "We need at least one article-text store "
        "with article UID + title + description + content, "
        "and saved validation probabilities keyed by pair_id."
    )

    print()
    print(
        "INTEGRATION STEP 02B FINISHED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()