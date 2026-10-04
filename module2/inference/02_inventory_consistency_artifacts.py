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

DATA_ROOT = (
    MODULE_ROOT
    / "data"
    / "processed"
    / "consistency"
)

MODELS_ROOT = (
    MODULE_ROOT
    / "models"
    / "consistency"
)


# ============================================================
# CONFIG
# ============================================================

MAX_PREVIEW_COLUMNS = 40


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

        return str(path)


def inspect_parquet(
    path,
):

    try:

        parquet_file = (
            pq.ParquetFile(
                path
            )
        )

        schema_names = list(
            parquet_file.schema.names
        )

        metadata = (
            parquet_file.metadata
        )

        rows = (
            metadata.num_rows
            if metadata is not None
            else None
        )

        return {
            "ok":
                True,

            "rows":
                rows,

            "columns":
                schema_names,
        }

    except Exception as exc:

        return {
            "ok":
                False,

            "error":
                repr(
                    exc
                ),
        }


def inspect_csv(
    path,
):

    try:

        frame = pd.read_csv(
            path,
            nrows=3,
        )

        return {
            "ok":
                True,

            "columns":
                list(
                    frame.columns
                ),
        }

    except Exception as exc:

        return {
            "ok":
                False,

            "error":
                repr(
                    exc
                ),
        }


def inspect_json(
    path,
):

    try:

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
            "ok":
                True,

            "keys":
                keys,
        }

    except Exception as exc:

        return {
            "ok":
                False,

            "error":
                repr(
                    exc
                ),
        }


def looks_like_prediction_file(
    columns,
):

    lowered = [
        str(
            column
        ).lower()

        for column in columns
    ]

    probability_terms = [
        "prob",
        "score",
        "logit",
        "prediction",
        "pred",
    ]

    id_terms = [
        "sample_id",
        "pair_id",
        "anchor_id",
        "id",
    ]

    has_probability = any(
        any(
            term in column
            for term in probability_terms
        )

        for column in lowered
    )

    has_id = any(
        any(
            term == column
            or term in column
            for term in id_terms
        )

        for column in lowered
    )

    return (
        has_probability
        and
        has_id
    )


def looks_like_consistency_dataset(
    columns,
):

    lowered = [
        str(
            column
        ).lower()

        for column in columns
    ]

    title_terms = [
        "title",
        "headline",
    ]

    description_terms = [
        "description",
        "lead",
        "summary",
    ]

    article_terms = [
        "article",
        "content",
        "body",
        "text",
    ]

    has_title = any(
        any(
            term in column
            for term in title_terms
        )

        for column in lowered
    )

    has_description = any(
        any(
            term in column
            for term in description_terms
        )

        for column in lowered
    )

    has_article = any(
        any(
            term in column
            for term in article_terms
        )

        for column in lowered
    )

    return {
        "has_title":
            has_title,

        "has_description":
            has_description,

        "has_article":
            has_article,

        "looks_like_input_dataset":
            bool(
                has_title
                and
                has_article
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
        "INTEGRATION STEP 02A"
    )
    print(
        "INVENTORY CONSISTENCY ARTIFACTS"
    )
    print("=" * 80)

    print()
    print(
        "READ-ONLY INSPECTION."
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

    # ========================================================
    # ROOT CHECK
    # ========================================================

    roots = [
        (
            "RESULTS",
            RESULTS_ROOT,
        ),
        (
            "PROCESSED DATA",
            DATA_ROOT,
        ),
        (
            "MODELS",
            MODELS_ROOT,
        ),
    ]

    print()
    print("=" * 80)
    print(
        "ROOTS"
    )
    print("=" * 80)

    for name, path in roots:

        print(
            f"{name:16s}: "
            f"{path}"
        )

        print(
            f"  exists: "
            f"{path.exists()}"
        )

    # ========================================================
    # RESULTS PARQUET
    # ========================================================

    print()
    print("=" * 80)
    print(
        "PARQUET FILES UNDER results/consistency"
    )
    print("=" * 80)

    result_parquets = []

    if RESULTS_ROOT.exists():

        result_parquets = sorted(
            RESULTS_ROOT.rglob(
                "*.parquet"
            )
        )

    prediction_candidates = []

    if not result_parquets:

        print()
        print(
            "No parquet files found."
        )

    for index, path in enumerate(
        result_parquets,
        start=1,
    ):

        info = inspect_parquet(
            path
        )

        print()
        print(
            f"[{index}] "
            f"{relative(path)}"
        )

        if not info[
            "ok"
        ]:

            print(
                f"  ERROR: "
                f"{info['error']}"
            )

            continue

        rows = info[
            "rows"
        ]

        columns = info[
            "columns"
        ]

        print(
            f"  rows: "
            f"{rows:,}"
            if rows is not None
            else
            "  rows: unknown"
        )

        print(
            "  columns:"
        )

        for column in (
            columns[
                :MAX_PREVIEW_COLUMNS
            ]
        ):

            print(
                f"    - {column}"
            )

        if (
            len(
                columns
            )
            >
            MAX_PREVIEW_COLUMNS
        ):

            print(
                f"    ... "
                f"{len(columns) - MAX_PREVIEW_COLUMNS} "
                f"more"
            )

        if looks_like_prediction_file(
            columns
        ):

            prediction_candidates.append(
                path
            )

            print(
                "  >>> PREDICTION CANDIDATE"
            )

    # ========================================================
    # RESULT CSV FILES
    # ========================================================

    print()
    print("=" * 80)
    print(
        "CSV FILES UNDER results/consistency"
    )
    print("=" * 80)

    result_csvs = []

    if RESULTS_ROOT.exists():

        result_csvs = sorted(
            RESULTS_ROOT.rglob(
                "*.csv"
            )
        )

    for index, path in enumerate(
        result_csvs,
        start=1,
    ):

        info = inspect_csv(
            path
        )

        print()
        print(
            f"[{index}] "
            f"{relative(path)}"
        )

        if not info[
            "ok"
        ]:

            print(
                f"  ERROR: "
                f"{info['error']}"
            )

            continue

        columns = info[
            "columns"
        ]

        print(
            "  columns:"
        )

        for column in (
            columns[
                :MAX_PREVIEW_COLUMNS
            ]
        ):

            print(
                f"    - {column}"
            )

        if looks_like_prediction_file(
            columns
        ):

            print(
                "  >>> POSSIBLE PREDICTION TABLE"
            )

    # ========================================================
    # JSON FILES
    # ========================================================

    print()
    print("=" * 80)
    print(
        "JSON FILES UNDER results/consistency"
    )
    print("=" * 80)

    result_jsons = []

    if RESULTS_ROOT.exists():

        result_jsons = sorted(
            RESULTS_ROOT.rglob(
                "*.json"
            )
        )

    for index, path in enumerate(
        result_jsons,
        start=1,
    ):

        info = inspect_json(
            path
        )

        print()
        print(
            f"[{index}] "
            f"{relative(path)}"
        )

        if not info[
            "ok"
        ]:

            print(
                f"  ERROR: "
                f"{info['error']}"
            )

            continue

        keys = info[
            "keys"
        ]

        print(
            "  top-level keys:"
        )

        for key in (
            keys[
                :MAX_PREVIEW_COLUMNS
            ]
        ):

            print(
                f"    - {key}"
            )

    # ========================================================
    # PROCESSED CONSISTENCY DATA
    # ========================================================

    print()
    print("=" * 80)
    print(
        "PROCESSED CONSISTENCY PARQUET FILES"
    )
    print("=" * 80)

    data_parquets = []

    if DATA_ROOT.exists():

        data_parquets = sorted(
            DATA_ROOT.rglob(
                "*.parquet"
            )
        )

    input_candidates = []

    if not data_parquets:

        print()
        print(
            "No parquet files found."
        )

    for index, path in enumerate(
        data_parquets,
        start=1,
    ):

        info = inspect_parquet(
            path
        )

        print()
        print(
            f"[{index}] "
            f"{relative(path)}"
        )

        if not info[
            "ok"
        ]:

            print(
                f"  ERROR: "
                f"{info['error']}"
            )

            continue

        columns = info[
            "columns"
        ]

        rows = info[
            "rows"
        ]

        print(
            f"  rows: "
            f"{rows:,}"
            if rows is not None
            else
            "  rows: unknown"
        )

        features = (
            looks_like_consistency_dataset(
                columns
            )
        )

        print(
            f"  title field candidate      : "
            f"{features['has_title']}"
        )

        print(
            f"  description field candidate: "
            f"{features['has_description']}"
        )

        print(
            f"  article field candidate    : "
            f"{features['has_article']}"
        )

        print(
            "  columns:"
        )

        for column in (
            columns[
                :MAX_PREVIEW_COLUMNS
            ]
        ):

            print(
                f"    - {column}"
            )

        if (
            features[
                "looks_like_input_dataset"
            ]
        ):

            input_candidates.append(
                path
            )

            print(
                "  >>> INPUT DATASET CANDIDATE"
            )

    # ========================================================
    # MODEL CHECKPOINTS
    # ========================================================

    print()
    print("=" * 80)
    print(
        "CONSISTENCY MODEL FILES"
    )
    print("=" * 80)

    model_files = []

    if MODELS_ROOT.exists():

        for pattern in [
            "*.pt",
            "*.pth",
            "*.bin",
            "*.safetensors",
        ]:

            model_files.extend(
                MODELS_ROOT.rglob(
                    pattern
                )
            )

    model_files = sorted(
        set(
            model_files
        )
    )

    if not model_files:

        print()
        print(
            "No model checkpoint files found."
        )

    for index, path in enumerate(
        model_files,
        start=1,
    ):

        size_mb = (
            path.stat().st_size
            /
            (
                1024
                *
                1024
            )
        )

        print(
            f"[{index}] "
            f"{relative(path)}"
        )

        print(
            f"    size = "
            f"{size_mb:.2f} MB"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print()
    print("=" * 80)
    print(
        "STEP 02A SUMMARY"
    )
    print("=" * 80)

    print()
    print(
        f"Results parquet files     : "
        f"{len(result_parquets):,}"
    )

    print(
        f"Prediction candidates     : "
        f"{len(prediction_candidates):,}"
    )

    print(
        f"Processed parquet files   : "
        f"{len(data_parquets):,}"
    )

    print(
        f"Input dataset candidates  : "
        f"{len(input_candidates):,}"
    )

    print(
        f"Model checkpoint files    : "
        f"{len(model_files):,}"
    )

    print()
    print(
        "PREDICTION CANDIDATES:"
    )

    if prediction_candidates:

        for path in prediction_candidates:

            print(
                f"  {relative(path)}"
            )

    else:

        print(
            "  NONE"
        )

    print()
    print(
        "INPUT DATASET CANDIDATES:"
    )

    if input_candidates:

        for path in input_candidates:

            print(
                f"  {relative(path)}"
            )

    else:

        print(
            "  NONE"
        )

    print()
    print(
        "INTEGRATION STEP 02A FINISHED SUCCESSFULLY"
    )


if __name__ == "__main__":
    main()