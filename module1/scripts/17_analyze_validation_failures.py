from pathlib import Path
from collections import Counter, defaultdict
import csv
import json


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vn_newsspell_v1"
    / "validation.jsonl"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "results"
    / "module1_phobert_mil_full"
    / "threshold_tuning"
    / "validation_probabilities.csv"
)

BEST_THRESHOLD_PATH = (
    PROJECT_ROOT
    / "results"
    / "module1_phobert_mil_full"
    / "threshold_tuning"
    / "best_threshold.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "module1_phobert_mil_full"
    / "threshold_tuning"
    / "validation_failure_analysis.json"
)


# ============================================================
# LOAD THRESHOLD
# ============================================================

def load_threshold():

    with open(
        BEST_THRESHOLD_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        data = json.load(f)

    return float(
        data["selected_threshold"]
    )


# ============================================================
# LOAD VALIDATION METADATA
# ============================================================

def load_validation():

    records = {}

    with open(
        VALIDATION_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            row = json.loads(line)

            records[
                row["variant_id"]
            ] = row

    return records


# ============================================================
# LOAD PROBABILITIES
# ============================================================

def load_predictions():

    rows = []

    with open(
        PREDICTIONS_PATH,
        "r",
        encoding="utf-8-sig"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            rows.append({
                "variant_id":
                    row["variant_id"],

                "label":
                    int(row["label"]),

                "probability":
                    float(
                        row["probability_error"]
                    ),
            })

    return rows


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("PHOBERT MIL - VALIDATION FAILURE ANALYSIS")
    print("=" * 80)

    threshold = load_threshold()

    print(
        f"\nLocked threshold: "
        f"{threshold:.4f}"
    )

    validation = load_validation()

    predictions = load_predictions()

    print(
        f"Validation records : "
        f"{len(validation):,}"
    )

    print(
        f"Prediction records : "
        f"{len(predictions):,}"
    )

    # ========================================================
    # COUNTERS
    # ========================================================

    false_negative_count = 0

    fn_severity = Counter()

    fn_error_count = Counter()

    fn_all_fields = Counter()

    fn_single_error_field = Counter()

    fn_error_types = Counter()

    fn_error_sources = Counter()

    fn_article_error_types = Counter()

    fn_without_article_error = 0

    fn_with_article_error = 0

    single_error_total = Counter()

    single_error_fn = Counter()

    # ========================================================
    # ANALYSIS
    # ========================================================

    for prediction_row in predictions:

        if prediction_row["label"] != 1:
            continue

        variant_id = prediction_row[
            "variant_id"
        ]

        record = validation.get(
            variant_id
        )

        if record is None:

            raise RuntimeError(
                f"Missing validation record: "
                f"{variant_id}"
            )

        errors = record.get(
            "errors",
            []
        )

        probability = prediction_row[
            "probability"
        ]

        predicted = int(
            probability >= threshold
        )

        # ----------------------------------------------------
        # All noisy records:
        # calculate denominator for one-error field recall.
        # ----------------------------------------------------

        if len(errors) == 1:

            field = errors[0].get(
                "field",
                "unknown"
            )

            single_error_total[
                field
            ] += 1

            if predicted == 0:

                single_error_fn[
                    field
                ] += 1

        # ----------------------------------------------------
        # Only false negatives from here.
        # ----------------------------------------------------

        if predicted == 1:
            continue

        false_negative_count += 1

        severity = record.get(
            "severity",
            "unknown"
        )

        fn_severity[
            severity
        ] += 1

        fn_error_count[
            len(errors)
        ] += 1

        fields_in_document = set()

        has_article_error = False

        for error in errors:

            field = error.get(
                "field",
                "unknown"
            )

            error_type = error.get(
                "error_type",
                "unknown"
            )

            error_source = error.get(
                "error_source",
                "unknown"
            )

            fields_in_document.add(
                field
            )

            fn_error_types[
                error_type
            ] += 1

            fn_error_sources[
                error_source
            ] += 1

            if field == "article":

                has_article_error = True

                fn_article_error_types[
                    error_type
                ] += 1

        for field in fields_in_document:

            fn_all_fields[
                field
            ] += 1

        if len(errors) == 1:

            field = errors[0].get(
                "field",
                "unknown"
            )

            fn_single_error_field[
                field
            ] += 1

        if has_article_error:

            fn_with_article_error += 1

        else:

            fn_without_article_error += 1

    # ========================================================
    # SINGLE-ERROR RECALL BY FIELD
    # ========================================================

    single_error_field_stats = {}

    for field in sorted(
        single_error_total.keys()
    ):

        total = single_error_total[
            field
        ]

        fn = single_error_fn[
            field
        ]

        detected = (
            total - fn
        )

        recall = (
            detected / total
            if total
            else 0
        )

        single_error_field_stats[
            field
        ] = {
            "total":
                total,

            "false_negative":
                fn,

            "detected":
                detected,

            "recall":
                recall,
        }

    # ========================================================
    # PRINT
    # ========================================================

    print("\n" + "=" * 80)
    print("FALSE NEGATIVES")
    print("=" * 80)

    print(
        f"\nTotal false negatives: "
        f"{false_negative_count:,}"
    )

    print(
        "\nBy severity:"
    )

    for key, value in (
        fn_severity.most_common()
    ):

        print(
            f"  {key:12s}: "
            f"{value:,}"
        )

    print(
        "\nBy error count:"
    )

    for key, value in sorted(
        fn_error_count.items()
    ):

        print(
            f"  {key:2d} errors: "
            f"{value:,}"
        )

    print(
        "\nSingle-error false negatives "
        "by field:"
    )

    for key, value in (
        fn_single_error_field
        .most_common()
    ):

        print(
            f"  {key:10s}: "
            f"{value:,}"
        )

    print(
        "\nFalse negatives with "
        "at least one article error:"
    )

    print(
        f"  {fn_with_article_error:,}"
    )

    print(
        "\nFalse negatives with NO "
        "article error:"
    )

    print(
        f"  {fn_without_article_error:,}"
    )

    print(
        "\nSingle-error performance "
        "by field:"
    )

    for field, stats in (
        single_error_field_stats.items()
    ):

        print(
            f"\n  {field}"
        )

        print(
            f"    total     : "
            f"{stats['total']:,}"
        )

        print(
            f"    FN        : "
            f"{stats['false_negative']:,}"
        )

        print(
            f"    detected  : "
            f"{stats['detected']:,}"
        )

        print(
            f"    recall    : "
            f"{stats['recall']:.6f}"
        )

    print(
        "\nTop FN error types:"
    )

    for error_type, count in (
        fn_error_types.most_common(
            15
        )
    ):

        print(
            f"  {error_type:25s}: "
            f"{count:,}"
        )

    print(
        "\nTop ARTICLE FN error types:"
    )

    for error_type, count in (
        fn_article_error_types
        .most_common(
            15
        )
    ):

        print(
            f"  {error_type:25s}: "
            f"{count:,}"
        )

    print(
        "\nFN error sources:"
    )

    for source, count in (
        fn_error_sources.most_common()
    ):

        print(
            f"  {source:15s}: "
            f"{count:,}"
        )

    # ========================================================
    # SAVE
    # ========================================================

    output = {
        "threshold":
            threshold,

        "false_negative_count":
            false_negative_count,

        "false_negative_by_severity":
            dict(
                fn_severity
            ),

        "false_negative_by_error_count":
            {
                str(k): v
                for k, v
                in sorted(
                    fn_error_count.items()
                )
            },

        "false_negative_fields":
            dict(
                fn_all_fields
            ),

        "single_error_false_negative_fields":
            dict(
                fn_single_error_field
            ),

        "false_negatives_with_article_error":
            fn_with_article_error,

        "false_negatives_without_article_error":
            fn_without_article_error,

        "single_error_field_statistics":
            single_error_field_stats,

        "false_negative_error_types":
            dict(
                fn_error_types.most_common()
            ),

        "article_false_negative_error_types":
            dict(
                fn_article_error_types
                .most_common()
            ),

        "false_negative_error_sources":
            dict(
                fn_error_sources
            ),
    }

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\n" + "=" * 80)
    print("ANALYSIS FINISHED")
    print("=" * 80)

    print(
        f"\nSaved:\n"
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()