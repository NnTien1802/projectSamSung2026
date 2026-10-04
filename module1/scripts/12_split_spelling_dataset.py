from pathlib import Path
from collections import Counter, defaultdict
import json
import random


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "vn_newsspell_hybrid_v03_full"
    / "vn_newsspell_all.jsonl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vn_newsspell_v1"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


TRAIN_PATH = (
    OUTPUT_DIR
    / "train.jsonl"
)

VALIDATION_PATH = (
    OUTPUT_DIR
    / "validation.jsonl"
)

TEST_PATH = (
    OUTPUT_DIR
    / "test.jsonl"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "split_summary.json"
)


# ============================================================
# SPLIT CONFIG
# ============================================================

TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10

RANDOM_SEED = 20260918


# ============================================================
# LOAD DATA
# ============================================================

def load_records():

    print("=" * 80)
    print("VN-NEWSSPELL GROUPED DATASET SPLIT")
    print("=" * 80)

    print(
        f"\nInput:\n{INPUT_PATH}"
    )

    records = []

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(line)
            )

    print(
        f"\nTotal records: "
        f"{len(records):,}"
    )

    return records


# ============================================================
# GROUP BY SOURCE
# ============================================================

def group_records(records):

    groups = defaultdict(list)

    for row in records:

        source_id = row[
            "source_id"
        ]

        groups[
            source_id
        ].append(
            row
        )

    print(
        f"Unique source IDs: "
        f"{len(groups):,}"
    )

    return groups


# ============================================================
# VALIDATE SOURCE GROUPS BEFORE SPLIT
# ============================================================

def validate_groups(groups):

    invalid = []

    for source_id, rows in groups.items():

        label_counter = Counter(
            row[
                "label"
            ]
            for row in rows
        )

        clean_count = (
            label_counter.get(
                0,
                0
            )
        )

        noisy_count = (
            label_counter.get(
                1,
                0
            )
        )

        if (
            clean_count != 1
            or noisy_count != 1
        ):

            invalid.append({
                "source_id":
                    source_id,

                "clean_count":
                    clean_count,

                "noisy_count":
                    noisy_count,
            })

    print(
        f"Invalid source groups: "
        f"{len(invalid):,}"
    )

    if invalid:

        print(
            "\nFirst invalid groups:"
        )

        for item in invalid[:10]:

            print(item)

        raise RuntimeError(
            "Dataset contains invalid "
            "source groups."
        )


# ============================================================
# CREATE GROUP SPLIT
# ============================================================

def create_split(groups):

    source_ids = list(
        groups.keys()
    )

    rng = random.Random(
        RANDOM_SEED
    )

    rng.shuffle(
        source_ids
    )

    total_sources = len(
        source_ids
    )

    train_count = int(
        total_sources
        * TRAIN_RATIO
    )

    validation_count = int(
        total_sources
        * VALIDATION_RATIO
    )

    # Phần còn lại vào test.
    test_count = (
        total_sources
        - train_count
        - validation_count
    )

    train_ids = set(
        source_ids[
            :train_count
        ]
    )

    validation_ids = set(
        source_ids[
            train_count:
            train_count
            + validation_count
        ]
    )

    test_ids = set(
        source_ids[
            train_count
            + validation_count:
        ]
    )

    assert len(
        train_ids
        & validation_ids
    ) == 0

    assert len(
        train_ids
        & test_ids
    ) == 0

    assert len(
        validation_ids
        & test_ids
    ) == 0

    assert (
        len(train_ids)
        + len(validation_ids)
        + len(test_ids)
        == total_sources
    )

    print("\nSOURCE SPLIT")

    print(
        f"Train      : "
        f"{len(train_ids):,}"
    )

    print(
        f"Validation : "
        f"{len(validation_ids):,}"
    )

    print(
        f"Test       : "
        f"{len(test_ids):,}"
    )

    return {
        "train":
            train_ids,

        "validation":
            validation_ids,

        "test":
            test_ids,
    }


# ============================================================
# WRITE SPLIT
# ============================================================

def write_split(
    name,
    source_ids,
    groups,
    output_path
):

    record_count = 0

    label_counter = Counter()

    severity_counter = Counter()

    error_type_counter = Counter()

    error_source_counter = Counter()

    original_source_split_counter = (
        Counter()
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        # Sort giúp file reproducible hơn.
        for source_id in sorted(
            source_ids
        ):

            rows = groups[
                source_id
            ]

            # clean trước, noisy sau.
            rows = sorted(
                rows,
                key=lambda row:
                    row[
                        "label"
                    ]
            )

            for row in rows:

                # ============================================
                # Project split
                # ============================================

                output_row = dict(
                    row
                )

                output_row[
                    "project_split"
                ] = name

                f.write(
                    json.dumps(
                        output_row,
                        ensure_ascii=False
                    )
                    + "\n"
                )

                record_count += 1

                label_counter[
                    output_row[
                        "label"
                    ]
                ] += 1

                original_source_split_counter[
                    output_row.get(
                        "source_split",
                        "unknown"
                    )
                ] += 1

                # ============================================
                # Noisy-only metadata
                # ============================================

                if (
                    output_row[
                        "label"
                    ]
                    == 1
                ):

                    severity_counter[
                        output_row.get(
                            "severity",
                            "unknown"
                        )
                    ] += 1

                    for error in (
                        output_row.get(
                            "errors",
                            []
                        )
                    ):

                        error_type_counter[
                            error.get(
                                "error_type",
                                "unknown"
                            )
                        ] += 1

                        error_source_counter[
                            error.get(
                                "error_source",
                                "unknown"
                            )
                        ] += 1

    return {
        "source_count":
            len(
                source_ids
            ),

        "record_count":
            record_count,

        "label_distribution":
            dict(
                label_counter
            ),

        "severity_distribution":
            dict(
                severity_counter
            ),

        "error_type_distribution":
            dict(
                error_type_counter
            ),

        "error_source_distribution":
            dict(
                error_source_counter
            ),

        "original_source_split_distribution":
            dict(
                original_source_split_counter
            ),

        "path":
            str(
                output_path
            ),
    }


# ============================================================
# CHECK LEAKAGE
# ============================================================

def validate_no_leakage(
    split_ids
):

    train_ids = split_ids[
        "train"
    ]

    validation_ids = split_ids[
        "validation"
    ]

    test_ids = split_ids[
        "test"
    ]

    train_validation_overlap = (
        train_ids
        & validation_ids
    )

    train_test_overlap = (
        train_ids
        & test_ids
    )

    validation_test_overlap = (
        validation_ids
        & test_ids
    )

    print("\nLEAKAGE CHECK")

    print(
        "train ∩ validation : "
        f"{len(train_validation_overlap):,}"
    )

    print(
        "train ∩ test       : "
        f"{len(train_test_overlap):,}"
    )

    print(
        "validation ∩ test  : "
        f"{len(validation_test_overlap):,}"
    )

    if (
        train_validation_overlap
        or train_test_overlap
        or validation_test_overlap
    ):

        raise RuntimeError(
            "SOURCE LEAKAGE DETECTED"
        )


# ============================================================
# PRINT SPLIT SUMMARY
# ============================================================

def print_split_summary(
    name,
    summary
):

    print("\n" + "=" * 80)

    print(
        f"{name.upper()} SUMMARY"
    )

    print("=" * 80)

    print(
        f"Sources : "
        f"{summary['source_count']:,}"
    )

    print(
        f"Records : "
        f"{summary['record_count']:,}"
    )

    print("\nLabels:")

    for label, count in sorted(
        summary[
            "label_distribution"
        ].items(),
        key=lambda item:
            int(item[0])
    ):

        print(
            f"  label={label}: "
            f"{count:,}"
        )

    print("\nSeverity:")

    for severity, count in sorted(
        summary[
            "severity_distribution"
        ].items()
    ):

        print(
            f"  {severity:12s}: "
            f"{count:,}"
        )

    print("\nOriginal VietNews split:")

    for source_split, count in sorted(
        summary[
            "original_source_split_distribution"
        ].items()
    ):

        print(
            f"  {source_split:12s}: "
            f"{count:,}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # LOAD
    # ========================================================

    records = load_records()

    groups = group_records(
        records
    )

    # ========================================================
    # PRE-SPLIT VALIDATION
    # ========================================================

    validate_groups(
        groups
    )

    # ========================================================
    # GROUP SPLIT
    # ========================================================

    split_ids = create_split(
        groups
    )

    validate_no_leakage(
        split_ids
    )

    # ========================================================
    # WRITE
    # ========================================================

    train_summary = write_split(
        "train",
        split_ids[
            "train"
        ],
        groups,
        TRAIN_PATH
    )

    validation_summary = write_split(
        "validation",
        split_ids[
            "validation"
        ],
        groups,
        VALIDATION_PATH
    )

    test_summary = write_split(
        "test",
        split_ids[
            "test"
        ],
        groups,
        TEST_PATH
    )

    # ========================================================
    # PRINT
    # ========================================================

    print_split_summary(
        "train",
        train_summary
    )

    print_split_summary(
        "validation",
        validation_summary
    )

    print_split_summary(
        "test",
        test_summary
    )

    # ========================================================
    # FINAL CONSISTENCY
    # ========================================================

    output_source_count = (
        train_summary[
            "source_count"
        ]
        + validation_summary[
            "source_count"
        ]
        + test_summary[
            "source_count"
        ]
    )

    output_record_count = (
        train_summary[
            "record_count"
        ]
        + validation_summary[
            "record_count"
        ]
        + test_summary[
            "record_count"
        ]
    )

    assert (
        output_source_count
        == len(groups)
    )

    assert (
        output_record_count
        == len(records)
    )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary = {
        "dataset":
            "VN-NewsSpell v1",

        "input_path":
            str(
                INPUT_PATH
            ),

        "random_seed":
            RANDOM_SEED,

        "split_strategy":
            "grouped_by_source_id",

        "ratios": {
            "train":
                TRAIN_RATIO,

            "validation":
                VALIDATION_RATIO,

            "test":
                TEST_RATIO,
        },

        "total_unique_sources":
            len(groups),

        "total_records":
            len(records),

        "leakage_check": {
            "train_validation":
                len(
                    split_ids[
                        "train"
                    ]
                    & split_ids[
                        "validation"
                    ]
                ),

            "train_test":
                len(
                    split_ids[
                        "train"
                    ]
                    & split_ids[
                        "test"
                    ]
                ),

            "validation_test":
                len(
                    split_ids[
                        "validation"
                    ]
                    & split_ids[
                        "test"
                    ]
                ),
        },

        "splits": {
            "train":
                train_summary,

            "validation":
                validation_summary,

            "test":
                test_summary,
        },
    }

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\n" + "=" * 80)
    print("SPLIT FINISHED")
    print("=" * 80)

    print(
        f"\nTotal sources: "
        f"{output_source_count:,}"
    )

    print(
        f"Total records: "
        f"{output_record_count:,}"
    )

    print("\nFiles:")

    print(
        f"  Train      : "
        f"{TRAIN_PATH}"
    )

    print(
        f"  Validation : "
        f"{VALIDATION_PATH}"
    )

    print(
        f"  Test       : "
        f"{TEST_PATH}"
    )

    print(
        f"  Summary    : "
        f"{SUMMARY_PATH}"
    )

    print(
        "\nNo source_id leakage "
        "between project splits."
    )


if __name__ == "__main__":
    main()