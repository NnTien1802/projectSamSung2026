from pathlib import Path
from collections import defaultdict, Counter
import csv
import json
import time
import unicodedata


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vn_newsspell_wordloc_v1"
)

TRAIN_PATH = (
    DATA_DIR
    / "train.jsonl"
)

VALIDATION_PATH = (
    DATA_DIR
    / "validation.jsonl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vn_newsspell_correction_v1"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LOOKUP_PATH = (
    OUTPUT_DIR
    / "correction_lookup.json"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "validation_summary.json"
)

ERROR_ANALYSIS_PATH = (
    OUTPUT_DIR
    / "validation_correction_errors.csv"
)


# ============================================================
# CONFIG
# ============================================================

TOP_K_PER_KEY = 5

MAX_ERROR_EXAMPLES = 5000


# ============================================================
# NORMALIZATION
# ============================================================

def nfc(text):

    return unicodedata.normalize(
        "NFC",
        str(text),
    )


def canonical(text):

    return (
        nfc(text)
        .casefold()
    )


# ============================================================
# CASING
# ============================================================

def restore_case(
    source,
    candidate,
):

    if not source:

        return candidate

    if source.isupper():

        return candidate.upper()

    if (
        source[0].isupper()
        and
        source[1:].islower()
    ):

        return (
            candidate[:1].upper()
            + candidate[1:]
        )

    return candidate


# ============================================================
# PREFIX / CORE / SUFFIX
#
# Example:
#
# "việc." -> "", "việc", "."
# "(sai)" -> "(", "sai", ")"
# ============================================================

def is_core_character(char):

    category = (
        unicodedata.category(
            char
        )
    )

    return (
        category.startswith("L")
        or
        category.startswith("M")
        or
        category.startswith("N")
    )


def split_affixes(text):

    text = nfc(
        text
    )

    if not text:

        return (
            "",
            "",
            "",
        )

    start = 0

    while (
        start < len(text)
        and
        not is_core_character(
            text[start]
        )
    ):

        start += 1

    end = len(
        text
    )

    while (
        end > start
        and
        not is_core_character(
            text[
                end - 1
            ]
        )
    ):

        end -= 1

    return (
        text[
            :start
        ],

        text[
            start:end
        ],

        text[
            end:
        ],
    )


# ============================================================
# LOAD WORD ERRORS
# ============================================================

def iter_word_errors(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            line = line.strip()

            if not line:

                continue

            row = json.loads(
                line
            )

            errors = row.get(
                "word_errors",
                [],
            )

            if errors is None:

                continue

            for error in errors:

                noisy = nfc(
                    error.get(
                        "noisy",
                        "",
                    )
                )

                clean = nfc(
                    error.get(
                        "clean",
                        "",
                    )
                )

                if not noisy:
                    continue

                if not clean:
                    continue

                if noisy == clean:
                    continue

                yield {
                    "source_id":
                        row.get(
                            "source_id",
                            "",
                        ),

                    "severity":
                        row.get(
                            "severity",
                            "",
                        ),

                    "field":
                        error.get(
                            "field",
                            "",
                        ),

                    "noisy":
                        noisy,

                    "clean":
                        clean,
                }


# ============================================================
# BUILD LOOKUPS FROM TRAIN ONLY
# ============================================================

def build_train_lookup():

    print(
        f"\nBuilding correction lookup "
        f"from TRAIN only:\n{TRAIN_PATH}"
    )

    start_time = time.time()

    exact_counts = defaultdict(
        Counter
    )

    component_counts = defaultdict(
        Counter
    )

    total_pairs = 0

    component_pairs = 0

    component_alignment_mismatch = 0

    for item in iter_word_errors(
        TRAIN_PATH
    ):

        noisy = item[
            "noisy"
        ]

        clean = item[
            "clean"
        ]

        total_pairs += 1

        # ====================================================
        # WHOLE-WORD LOOKUP
        # ====================================================

        noisy_key = canonical(
            noisy
        )

        clean_key = canonical(
            clean
        )

        exact_counts[
            noisy_key
        ][
            clean_key
        ] += 1

        # ====================================================
        # COMPONENT LOOKUP
        #
        # Example:
        #
        # quyêt_định
        # quyết_định
        #
        # =>
        #
        # quyêt -> quyết
        # ====================================================

        noisy_parts = noisy.split(
            "_"
        )

        clean_parts = clean.split(
            "_"
        )

        if (
            len(
                noisy_parts
            )
            != len(
                clean_parts
            )
        ):

            component_alignment_mismatch += 1

            continue

        for (
            noisy_part,
            clean_part,
        ) in zip(
            noisy_parts,
            clean_parts,
        ):

            (
                noisy_prefix,
                noisy_core,
                noisy_suffix,
            ) = split_affixes(
                noisy_part
            )

            (
                clean_prefix,
                clean_core,
                clean_suffix,
            ) = split_affixes(
                clean_part
            )

            if not noisy_core:

                continue

            if not clean_core:

                continue

            # Generator không thay đổi punctuation.
            if (
                noisy_prefix
                != clean_prefix
                or
                noisy_suffix
                != clean_suffix
            ):

                continue

            noisy_component_key = (
                canonical(
                    noisy_core
                )
            )

            clean_component_key = (
                canonical(
                    clean_core
                )
            )

            if (
                noisy_component_key
                ==
                clean_component_key
            ):

                continue

            component_counts[
                noisy_component_key
            ][
                clean_component_key
            ] += 1

            component_pairs += 1

    elapsed = (
        time.time()
        - start_time
    )

    print(
        f"\nTraining changed words : "
        f"{total_pairs:,}"
    )

    print(
        f"Exact noisy keys       : "
        f"{len(exact_counts):,}"
    )

    print(
        f"Component pairs        : "
        f"{component_pairs:,}"
    )

    print(
        f"Component noisy keys   : "
        f"{len(component_counts):,}"
    )

    print(
        f"Alignment mismatch     : "
        f"{component_alignment_mismatch:,}"
    )

    print(
        f"Build time             : "
        f"{elapsed:.2f}s"
    )

    return (
        exact_counts,
        component_counts,
        {
            "training_word_pairs":
                total_pairs,

            "exact_keys":
                len(
                    exact_counts
                ),

            "component_pairs":
                component_pairs,

            "component_keys":
                len(
                    component_counts
                ),

            "component_alignment_mismatch":
                component_alignment_mismatch,

            "build_seconds":
                elapsed,
        },
    )


# ============================================================
# SERIALIZE COUNTER MAP
# ============================================================

def serialize_lookup(
    counts,
):

    result = {}

    for key, counter in (
        counts.items()
    ):

        total = sum(
            counter.values()
        )

        top_candidates = (
            counter.most_common(
                TOP_K_PER_KEY
            )
        )

        result[
            key
        ] = []

        for (
            candidate,
            count,
        ) in top_candidates:

            result[
                key
            ].append({
                "candidate":
                    candidate,

                "count":
                    int(
                        count
                    ),

                "probability":
                    float(
                        count
                        / total
                    ),
            })

    return result


# ============================================================
# COMPONENT CANDIDATES
# ============================================================

def get_component_candidates(
    noisy_word,
    component_lookup,
):

    parts = nfc(
        noisy_word
    ).split(
        "_"
    )

    candidates = {}

    for component_index, part in enumerate(
        parts
    ):

        (
            prefix,
            core,
            suffix,
        ) = split_affixes(
            part
        )

        if not core:

            continue

        key = canonical(
            core
        )

        entries = (
            component_lookup.get(
                key
            )
        )

        if not entries:

            continue

        for entry in entries:

            clean_core = restore_case(
                core,
                entry[
                    "candidate"
                ],
            )

            corrected_part = (
                prefix
                + clean_core
                + suffix
            )

            corrected_parts = list(
                parts
            )

            corrected_parts[
                component_index
            ] = corrected_part

            candidate = "_".join(
                corrected_parts
            )

            if (
                canonical(
                    candidate
                )
                ==
                canonical(
                    noisy_word
                )
            ):

                continue

            candidate_key = nfc(
                candidate
            )

            score = float(
                entry[
                    "probability"
                ]
            )

            support = int(
                entry[
                    "count"
                ]
            )

            old = candidates.get(
                candidate_key
            )

            value = {
                "candidate":
                    candidate_key,

                "method":
                    "component",

                "component_index":
                    component_index,

                "support":
                    support,

                "conditional_probability":
                    score,
            }

            if old is None:

                candidates[
                    candidate_key
                ] = value

            else:

                if (
                    score
                    >
                    old[
                        "conditional_probability"
                    ]
                ):

                    candidates[
                        candidate_key
                    ] = value

                elif (
                    score
                    ==
                    old[
                        "conditional_probability"
                    ]
                    and
                    support
                    >
                    old[
                        "support"
                    ]
                ):

                    candidates[
                        candidate_key
                    ] = value

    result = list(
        candidates.values()
    )

    result.sort(
        key=lambda item: (
            item[
                "conditional_probability"
            ],
            item[
                "support"
            ],
        ),
        reverse=True,
    )

    return result[
        :TOP_K_PER_KEY
    ]


# ============================================================
# CORRECTION CANDIDATES
# ============================================================

def get_correction_candidates(
    noisy_word,
    exact_lookup,
    component_lookup,
):

    noisy_word = nfc(
        noisy_word
    )

    exact_key = canonical(
        noisy_word
    )

    # ========================================================
    # METHOD 1:
    # exact noisy whole word seen in training
    # ========================================================

    entries = (
        exact_lookup.get(
            exact_key
        )
    )

    if entries:

        candidates = []

        for entry in entries:

            candidate = restore_case(
                noisy_word,
                entry[
                    "candidate"
                ],
            )

            candidates.append({
                "candidate":
                    candidate,

                "method":
                    "exact",

                "component_index":
                    None,

                "support":
                    int(
                        entry[
                            "count"
                        ]
                    ),

                "conditional_probability":
                    float(
                        entry[
                            "probability"
                        ]
                    ),
            })

        return candidates

    # ========================================================
    # METHOD 2:
    # correct one underscore component
    # ========================================================

    return get_component_candidates(
        noisy_word=
            noisy_word,

        component_lookup=
            component_lookup,
    )


# ============================================================
# EVALUATE VALIDATION
# ============================================================

def evaluate_validation(
    exact_lookup,
    component_lookup,
):

    print(
        "\n" + "=" * 80
    )

    print(
        "VALIDATION CORRECTION EVALUATION"
    )

    print(
        "=" * 80
    )

    print(
        "\nValidation only."
    )

    print(
        "Test set is NOT read or used."
    )

    start_time = time.time()

    total = 0

    covered = 0

    exact_method = 0

    component_method = 0

    no_candidate = 0

    top1_correct = 0

    top3_correct = 0

    top5_correct = 0

    exact_top1_correct = 0

    component_top1_correct = 0

    exact_method_total = 0

    component_method_total = 0

    error_rows = []

    for item in iter_word_errors(
        VALIDATION_PATH
    ):

        total += 1

        noisy = item[
            "noisy"
        ]

        clean = item[
            "clean"
        ]

        candidates = (
            get_correction_candidates(
                noisy_word=
                    noisy,

                exact_lookup=
                    exact_lookup,

                component_lookup=
                    component_lookup,
            )
        )

        if not candidates:

            no_candidate += 1

            if (
                len(
                    error_rows
                )
                <
                MAX_ERROR_EXAMPLES
            ):

                error_rows.append({
                    "source_id":
                        item[
                            "source_id"
                        ],

                    "field":
                        item[
                            "field"
                        ],

                    "severity":
                        item[
                            "severity"
                        ],

                    "noisy":
                        noisy,

                    "clean":
                        clean,

                    "prediction":
                        "",

                    "method":
                        "none",

                    "top1_correct":
                        0,
                })

            continue

        covered += 1

        method = candidates[
            0
        ][
            "method"
        ]

        if method == "exact":

            exact_method += 1

            exact_method_total += 1

        else:

            component_method += 1

            component_method_total += 1

        predicted = (
            candidates[
                0
            ][
                "candidate"
            ]
        )

        clean_nfc = nfc(
            clean
        )

        is_top1_correct = (
            nfc(
                predicted
            )
            ==
            clean_nfc
        )

        if is_top1_correct:

            top1_correct += 1

            if method == "exact":

                exact_top1_correct += 1

            else:

                component_top1_correct += 1

        candidate_strings = [
            nfc(
                candidate[
                    "candidate"
                ]
            )
            for candidate
            in candidates
        ]

        if (
            clean_nfc
            in candidate_strings[
                :3
            ]
        ):

            top3_correct += 1

        if (
            clean_nfc
            in candidate_strings[
                :5
            ]
        ):

            top5_correct += 1

        if (
            not is_top1_correct
            and
            len(
                error_rows
            )
            <
            MAX_ERROR_EXAMPLES
        ):

            error_rows.append({
                "source_id":
                    item[
                        "source_id"
                    ],

                "field":
                    item[
                        "field"
                    ],

                "severity":
                    item[
                        "severity"
                    ],

                "noisy":
                    noisy,

                "clean":
                    clean,

                "prediction":
                    predicted,

                "method":
                    method,

                "top1_correct":
                    0,
            })

    elapsed = (
        time.time()
        - start_time
    )

    coverage = (
        covered
        / total
        if total
        else 0.0
    )

    top1_accuracy = (
        top1_correct
        / total
        if total
        else 0.0
    )

    top3_accuracy = (
        top3_correct
        / total
        if total
        else 0.0
    )

    top5_accuracy = (
        top5_correct
        / total
        if total
        else 0.0
    )

    accuracy_when_covered = (
        top1_correct
        / covered
        if covered
        else 0.0
    )

    exact_accuracy = (
        exact_top1_correct
        / exact_method_total
        if exact_method_total
        else 0.0
    )

    component_accuracy = (
        component_top1_correct
        / component_method_total
        if component_method_total
        else 0.0
    )

    print(
        f"\nValidation changed words : "
        f"{total:,}"
    )

    print(
        f"Candidates available     : "
        f"{covered:,}"
    )

    print(
        f"No candidate             : "
        f"{no_candidate:,}"
    )

    print(
        f"Correction coverage      : "
        f"{coverage:.6f}"
    )

    print(
        "\nCandidate source:"
    )

    print(
        f"  exact word : "
        f"{exact_method:,}"
    )

    print(
        f"  component  : "
        f"{component_method:,}"
    )

    print(
        "\nCorrection accuracy:"
    )

    print(
        f"  Top-1 overall       : "
        f"{top1_accuracy:.6f}"
    )

    print(
        f"  Top-3 overall       : "
        f"{top3_accuracy:.6f}"
    )

    print(
        f"  Top-5 overall       : "
        f"{top5_accuracy:.6f}"
    )

    print(
        f"  Top-1 when covered  : "
        f"{accuracy_when_covered:.6f}"
    )

    print(
        f"  Exact lookup Top-1  : "
        f"{exact_accuracy:.6f}"
    )

    print(
        f"  Component Top-1     : "
        f"{component_accuracy:.6f}"
    )

    print(
        f"\nEvaluation time: "
        f"{elapsed:.2f}s"
    )

    # ========================================================
    # SAVE ERROR ANALYSIS
    # ========================================================

    with open(
        ERROR_ANALYSIS_PATH,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        fieldnames = [
            "source_id",
            "field",
            "severity",
            "noisy",
            "clean",
            "prediction",
            "method",
            "top1_correct",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in error_rows:

            writer.writerow(
                row
            )

    return {
        "dataset":
            "validation",

        "test_used":
            False,

        "total_error_words":
            total,

        "covered":
            covered,

        "uncovered":
            no_candidate,

        "coverage":
            coverage,

        "exact_method":
            exact_method,

        "component_method":
            component_method,

        "top1_correct":
            top1_correct,

        "top1_accuracy":
            top1_accuracy,

        "top3_accuracy":
            top3_accuracy,

        "top5_accuracy":
            top5_accuracy,

        "top1_accuracy_when_covered":
            accuracy_when_covered,

        "exact_top1_accuracy":
            exact_accuracy,

        "component_top1_accuracy":
            component_accuracy,

        "evaluation_seconds":
            elapsed,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)

    print(
        "VN-NEWSSPELL - BUILD CORRECTION LOOKUP"
    )

    print("=" * 80)

    print(
        "\nLookup is built from TRAIN only."
    )

    print(
        "Validation is used for evaluation only."
    )

    print(
        "Test set is NOT read or used."
    )

    # ========================================================
    # BUILD
    # ========================================================

    (
        exact_counts,
        component_counts,
        train_stats,
    ) = build_train_lookup()

    # ========================================================
    # AMBIGUITY
    # ========================================================

    ambiguous_exact = sum(
        1
        for counter
        in exact_counts.values()
        if len(
            counter
        ) > 1
    )

    ambiguous_component = sum(
        1
        for counter
        in component_counts.values()
        if len(
            counter
        ) > 1
    )

    print(
        f"\nAmbiguous exact keys     : "
        f"{ambiguous_exact:,}"
    )

    print(
        f"Ambiguous component keys : "
        f"{ambiguous_component:,}"
    )

    # ========================================================
    # SERIALIZE
    # ========================================================

    print(
        "\nPreparing lookup..."
    )

    exact_lookup = (
        serialize_lookup(
            exact_counts
        )
    )

    component_lookup = (
        serialize_lookup(
            component_counts
        )
    )

    lookup = {
        "version":
            "vn_newsspell_correction_v1",

        "built_from":
            "train_only",

        "test_used":
            False,

        "top_k_per_key":
            TOP_K_PER_KEY,

        "exact_lookup":
            exact_lookup,

        "component_lookup":
            component_lookup,
    }

    with open(
        LOOKUP_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            lookup,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"\nCorrection lookup:\n"
        f"{LOOKUP_PATH}"
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    validation_stats = (
        evaluate_validation(
            exact_lookup=
                exact_lookup,

            component_lookup=
                component_lookup,
        )
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {
        "train":
            train_stats,

        "ambiguity": {
            "ambiguous_exact_keys":
                ambiguous_exact,

            "ambiguous_component_keys":
                ambiguous_component,
        },

        "validation":
            validation_stats,
    }

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "\n" + "=" * 80
    )

    print(
        "CORRECTION LOOKUP FINISHED"
    )

    print(
        "=" * 80
    )

    print(
        f"\nLookup:\n"
        f"{LOOKUP_PATH}"
    )

    print(
        f"\nSummary:\n"
        f"{SUMMARY_PATH}"
    )

    print(
        f"\nValidation errors:\n"
        f"{ERROR_ANALYSIS_PATH}"
    )

    print(
        "\nTest set was NOT read or used."
    )


if __name__ == "__main__":

    main()