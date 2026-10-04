from pathlib import Path
from collections import Counter
import json
import time

from transformers import AutoTokenizer


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vn_newsspell_v1"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "vn_newsspell_wordloc_v1"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TOKENIZER_DIR = (
    PROJECT_ROOT
    / "models"
    / "module1_phobert_mil_all_fields_4chunks"
    / "tokenizer"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "dataset_summary.json"
)


# ============================================================
# CONFIG
# ============================================================

SPLITS = [
    "train",
    "validation",
]

TEXT_FIELDS = [
    "title_segmented",
    "abstract_segmented",
    "article_segmented",
]

FIELD_NAMES = {
    "title_segmented": "title",
    "abstract_segmented": "abstract",
    "article_segmented": "article",
}

# Chính là separator đang dùng trong all-fields model:
#
# title . abstract . article
SEPARATOR_WORD = "."

# Chỉ dùng để kiểm tra rằng tokenize từng word rồi nối lại
# cho cùng token IDs với tokenize cả document.
TOKENIZER_CHECK_LIMIT_PER_SPLIT = 2000

EXAMPLE_LIMIT = 15


# ============================================================
# LOAD CLEAN / NOISY PAIRS
# ============================================================

def load_pairs(path):

    print(
        f"\nLoading:\n{path}"
    )

    start_time = time.time()

    groups = {}

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

            source_id = row[
                "source_id"
            ]

            label = int(
                row[
                    "label"
                ]
            )

            if source_id not in groups:

                groups[
                    source_id
                ] = {}

            if label == 0:

                groups[
                    source_id
                ][
                    "clean"
                ] = row

            elif label == 1:

                groups[
                    source_id
                ][
                    "noisy"
                ] = row

    invalid = []

    pairs = []

    for source_id, group in (
        groups.items()
    ):

        if (
            "clean" not in group
            or
            "noisy" not in group
        ):

            invalid.append(
                source_id
            )

            continue

        pairs.append({
            "source_id":
                source_id,

            "clean":
                group[
                    "clean"
                ],

            "noisy":
                group[
                    "noisy"
                ],
        })

    if invalid:

        raise RuntimeError(
            f"Invalid source pairs: "
            f"{len(invalid):,}"
        )

    elapsed = (
        time.time()
        - start_time
    )

    print(
        f"Source pairs: "
        f"{len(pairs):,}"
    )

    print(
        f"Load time: "
        f"{elapsed:.2f}s"
    )

    return pairs


# ============================================================
# BUILD WORD SEQUENCE
# ============================================================

def build_words_and_fields(row):

    """
    Reconstruct:

        title . abstract . article

    at whitespace-word level.

    Returns:
        words
        fields

    Both lists have exactly the same length.
    """

    words = []
    fields = []

    non_empty_fields = []

    for field in TEXT_FIELDS:

        text = row.get(
            field,
            "",
        )

        if text is None:
            text = ""

        text = str(
            text
        ).strip()

        if not text:
            continue

        field_words = (
            text.split()
        )

        if not field_words:
            continue

        non_empty_fields.append(
            (
                field,
                field_words,
            )
        )

    for index, (
        field,
        field_words,
    ) in enumerate(
        non_empty_fields
    ):

        if index > 0:

            words.append(
                SEPARATOR_WORD
            )

            fields.append(
                "separator"
            )

        simple_field = (
            FIELD_NAMES[
                field
            ]
        )

        words.extend(
            field_words
        )

        fields.extend(
            [
                simple_field
            ]
            * len(
                field_words
            )
        )

    return (
        words,
        fields,
    )


# ============================================================
# TOKENIZER CONSISTENCY CHECK
# ============================================================

def tokenizer_consistency_check(
    tokenizer,
    words,
):

    """
    Test whether:

      tokenizer(" ".join(words))

    equals:

      tokenizer(word1)
      + tokenizer(word2)
      + ...

    This is important for the next step, where each
    whitespace word will need to be mapped to PhoBERT
    subword tokens.
    """

    document_text = (
        " ".join(
            words
        )
    )

    whole_ids = tokenizer(
        document_text,
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_attention_mask=False,
    )[
        "input_ids"
    ]

    if not words:

        return (
            len(
                whole_ids
            )
            == 0
        )

    encoded_words = tokenizer(
        words,
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_attention_mask=False,
    )[
        "input_ids"
    ]

    concatenated = []

    for ids in encoded_words:

        concatenated.extend(
            ids
        )

    return (
        whole_ids
        == concatenated
    )


# ============================================================
# PROCESS SPLIT
# ============================================================

def process_split(
    split_name,
    tokenizer,
):

    input_path = (
        INPUT_DIR
        / f"{split_name}.jsonl"
    )

    output_path = (
        OUTPUT_DIR
        / f"{split_name}.jsonl"
    )

    pairs = load_pairs(
        input_path
    )

    stats = {
        "split":
            split_name,

        "source_pairs":
            len(
                pairs
            ),

        "written_records":
            0,

        "word_length_mismatch":
            0,

        "field_alignment_mismatch":
            0,

        "zero_changed_word_documents":
            0,

        "total_word_positions":
            0,

        "changed_word_positions":
            0,

        "metadata_error_count":
            0,

        "changed_word_count_equals_error_count":
            0,

        "changed_word_count_differs_error_count":
            0,

        "tokenizer_checks":
            0,

        "tokenizer_checks_passed":
            0,

        "tokenizer_checks_failed":
            0,
    }

    changed_words_per_doc = (
        Counter()
    )

    changed_fields = (
        Counter()
    )

    severity_counter = (
        Counter()
    )

    examples = []

    print(
        f"\nBuilding {split_name}..."
    )

    start_time = time.time()

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as out_file:

        for pair_index, pair in enumerate(
            pairs,
            start=1,
        ):

            clean = pair[
                "clean"
            ]

            noisy = pair[
                "noisy"
            ]

            (
                clean_words,
                clean_fields,
            ) = build_words_and_fields(
                clean
            )

            (
                noisy_words,
                noisy_fields,
            ) = build_words_and_fields(
                noisy
            )

            # =================================================
            # STRICT ALIGNMENT CHECK
            #
            # Dataset generator only mutates characters inside
            # words, so whitespace word positions should stay
            # aligned.
            # =================================================

            if (
                len(
                    clean_words
                )
                != len(
                    noisy_words
                )
            ):

                stats[
                    "word_length_mismatch"
                ] += 1

                continue

            if (
                clean_fields
                != noisy_fields
            ):

                stats[
                    "field_alignment_mismatch"
                ] += 1

                continue

            # =================================================
            # WORD ERROR LABELS
            # =================================================

            error_word_indices = []

            word_errors = []

            for word_index, (
                clean_word,
                noisy_word,
                field,
            ) in enumerate(
                zip(
                    clean_words,
                    noisy_words,
                    noisy_fields,
                )
            ):

                if (
                    clean_word
                    == noisy_word
                ):

                    continue

                # Field separators must never change.
                if field == "separator":

                    raise RuntimeError(
                        "Field separator changed between "
                        "clean/noisy pair."
                    )

                error_word_indices.append(
                    word_index
                )

                changed_fields[
                    field
                ] += 1

                word_errors.append({
                    "index":
                        word_index,

                    "field":
                        field,

                    "noisy":
                        noisy_word,

                    "clean":
                        clean_word,
                })

            changed_count = len(
                error_word_indices
            )

            metadata_error_count = int(
                noisy.get(
                    "error_count",
                    0,
                )
            )

            stats[
                "total_word_positions"
            ] += len(
                noisy_words
            )

            stats[
                "changed_word_positions"
            ] += changed_count

            stats[
                "metadata_error_count"
            ] += metadata_error_count

            changed_words_per_doc[
                changed_count
            ] += 1

            severity = noisy.get(
                "severity",
                "unknown",
            )

            severity_counter[
                severity
            ] += 1

            if changed_count == 0:

                stats[
                    "zero_changed_word_documents"
                ] += 1

            if (
                changed_count
                == metadata_error_count
            ):

                stats[
                    "changed_word_count_equals_error_count"
                ] += 1

            else:

                stats[
                    "changed_word_count_differs_error_count"
                ] += 1

            # =================================================
            # TOKENIZER CHECK
            # =================================================

            if (
                stats[
                    "tokenizer_checks"
                ]
                <
                TOKENIZER_CHECK_LIMIT_PER_SPLIT
            ):

                stats[
                    "tokenizer_checks"
                ] += 1

                tokenizer_ok = (
                    tokenizer_consistency_check(
                        tokenizer,
                        noisy_words,
                    )
                )

                if tokenizer_ok:

                    stats[
                        "tokenizer_checks_passed"
                    ] += 1

                else:

                    stats[
                        "tokenizer_checks_failed"
                    ] += 1

            # =================================================
            # SAVE ONE NOISY DOCUMENT WITH SPARSE WORD LABELS
            # =================================================

            output_row = {
                "source_id":
                    pair[
                        "source_id"
                    ],

                "variant_id":
                    noisy.get(
                        "variant_id",
                        "",
                    ),

                "severity":
                    severity,

                "metadata_error_count":
                    metadata_error_count,

                "changed_word_count":
                    changed_count,

                "words":
                    noisy_words,

                "word_fields":
                    noisy_fields,

                "error_word_indices":
                    error_word_indices,

                "word_errors":
                    word_errors,
            }

            out_file.write(
                json.dumps(
                    output_row,
                    ensure_ascii=False,
                )
                + "\n"
            )

            stats[
                "written_records"
            ] += 1

            # =================================================
            # EXAMPLES
            # =================================================

            if (
                len(
                    examples
                )
                < EXAMPLE_LIMIT
                and
                changed_count > 0
            ):

                examples.append({
                    "source_id":
                        pair[
                            "source_id"
                        ],

                    "severity":
                        severity,

                    "metadata_error_count":
                        metadata_error_count,

                    "changed_word_count":
                        changed_count,

                    "word_errors":
                        word_errors[
                            :10
                        ],
                })

            if (
                pair_index
                % 5000
                == 0
            ):

                print(
                    f"\rProcessed "
                    f"{pair_index:,}/"
                    f"{len(pairs):,}",
                    end="",
                    flush=True,
                )

    print()

    elapsed = (
        time.time()
        - start_time
    )

    # =========================================================
    # DERIVED STATS
    # =========================================================

    if (
        stats[
            "total_word_positions"
        ]
        > 0
    ):

        positive_rate = (
            stats[
                "changed_word_positions"
            ]
            /
            stats[
                "total_word_positions"
            ]
        )

    else:

        positive_rate = 0.0

    stats[
        "changed_word_rate"
    ] = positive_rate

    stats[
        "changed_words_per_document"
    ] = {
        str(key):
            value
        for key, value in sorted(
            changed_words_per_doc.items()
        )
    }

    stats[
        "changed_fields"
    ] = dict(
        changed_fields
    )

    stats[
        "severity"
    ] = dict(
        severity_counter
    )

    stats[
        "examples"
    ] = examples

    stats[
        "build_seconds"
    ] = elapsed

    # =========================================================
    # PRINT
    # =========================================================

    print(
        "\n" + "=" * 80
    )

    print(
        f"{split_name.upper()} SUMMARY"
    )

    print(
        "=" * 80
    )

    print(
        f"\nSource pairs                 : "
        f"{stats['source_pairs']:,}"
    )

    print(
        f"Written records              : "
        f"{stats['written_records']:,}"
    )

    print(
        f"Word length mismatch         : "
        f"{stats['word_length_mismatch']:,}"
    )

    print(
        f"Field alignment mismatch     : "
        f"{stats['field_alignment_mismatch']:,}"
    )

    print(
        f"Zero changed-word documents  : "
        f"{stats['zero_changed_word_documents']:,}"
    )

    print(
        f"\nTotal word positions         : "
        f"{stats['total_word_positions']:,}"
    )

    print(
        f"Changed word positions       : "
        f"{stats['changed_word_positions']:,}"
    )

    print(
        f"Changed-word rate            : "
        f"{positive_rate * 100:.4f}%"
    )

    print(
        f"\nMetadata error instances     : "
        f"{stats['metadata_error_count']:,}"
    )

    print(
        f"Docs changed_words == errors : "
        f"{stats['changed_word_count_equals_error_count']:,}"
    )

    print(
        f"Docs changed_words != errors : "
        f"{stats['changed_word_count_differs_error_count']:,}"
    )

    print(
        "\nChanged-word field distribution:"
    )

    for field, count in (
        changed_fields.most_common()
    ):

        print(
            f"  {field:10s}: "
            f"{count:,}"
        )

    print(
        "\nChanged words/document:"
    )

    for count, documents in sorted(
        changed_words_per_doc.items()
    ):

        print(
            f"  {count:2d}: "
            f"{documents:,}"
        )

    print(
        "\nTokenizer word-concatenation check:"
    )

    print(
        f"  checked : "
        f"{stats['tokenizer_checks']:,}"
    )

    print(
        f"  passed  : "
        f"{stats['tokenizer_checks_passed']:,}"
    )

    print(
        f"  failed  : "
        f"{stats['tokenizer_checks_failed']:,}"
    )

    print(
        f"\nBuild time: "
        f"{elapsed:.2f}s"
    )

    print(
        f"\nOutput:\n"
        f"{output_path}"
    )

    return stats


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)

    print(
        "VN-NEWSSPELL - BUILD WORD-LEVEL "
        "LOCALIZATION DATASET"
    )

    print("=" * 80)

    print(
        "\nTrain + validation only."
    )

    print(
        "Test set is NOT read or used."
    )

    # ========================================================
    # TOKENIZER
    # ========================================================

    print(
        "\nLoading saved PhoBERT tokenizer..."
    )

    tokenizer = (
        AutoTokenizer
        .from_pretrained(
            TOKENIZER_DIR,
            use_fast=False,
        )
    )

    print(
        f"Tokenizer fast: "
        f"{tokenizer.is_fast}"
    )

    # ========================================================
    # SPLITS
    # ========================================================

    summary = {}

    for split_name in SPLITS:

        summary[
            split_name
        ] = process_split(
            split_name,
            tokenizer,
        )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

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

    # ========================================================
    # FINAL SAFETY CHECK
    # ========================================================

    total_length_mismatch = sum(
        summary[
            split
        ][
            "word_length_mismatch"
        ]
        for split in SPLITS
    )

    total_field_mismatch = sum(
        summary[
            split
        ][
            "field_alignment_mismatch"
        ]
        for split in SPLITS
    )

    total_zero_change = sum(
        summary[
            split
        ][
            "zero_changed_word_documents"
        ]
        for split in SPLITS
    )

    total_tokenizer_fail = sum(
        summary[
            split
        ][
            "tokenizer_checks_failed"
        ]
        for split in SPLITS
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "FINAL VALIDATION"
    )

    print(
        "=" * 80
    )

    print(
        f"\nWord length mismatches : "
        f"{total_length_mismatch:,}"
    )

    print(
        f"Field mismatches       : "
        f"{total_field_mismatch:,}"
    )

    print(
        f"Zero-change noisy docs : "
        f"{total_zero_change:,}"
    )

    print(
        f"Tokenizer check fails  : "
        f"{total_tokenizer_fail:,}"
    )

    print(
        f"\nSummary:\n"
        f"{SUMMARY_PATH}"
    )

    print(
        "\nTest set was NOT read or used."
    )

    if (
        total_length_mismatch == 0
        and
        total_field_mismatch == 0
        and
        total_zero_change == 0
        and
        total_tokenizer_fail == 0
    ):

        print(
            "\nWORD-LEVEL DATASET: PASSED"
        )

    else:

        print(
            "\nWORD-LEVEL DATASET: "
            "REQUIRES REVIEW"
        )


if __name__ == "__main__":

    main()