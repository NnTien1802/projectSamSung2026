from pathlib import Path
from collections import Counter, defaultdict
import csv
import json
import math
import unicodedata


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "real_error_patterns_v02"
    / "filtered_error_patterns.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "real_error_patterns_v03"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TRUSTED_CSV = (
    OUTPUT_DIR
    / "trusted_error_patterns.csv"
)

REJECTED_CSV = (
    OUTPUT_DIR
    / "rejected_from_training.csv"
)

LOOKUP_JSON = (
    OUTPUT_DIR
    / "trusted_error_lookup.json"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "trusted_error_summary.json"
)


# ============================================================
# CONFIG
# ============================================================

# Chỉ giữ những loại lỗi có cấu trúc rõ ràng.
TRUSTED_ERROR_TYPES = {
    "diacritic_confusion",
    "delete_character",
    "insert_character",
    "transpose_character",
    "onset_confusion",
}


# Pattern phải có ít nhất N evidence
MIN_PATTERN_FREQUENCY = 3


# correct phải thực sự xuất hiện trong news corpus.
MIN_CORRECT_VIETNEWS_FREQUENCY = 5


# Nếu "wrong" xuất hiện cực nhiều trong VietNews,
# khả năng cao nó là một từ thực / contextual word.
#
# Ngưỡng cũ = 5 quá thấp.
MAX_WRONG_VIETNEWS_FREQUENCY = 100


# Không cho một correct word có quá nhiều biến thể.
MAX_VARIANTS_PER_WORD = 20


# ============================================================
# HELPERS
# ============================================================

def normalize_word(word):

    return unicodedata.normalize(
        "NFC",
        str(word).strip().lower()
    )


def levenshtein_distance(a, b):

    if a == b:
        return 0

    if len(a) < len(b):
        a, b = b, a

    previous = list(
        range(len(b) + 1)
    )

    for i, char_a in enumerate(
        a,
        start=1
    ):

        current = [i]

        for j, char_b in enumerate(
            b,
            start=1
        ):

            insertion = (
                current[j - 1]
                + 1
            )

            deletion = (
                previous[j]
                + 1
            )

            substitution = (
                previous[j - 1]
                + (
                    0
                    if char_a == char_b
                    else 1
                )
            )

            current.append(
                min(
                    insertion,
                    deletion,
                    substitution
                )
            )

        previous = current

    return previous[-1]


# ============================================================
# QUALITY FILTER
# ============================================================

def evaluate_pattern(row):

    correct = normalize_word(
        row["correct"]
    )

    wrong = normalize_word(
        row["wrong"]
    )

    frequency = int(
        row["frequency"]
    )

    error_type = (
        row["error_type"]
    )

    correct_vn_freq = int(
        row[
            "correct_vietnews_frequency"
        ]
    )

    wrong_vn_freq = int(
        row[
            "wrong_vietnews_frequency"
        ]
    )

    vsec_frequency = int(
        row["vsec_frequency"]
    )

    spelling_frequency = int(
        row["spelling_frequency"]
    )

    # --------------------------------------------------------
    # Basic
    # --------------------------------------------------------

    if correct == wrong:
        return False, "same_word"

    if frequency < MIN_PATTERN_FREQUENCY:
        return False, "low_frequency"

    if (
        correct_vn_freq
        < MIN_CORRECT_VIETNEWS_FREQUENCY
    ):
        return False, "weak_correct_word_support"

    # --------------------------------------------------------
    # IMPORTANT:
    # Generic 1/2 edit không được dùng cho training chính.
    #
    # Ví dụ:
    # của -> cuta
    # các -> cadc
    # không -> khug
    # --------------------------------------------------------

    if error_type not in TRUSTED_ERROR_TYPES:

        return (
            False,
            "untrusted_generic_edit_type"
        )

    # --------------------------------------------------------
    # Wrong word xuất hiện rất nhiều trong news corpus.
    #
    # Có khả năng là một từ thật:
    #
    # này -> nay
    # là   -> la
    # một  -> mộc
    # các  -> cát
    # --------------------------------------------------------

    if (
        wrong_vn_freq
        >= MAX_WRONG_VIETNEWS_FREQUENCY
    ):

        return (
            False,
            "possible_real_word"
        )

    # --------------------------------------------------------
    # Edit distance safety
    # --------------------------------------------------------

    distance = levenshtein_distance(
        correct,
        wrong
    )

    # Các error type trusted hầu hết
    # không cần quá nhiều edit.
    if distance > 2:

        return (
            False,
            "too_many_edits"
        )

    # --------------------------------------------------------
    # Very large length difference
    # --------------------------------------------------------

    if (
        abs(
            len(correct)
            - len(wrong)
        )
        > 2
    ):

        return (
            False,
            "large_length_difference"
        )

    # --------------------------------------------------------
    # Additional evidence metadata
    # --------------------------------------------------------

    if (
        vsec_frequency == 0
        and spelling_frequency == 0
    ):

        return (
            False,
            "no_source_evidence"
        )

    return True, "trusted"


# ============================================================
# SAMPLING WEIGHT
# ============================================================

def calculate_sampling_weight(
    frequency,
    vsec_frequency,
    spelling_frequency
):
    """
    Không dùng raw frequency trực tiếp.

    log1p tránh pattern cực phổ biến thống trị.

    VSEC được cộng thêm một chút trọng số
    vì đây là nguồn annotation lỗi rõ ràng hơn.
    """

    weight = math.log1p(
        frequency
    )

    if vsec_frequency > 0:

        weight *= 1.20

    if (
        vsec_frequency > 0
        and spelling_frequency > 0
    ):

        weight *= 1.15

    return weight


# ============================================================
# LOAD
# ============================================================

def load_rows():

    rows = []

    with open(
        INPUT_CSV,
        "r",
        encoding="utf-8-sig"
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            rows.append(row)

    return rows


# ============================================================
# SAVE CSV
# ============================================================

FIELDNAMES = [
    "correct",
    "wrong",
    "frequency",
    "error_type",
    "reason",
    "correct_vietnews_frequency",
    "wrong_vietnews_frequency",
    "vsec_frequency",
    "spelling_frequency",
    "sampling_weight",
]


def save_csv(path, rows):

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=FIELDNAMES
        )

        writer.writeheader()

        writer.writerows(rows)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("BUILD TRUSTED REAL ERROR LOOKUP")
    print("=" * 80)

    rows = load_rows()

    print(
        f"\nInput patterns: "
        f"{len(rows):,}"
    )

    trusted = []

    rejected = []

    reject_reasons = Counter()

    for row in rows:

        keep, reason = (
            evaluate_pattern(row)
        )

        correct = normalize_word(
            row["correct"]
        )

        wrong = normalize_word(
            row["wrong"]
        )

        frequency = int(
            row["frequency"]
        )

        vsec_frequency = int(
            row["vsec_frequency"]
        )

        spelling_frequency = int(
            row["spelling_frequency"]
        )

        output = {
            "correct":
                correct,

            "wrong":
                wrong,

            "frequency":
                frequency,

            "error_type":
                row["error_type"],

            "reason":
                reason,

            "correct_vietnews_frequency":
                int(
                    row[
                        "correct_vietnews_frequency"
                    ]
                ),

            "wrong_vietnews_frequency":
                int(
                    row[
                        "wrong_vietnews_frequency"
                    ]
                ),

            "vsec_frequency":
                vsec_frequency,

            "spelling_frequency":
                spelling_frequency,

            "sampling_weight":
                calculate_sampling_weight(
                    frequency,
                    vsec_frequency,
                    spelling_frequency
                ),
        }

        if keep:

            trusted.append(
                output
            )

        else:

            rejected.append(
                output
            )

            reject_reasons[
                reason
            ] += 1

    # ========================================================
    # Group by correct word
    # ========================================================

    grouped = defaultdict(list)

    for row in trusted:

        grouped[
            row["correct"]
        ].append(
            row
        )

    # ========================================================
    # Limit variants per word
    # ========================================================

    final_rows = []

    lookup = {}

    for correct, variants in grouped.items():

        variants.sort(
            key=lambda row:
                (
                    row[
                        "sampling_weight"
                    ],
                    row[
                        "frequency"
                    ],
                ),
            reverse=True
        )

        variants = variants[
            :MAX_VARIANTS_PER_WORD
        ]

        lookup[
            correct
        ] = []

        for row in variants:

            final_rows.append(
                row
            )

            lookup[
                correct
            ].append({
                "wrong":
                    row["wrong"],

                "error_type":
                    row["error_type"],

                "frequency":
                    row["frequency"],

                "sampling_weight":
                    row[
                        "sampling_weight"
                    ],

                "vsec_frequency":
                    row[
                        "vsec_frequency"
                    ],

                "spelling_frequency":
                    row[
                        "spelling_frequency"
                    ],
            })

    # ========================================================
    # SORT CSV
    # ========================================================

    final_rows.sort(
        key=lambda row:
            row["frequency"],
        reverse=True
    )

    rejected.sort(
        key=lambda row:
            row["frequency"],
        reverse=True
    )

    # ========================================================
    # SAVE
    # ========================================================

    save_csv(
        TRUSTED_CSV,
        final_rows
    )

    save_csv(
        REJECTED_CSV,
        rejected
    )

    with open(
        LOOKUP_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            lookup,
            f,
            ensure_ascii=False
        )

    # ========================================================
    # TYPE DISTRIBUTION
    # ========================================================

    type_counter = Counter(
        row["error_type"]
        for row in final_rows
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {
        "input_patterns":
            len(rows),

        "trusted_before_variant_cap":
            len(trusted),

        "trusted_final_patterns":
            len(final_rows),

        "lookup_words":
            len(lookup),

        "rejected_patterns":
            len(rejected),

        "error_type_distribution":
            dict(
                type_counter
            ),

        "reject_reasons":
            dict(
                reject_reasons
            ),

        "config": {
            "trusted_error_types":
                sorted(
                    TRUSTED_ERROR_TYPES
                ),

            "min_pattern_frequency":
                MIN_PATTERN_FREQUENCY,

            "min_correct_vietnews_frequency":
                MIN_CORRECT_VIETNEWS_FREQUENCY,

            "max_wrong_vietnews_frequency":
                MAX_WRONG_VIETNEWS_FREQUENCY,

            "max_variants_per_word":
                MAX_VARIANTS_PER_WORD,
        },
    }

    with open(
        SUMMARY_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2
        )

    # ========================================================
    # DISPLAY
    # ========================================================

    print("\n" + "=" * 80)
    print("RESULT")
    print("=" * 80)

    print(
        f"\nTrusted before cap : "
        f"{len(trusted):,}"
    )

    print(
        f"Trusted final      : "
        f"{len(final_rows):,}"
    )

    print(
        f"Lookup words       : "
        f"{len(lookup):,}"
    )

    print(
        f"Rejected           : "
        f"{len(rejected):,}"
    )

    print("\nTrusted types:")

    for error_type, count in (
        type_counter.most_common()
    ):

        print(
            f"  {error_type:25s}: "
            f"{count:,}"
        )

    print("\nReject reasons:")

    for reason, count in (
        reject_reasons.most_common()
    ):

        print(
            f"  {reason:35s}: "
            f"{count:,}"
        )

    # ========================================================
    # EXAMPLES
    # ========================================================

    print("\n" + "=" * 80)
    print("TOP TRUSTED PATTERNS")
    print("=" * 80)

    for row in final_rows[:100]:

        print(
            f"{row['frequency']:7,} | "
            f"{row['error_type']:25s} | "
            f"{row['correct']:18s}"
            f" -> "
            f"{row['wrong']:18s} | "
            f"VNwrong="
            f"{row['wrong_vietnews_frequency']:4d} | "
            f"VSEC="
            f"{row['vsec_frequency']:3d}"
        )

    print("\n" + "=" * 80)
    print("FILES")
    print("=" * 80)

    print(
        f"\nTrusted CSV:\n"
        f"{TRUSTED_CSV}"
    )

    print(
        f"\nLookup JSON:\n"
        f"{LOOKUP_JSON}"
    )

    print(
        f"\nRejected CSV:\n"
        f"{REJECTED_CSV}"
    )

    print(
        f"\nSummary:\n"
        f"{SUMMARY_JSON}"
    )


if __name__ == "__main__":
    main()