from pathlib import Path
from collections import Counter, defaultdict
import csv
import json
import math
import re
import unicodedata

from datasets import load_from_disk
from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIETNEWS_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "vietnews"
)

INPUT_CSV = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "real_error_patterns_v01"
    / "real_error_pairs.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "real_error_patterns_v02"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


ALL_FILTERED_CSV = (
    OUTPUT_DIR
    / "filtered_error_patterns.csv"
)

HIGH_CONF_CSV = (
    OUTPUT_DIR
    / "high_confidence_patterns.csv"
)

CONTEXTUAL_CSV = (
    OUTPUT_DIR
    / "contextual_patterns.csv"
)

INFORMAL_CSV = (
    OUTPUT_DIR
    / "informal_patterns.csv"
)

HYBRID_JSON = (
    OUTPUT_DIR
    / "hybrid_error_lookup.json"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "filter_summary.json"
)


# ============================================================
# CONFIG
# ============================================================

# Pattern phải xuất hiện ít nhất N lần
# trong VSEC + spelling corpus.
MIN_PATTERN_FREQUENCY = 3


# Correct word phải xuất hiện trong VietNews.
MIN_CORRECT_VIETNEWS_FREQ = 2


# Nếu wrong xuất hiện rất nhiều trong VietNews,
# có khả năng nó là một từ hợp lệ.
CONTEXTUAL_WRONG_FREQ = 5


# Những nhóm tương đối đáng tin.
HIGH_CONFIDENCE_TYPES = {
    "diacritic_confusion",
    "delete_character",
    "insert_character",
    "transpose_character",
    "onset_confusion",
}


# Các nhóm heterogeneous hơn.
GENERIC_EDIT_TYPES = {
    "single_character_substitution",
    "two_character_edit",
}


# ============================================================
# TOKENIZATION
# ============================================================

WORD_RE = re.compile(
    r"[^\W\d_]+",
    re.UNICODE
)


def normalize_word(word):

    return unicodedata.normalize(
        "NFC",
        str(word).strip().lower()
    )


def tokenize(text):

    if text is None:
        return []

    text = unicodedata.normalize(
        "NFC",
        str(text)
    )

    return [
        word.lower()
        for word in WORD_RE.findall(text)
    ]


# ============================================================
# BUILD VIETNEWS VOCABULARY
# ============================================================

def build_vietnews_vocabulary():

    print("=" * 80)
    print("BUILDING VIETNEWS VOCABULARY")
    print("=" * 80)

    dataset = load_from_disk(
        str(VIETNEWS_DIR)
    )

    vocabulary = Counter()

    total_articles = sum(
        len(dataset[split])
        for split in dataset.keys()
    )

    progress = tqdm(
        total=total_articles,
        desc="VietNews vocab"
    )

    for split in dataset.keys():

        for row in dataset[split]:

            # Chỉ article để tránh title/abstract
            # làm cùng nội dung bị đếm lặp quá nhiều.
            words = tokenize(
                row["article"]
            )

            vocabulary.update(
                words
            )

            progress.update(1)

    progress.close()

    print(
        f"\nVocabulary size: "
        f"{len(vocabulary):,}"
    )

    print(
        f"Total word occurrences: "
        f"{sum(vocabulary.values()):,}"
    )

    return vocabulary


# ============================================================
# INFORMAL / TEEN-CODE DETECTION
# ============================================================

def looks_informal(correct, wrong):
    """
    Nhận diện một số biến đổi chat/teen-code
    xuất hiện rất nhiều trong spelling corpus.

    Không nói chúng sai về mặt dataset nguồn;
    chỉ không ưu tiên cho bài báo chính thống.
    """

    c = correct.lower()
    w = wrong.lower()

    # --------------------------------------------
    # c / k
    #
    # của -> kủa
    # có  -> kó
    # các -> kák
    # --------------------------------------------

    if c.startswith("c") and w.startswith("k"):
        return True

    # --------------------------------------------
    # ph -> f
    #
    # pháp -> fáp
    # phân -> fân
    # phải -> fải
    # --------------------------------------------

    if c.startswith("ph") and w.startswith("f"):
        return True

    # --------------------------------------------
    # qu -> w
    #
    # quan -> wan
    # quân -> wân
    # --------------------------------------------

    if c.startswith("qu") and w.startswith("w"):
        return True

    # --------------------------------------------
    # gi -> j
    #
    # giáo -> jáo
    # gia  -> ja
    # --------------------------------------------

    if c.startswith("gi") and w.startswith("j"):
        return True

    # --------------------------------------------
    # cuối âm c -> k
    #
    # học  -> họk
    # khác -> khák
    # thực -> thựk
    # --------------------------------------------

    if (
        c.endswith("c")
        and w.endswith("k")
    ):
        return True

    # --------------------------------------------
    # Initial ng -> lg...
    #
    # người -> lgười
    # những -> lhững
    #
    # Những pattern này rất không phù hợp
    # với báo chí.
    # --------------------------------------------

    if (
        c.startswith(("ng", "nh"))
        and w.startswith(("lg", "lh"))
    ):
        return True

    return False


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_pattern(
    row,
    vocabulary
):

    correct = normalize_word(
        row["correct"]
    )

    wrong = normalize_word(
        row["wrong"]
    )

    frequency = int(
        row["frequency"]
    )

    error_type = row[
        "error_type"
    ]

    correct_vn_freq = vocabulary.get(
        correct,
        0
    )

    wrong_vn_freq = vocabulary.get(
        wrong,
        0
    )

    # --------------------------------------------------------
    # Basic reject
    # --------------------------------------------------------

    if correct == wrong:

        return (
            "rejected",
            "same_word"
        )

    if frequency < MIN_PATTERN_FREQUENCY:

        return (
            "rejected",
            "low_pattern_frequency"
        )

    if (
        correct_vn_freq
        < MIN_CORRECT_VIETNEWS_FREQ
    ):

        return (
            "rejected",
            "correct_not_supported_by_vietnews"
        )

    # --------------------------------------------------------
    # Informal / chat-style
    # --------------------------------------------------------

    if looks_informal(
        correct,
        wrong
    ):

        return (
            "informal",
            "chat_or_informal_orthography"
        )

    # --------------------------------------------------------
    # High-confidence edit type
    # --------------------------------------------------------

    if (
        error_type
        in HIGH_CONFIDENCE_TYPES
    ):

        # Wrong token itself appears often
        # in clean news => contextual real-word error.
        if (
            wrong_vn_freq
            >= CONTEXTUAL_WRONG_FREQ
        ):

            return (
                "contextual",
                "wrong_is_existing_vietnews_word"
            )

        return (
            "high_confidence",
            "high_confidence_orthographic"
        )

    # --------------------------------------------------------
    # Generic edit
    # --------------------------------------------------------

    if (
        error_type
        in GENERIC_EDIT_TYPES
    ):

        # Generic edit nhưng wrong không phải
        # từ thông dụng trong clean corpus.
        if wrong_vn_freq == 0:

            return (
                "high_confidence",
                "generic_edit_nonword"
            )

        return (
            "contextual",
            "generic_edit_existing_word"
        )

    return (
        "rejected",
        "unsupported_error_type"
    )


# ============================================================
# LOAD MINED PAIRS
# ============================================================

def load_patterns():

    rows = []

    with open(
        INPUT_CSV,
        "r",
        encoding="utf-8-sig"
    ) as f:

        reader = csv.DictReader(
            f
        )

        for row in reader:

            rows.append(
                row
            )

    print(
        f"\nLoaded mined patterns: "
        f"{len(rows):,}"
    )

    return rows


# ============================================================
# CREATE HYBRID WEIGHT
# ============================================================

def calculate_sampling_weight(
    frequency
):
    """
    Không dùng raw frequency trực tiếp.

    Nếu dùng frequency thẳng:
        của -> kủa
    có thể thống trị toàn dataset.

    log1p giúp làm distribution mềm hơn.
    """

    return math.log1p(
        frequency
    )


# ============================================================
# PROCESS
# ============================================================

def process_patterns(
    rows,
    vocabulary
):

    output = []

    bucket_counter = Counter()
    reason_counter = Counter()
    type_counter = Counter()

    for row in tqdm(
        rows,
        desc="Filtering patterns"
    ):

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

        bucket, reason = (
            classify_pattern(
                row,
                vocabulary
            )
        )

        correct_vn_freq = (
            vocabulary.get(
                correct,
                0
            )
        )

        wrong_vn_freq = (
            vocabulary.get(
                wrong,
                0
            )
        )

        output_row = {
            "correct":
                correct,

            "wrong":
                wrong,

            "frequency":
                frequency,

            "error_type":
                error_type,

            "bucket":
                bucket,

            "reason":
                reason,

            "correct_vietnews_frequency":
                correct_vn_freq,

            "wrong_vietnews_frequency":
                wrong_vn_freq,

            "vsec_frequency":
                int(
                    row[
                        "vsec_frequency"
                    ]
                ),

            "spelling_frequency":
                int(
                    row[
                        "spelling_frequency"
                    ]
                ),

            "sampling_weight":
                calculate_sampling_weight(
                    frequency
                ),
        }

        output.append(
            output_row
        )

        bucket_counter[
            bucket
        ] += 1

        reason_counter[
            reason
        ] += 1

        type_counter[
            (
                bucket,
                error_type
            )
        ] += 1

    return (
        output,
        bucket_counter,
        reason_counter,
        type_counter
    )


# ============================================================
# SAVE CSV
# ============================================================

FIELDNAMES = [
    "correct",
    "wrong",
    "frequency",
    "error_type",
    "bucket",
    "reason",
    "correct_vietnews_frequency",
    "wrong_vietnews_frequency",
    "vsec_frequency",
    "spelling_frequency",
    "sampling_weight",
]


def save_csv(
    path,
    rows
):

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

        writer.writerows(
            rows
        )


# ============================================================
# SAVE LOOKUP
# ============================================================

def save_hybrid_lookup(
    rows
):
    """
    Chỉ high_confidence được cho vào lookup chính.

    contextual / informal vẫn được giữ riêng
    để sau này thí nghiệm.
    """

    lookup = defaultdict(list)

    for row in rows:

        if (
            row["bucket"]
            != "high_confidence"
        ):
            continue

        lookup[
            row["correct"]
        ].append({
            "wrong":
                row["wrong"],

            "frequency":
                row["frequency"],

            "sampling_weight":
                row["sampling_weight"],

            "error_type":
                row["error_type"],

            "source":
                "mined_pattern",
        })

    # Sort alternatives của mỗi từ.
    for correct in lookup:

        lookup[correct].sort(
            key=lambda item:
                item["sampling_weight"],
            reverse=True
        )

    with open(
        HYBRID_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            lookup,
            f,
            ensure_ascii=False
        )

    return lookup


# ============================================================
# DISPLAY EXAMPLES
# ============================================================

def show_examples(
    rows,
    bucket,
    n=30
):

    print("\n" + "=" * 80)
    print(
        f"{bucket.upper()} EXAMPLES"
    )
    print("=" * 80)

    filtered = [
        row
        for row in rows
        if row["bucket"] == bucket
    ]

    filtered.sort(
        key=lambda row:
            row["frequency"],
        reverse=True
    )

    for row in filtered[:n]:

        print(
            f"{row['frequency']:7,} | "
            f"{row['error_type']:28s} | "
            f"{row['correct']:18s}"
            f" -> "
            f"{row['wrong']:18s} | "
            f"VN wrong={row['wrong_vietnews_frequency']}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("REAL ERROR PATTERN FILTER V0.2")
    print("=" * 80)

    vocabulary = (
        build_vietnews_vocabulary()
    )

    rows = load_patterns()

    (
        filtered_rows,
        bucket_counter,
        reason_counter,
        type_counter
    ) = process_patterns(
        rows,
        vocabulary
    )

    # --------------------------------------------------------
    # Split buckets
    # --------------------------------------------------------

    high_conf = [
        row
        for row in filtered_rows
        if (
            row["bucket"]
            == "high_confidence"
        )
    ]

    contextual = [
        row
        for row in filtered_rows
        if (
            row["bucket"]
            == "contextual"
        )
    ]

    informal = [
        row
        for row in filtered_rows
        if (
            row["bucket"]
            == "informal"
        )
    ]

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_csv(
        ALL_FILTERED_CSV,
        filtered_rows
    )

    save_csv(
        HIGH_CONF_CSV,
        high_conf
    )

    save_csv(
        CONTEXTUAL_CSV,
        contextual
    )

    save_csv(
        INFORMAL_CSV,
        informal
    )

    lookup = save_hybrid_lookup(
        filtered_rows
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {
        "total_input_patterns":
            len(rows),

        "bucket_distribution":
            dict(
                bucket_counter
            ),

        "reason_distribution":
            dict(
                reason_counter
            ),

        "high_confidence_patterns":
            len(high_conf),

        "contextual_patterns":
            len(contextual),

        "informal_patterns":
            len(informal),

        "hybrid_lookup_words":
            len(lookup),

        "config": {
            "min_pattern_frequency":
                MIN_PATTERN_FREQUENCY,

            "min_correct_vietnews_frequency":
                MIN_CORRECT_VIETNEWS_FREQ,

            "contextual_wrong_frequency":
                CONTEXTUAL_WRONG_FREQ,
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

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("FILTER RESULT")
    print("=" * 80)

    print(
        f"\nInput patterns: "
        f"{len(rows):,}"
    )

    for bucket, count in (
        bucket_counter.most_common()
    ):

        percentage = (
            count
            / len(rows)
            * 100
        )

        print(
            f"{bucket:18s}: "
            f"{count:7,} "
            f"({percentage:6.2f}%)"
        )

    print(
        f"\nHybrid lookup words: "
        f"{len(lookup):,}"
    )

    show_examples(
        filtered_rows,
        "high_confidence"
    )

    show_examples(
        filtered_rows,
        "contextual"
    )

    show_examples(
        filtered_rows,
        "informal"
    )

    print("\n" + "=" * 80)
    print("FILES")
    print("=" * 80)

    print(
        f"\nAll:\n{ALL_FILTERED_CSV}"
    )

    print(
        f"\nHigh confidence:\n{HIGH_CONF_CSV}"
    )

    print(
        f"\nContextual:\n{CONTEXTUAL_CSV}"
    )

    print(
        f"\nInformal:\n{INFORMAL_CSV}"
    )

    print(
        f"\nHybrid lookup:\n{HYBRID_JSON}"
    )

    print(
        f"\nSummary:\n{SUMMARY_JSON}"
    )


if __name__ == "__main__":
    main()