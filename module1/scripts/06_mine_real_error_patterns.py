from pathlib import Path
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import csv
import json
import re
import unicodedata

from datasets import load_from_disk
from tqdm import tqdm


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VSEC_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "vsec"
)

SPELLING_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "vi_spelling_correction"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "real_error_patterns_v01"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


PAIR_CSV_PATH = (
    OUTPUT_DIR
    / "real_error_pairs.csv"
)

PAIR_JSON_PATH = (
    OUTPUT_DIR
    / "real_error_pairs.json"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "mining_summary.json"
)


# ============================================================
# CONFIG
# ============================================================

# Giữ những pair xuất hiện ít nhất N lần.
MIN_PAIR_FREQUENCY = 2

# Giới hạn chiều dài từ để loại noise.
MIN_WORD_LENGTH = 2
MAX_WORD_LENGTH = 30


# ============================================================
# TOKENIZATION
# ============================================================

# Unicode word, bỏ số và "_".
WORD_RE = re.compile(
    r"[^\W\d_]+",
    re.UNICODE
)


def normalize_text(text):

    if text is None:
        return ""

    return unicodedata.normalize(
        "NFC",
        str(text)
    )


def tokenize_words(text):

    text = normalize_text(text)

    return WORD_RE.findall(text)


# ============================================================
# VIETNAMESE NORMALIZATION
# ============================================================

def remove_diacritics(text):

    result = []

    for char in text:

        if char == "đ":
            result.append("d")
            continue

        if char == "Đ":
            result.append("D")
            continue

        decomposed = unicodedata.normalize(
            "NFD",
            char
        )

        base = "".join(
            c
            for c in decomposed
            if not unicodedata.combining(c)
        )

        result.append(base)

    return "".join(result)


# ============================================================
# EDIT DISTANCE
# ============================================================

def levenshtein_distance(a, b):
    """
    Levenshtein distance không dùng thư viện ngoài.
    """

    if a == b:
        return 0

    if len(a) < len(b):
        a, b = b, a

    previous = list(
        range(
            len(b) + 1
        )
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
# SIMPLE EDIT CHECKS
# ============================================================

def is_single_deletion(correct, wrong):
    """
    correct -> wrong mất đúng 1 ký tự.

    Ví dụ:
        chính -> chín
    """

    if len(correct) != len(wrong) + 1:
        return False

    for i in range(len(correct)):

        candidate = (
            correct[:i]
            + correct[i + 1:]
        )

        if candidate == wrong:
            return True

    return False


def is_single_insertion(correct, wrong):
    """
    correct -> wrong thêm đúng 1 ký tự.
    """

    return is_single_deletion(
        wrong,
        correct
    )


def is_transposition(correct, wrong):

    if len(correct) != len(wrong):
        return False

    differences = [
        i
        for i in range(len(correct))
        if correct[i] != wrong[i]
    ]

    if len(differences) != 2:
        return False

    i, j = differences

    if j != i + 1:
        return False

    return (
        correct[i] == wrong[j]
        and correct[j] == wrong[i]
    )


# ============================================================
# ONSET CHECK
# ============================================================

ONSET_CONFUSION_PAIRS = {
    ("ch", "tr"),
    ("tr", "ch"),

    ("s", "x"),
    ("x", "s"),

    ("d", "gi"),
    ("gi", "d"),

    ("d", "r"),
    ("r", "d"),

    ("gi", "r"),
    ("r", "gi"),

    ("l", "n"),
    ("n", "l"),
}


VIETNAMESE_ONSETS = [
    "ngh",
    "ch",
    "gh",
    "gi",
    "kh",
    "ng",
    "nh",
    "ph",
    "qu",
    "th",
    "tr",

    "b",
    "c",
    "d",
    "đ",
    "g",
    "h",
    "k",
    "l",
    "m",
    "n",
    "p",
    "r",
    "s",
    "t",
    "v",
    "x",
]


def get_onset(word):

    lower = word.lower()

    for onset in sorted(
        VIETNAMESE_ONSETS,
        key=len,
        reverse=True
    ):

        if lower.startswith(onset):
            return onset

    return ""


def is_onset_confusion(
    correct,
    wrong
):

    c_onset = get_onset(
        correct
    )

    w_onset = get_onset(
        wrong
    )

    if (
        c_onset,
        w_onset
    ) not in ONSET_CONFUSION_PAIRS:

        return False

    correct_tail = (
        correct[
            len(c_onset):
        ]
    )

    wrong_tail = (
        wrong[
            len(w_onset):
        ]
    )

    return (
        correct_tail
        == wrong_tail
    )


# ============================================================
# CLASSIFY REAL ERROR
# ============================================================

def classify_error(
    correct,
    wrong
):

    correct = correct.lower()
    wrong = wrong.lower()

    if correct == wrong:
        return None

    # --------------------------------------------------------
    # Diacritic / tone related
    # --------------------------------------------------------

    if (
        remove_diacritics(correct)
        == remove_diacritics(wrong)
    ):

        return "diacritic_confusion"

    # --------------------------------------------------------
    # Vietnamese onset
    # --------------------------------------------------------

    if is_onset_confusion(
        correct,
        wrong
    ):

        return "onset_confusion"

    # --------------------------------------------------------
    # Single delete
    # --------------------------------------------------------

    if is_single_deletion(
        correct,
        wrong
    ):

        return "delete_character"

    # --------------------------------------------------------
    # Single insertion
    # --------------------------------------------------------

    if is_single_insertion(
        correct,
        wrong
    ):

        return "insert_character"

    # --------------------------------------------------------
    # Adjacent transposition
    # --------------------------------------------------------

    if is_transposition(
        correct,
        wrong
    ):

        return "transpose_character"

    # --------------------------------------------------------
    # General substitution / multi edit
    # --------------------------------------------------------

    distance = levenshtein_distance(
        correct,
        wrong
    )

    if distance == 1:
        return "single_character_substitution"

    if distance == 2:
        return "two_character_edit"

    return "other"


# ============================================================
# PAIR QUALITY FILTER
# ============================================================

def valid_pair(
    correct,
    wrong
):

    correct = normalize_text(
        correct
    ).lower()

    wrong = normalize_text(
        wrong
    ).lower()

    if correct == wrong:
        return False

    if (
        len(correct) < MIN_WORD_LENGTH
        or len(wrong) < MIN_WORD_LENGTH
    ):
        return False

    if (
        len(correct) > MAX_WORD_LENGTH
        or len(wrong) > MAX_WORD_LENGTH
    ):
        return False

    if not correct.isalpha():
        return False

    if not wrong.isalpha():
        return False

    # --------------------------------------------------------
    # High confidence spelling relationships
    # --------------------------------------------------------

    if (
        remove_diacritics(correct)
        == remove_diacritics(wrong)
    ):
        return True

    if is_onset_confusion(
        correct,
        wrong
    ):
        return True

    distance = levenshtein_distance(
        correct,
        wrong
    )

    # Conservative:
    # chỉ nhận <= 2 edit.
    if distance <= 2:
        return True

    return False


# ============================================================
# EXTRACT TOKEN PAIRS
# ============================================================

def extract_pairs(
    noisy_text,
    clean_text
):
    """
    Return:
        [(correct_word, wrong_word), ...]

    noisy_text:
        câu có lỗi

    clean_text:
        câu đúng
    """

    noisy_tokens = tokenize_words(
        noisy_text
    )

    clean_tokens = tokenize_words(
        clean_text
    )

    if not noisy_tokens:
        return []

    if not clean_tokens:
        return []

    pairs = []

    # --------------------------------------------------------
    # FAST PATH
    # --------------------------------------------------------
    # Nếu số token bằng nhau,
    # pair từng vị trí trực tiếp.
    # --------------------------------------------------------

    if (
        len(noisy_tokens)
        == len(clean_tokens)
    ):

        for noisy, clean in zip(
            noisy_tokens,
            clean_tokens
        ):

            noisy_lower = (
                noisy.lower()
            )

            clean_lower = (
                clean.lower()
            )

            if (
                noisy_lower
                == clean_lower
            ):
                continue

            if valid_pair(
                clean_lower,
                noisy_lower
            ):

                pairs.append(
                    (
                        clean_lower,
                        noisy_lower
                    )
                )

        return pairs

    # --------------------------------------------------------
    # Different token count
    # --------------------------------------------------------

    matcher = SequenceMatcher(
        None,
        clean_tokens,
        noisy_tokens,
        autojunk=False
    )

    for (
        tag,
        c1,
        c2,
        n1,
        n2
    ) in matcher.get_opcodes():

        if tag != "replace":
            continue

        clean_span = (
            clean_tokens[c1:c2]
        )

        noisy_span = (
            noisy_tokens[n1:n2]
        )

        # Chỉ lấy 1 clean token
        # ↔ 1 noisy token.
        if (
            len(clean_span) != 1
            or len(noisy_span) != 1
        ):
            continue

        clean = clean_span[0].lower()
        noisy = noisy_span[0].lower()

        if valid_pair(
            clean,
            noisy
        ):

            pairs.append(
                (
                    clean,
                    noisy
                )
            )

    return pairs


# ============================================================
# STORAGE
# ============================================================

pair_counter = Counter()

source_counter = defaultdict(
    Counter
)

type_counter = Counter()

examples = defaultdict(list)


def register_pair(
    correct,
    wrong,
    source
):

    pair = (
        correct,
        wrong
    )

    pair_counter[pair] += 1

    source_counter[pair][
        source
    ] += 1

    error_type = classify_error(
        correct,
        wrong
    )

    type_counter[
        error_type
    ] += 1

    if (
        len(examples[
            error_type
        ])
        < 20
    ):

        examples[
            error_type
        ].append(
            (
                correct,
                wrong
            )
        )


# ============================================================
# MINE VSEC
# ============================================================

def mine_vsec():

    print("\n" + "=" * 80)
    print("MINING VSEC")
    print("=" * 80)

    dataset = load_from_disk(
        str(VSEC_DIR)
    )

    train = dataset["train"]

    print(
        f"Rows: {len(train):,}"
    )

    extracted = 0

    for row in tqdm(
        train,
        desc="VSEC"
    ):

        noisy_text = (
            row["text"]
        )

        clean_text = (
            row["corrected_text"]
        )

        pairs = extract_pairs(
            noisy_text,
            clean_text
        )

        for (
            correct,
            wrong
        ) in pairs:

            register_pair(
                correct,
                wrong,
                "vsec"
            )

            extracted += 1

    print(
        f"Extracted pairs: "
        f"{extracted:,}"
    )

    return extracted


# ============================================================
# MINE VI SPELLING CORRECTION
# ============================================================

def mine_spelling_correction():

    print("\n" + "=" * 80)
    print("MINING VI_SPELLING_CORRECTION")
    print("=" * 80)

    dataset = load_from_disk(
        str(SPELLING_DIR)
    )

    # IMPORTANT:
    # Chỉ dùng TRAIN để mining.
    #
    # TEST được giữ nguyên cho future evaluation.
    train = dataset["train"]

    print(
        f"Training rows: "
        f"{len(train):,}"
    )

    if "test" in dataset:

        print(
            f"Reserved test rows: "
            f"{len(dataset['test']):,}"
        )

    extracted = 0

    for row in tqdm(
        train,
        desc="Spelling train"
    ):

        # Dataset:
        # source = noisy
        # target = correct
        noisy_text = (
            row["source"]
        )

        clean_text = (
            row["target"]
        )

        pairs = extract_pairs(
            noisy_text,
            clean_text
        )

        for (
            correct,
            wrong
        ) in pairs:

            register_pair(
                correct,
                wrong,
                "vi_spelling_correction"
            )

            extracted += 1

    print(
        f"Extracted pairs: "
        f"{extracted:,}"
    )

    return extracted


# ============================================================
# SAVE
# ============================================================

def save_results(
    vsec_count,
    spelling_count
):

    print("\n" + "=" * 80)
    print("FILTERING PAIRS")
    print("=" * 80)

    rows = []

    for (
        correct,
        wrong
    ), frequency in (
        pair_counter.items()
    ):

        if (
            frequency
            < MIN_PAIR_FREQUENCY
        ):
            continue

        error_type = classify_error(
            correct,
            wrong
        )

        sources = (
            source_counter[
                (correct, wrong)
            ]
        )

        rows.append({
            "correct":
                correct,

            "wrong":
                wrong,

            "frequency":
                frequency,

            "error_type":
                error_type,

            "vsec_frequency":
                sources.get(
                    "vsec",
                    0
                ),

            "spelling_frequency":
                sources.get(
                    "vi_spelling_correction",
                    0
                ),
        })

    rows.sort(
        key=lambda row:
            row["frequency"],
        reverse=True
    )

    print(
        f"Unique raw pairs: "
        f"{len(pair_counter):,}"
    )

    print(
        f"Pairs with frequency >= "
        f"{MIN_PAIR_FREQUENCY}: "
        f"{len(rows):,}"
    )

    # ========================================================
    # CSV
    # ========================================================

    with open(
        PAIR_CSV_PATH,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        fieldnames = [
            "correct",
            "wrong",
            "frequency",
            "error_type",
            "vsec_frequency",
            "spelling_frequency",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    # ========================================================
    # JSON lookup
    # ========================================================

    lookup = defaultdict(list)

    for row in rows:

        lookup[
            row["correct"]
        ].append({
            "wrong":
                row["wrong"],

            "frequency":
                row["frequency"],

            "error_type":
                row["error_type"],
        })

    with open(
        PAIR_JSON_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            lookup,
            f,
            ensure_ascii=False,
            indent=2
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {
        "vsec_extracted_pairs":
            vsec_count,

        "spelling_train_extracted_pairs":
            spelling_count,

        "unique_raw_pairs":
            len(pair_counter),

        "retained_pairs":
            len(rows),

        "minimum_frequency":
            MIN_PAIR_FREQUENCY,

        "error_type_distribution":
            dict(
                type_counter
            ),

        "vi_spelling_test_used":
            False,
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

    return rows


# ============================================================
# DISPLAY
# ============================================================

def show_results(rows):

    print("\n" + "=" * 80)
    print("TOP REAL ERROR PAIRS")
    print("=" * 80)

    for row in rows[:100]:

        print(
            f"{row['frequency']:7,} | "
            f"{row['error_type']:28s} | "
            f"{row['correct']:20s} "
            f"→ {row['wrong']}"
        )

    print("\n" + "=" * 80)
    print("ERROR TYPE EXAMPLES")
    print("=" * 80)

    for error_type in sorted(
        examples.keys()
    ):

        print(
            f"\n[{error_type}]"
        )

        seen = set()

        shown = 0

        for (
            correct,
            wrong
        ) in examples[
            error_type
        ]:

            pair = (
                correct,
                wrong
            )

            if pair in seen:
                continue

            seen.add(pair)

            print(
                f"  {correct:20s} "
                f"→ {wrong}"
            )

            shown += 1

            if shown >= 10:
                break


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("REAL VIETNAMESE SPELLING ERROR PATTERN MINER")
    print("=" * 80)

    print(
        "\nVSEC:"
    )

    print(
        VSEC_DIR
    )

    print(
        "\nVI SPELLING:"
    )

    print(
        SPELLING_DIR
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "vi_spelling_correction TEST split "
        "will NOT be used."
    )

    vsec_count = mine_vsec()

    spelling_count = (
        mine_spelling_correction()
    )

    rows = save_results(
        vsec_count,
        spelling_count
    )

    show_results(
        rows
    )

    print("\n" + "=" * 80)
    print("MINING FINISHED")
    print("=" * 80)

    print(
        f"\nCSV:\n{PAIR_CSV_PATH}"
    )

    print(
        f"\nJSON lookup:\n{PAIR_JSON_PATH}"
    )

    print(
        f"\nSummary:\n{SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()