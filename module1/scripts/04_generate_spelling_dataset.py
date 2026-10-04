from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import random
import re
import unicodedata

from datasets import load_from_disk
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "vietnews"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "vn_newsspell_v01"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# FIRST RUN
# ------------------------------------------------------------
# Để test pipeline trước:
MAX_ARTICLES = 5000

# Sau khi kiểm tra dataset ổn:
# MAX_ARTICLES = None


# 1 clean + 1 noisy cho mỗi bài
NOISY_VARIANTS_PER_ARTICLE = 1

RANDOM_SEED = 20260918


# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

ALL_DATA_PATH = (
    OUTPUT_DIR
    / "vn_newsspell_all.jsonl"
)

NOISY_DATA_PATH = (
    OUTPUT_DIR
    / "vn_newsspell_noisy.jsonl"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "generation_summary.json"
)

ERROR_SAMPLE_PATH = (
    OUTPUT_DIR
    / "error_samples.csv"
)


# ============================================================
# ERROR DISTRIBUTION
# ============================================================

ERROR_WEIGHTS = {
    "missing_diacritic": 0.20,
    "wrong_tone": 0.17,
    "delete_character": 0.12,
    "duplicate_character": 0.10,
    "transpose_character": 0.10,
    "keyboard_neighbor": 0.10,
    "onset_confusion": 0.09,
    "vowel_confusion": 0.06,
    "telex_unconverted": 0.06,
}


SEVERITY_WEIGHTS = {
    "minor": 0.50,
    "moderate": 0.35,
    "severe": 0.15,
}


# Error density / 1000 syllables
ERROR_DENSITY = {
    "minor": (1, 3),
    "moderate": (4, 8),
    "severe": (9, 15),
}


# ============================================================
# REGEX
# ============================================================

# Unicode letters.
# Không tính digit và "_".
LETTER_RE = re.compile(
    r"[^\W\d_]+",
    re.UNICODE
)


# ============================================================
# BASIC UTILITIES
# ============================================================

def normalize_text(text):
    """
    Chỉ normalize Unicode.
    KHÔNG xóa "_".
    KHÔNG sửa chính tả.
    """

    if text is None:
        return ""

    text = str(text)

    return unicodedata.normalize(
        "NFC",
        text
    )


def to_natural(text):
    """
    Representation giống bài báo bình thường.
    Không dùng để xác định spelling label.
    """

    return text.replace(
        "_",
        " "
    )


def article_id(article):
    """
    GUID của VietNews không unique,
    do đó tạo ID dựa trên nội dung.
    """

    return hashlib.sha1(
        article.encode("utf-8")
    ).hexdigest()[:20]


def weighted_choice(rng, weights):
    names = list(weights.keys())
    values = list(weights.values())

    return rng.choices(
        names,
        weights=values,
        k=1
    )[0]


# ============================================================
# 1. MISSING DIACRITIC
# ============================================================

def remove_diacritics(word, rng):

    result = []

    for char in word:

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

    result = "".join(result)

    if result == word:
        return None

    return result


# ============================================================
# 2. WRONG TONE
# ============================================================

TONE_MARKS = [
    "\u0300",  # huyền
    "\u0301",  # sắc
    "\u0303",  # ngã
    "\u0309",  # hỏi
    "\u0323",  # nặng
]


def wrong_tone(word, rng):

    chars = list(
        unicodedata.normalize(
            "NFD",
            word
        )
    )

    tone_positions = [
        i
        for i, char in enumerate(chars)
        if char in TONE_MARKS
    ]

    if not tone_positions:
        return None

    pos = rng.choice(
        tone_positions
    )

    old_tone = chars[pos]

    # Có thể đổi sang tone khác
    # hoặc mất dấu tone.
    choices = [
        tone
        for tone in TONE_MARKS
        if tone != old_tone
    ]

    choices.append(None)

    new_tone = rng.choice(
        choices
    )

    if new_tone is None:
        del chars[pos]
    else:
        chars[pos] = new_tone

    result = unicodedata.normalize(
        "NFC",
        "".join(chars)
    )

    if result == word:
        return None

    return result


# ============================================================
# 3. DELETE CHARACTER
# ============================================================

def delete_character(word, rng):

    if len(word) < 3:
        return None

    pos = rng.randrange(
        len(word)
    )

    result = (
        word[:pos]
        + word[pos + 1:]
    )

    if not result:
        return None

    return result


# ============================================================
# 4. DUPLICATE CHARACTER
# ============================================================

def duplicate_character(word, rng):

    if len(word) < 2:
        return None

    pos = rng.randrange(
        len(word)
    )

    return (
        word[:pos]
        + word[pos]
        + word[pos:]
    )


# ============================================================
# 5. TRANSPOSE
# ============================================================

def transpose_character(word, rng):

    if len(word) < 3:
        return None

    candidates = [
        i
        for i in range(
            len(word) - 1
        )
        if word[i] != word[i + 1]
    ]

    if not candidates:
        return None

    pos = rng.choice(
        candidates
    )

    chars = list(word)

    chars[pos], chars[pos + 1] = (
        chars[pos + 1],
        chars[pos],
    )

    return "".join(chars)


# ============================================================
# 6. KEYBOARD NEIGHBOUR
# ============================================================

KEYBOARD_NEIGHBORS = {
    "q": "wa",
    "w": "qeas",
    "e": "wrsd",
    "r": "etdf",
    "t": "ryfg",
    "y": "tugh",
    "u": "yihj",
    "i": "uojk",
    "o": "ipkl",
    "p": "ol",

    "a": "qwsz",
    "s": "awedxz",
    "d": "serfcx",
    "f": "drtgvc",
    "g": "ftyhbv",
    "h": "gyujnb",
    "j": "huikmn",
    "k": "jiolm",
    "l": "kop",

    "z": "asx",
    "x": "zsdc",
    "c": "xdfv",
    "v": "cfgb",
    "b": "vghn",
    "n": "bhjm",
    "m": "njk",
}


def keyboard_neighbor(word, rng):

    candidates = []

    for i, char in enumerate(word):

        lower = char.lower()

        if lower in KEYBOARD_NEIGHBORS:
            candidates.append(i)

    if not candidates:
        return None

    pos = rng.choice(
        candidates
    )

    original = word[pos]

    replacement = rng.choice(
        KEYBOARD_NEIGHBORS[
            original.lower()
        ]
    )

    if original.isupper():
        replacement = replacement.upper()

    return (
        word[:pos]
        + replacement
        + word[pos + 1:]
    )


# ============================================================
# 7. VIETNAMESE INITIAL CONFUSION
# ============================================================

ONSET_CONFUSIONS = {
    "ch": ["tr"],
    "tr": ["ch"],

    "s": ["x"],
    "x": ["s"],

    "d": ["gi", "r"],
    "gi": ["d", "r"],
    "r": ["d", "gi"],

    "l": ["n"],
    "n": ["l"],
}


def preserve_case(original, replacement):

    if original.isupper():
        return replacement.upper()

    if original[0].isupper():
        return (
            replacement[0].upper()
            + replacement[1:]
        )

    return replacement


def onset_confusion(word, rng):

    lower = word.lower()

    # Ưu tiên prefix dài.
    prefixes = sorted(
        ONSET_CONFUSIONS.keys(),
        key=len,
        reverse=True
    )

    for prefix in prefixes:

        if lower.startswith(prefix):

            alternatives = (
                ONSET_CONFUSIONS[prefix]
            )

            replacement = rng.choice(
                alternatives
            )

            original_prefix = (
                word[:len(prefix)]
            )

            replacement = preserve_case(
                original_prefix,
                replacement
            )

            result = (
                replacement
                + word[len(prefix):]
            )

            if result != word:
                return result

    return None


# ============================================================
# 8. VOWEL CONFUSION
# ============================================================

VOWEL_CONFUSIONS = {
    "a": ["ă", "â"],
    "ă": ["a", "â"],
    "â": ["a", "ă"],

    "e": ["ê"],
    "ê": ["e"],

    "o": ["ô", "ơ"],
    "ô": ["o", "ơ"],
    "ơ": ["o", "ô"],

    "u": ["ư"],
    "ư": ["u"],

    "i": ["y"],
    "y": ["i"],
}


def vowel_confusion(word, rng):

    candidates = []

    for i, char in enumerate(word):

        if char.lower() in VOWEL_CONFUSIONS:
            candidates.append(i)

    if not candidates:
        return None

    pos = rng.choice(
        candidates
    )

    original = word[pos]

    replacement = rng.choice(
        VOWEL_CONFUSIONS[
            original.lower()
        ]
    )

    if original.isupper():
        replacement = replacement.upper()

    result = (
        word[:pos]
        + replacement
        + word[pos + 1:]
    )

    if result == word:
        return None

    return result


# ============================================================
# 9. TELEX NOT CONVERTED
# ============================================================

SHAPE_BREVE = "\u0306"
SHAPE_CIRCUMFLEX = "\u0302"
SHAPE_HORN = "\u031B"

TELEX_TONES = {
    "\u0301": "s",  # sắc
    "\u0300": "f",  # huyền
    "\u0309": "r",  # hỏi
    "\u0303": "x",  # ngã
    "\u0323": "j",  # nặng
}


def telex_unconverted(word, rng):

    result = []
    tone_key = None
    changed = False

    for char in word:

        if char == "đ":
            result.append("dd")
            changed = True
            continue

        if char == "Đ":
            result.append("DD")
            changed = True
            continue

        decomposed = unicodedata.normalize(
            "NFD",
            char
        )

        base = decomposed[0]
        marks = decomposed[1:]

        result.append(base)

        if SHAPE_BREVE in marks:
            result.append("w")
            changed = True

        elif SHAPE_CIRCUMFLEX in marks:
            result.append(base.lower())
            changed = True

        elif SHAPE_HORN in marks:
            result.append("w")
            changed = True

        for mark in marks:

            if mark in TELEX_TONES:
                tone_key = TELEX_TONES[mark]
                changed = True

    if tone_key is not None:
        result.append(tone_key)

    result = "".join(result)

    if not changed:
        return None

    if result == word:
        return None

    return result


# ============================================================
# ERROR FUNCTIONS
# ============================================================

ERROR_FUNCTIONS = {
    "missing_diacritic":
        remove_diacritics,

    "wrong_tone":
        wrong_tone,

    "delete_character":
        delete_character,

    "duplicate_character":
        duplicate_character,

    "transpose_character":
        transpose_character,

    "keyboard_neighbor":
        keyboard_neighbor,

    "onset_confusion":
        onset_confusion,

    "vowel_confusion":
        vowel_confusion,

    "telex_unconverted":
        telex_unconverted,
}


# ============================================================
# CANDIDATE EXTRACTION
# ============================================================

def find_candidates(field_name, text):

    candidates = []

    for match in LETTER_RE.finditer(text):

        word = match.group()

        # Tránh các ký tự quá ngắn.
        if len(word) < 2:
            continue

        # Tránh abbreviation kiểu:
        # TP, TAND, UBND...
        if word.isupper() and len(word) > 1:
            continue

        candidates.append({
            "field": field_name,
            "start": match.start(),
            "end": match.end(),
            "word": word,
        })

    return candidates


# ============================================================
# GENERATE ERROR
# ============================================================

def mutate_word(
    word,
    rng,
    used_error_types
):

    error_types = list(
        ERROR_WEIGHTS.keys()
    )

    weights = list(
        ERROR_WEIGHTS.values()
    )

    # Thử nhiều lần vì không phải lỗi nào
    # cũng áp dụng được với mọi từ.
    for attempt in range(30):

        # Cố gắng đa dạng error type
        # trong cùng một bài.
        unused = [
            t
            for t in error_types
            if t not in used_error_types
        ]

        if unused and attempt < 10:

            error_type = rng.choice(
                unused
            )

        else:

            error_type = rng.choices(
                error_types,
                weights=weights,
                k=1
            )[0]

        function = (
            ERROR_FUNCTIONS[
                error_type
            ]
        )

        result = function(
            word,
            rng
        )

        if (
            result is not None
            and result != word
        ):

            return (
                result,
                error_type
            )

    return None, None


# ============================================================
# SEVERITY / ERROR COUNT
# ============================================================

def choose_severity(rng):

    return weighted_choice(
        rng,
        SEVERITY_WEIGHTS
    )


def choose_error_count(
    syllable_count,
    severity,
    rng
):

    low, high = (
        ERROR_DENSITY[
            severity
        ]
    )

    density = rng.uniform(
        low,
        high
    )

    count = round(
        syllable_count
        * density
        / 1000
    )

    return max(
        1,
        count
    )


# ============================================================
# APPLY EDITS
# ============================================================

def apply_edits(text, edits):

    if not edits:
        return text, edits

    # --------------------------------------------------------
    # Calculate positions in noisy text
    # --------------------------------------------------------

    sorted_edits = sorted(
        edits,
        key=lambda x: x["start"]
    )

    delta = 0

    for edit in sorted_edits:

        noisy_start = (
            edit["start"]
            + delta
        )

        noisy_end = (
            noisy_start
            + len(edit["corrupted"])
        )

        edit["noisy_start"] = (
            noisy_start
        )

        edit["noisy_end"] = (
            noisy_end
        )

        delta += (
            len(edit["corrupted"])
            - len(edit["original"])
        )

    # --------------------------------------------------------
    # Apply backwards so original offsets remain valid
    # --------------------------------------------------------

    result = text

    for edit in reversed(
        sorted_edits
    ):

        result = (
            result[:edit["start"]]
            + edit["corrupted"]
            + result[edit["end"]:]
        )

    return (
        result,
        sorted_edits
    )


# ============================================================
# CORRUPT ARTICLE
# ============================================================

def corrupt_record(
    clean_fields,
    rng
):

    candidates = []

    for field_name, text in clean_fields.items():

        candidates.extend(
            find_candidates(
                field_name,
                text
            )
        )

    if not candidates:
        return None

    severity = choose_severity(
        rng
    )

    requested_error_count = (
        choose_error_count(
            len(candidates),
            severity,
            rng
        )
    )

    requested_error_count = min(
        requested_error_count,
        len(candidates)
    )

    rng.shuffle(
        candidates
    )

    field_edits = {
        "title": [],
        "abstract": [],
        "article": [],
    }

    used_error_types = set()

    errors_created = 0

    for candidate in candidates:

        if (
            errors_created
            >= requested_error_count
        ):
            break

        corrupted, error_type = (
            mutate_word(
                candidate["word"],
                rng,
                used_error_types
            )
        )

        if corrupted is None:
            continue

        edit = {
            "field":
                candidate["field"],

            "start":
                candidate["start"],

            "end":
                candidate["end"],

            "original":
                candidate["word"],

            "corrupted":
                corrupted,

            "error_type":
                error_type,
        }

        field_edits[
            candidate["field"]
        ].append(edit)

        used_error_types.add(
            error_type
        )

        errors_created += 1

    if errors_created == 0:
        return None

    noisy_fields = {}
    all_errors = []

    for field_name in [
        "title",
        "abstract",
        "article"
    ]:

        noisy_text, edits = (
            apply_edits(
                clean_fields[
                    field_name
                ],
                field_edits[
                    field_name
                ],
            )
        )

        noisy_fields[
            field_name
        ] = noisy_text

        for edit in edits:

            all_errors.append(
                edit
            )

    return {
        "severity": severity,
        "error_count": len(
            all_errors
        ),
        "errors": all_errors,
        "fields": noisy_fields,
        "candidate_syllables": len(
            candidates
        ),
    }


# ============================================================
# CREATE RECORD
# ============================================================

def create_output_record(
    source_id,
    source_split,
    source_guid,
    variant_id,
    label,
    severity,
    fields,
    errors,
):

    error_types = sorted(
        set(
            error["error_type"]
            for error in errors
        )
    )

    return {
        "source_id":
            source_id,

        "variant_id":
            variant_id,

        "source_dataset":
            "nam194/vietnews",

        "source_split":
            source_split,

        "source_guid":
            source_guid,

        "language":
            "vi",

        "label":
            label,

        "has_error":
            bool(label),

        "severity":
            severity,

        "error_count":
            len(errors),

        "error_types":
            error_types,

        # -----------------------------
        # Segmented representation
        # -----------------------------

        "title_segmented":
            fields["title"],

        "abstract_segmented":
            fields["abstract"],

        "article_segmented":
            fields["article"],

        # -----------------------------
        # Natural representation
        # -----------------------------

        "title_natural":
            to_natural(
                fields["title"]
            ),

        "abstract_natural":
            to_natural(
                fields["abstract"]
            ),

        "article_natural":
            to_natural(
                fields["article"]
            ),

        # -----------------------------
        # Error metadata
        # -----------------------------

        "errors":
            errors,

        "generation_method":
            (
                "clean_original"
                if label == 0
                else "synthetic_v1"
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("VN-NEWSSPELL V0.1")
    print("DOCUMENT-LEVEL SPELLING DATA GENERATOR")
    print("=" * 80)

    print(
        f"\nInput:\n{INPUT_DIR}"
    )

    print(
        f"\nOutput:\n{OUTPUT_DIR}"
    )

    dataset = load_from_disk(
        str(INPUT_DIR)
    )

    total_available = sum(
        len(dataset[split])
        for split in dataset.keys()
    )

    if MAX_ARTICLES is None:
        target_articles = total_available
    else:
        target_articles = min(
            MAX_ARTICLES,
            total_available
        )

    print(
        f"\nAvailable articles: "
        f"{total_available:,}"
    )

    print(
        f"Articles to process: "
        f"{target_articles:,}"
    )

    print(
        f"Noisy variants/article: "
        f"{NOISY_VARIANTS_PER_ARTICLE}"
    )

    print("\nError types:")

    for name, weight in ERROR_WEIGHTS.items():

        print(
            f"  {name:25s} "
            f"{weight:.2f}"
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    error_type_counter = Counter()
    severity_counter = Counter()
    field_counter = Counter()

    total_clean = 0
    total_noisy = 0
    total_errors = 0

    error_samples = []

    # --------------------------------------------------------
    # Write JSONL
    # --------------------------------------------------------

    with (
        open(
            ALL_DATA_PATH,
            "w",
            encoding="utf-8"
        ) as all_file,

        open(
            NOISY_DATA_PATH,
            "w",
            encoding="utf-8"
        ) as noisy_file
    ):

        processed = 0

        progress = tqdm(
            total=target_articles,
            desc="Generating dataset"
        )

        for split_name in dataset.keys():

            for row in dataset[split_name]:

                if (
                    processed
                    >= target_articles
                ):
                    break

                clean_fields = {
                    "title":
                        normalize_text(
                            row["title"]
                        ),

                    "abstract":
                        normalize_text(
                            row["abstract"]
                        ),

                    "article":
                        normalize_text(
                            row["article"]
                        ),
                }

                source_id = article_id(
                    clean_fields[
                        "article"
                    ]
                )

                # ============================================
                # CLEAN RECORD
                # ============================================

                clean_record = (
                    create_output_record(
                        source_id=source_id,

                        source_split=
                            split_name,

                        source_guid=
                            row["guid"],

                        variant_id=
                            f"{source_id}_clean",

                        label=0,

                        severity="clean",

                        fields=
                            clean_fields,

                        errors=[],
                    )
                )

                all_file.write(
                    json.dumps(
                        clean_record,
                        ensure_ascii=False
                    )
                    + "\n"
                )

                total_clean += 1

                # ============================================
                # NOISY RECORDS
                # ============================================

                for variant_index in range(
                    NOISY_VARIANTS_PER_ARTICLE
                ):

                    # Seed riêng cho mỗi article
                    # => reproducible.
                    seed_string = (
                        f"{RANDOM_SEED}-"
                        f"{source_id}-"
                        f"{variant_index}"
                    )

                    seed_value = int(
                        hashlib.sha1(
                            seed_string.encode()
                        ).hexdigest()[:16],
                        16
                    )

                    rng = random.Random(
                        seed_value
                    )

                    corrupted = corrupt_record(
                        clean_fields,
                        rng
                    )

                    if corrupted is None:
                        continue

                    noisy_record = (
                        create_output_record(
                            source_id=
                                source_id,

                            source_split=
                                split_name,

                            source_guid=
                                row["guid"],

                            variant_id=
                                (
                                    f"{source_id}"
                                    f"_noisy_"
                                    f"{variant_index}"
                                ),

                            label=1,

                            severity=
                                corrupted[
                                    "severity"
                                ],

                            fields=
                                corrupted[
                                    "fields"
                                ],

                            errors=
                                corrupted[
                                    "errors"
                                ],
                        )
                    )

                    line = json.dumps(
                        noisy_record,
                        ensure_ascii=False
                    )

                    all_file.write(
                        line + "\n"
                    )

                    noisy_file.write(
                        line + "\n"
                    )

                    total_noisy += 1

                    severity_counter[
                        corrupted[
                            "severity"
                        ]
                    ] += 1

                    total_errors += (
                        corrupted[
                            "error_count"
                        ]
                    )

                    for error in (
                        corrupted[
                            "errors"
                        ]
                    ):

                        error_type_counter[
                            error[
                                "error_type"
                            ]
                        ] += 1

                        field_counter[
                            error["field"]
                        ] += 1

                        if (
                            len(
                                error_samples
                            )
                            < 1000
                        ):

                            error_samples.append({
                                "source_id":
                                    source_id,

                                "field":
                                    error[
                                        "field"
                                    ],

                                "error_type":
                                    error[
                                        "error_type"
                                    ],

                                "original":
                                    error[
                                        "original"
                                    ],

                                "corrupted":
                                    error[
                                        "corrupted"
                                    ],

                                "severity":
                                    corrupted[
                                        "severity"
                                    ],
                            })

                processed += 1

                progress.update(1)

            if (
                processed
                >= target_articles
            ):
                break

        progress.close()

    # ========================================================
    # SAVE ERROR SAMPLES
    # ========================================================

    with open(
        ERROR_SAMPLE_PATH,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as csv_file:

        fieldnames = [
            "source_id",
            "field",
            "error_type",
            "original",
            "corrupted",
            "severity",
        ]

        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            error_samples
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {
        "source_dataset":
            "nam194/vietnews",

        "processed_articles":
            processed,

        "clean_records":
            total_clean,

        "noisy_records":
            total_noisy,

        "total_records":
            total_clean
            + total_noisy,

        "total_synthetic_errors":
            total_errors,

        "average_errors_per_noisy_article":
            (
                total_errors
                / total_noisy
                if total_noisy
                else 0
            ),

        "severity_distribution":
            dict(
                severity_counter
            ),

        "error_type_distribution":
            dict(
                error_type_counter
            ),

        "error_field_distribution":
            dict(
                field_counter
            ),

        "configuration": {
            "max_articles":
                MAX_ARTICLES,

            "noisy_variants_per_article":
                NOISY_VARIANTS_PER_ARTICLE,

            "random_seed":
                RANDOM_SEED,

            "error_weights":
                ERROR_WEIGHTS,

            "severity_weights":
                SEVERITY_WEIGHTS,

            "error_density":
                ERROR_DENSITY,
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

    # ========================================================
    # RESULT
    # ========================================================

    print("\n" + "=" * 80)
    print("GENERATION FINISHED")
    print("=" * 80)

    print(
        f"\nClean articles : "
        f"{total_clean:,}"
    )

    print(
        f"Noisy articles : "
        f"{total_noisy:,}"
    )

    print(
        f"Total records  : "
        f"{total_clean + total_noisy:,}"
    )

    print(
        f"Total errors   : "
        f"{total_errors:,}"
    )

    if total_noisy:

        print(
            f"Errors/noisy article: "
            f"{total_errors / total_noisy:.2f}"
        )

    print("\nSeverity:")

    for key, value in (
        severity_counter.items()
    ):

        print(
            f"  {key:10s}: "
            f"{value:,}"
        )

    print("\nError types:")

    for key, value in (
        error_type_counter.most_common()
    ):

        print(
            f"  {key:25s}: "
            f"{value:,}"
        )

    print("\nFiles:")

    print(
        f"  {ALL_DATA_PATH}"
    )

    print(
        f"  {NOISY_DATA_PATH}"
    )

    print(
        f"  {ERROR_SAMPLE_PATH}"
    )

    print(
        f"  {SUMMARY_PATH}"
    )

    print(
        "\nNo train/validation/test "
        "split was created."
    )

    print(
        "Raw VietNews was not modified."
    )


if __name__ == "__main__":
    main()