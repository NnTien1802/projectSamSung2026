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

VIETNEWS_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "vietnews"
)

TRUSTED_LOOKUP_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "real_error_patterns_v03"
    / "trusted_error_lookup.json"
)

# Bản cuối có dedup trực tiếp trong generator.
OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "vn_newsspell_hybrid_v03_full"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DATASET SIZE
# ============================================================

# FULL DATASET
MAX_ARTICLES = None

# Nếu sau này muốn chạy pilot:
# MAX_ARTICLES = 5000


NOISY_VARIANTS_PER_ARTICLE = 1

RANDOM_SEED = 20260918


# ============================================================
# HYBRID CONFIG
# ============================================================

# Nếu một từ có mined pattern:
# 60% thử mined trước.
MINED_PATTERN_PROBABILITY = 0.60


# Không cho mined transpose trên từ quá ngắn.
#
# Tránh:
#   là -> àl
#   và -> àv
MIN_MINED_TRANSPOSE_LENGTH = 3


# ============================================================
# SEVERITY
# ============================================================

SEVERITY_WEIGHTS = {
    "minor": 0.50,
    "moderate": 0.35,
    "severe": 0.15,
}


# Số lỗi / 1000 candidate syllables.
ERROR_DENSITY = {
    "minor": (1, 3),
    "moderate": (4, 8),
    "severe": (9, 15),
}


# ============================================================
# RULE-BASED ERROR WEIGHTS
# ============================================================

RULE_ERROR_WEIGHTS = {
    "missing_diacritic": 0.24,
    "wrong_tone": 0.18,
    "delete_character": 0.12,
    "duplicate_character": 0.10,
    "transpose_character": 0.10,
    "keyboard_neighbor": 0.10,
    "onset_confusion": 0.10,
    "telex_unconverted": 0.06,
}


# ============================================================
# OUTPUT FILES
# ============================================================

ALL_DATA_PATH = (
    OUTPUT_DIR
    / "vn_newsspell_all.jsonl"
)

NOISY_DATA_PATH = (
    OUTPUT_DIR
    / "vn_newsspell_noisy.jsonl"
)

ERROR_SAMPLE_PATH = (
    OUTPUT_DIR
    / "error_samples.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "generation_summary.json"
)


# ============================================================
# REGEX
# ============================================================

WORD_RE = re.compile(
    r"[^\W\d_]+",
    re.UNICODE
)


# ============================================================
# BASIC UTILITIES
# ============================================================

def normalize_text(text):

    if text is None:
        return ""

    return unicodedata.normalize(
        "NFC",
        str(text)
    )


def to_natural(text):

    return text.replace(
        "_",
        " "
    )


def create_article_id(article):
    """
    source_id được tạo từ article sau NFC normalization.

    Hai article chỉ khác Unicode representation nhưng
    sau NFC giống nhau sẽ nhận cùng source_id.

    Đây chính là hành vi chúng ta muốn để deduplicate.
    """

    article = normalize_text(
        article
    )

    return hashlib.sha1(
        article.encode("utf-8")
    ).hexdigest()[:20]


def sha256_text(text):
    """
    Chỉ dùng cho metadata/debug duplicate.
    """

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def weighted_choice(
    rng,
    weights
):

    names = list(
        weights.keys()
    )

    values = list(
        weights.values()
    )

    return rng.choices(
        names,
        weights=values,
        k=1
    )[0]


# ============================================================
# LOAD TRUSTED LOOKUP
# ============================================================

def load_trusted_lookup():

    print(
        f"\nLoading trusted lookup:\n"
        f"{TRUSTED_LOOKUP_PATH}"
    )

    with open(
        TRUSTED_LOOKUP_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        lookup = json.load(f)

    print(
        f"Trusted words: "
        f"{len(lookup):,}"
    )

    return lookup


# ============================================================
# CASE PRESERVATION
# ============================================================

def preserve_word_case(
    original,
    replacement
):

    if not original:
        return replacement

    if original.isupper():
        return replacement.upper()

    if original[0].isupper():

        if not replacement:
            return replacement

        return (
            replacement[0].upper()
            + replacement[1:]
        )

    return replacement


# ============================================================
# RULE 1 - MISSING DIACRITIC
# ============================================================

def remove_diacritics(
    word,
    rng
):

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

        result.append(
            base
        )

    result = "".join(
        result
    )

    if result == word:
        return None

    return result


# ============================================================
# RULE 2 - WRONG TONE
# ============================================================

TONE_GROUPS = [
    "aáàảãạ",
    "ăắằẳẵặ",
    "âấầẩẫậ",

    "eéèẻẽẹ",
    "êếềểễệ",

    "iíìỉĩị",

    "oóòỏõọ",
    "ôốồổỗộ",
    "ơớờởỡợ",

    "uúùủũụ",
    "ưứừửữự",

    "yýỳỷỹỵ",
]


TONE_LOOKUP = {}

for group in TONE_GROUPS:

    for char in group:

        TONE_LOOKUP[
            char
        ] = group

        TONE_LOOKUP[
            char.upper()
        ] = group.upper()


def wrong_tone(
    word,
    rng
):

    candidates = []

    for index, char in enumerate(
        word
    ):

        if char not in TONE_LOOKUP:
            continue

        group = TONE_LOOKUP[
            char
        ]

        # group[0] = không có dấu thanh
        if char == group[0]:
            continue

        candidates.append(
            index
        )

    if not candidates:
        return None

    pos = rng.choice(
        candidates
    )

    original_char = word[
        pos
    ]

    group = TONE_LOOKUP[
        original_char
    ]

    alternatives = [
        char
        for char in group
        if char != original_char
    ]

    replacement = rng.choice(
        alternatives
    )

    result = (
        word[:pos]
        + replacement
        + word[pos + 1:]
    )

    result = unicodedata.normalize(
        "NFC",
        result
    )

    if result == word:
        return None

    return result


# ============================================================
# RULE 3 - DELETE CHARACTER
# ============================================================

def delete_character(
    word,
    rng
):

    if len(word) < 4:
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

    if result == word:
        return None

    return result


# ============================================================
# RULE 4 - DUPLICATE CHARACTER
# ============================================================

def duplicate_character(
    word,
    rng
):

    if len(word) < 3:
        return None

    pos = rng.randrange(
        len(word)
    )

    result = (
        word[:pos]
        + word[pos]
        + word[pos:]
    )

    if result == word:
        return None

    return result


# ============================================================
# RULE 5 - TRANSPOSE CHARACTER
# ============================================================

def transpose_character(
    word,
    rng
):

    if len(word) < 4:
        return None

    candidates = [
        i
        for i in range(
            len(word) - 1
        )
        if (
            word[i]
            != word[i + 1]
        )
    ]

    if not candidates:
        return None

    pos = rng.choice(
        candidates
    )

    chars = list(
        word
    )

    (
        chars[pos],
        chars[pos + 1]
    ) = (
        chars[pos + 1],
        chars[pos]
    )

    result = "".join(
        chars
    )

    if result == word:
        return None

    return result


# ============================================================
# RULE 6 - KEYBOARD NEIGHBOR
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


def keyboard_neighbor(
    word,
    rng
):

    candidates = []

    for index, char in enumerate(
        word
    ):

        lower = char.lower()

        if lower in KEYBOARD_NEIGHBORS:

            candidates.append(
                index
            )

    if not candidates:
        return None

    pos = rng.choice(
        candidates
    )

    original = word[
        pos
    ]

    replacement = rng.choice(
        KEYBOARD_NEIGHBORS[
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
# RULE 7 - ONSET CONFUSION
# ============================================================

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


def get_vietnamese_onset(
    word
):

    lower = word.lower()

    for onset in sorted(
        VIETNAMESE_ONSETS,
        key=len,
        reverse=True
    ):

        if lower.startswith(
            onset
        ):

            return onset

    return ""


def onset_confusion(
    word,
    rng
):

    onset = get_vietnamese_onset(
        word
    )

    if onset not in ONSET_CONFUSIONS:
        return None

    replacement = rng.choice(
        ONSET_CONFUSIONS[
            onset
        ]
    )

    original_onset = word[
        :len(onset)
    ]

    replacement = preserve_word_case(
        original_onset,
        replacement
    )

    result = (
        replacement
        + word[len(onset):]
    )

    if result == word:
        return None

    return result


# ============================================================
# RULE 8 - TELEX UNCONVERTED
# ============================================================

SHAPE_BREVE = "\u0306"
SHAPE_CIRCUMFLEX = "\u0302"
SHAPE_HORN = "\u031B"

TELEX_TONES = {
    "\u0301": "s",
    "\u0300": "f",
    "\u0309": "r",
    "\u0303": "x",
    "\u0323": "j",
}


def telex_unconverted(
    word,
    rng
):

    result = []

    tone_key = None
    changed = False

    for char in word:

        if char == "đ":

            result.append(
                "dd"
            )

            changed = True
            continue

        if char == "Đ":

            result.append(
                "DD"
            )

            changed = True
            continue

        decomposed = unicodedata.normalize(
            "NFD",
            char
        )

        base = decomposed[0]

        marks = decomposed[
            1:
        ]

        if SHAPE_BREVE in marks:

            result.append(
                base
            )

            result.append(
                "w"
            )

            changed = True

        elif SHAPE_CIRCUMFLEX in marks:

            result.append(
                base
            )

            result.append(
                base.lower()
            )

            changed = True

        elif SHAPE_HORN in marks:

            result.append(
                base
            )

            result.append(
                "w"
            )

            changed = True

        else:

            result.append(
                base
            )

        for mark in marks:

            if mark in TELEX_TONES:

                tone_key = TELEX_TONES[
                    mark
                ]

                changed = True

    if tone_key is not None:

        result.append(
            tone_key
        )

    result = "".join(
        result
    )

    if not changed:
        return None

    if result == word:
        return None

    return result


# ============================================================
# RULE FUNCTION MAP
# ============================================================

RULE_FUNCTIONS = {
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

    "telex_unconverted":
        telex_unconverted,
}


# ============================================================
# RULE-BASED MUTATION
# ============================================================

def rule_based_mutation(
    word,
    rng
):

    remaining_types = list(
        RULE_ERROR_WEIGHTS.keys()
    )

    while remaining_types:

        weights = [
            RULE_ERROR_WEIGHTS[
                error_type
            ]
            for error_type
            in remaining_types
        ]

        error_type = rng.choices(
            remaining_types,
            weights=weights,
            k=1
        )[0]

        function = RULE_FUNCTIONS[
            error_type
        ]

        result = function(
            word,
            rng
        )

        if result is not None:

            result = unicodedata.normalize(
                "NFC",
                result
            )

        if (
            result is not None
            and result != word
        ):

            return {
                "corrupted":
                    result,

                "error_type":
                    error_type,

                "error_source":
                    "rule_based",
            }

        remaining_types.remove(
            error_type
        )

    return None


# ============================================================
# MINED INSERTION FILTER
# ============================================================

def is_reasonable_mined_insertion(
    correct,
    wrong
):
    """
    Chỉ giữ mined insert_character
    dạng duplicate-like.

    Giữ:
        bộ   -> bbộ
        phát -> phátt
        là   -> llà

    Loại:
        bộ    -> bộv
        được  -> yđược
        phát  -> pihát
    """

    correct = normalize_text(
        correct
    )

    wrong = normalize_text(
        wrong
    )

    if (
        len(wrong)
        != len(correct) + 1
    ):

        return False

    for i in range(
        len(wrong)
    ):

        candidate = (
            wrong[:i]
            + wrong[i + 1:]
        )

        if candidate != correct:
            continue

        inserted = wrong[
            i
        ]

        left_same = (
            i > 0
            and wrong[i - 1]
            == inserted
        )

        right_same = (
            i + 1 < len(wrong)
            and wrong[i + 1]
            == inserted
        )

        if left_same or right_same:

            return True

    return False


# ============================================================
# MINED MUTATION
# ============================================================

def mined_mutation(
    word,
    trusted_lookup,
    rng
):

    key = word.lower()

    variants = trusted_lookup.get(
        key
    )

    if not variants:
        return None

    usable = []

    for variant in variants:

        error_type = variant[
            "error_type"
        ]

        wrong = variant[
            "wrong"
        ]

        if not wrong:
            continue

        if wrong == key:
            continue

        # ----------------------------------------------------
        # Transpose safety
        # ----------------------------------------------------

        if (
            error_type
            == "transpose_character"
            and len(word)
            < MIN_MINED_TRANSPOSE_LENGTH
        ):

            continue

        # ----------------------------------------------------
        # Insertion safety
        # ----------------------------------------------------

        if (
            error_type
            == "insert_character"
            and not is_reasonable_mined_insertion(
                key,
                wrong
            )
        ):

            continue

        usable.append(
            variant
        )

    if not usable:
        return None

    weights = [
        float(
            variant.get(
                "sampling_weight",
                1.0
            )
        )
        for variant in usable
    ]

    selected = rng.choices(
        usable,
        weights=weights,
        k=1
    )[0]

    corrupted = preserve_word_case(
        word,
        selected[
            "wrong"
        ]
    )

    corrupted = unicodedata.normalize(
        "NFC",
        corrupted
    )

    if corrupted == word:
        return None

    return {
        "corrupted":
            corrupted,

        "error_type":
            selected[
                "error_type"
            ],

        "error_source":
            "mined",

        "pattern_frequency":
            selected.get(
                "frequency",
                0
            ),

        "vsec_frequency":
            selected.get(
                "vsec_frequency",
                0
            ),

        "spelling_frequency":
            selected.get(
                "spelling_frequency",
                0
            ),
    }


# ============================================================
# HYBRID MUTATION
# ============================================================

def hybrid_mutation(
    word,
    trusted_lookup,
    rng
):

    key = word.lower()

    has_mined_pattern = (
        key in trusted_lookup
    )

    # --------------------------------------------------------
    # 1. Try mined
    # --------------------------------------------------------

    if (
        has_mined_pattern
        and rng.random()
        < MINED_PATTERN_PROBABILITY
    ):

        result = mined_mutation(
            word,
            trusted_lookup,
            rng
        )

        if result is not None:
            return result

    # --------------------------------------------------------
    # 2. Rule-based fallback
    # --------------------------------------------------------

    result = rule_based_mutation(
        word,
        rng
    )

    if result is not None:
        return result

    # --------------------------------------------------------
    # 3. Final mined fallback
    # --------------------------------------------------------

    if has_mined_pattern:

        result = mined_mutation(
            word,
            trusted_lookup,
            rng
        )

        if result is not None:
            return result

    return None


# ============================================================
# PROPER NOUN FILTER
# ============================================================

def is_probable_proper_noun(
    text,
    match
):

    word = match.group()

    if not word:
        return False

    if not word[0].isupper():
        return False

    start = match.start()

    before = text[
        :start
    ].rstrip()

    # Đầu field
    if not before:
        return False

    previous_char = before[
        -1
    ]

    # Đầu câu
    if previous_char in ".!?":
        return False

    return True


# ============================================================
# FIND CANDIDATES
# ============================================================

def find_candidates(
    field_name,
    text
):

    candidates = []

    for match in WORD_RE.finditer(
        text
    ):

        word = match.group()

        if len(word) < 2:
            continue

        # Abbreviation:
        # UBND, TAND, TP...
        if (
            word.isupper()
            and len(word) > 1
        ):

            continue

        if is_probable_proper_noun(
            text,
            match
        ):

            continue

        candidates.append({
            "field":
                field_name,

            "start":
                match.start(),

            "end":
                match.end(),

            "word":
                word,
        })

    return candidates


# ============================================================
# SEVERITY
# ============================================================

def choose_severity(
    rng
):

    return weighted_choice(
        rng,
        SEVERITY_WEIGHTS
    )


def choose_error_count(
    candidate_count,
    severity,
    rng
):

    low, high = ERROR_DENSITY[
        severity
    ]

    density = rng.uniform(
        low,
        high
    )

    error_count = round(
        candidate_count
        * density
        / 1000
    )

    return max(
        1,
        error_count
    )


# ============================================================
# APPLY EDITS
# ============================================================

def apply_edits(
    text,
    edits
):

    if not edits:
        return text, []

    sorted_edits = sorted(
        edits,
        key=lambda item:
            item["start"]
    )

    delta = 0

    # --------------------------------------------------------
    # Calculate noisy offsets
    # --------------------------------------------------------

    for edit in sorted_edits:

        noisy_start = (
            edit[
                "start"
            ]
            + delta
        )

        noisy_end = (
            noisy_start
            + len(
                edit[
                    "corrupted"
                ]
            )
        )

        edit[
            "noisy_start"
        ] = noisy_start

        edit[
            "noisy_end"
        ] = noisy_end

        delta += (
            len(
                edit[
                    "corrupted"
                ]
            )
            - len(
                edit[
                    "original"
                ]
            )
        )

    # --------------------------------------------------------
    # Apply edits from right -> left
    # --------------------------------------------------------

    result = text

    for edit in reversed(
        sorted_edits
    ):

        result = (
            result[
                :edit["start"]
            ]
            + edit[
                "corrupted"
            ]
            + result[
                edit["end"]:
            ]
        )

    return (
        result,
        sorted_edits
    )


# ============================================================
# CORRUPT RECORD
# ============================================================

def corrupt_record(
    clean_fields,
    trusted_lookup,
    rng
):

    candidates = []

    for (
        field_name,
        text
    ) in clean_fields.items():

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

    used_positions = set()

    errors_created = 0

    for candidate in candidates:

        if (
            errors_created
            >= requested_error_count
        ):

            break

        position_key = (
            candidate[
                "field"
            ],
            candidate[
                "start"
            ],
            candidate[
                "end"
            ],
        )

        if position_key in used_positions:
            continue

        mutation = hybrid_mutation(
            candidate[
                "word"
            ],
            trusted_lookup,
            rng
        )

        if mutation is None:
            continue

        edit = {
            "field":
                candidate[
                    "field"
                ],

            "start":
                candidate[
                    "start"
                ],

            "end":
                candidate[
                    "end"
                ],

            "original":
                candidate[
                    "word"
                ],

            "corrupted":
                mutation[
                    "corrupted"
                ],

            "error_type":
                mutation[
                    "error_type"
                ],

            "error_source":
                mutation[
                    "error_source"
                ],
        }

        if (
            mutation[
                "error_source"
            ]
            == "mined"
        ):

            edit[
                "pattern_frequency"
            ] = mutation.get(
                "pattern_frequency",
                0
            )

            edit[
                "vsec_frequency"
            ] = mutation.get(
                "vsec_frequency",
                0
            )

            edit[
                "spelling_frequency"
            ] = mutation.get(
                "spelling_frequency",
                0
            )

        field_edits[
            candidate[
                "field"
            ]
        ].append(
            edit
        )

        used_positions.add(
            position_key
        )

        errors_created += 1

    if errors_created == 0:
        return None

    noisy_fields = {}

    all_errors = []

    for field_name in [
        "title",
        "abstract",
        "article",
    ]:

        (
            noisy_text,
            edits
        ) = apply_edits(
            clean_fields[
                field_name
            ],
            field_edits[
                field_name
            ],
        )

        noisy_fields[
            field_name
        ] = noisy_text

        all_errors.extend(
            edits
        )

    return {
        "severity":
            severity,

        "error_count":
            len(
                all_errors
            ),

        "fields":
            noisy_fields,

        "errors":
            all_errors,

        "candidate_count":
            len(
                candidates
            ),
    }


# ============================================================
# OUTPUT RECORD
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
            error[
                "error_type"
            ]
            for error in errors
        )
    )

    error_sources = sorted(
        set(
            error[
                "error_source"
            ]
            for error in errors
            if (
                "error_source"
                in error
            )
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

        "error_sources":
            error_sources,

        # ----------------------------------------------------
        # Segmented representation
        # ----------------------------------------------------

        "title_segmented":
            fields[
                "title"
            ],

        "abstract_segmented":
            fields[
                "abstract"
            ],

        "article_segmented":
            fields[
                "article"
            ],

        # ----------------------------------------------------
        # Natural representation
        # ----------------------------------------------------

        "title_natural":
            to_natural(
                fields[
                    "title"
                ]
            ),

        "abstract_natural":
            to_natural(
                fields[
                    "abstract"
                ]
            ),

        "article_natural":
            to_natural(
                fields[
                    "article"
                ]
            ),

        # ----------------------------------------------------
        # Error metadata
        # ----------------------------------------------------

        "errors":
            errors,

        "generation_method":
            (
                "clean_original"
                if label == 0
                else "hybrid_v1_1_dedup"
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("VN-NEWSSPELL HYBRID V1.2 - DEDUP")
    print("=" * 80)

    print(
        f"\nVietNews:\n"
        f"{VIETNEWS_DIR}"
    )

    print(
        f"\nOutput:\n"
        f"{OUTPUT_DIR}"
    )

    # ========================================================
    # LOAD LOOKUP
    # ========================================================

    trusted_lookup = (
        load_trusted_lookup()
    )

    # ========================================================
    # LOAD VIETNEWS
    # ========================================================

    dataset = load_from_disk(
        str(VIETNEWS_DIR)
    )

    total_available = sum(
        len(
            dataset[split]
        )
        for split in dataset.keys()
    )

    if MAX_ARTICLES is None:

        target_articles = (
            total_available
        )

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
        f"Raw articles to scan: "
        f"{target_articles:,}"
    )

    print(
        f"Mined preference: "
        f"{MINED_PATTERN_PROBABILITY:.0%}"
    )

    print(
        "Mined insertion filter: "
        "duplicate-like only"
    )

    print(
        "Canonical deduplication: "
        "ENABLED"
    )

    # ========================================================
    # DEDUP TRACKING
    # ========================================================

    seen_source_ids = set()

    seen_source_metadata = {}

    duplicate_sources_skipped = 0

    duplicate_details = []

    # ========================================================
    # STATISTICS
    # ========================================================

    source_counter = Counter()

    error_type_counter = Counter()

    severity_counter = Counter()

    field_counter = Counter()

    total_clean = 0

    total_noisy = 0

    total_errors = 0

    unique_sources_written = 0

    raw_articles_scanned = 0

    error_samples = []

    # ========================================================
    # WRITE OUTPUT
    # ========================================================

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

        progress = tqdm(
            total=target_articles,
            desc="Hybrid generation"
        )

        stop_processing = False

        for split_name in dataset.keys():

            if stop_processing:
                break

            for row_index, row in enumerate(
                dataset[
                    split_name
                ]
            ):

                if (
                    raw_articles_scanned
                    >= target_articles
                ):

                    stop_processing = True
                    break

                # ============================================
                # COUNT RAW INPUT
                # ============================================

                raw_articles_scanned += 1

                progress.update(1)

                # ============================================
                # NORMALIZE CLEAN FIELDS
                # ============================================

                clean_fields = {
                    "title":
                        normalize_text(
                            row[
                                "title"
                            ]
                        ),

                    "abstract":
                        normalize_text(
                            row[
                                "abstract"
                            ]
                        ),

                    "article":
                        normalize_text(
                            row[
                                "article"
                            ]
                        ),
                }

                source_id = create_article_id(
                    clean_fields[
                        "article"
                    ]
                )

                # ============================================
                # CANONICAL DEDUPLICATION
                # ============================================

                if (
                    source_id
                    in seen_source_ids
                ):

                    duplicate_sources_skipped += 1

                    first = seen_source_metadata[
                        source_id
                    ]

                    duplicate_details.append({
                        "source_id":
                            source_id,

                        "article_sha256":
                            sha256_text(
                                clean_fields[
                                    "article"
                                ]
                            ),

                        "kept_split":
                            first[
                                "source_split"
                            ],

                        "kept_guid":
                            first[
                                "source_guid"
                            ],

                        "kept_row_index":
                            first[
                                "row_index"
                            ],

                        "skipped_split":
                            split_name,

                        "skipped_guid":
                            row[
                                "guid"
                            ],

                        "skipped_row_index":
                            row_index,

                        "title":
                            clean_fields[
                                "title"
                            ],
                    })

                    continue

                # ============================================
                # FIRST TIME WE SEE THIS SOURCE
                # ============================================

                seen_source_ids.add(
                    source_id
                )

                seen_source_metadata[
                    source_id
                ] = {
                    "source_split":
                        split_name,

                    "source_guid":
                        row[
                            "guid"
                        ],

                    "row_index":
                        row_index,
                }

                unique_sources_written += 1

                # ============================================
                # CLEAN RECORD
                # ============================================

                clean_record = (
                    create_output_record(
                        source_id=
                            source_id,

                        source_split=
                            split_name,

                        source_guid=
                            row[
                                "guid"
                            ],

                        variant_id=
                            (
                                f"{source_id}"
                                "_clean"
                            ),

                        label=0,

                        severity=
                            "clean",

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
                # NOISY VARIANTS
                # ============================================

                for variant_index in range(
                    NOISY_VARIANTS_PER_ARTICLE
                ):

                    seed_string = (
                        f"{RANDOM_SEED}-"
                        f"{source_id}-"
                        f"{variant_index}"
                    )

                    seed_value = int(
                        hashlib.sha1(
                            seed_string.encode(
                                "utf-8"
                            )
                        ).hexdigest()[:16],
                        16
                    )

                    rng = random.Random(
                        seed_value
                    )

                    corrupted = corrupt_record(
                        clean_fields,
                        trusted_lookup,
                        rng
                    )

                    if corrupted is None:

                        # Với dataset hiện tại gần như
                        # không xảy ra, nhưng vẫn log được
                        # thông qua clean/noisy validation.
                        continue

                    noisy_record = (
                        create_output_record(
                            source_id=
                                source_id,

                            source_split=
                                split_name,

                            source_guid=
                                row[
                                    "guid"
                                ],

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
                        line
                        + "\n"
                    )

                    noisy_file.write(
                        line
                        + "\n"
                    )

                    total_noisy += 1

                    severity_counter[
                        corrupted[
                            "severity"
                        ]
                    ] += 1

                    # ========================================
                    # ERROR STATS
                    # ========================================

                    for error in corrupted[
                        "errors"
                    ]:

                        total_errors += 1

                        error_type_counter[
                            error[
                                "error_type"
                            ]
                        ] += 1

                        source_counter[
                            error[
                                "error_source"
                            ]
                        ] += 1

                        field_counter[
                            error[
                                "field"
                            ]
                        ] += 1

                        if (
                            len(
                                error_samples
                            )
                            < 5000
                        ):

                            error_samples.append({
                                "source_id":
                                    source_id,

                                "field":
                                    error[
                                        "field"
                                    ],

                                "error_source":
                                    error[
                                        "error_source"
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

        progress.close()

    # ========================================================
    # SAVE ERROR SAMPLE CSV
    # ========================================================

    with open(
        ERROR_SAMPLE_PATH,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        fieldnames = [
            "source_id",
            "field",
            "error_source",
            "error_type",
            "original",
            "corrupted",
            "severity",
        ]

        writer = csv.DictWriter(
            f,
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
        "version":
            "hybrid_v1_2_dedup",

        "raw_articles_available":
            total_available,

        "raw_articles_scanned":
            raw_articles_scanned,

        "duplicate_sources_skipped":
            duplicate_sources_skipped,

        "unique_sources_written":
            unique_sources_written,

        "clean_records":
            total_clean,

        "noisy_records":
            total_noisy,

        "total_records":
            (
                total_clean
                + total_noisy
            ),

        "total_errors":
            total_errors,

        "average_errors_per_noisy_article":
            (
                total_errors
                / total_noisy
                if total_noisy
                else 0
            ),

        "error_source_distribution":
            dict(
                source_counter
            ),

        "error_type_distribution":
            dict(
                error_type_counter
            ),

        "severity_distribution":
            dict(
                severity_counter
            ),

        "field_distribution":
            dict(
                field_counter
            ),

        "trusted_lookup_words":
            len(
                trusted_lookup
            ),

        "duplicate_details":
            duplicate_details,

        "configuration": {
            "max_articles":
                MAX_ARTICLES,

            "noisy_variants_per_article":
                NOISY_VARIANTS_PER_ARTICLE,

            "mined_pattern_probability":
                MINED_PATTERN_PROBABILITY,

            "min_mined_transpose_length":
                MIN_MINED_TRANSPOSE_LENGTH,

            "mined_insert_filter":
                "duplicate_like_only",

            "canonical_deduplication":
                True,

            "canonical_normalization":
                "NFC",

            "source_id":
                "SHA1(normalized_article)[:20]",

            "random_seed":
                RANDOM_SEED,

            "rule_error_weights":
                RULE_ERROR_WEIGHTS,

            "severity_weights":
                SEVERITY_WEIGHTS,

            "error_density":
                ERROR_DENSITY,

            "preserve_underscore_segmentation":
                True,

            "protect_probable_proper_nouns":
                True,
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
    # PRINT RESULT
    # ========================================================

    print("\n" + "=" * 80)
    print("HYBRID GENERATION FINISHED")
    print("=" * 80)

    print(
        f"\nRaw articles scanned : "
        f"{raw_articles_scanned:,}"
    )

    print(
        f"Duplicate sources    : "
        f"{duplicate_sources_skipped:,}"
    )

    print(
        f"Unique sources       : "
        f"{unique_sources_written:,}"
    )

    print(
        f"Clean articles       : "
        f"{total_clean:,}"
    )

    print(
        f"Noisy articles       : "
        f"{total_noisy:,}"
    )

    print(
        f"Total records        : "
        f"{total_clean + total_noisy:,}"
    )

    print(
        f"Total errors         : "
        f"{total_errors:,}"
    )

    if total_noisy:

        print(
            f"Errors/noisy article : "
            f"{total_errors / total_noisy:.2f}"
        )

    # ========================================================
    # DUPLICATES
    # ========================================================

    print("\nDUPLICATE SOURCES")

    if not duplicate_details:

        print(
            "  None"
        )

    else:

        for item in duplicate_details:

            print(
                f"  source_id="
                f"{item['source_id']} | "
                f"kept="
                f"{item['kept_split']}/"
                f"{item['kept_guid']} | "
                f"skipped="
                f"{item['skipped_split']}/"
                f"{item['skipped_guid']}"
            )

    # ========================================================
    # ERROR SOURCE
    # ========================================================

    print("\nERROR SOURCE")

    for source, count in (
        source_counter.most_common()
    ):

        percentage = (
            count
            / total_errors
            * 100
            if total_errors
            else 0
        )

        print(
            f"  {source:15s}: "
            f"{count:7,} "
            f"({percentage:6.2f}%)"
        )

    # ========================================================
    # ERROR TYPE
    # ========================================================

    print("\nERROR TYPE")

    for error_type, count in (
        error_type_counter.most_common()
    ):

        percentage = (
            count
            / total_errors
            * 100
            if total_errors
            else 0
        )

        print(
            f"  {error_type:25s}: "
            f"{count:7,} "
            f"({percentage:6.2f}%)"
        )

    # ========================================================
    # SEVERITY
    # ========================================================

    print("\nSEVERITY")

    for severity, count in (
        severity_counter.most_common()
    ):

        percentage = (
            count
            / total_noisy
            * 100
            if total_noisy
            else 0
        )

        print(
            f"  {severity:10s}: "
            f"{count:7,} "
            f"({percentage:6.2f}%)"
        )

    # ========================================================
    # FIELD
    # ========================================================

    print("\nFIELD")

    for field, count in (
        field_counter.most_common()
    ):

        percentage = (
            count
            / total_errors
            * 100
            if total_errors
            else 0
        )

        print(
            f"  {field:10s}: "
            f"{count:7,} "
            f"({percentage:6.2f}%)"
        )

    # ========================================================
    # FILES
    # ========================================================

    print("\nFILES")

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
        "\nRaw VietNews was NOT modified."
    )

    print(
        "No train/validation/test split "
        "was created."
    )


if __name__ == "__main__":
    main()