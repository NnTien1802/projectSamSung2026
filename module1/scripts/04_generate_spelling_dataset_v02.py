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
    / "vn_newsspell_v03"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PILOT DATASET
# ============================================================

# Chạy thử trước với 5.000 bài.
MAX_ARTICLES = 5000

# Khi generator đã được kiểm tra kỹ:
# MAX_ARTICLES = None


NOISY_VARIANTS_PER_ARTICLE = 1

RANDOM_SEED = 20260918


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

# Tổng = 1.0
ERROR_WEIGHTS = {
    "missing_diacritic": 0.24,
    "wrong_tone": 0.18,
    "delete_character": 0.12,
    "duplicate_character": 0.10,
    "transpose_character": 0.10,
    "keyboard_neighbor": 0.10,
    "onset_confusion": 0.10,
    "telex_unconverted": 0.06,
}


SEVERITY_WEIGHTS = {
    "minor": 0.50,
    "moderate": 0.35,
    "severe": 0.15,
}


# Số lỗi / 1000 syllable
ERROR_DENSITY = {
    "minor": (1, 3),
    "moderate": (4, 8),
    "severe": (9, 15),
}


# ============================================================
# REGEX
# ============================================================

# Lấy chuỗi chữ Unicode.
# "_" không được xem là một phần của syllable ở bước tạo lỗi.
LETTER_RE = re.compile(
    r"[^\W\d_]+",
    re.UNICODE
)


# ============================================================
# BASIC UTILITIES
# ============================================================

def normalize_text(text):
    """
    Chỉ Unicode normalize.

    KHÔNG:
    - sửa chính tả
    - xóa _
    - lowercase
    """

    if text is None:
        return ""

    return unicodedata.normalize(
        "NFC",
        str(text)
    )


def to_natural(text):
    """
    Representation dùng như văn bản tiếng Việt bình thường.

    VD:
        người_dân -> người dân

    Không thay đổi representation segmented gốc.
    """

    return text.replace("_", " ")


def create_article_id(article):
    """
    GUID của VietNews không unique.
    Tạo source_id riêng dựa trên nội dung article.
    """

    return hashlib.sha1(
        article.encode("utf-8")
    ).hexdigest()[:20]


def weighted_choice(rng, weights):

    keys = list(weights.keys())
    values = list(weights.values())

    return rng.choices(
        keys,
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
# 2. WRONG TONE - SAFE VIETNAMESE VERSION
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


# Tạo lookup:
#
# á -> aáàảãạ
# ấ -> âấầẩẫậ
# ộ -> ôốồổỗộ
# ...
TONE_LOOKUP = {}

for group in TONE_GROUPS:

    for char in group:

        TONE_LOOKUP[char] = group

        TONE_LOOKUP[
            char.upper()
        ] = group.upper()


def wrong_tone(word, rng):
    """
    Thay dấu thanh nhưng giữ nguyên loại nguyên âm.

    Ví dụ:
        á -> à / ả / ã / ạ / a
        ộ -> ô / ố / ồ / ổ / ỗ
        ệ -> ê / ế / ề / ể / ễ

    Không tạo combining Unicode bất thường.
    """

    candidates = []

    for index, char in enumerate(word):

        if char not in TONE_LOOKUP:
            continue

        group = TONE_LOOKUP[char]

        # Chỉ coi là candidate nếu ký tự
        # hiện tại thực sự có dấu thanh.
        #
        # index 0 trong group = không dấu thanh
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

    original_char = word[pos]

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
# 3. DELETE CHARACTER
# ============================================================

def delete_character(word, rng):

    # Tránh:
    # nạn -> nn
    # Hoà -> Ho
    #
    # Những từ quá ngắn dễ tạo lỗi phi thực tế.
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

    return result


# ============================================================
# 4. DUPLICATE CHARACTER
# ============================================================

def duplicate_character(word, rng):

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
# 5. TRANSPOSE CHARACTER
# ============================================================

def transpose_character(word, rng):

    if len(word) < 4:
        return None

    candidates = [
        i
        for i in range(len(word) - 1)
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
        chars[pos]
    )

    result = "".join(chars)

    if result == word:
        return None

    return result


# ============================================================
# 6. KEYBOARD NEIGHBOR
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

    result = (
        word[:pos]
        + replacement
        + word[pos + 1:]
    )

    if result == word:
        return None

    return result


# ============================================================
# 7. VIETNAMESE ONSET CONFUSION
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


def get_vietnamese_onset(word):

    lower = word.lower()

    for onset in sorted(
        VIETNAMESE_ONSETS,
        key=len,
        reverse=True
    ):

        if lower.startswith(onset):
            return onset

    return ""


def preserve_case(original, replacement):

    if original.isupper():
        return replacement.upper()

    if (
        original
        and original[0].isupper()
    ):

        return (
            replacement[0].upper()
            + replacement[1:]
        )

    return replacement


def onset_confusion(word, rng):

    onset = get_vietnamese_onset(
        word
    )

    # Ví dụ:
    # ngày -> onset = ng
    #
    # "ng" không nằm trong confusion map
    # nên sẽ KHÔNG thành lgày.
    if onset not in ONSET_CONFUSIONS:
        return None

    replacement = rng.choice(
        ONSET_CONFUSIONS[
            onset
        ]
    )

    original_onset = (
        word[:len(onset)]
    )

    replacement = preserve_case(
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
# 8. TELEX UNCONVERTED
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

        # ----------------------------------------------------
        # Character shape
        # ----------------------------------------------------

        if SHAPE_BREVE in marks:
            # ă -> aw
            result.append(base)
            result.append("w")
            changed = True

        elif SHAPE_CIRCUMFLEX in marks:
            # â -> aa
            # ê -> ee
            # ô -> oo
            result.append(base)
            result.append(base.lower())
            changed = True

        elif SHAPE_HORN in marks:
            # ơ -> ow
            # ư -> uw
            result.append(base)
            result.append("w")
            changed = True

        else:
            result.append(base)

        # ----------------------------------------------------
        # Tone
        # ----------------------------------------------------

        for mark in marks:

            if mark in TELEX_TONES:

                tone_key = (
                    TELEX_TONES[
                        mark
                    ]
                )

                changed = True

    if tone_key is not None:
        result.append(
            tone_key
        )

    result = "".join(result)

    if not changed:
        return None

    if result == word:
        return None

    return result


# ============================================================
# ERROR FUNCTION MAP
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

    "telex_unconverted":
        telex_unconverted,
}


# ============================================================
# CANDIDATE FILTER
# ============================================================

def is_probable_proper_noun(
    text,
    match
):
    """
    Filter đơn giản để hạn chế phá tên riêng.

    Từ viết hoa:
    - nếu nằm đầu câu -> vẫn cho phép
    - nếu nằm giữa câu -> bỏ qua

    Đây không phải NER hoàn chỉnh.
    """

    word = match.group()

    if not word:
        return False

    if not word[0].isupper():
        return False

    start = match.start()

    before = text[:start].rstrip()

    # Đầu field.
    if not before:
        return False

    previous_char = (
        before[-1]
    )

    # Sau dấu kết thúc câu:
    # có thể chỉ là chữ cái đầu câu.
    if previous_char in ".!?":
        return False

    return True


def find_candidates(
    field_name,
    text
):

    candidates = []

    for match in LETTER_RE.finditer(
        text
    ):

        word = match.group()

        # --------------------------------------------
        # Quá ngắn
        # --------------------------------------------

        if len(word) < 2:
            continue

        # --------------------------------------------
        # Abbreviation:
        # TP
        # UBND
        # TAND
        # ...
        # --------------------------------------------

        if (
            word.isupper()
            and len(word) > 1
        ):
            continue

        # --------------------------------------------
        # Hạn chế tên riêng
        # --------------------------------------------

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
# MUTATE WORD
# ============================================================

def mutate_word(
    word,
    rng
):
    """
    Chọn error type theo ERROR_WEIGHTS.

    Nếu lỗi được chọn không áp dụng được,
    loại lỗi đó khỏi candidates rồi thử loại khác.

    Không ép diversity theo từng article.
    """

    remaining_types = list(
        ERROR_WEIGHTS.keys()
    )

    while remaining_types:

        remaining_weights = [
            ERROR_WEIGHTS[t]
            for t in remaining_types
        ]

        error_type = rng.choices(
            remaining_types,
            weights=remaining_weights,
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

        if result is not None:
            result = unicodedata.normalize(
                "NFC",
            result
            )

        if (
            result is not None
            and result != word
        ):
            return (
                result,
                error_type
            )

        # Error type không áp dụng được
        # với từ này.
        remaining_types.remove(
            error_type
        )

    return None, None


# ============================================================
# SEVERITY
# ============================================================

def choose_severity(rng):

    return weighted_choice(
        rng,
        SEVERITY_WEIGHTS
    )


def choose_error_count(
    candidate_count,
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
        candidate_count
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

def apply_edits(
    text,
    edits
):

    if not edits:
        return text, []

    sorted_edits = sorted(
        edits,
        key=lambda x: x["start"]
    )

    # --------------------------------------------------------
    # Calculate offsets in noisy version
    # --------------------------------------------------------

    delta = 0

    for edit in sorted_edits:

        noisy_start = (
            edit["start"]
            + delta
        )

        noisy_end = (
            noisy_start
            + len(
                edit["corrupted"]
            )
        )

        edit["noisy_start"] = (
            noisy_start
        )

        edit["noisy_end"] = (
            noisy_end
        )

        delta += (
            len(
                edit["corrupted"]
            )
            - len(
                edit["original"]
            )
        )

    # --------------------------------------------------------
    # Apply backwards
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
# CORRUPT RECORD
# ============================================================

def corrupt_record(
    clean_fields,
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

    errors_created = 0

    used_positions = set()

    for candidate in candidates:

        if (
            errors_created
            >= requested_error_count
        ):
            break

        position_key = (
            candidate["field"],
            candidate["start"],
            candidate["end"],
        )

        if position_key in used_positions:
            continue

        (
            corrupted,
            error_type
        ) = mutate_word(
            candidate["word"],
            rng
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
            len(all_errors),

        "errors":
            all_errors,

        "fields":
            noisy_fields,

        "candidate_count":
            len(candidates),
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
            error["error_type"]
            for error in errors
        )
    )

    return {
        # ----------------------------------------------------
        # Source metadata
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Segmented representation
        # ----------------------------------------------------

        "title_segmented":
            fields["title"],

        "abstract_segmented":
            fields["abstract"],

        "article_segmented":
            fields["article"],

        # ----------------------------------------------------
        # Natural representation
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        "errors":
            errors,

        "generation_method":
            (
                "clean_original"
                if label == 0
                else "synthetic_v02"
            ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("VN-NEWSSPELL V0.2")
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
        f"Articles to process: "
        f"{target_articles:,}"
    )

    print(
        "Noisy variants/article: "
        f"{NOISY_VARIANTS_PER_ARTICLE}"
    )

    print("\nError weights:")

    for (
        error_type,
        weight
    ) in ERROR_WEIGHTS.items():

        print(
            f"  {error_type:25s}"
            f"{weight:7.2%}"
        )

    # ========================================================
    # STATISTICS
    # ========================================================

    error_type_counter = Counter()
    severity_counter = Counter()
    field_counter = Counter()

    total_clean = 0
    total_noisy = 0
    total_errors = 0

    error_samples = []

    # ========================================================
    # WRITE DATA
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

        processed = 0

        progress = tqdm(
            total=target_articles,
            desc="Generating dataset"
        )

        for split_name in dataset.keys():

            for row in dataset[
                split_name
            ]:

                if (
                    processed
                    >= target_articles
                ):
                    break

                # ==================================================
                # CLEAN FIELDS
                # ==================================================

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

                source_id = (
                    create_article_id(
                        clean_fields[
                            "article"
                        ]
                    )
                )

                # ==================================================
                # CLEAN RECORD
                # ==================================================

                clean_record = (
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

                # ==================================================
                # NOISY VERSIONS
                # ==================================================

                for variant_index in range(
                    NOISY_VARIANTS_PER_ARTICLE
                ):

                    # ----------------------------------------------
                    # Deterministic seed
                    # ----------------------------------------------

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

                    corrupted = (
                        corrupt_record(
                            clean_fields,
                            rng
                        )
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

                    # ----------------------------------------------
                    # Error statistics
                    # ----------------------------------------------

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
                            error[
                                "field"
                            ]
                        ] += 1

                        # Save samples for manual inspection.
                        if (
                            len(
                                error_samples
                            )
                            < 2000
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
    # SAVE ERROR SAMPLE CSV
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
        "version":
            "0.2",

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

            "protect_probable_proper_nouns":
                True,

            "preserve_underscore_segmentation":
                True,
        },
    }

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2
        )

    # ========================================================
    # PRINT RESULT
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

    for (
        key,
        value
    ) in severity_counter.items():

        print(
            f"  {key:10s}: "
            f"{value:,}"
        )

    print("\nError types:")

    for (
        key,
        value
    ) in error_type_counter.most_common():

        percentage = (
            value
            / total_errors
            * 100
            if total_errors
            else 0
        )

        print(
            f"  {key:25s}: "
            f"{value:7,} "
            f"({percentage:6.2f}%)"
        )

    print("\nFields:")

    for (
        key,
        value
    ) in field_counter.most_common():

        print(
            f"  {key:10s}: "
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
        "\nRaw VietNews was NOT modified."
    )

    print(
        "No train/validation/test split "
        "was created."
    )


if __name__ == "__main__":
    main()