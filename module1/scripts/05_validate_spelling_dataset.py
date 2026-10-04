from pathlib import Path
from collections import Counter, defaultdict
import json
import statistics
import unicodedata


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "interim"
    / "vn_newsspell_hybrid_v03_full"
    / "vn_newsspell_all.jsonl"
)

# Số example hiển thị cho mỗi:
#
#   error_source + error_type
#
EXAMPLES_PER_GROUP = 10


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("VN-NEWSSPELL SYNTHETIC DATA VALIDATION")
print("=" * 80)

print(
    f"\nDataset:\n{DATA_PATH}"
)

records = []

with open(
    DATA_PATH,
    "r",
    encoding="utf-8"
) as f:

    for line_number, line in enumerate(
        f,
        start=1
    ):

        line = line.strip()

        if not line:
            continue

        try:

            records.append(
                json.loads(line)
            )

        except json.JSONDecodeError as e:

            print(
                f"\nJSON ERROR at line "
                f"{line_number}: {e}"
            )

            raise


print(
    f"\nTotal records: "
    f"{len(records):,}"
)


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("LABEL DISTRIBUTION")
print("=" * 80)

label_counter = Counter(
    row.get("label")
    for row in records
)

for label, count in sorted(
    label_counter.items(),
    key=lambda x: str(x[0])
):

    print(
        f"  label={label}: "
        f"{count:,}"
    )


# ============================================================
# CLEAN / NOISY
# ============================================================

clean_records = [
    row
    for row in records
    if row.get("label") == 0
]

noisy_records = [
    row
    for row in records
    if row.get("label") == 1
]


print(
    f"\nClean records : "
    f"{len(clean_records):,}"
)

print(
    f"Noisy records : "
    f"{len(noisy_records):,}"
)


# ============================================================
# SOURCE GROUPS
# ============================================================

print("\n" + "=" * 80)
print("SOURCE PAIRS")
print("=" * 80)

source_groups = defaultdict(list)

for row in records:

    source_id = row.get(
        "source_id"
    )

    source_groups[
        source_id
    ].append(row)


invalid_source_groups = []

sources_without_clean = 0
sources_without_noisy = 0

for source_id, rows in source_groups.items():

    clean_count = sum(
        row.get("label") == 0
        for row in rows
    )

    noisy_count = sum(
        row.get("label") == 1
        for row in rows
    )

    if clean_count == 0:

        sources_without_clean += 1

    if noisy_count == 0:

        sources_without_noisy += 1

    # Với dataset hiện tại:
    # 1 clean + 1 noisy.
    #
    # Validator vẫn cho phép >1 noisy
    # để sau này có thể sinh nhiều variant.
    if (
        clean_count != 1
        or noisy_count < 1
    ):

        invalid_source_groups.append({
            "source_id":
                source_id,

            "clean_count":
                clean_count,

            "noisy_count":
                noisy_count,
        })


print(
    f"Unique source articles: "
    f"{len(source_groups):,}"
)

print(
    f"Sources without clean: "
    f"{sources_without_clean:,}"
)

print(
    f"Sources without noisy: "
    f"{sources_without_noisy:,}"
)

print(
    f"Invalid source groups: "
    f"{len(invalid_source_groups):,}"
)


if invalid_source_groups:

    print(
        "\nFirst invalid source groups:"
    )

    for item in (
        invalid_source_groups[:10]
    ):

        print(item)


# ============================================================
# VARIANT ID CHECK
# ============================================================

print("\n" + "=" * 80)
print("VARIANT ID CHECK")
print("=" * 80)

variant_ids = [
    row.get("variant_id")
    for row in records
]

variant_counter = Counter(
    variant_ids
)

duplicate_variant_ids = {
    variant_id: count
    for variant_id, count
    in variant_counter.items()
    if count > 1
}


print(
    f"Unique variant IDs: "
    f"{len(variant_counter):,}"
)

print(
    f"Duplicated variant IDs: "
    f"{len(duplicate_variant_ids):,}"
)


# ============================================================
# ERROR COUNT
# ============================================================

print("\n" + "=" * 80)
print("ERROR COUNT")
print("=" * 80)

error_counts = [
    int(
        row.get(
            "error_count",
            0
        )
    )
    for row in noisy_records
]


if error_counts:

    print(
        f"Min    : "
        f"{min(error_counts)}"
    )

    print(
        f"Max    : "
        f"{max(error_counts)}"
    )

    print(
        f"Mean   : "
        f"{statistics.mean(error_counts):.3f}"
    )

    print(
        f"Median : "
        f"{statistics.median(error_counts):.3f}"
    )


error_count_distribution = Counter(
    error_counts
)


print(
    "\nError count distribution:"
)

for count, frequency in sorted(
    error_count_distribution.items()
):

    print(
        f"  {count:2d} errors: "
        f"{frequency:,} articles"
    )


# ============================================================
# CHECK ERROR_COUNT AGAINST errors[]
# ============================================================

error_count_mismatch = []

for row in noisy_records:

    declared_count = int(
        row.get(
            "error_count",
            0
        )
    )

    actual_count = len(
        row.get(
            "errors",
            []
        )
    )

    if declared_count != actual_count:

        error_count_mismatch.append({
            "variant_id":
                row.get(
                    "variant_id"
                ),

            "declared":
                declared_count,

            "actual":
                actual_count,
        })


print(
    f"\nerror_count != len(errors): "
    f"{len(error_count_mismatch):,}"
)


# ============================================================
# SEVERITY
# ============================================================

print("\n" + "=" * 80)
print("SEVERITY DISTRIBUTION")
print("=" * 80)

severity_counter = Counter(
    row.get(
        "severity",
        "unknown"
    )
    for row in noisy_records
)


for severity, count in (
    severity_counter.most_common()
):

    percentage = (
        count
        / len(noisy_records)
        * 100
        if noisy_records
        else 0
    )

    print(
        f"  {severity:12s}: "
        f"{count:6,} "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# ERROR TYPE DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("ERROR TYPE DISTRIBUTION")
print("=" * 80)

error_type_counter = Counter()

for row in noisy_records:

    for error in row.get(
        "errors",
        []
    ):

        error_type = error.get(
            "error_type",
            "unknown"
        )

        error_type_counter[
            error_type
        ] += 1


total_errors = sum(
    error_type_counter.values()
)


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
        f"{count:6,} "
        f"({percentage:6.2f}%)"
    )


print(
    f"\nTotal errors from metadata: "
    f"{total_errors:,}"
)


# ============================================================
# ERROR FIELD DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("ERROR FIELD DISTRIBUTION")
print("=" * 80)

field_counter = Counter()

for row in noisy_records:

    for error in row.get(
        "errors",
        []
    ):

        field_counter[
            error.get(
                "field",
                "unknown"
            )
        ] += 1


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
        f"  {field:12s}: "
        f"{count:6,} "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# ERROR SOURCE DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("ERROR SOURCE DISTRIBUTION")
print("=" * 80)

source_counter = Counter()

for row in noisy_records:

    for error in row.get(
        "errors",
        []
    ):

        source = error.get(
            "error_source",
            "unknown"
        )

        source_counter[
            source
        ] += 1


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
        f"{count:6,} "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# ERROR SOURCE × TYPE
# ============================================================

print("\n" + "=" * 80)
print("ERROR SOURCE x TYPE")
print("=" * 80)

source_type_counter = Counter()

for row in noisy_records:

    for error in row.get(
        "errors",
        []
    ):

        source = error.get(
            "error_source",
            "unknown"
        )

        error_type = error.get(
            "error_type",
            "unknown"
        )

        source_type_counter[
            (
                source,
                error_type
            )
        ] += 1


sources = sorted(
    set(
        source
        for source, _
        in source_type_counter.keys()
    )
)


for source in sources:

    print(
        f"\n[{source}]"
    )

    source_items = [
        (
            error_type,
            count
        )
        for (
            item_source,
            error_type
        ), count
        in source_type_counter.items()
        if item_source == source
    ]

    source_items.sort(
        key=lambda x:
            x[1],
        reverse=True
    )

    source_total = sum(
        count
        for _, count in source_items
    )

    for error_type, count in source_items:

        percentage = (
            count
            / source_total
            * 100
            if source_total
            else 0
        )

        print(
            f"  {error_type:25s}: "
            f"{count:6,} "
            f"({percentage:6.2f}%)"
        )


# ============================================================
# INVALID MUTATIONS
# ============================================================

print("\n" + "=" * 80)
print("INVALID MUTATIONS")
print("=" * 80)

unchanged_errors = []

empty_original = []

empty_corrupted = []


for row in noisy_records:

    for error in row.get(
        "errors",
        []
    ):

        original = error.get(
            "original",
            ""
        )

        corrupted = error.get(
            "corrupted",
            ""
        )

        if original == corrupted:

            unchanged_errors.append(
                error
            )

        if not original:

            empty_original.append(
                error
            )

        if not corrupted:

            empty_corrupted.append(
                error
            )


print(
    f"original == corrupted: "
    f"{len(unchanged_errors):,}"
)

print(
    f"Empty original: "
    f"{len(empty_original):,}"
)

print(
    f"Empty corrupted: "
    f"{len(empty_corrupted):,}"
)


# ============================================================
# CLEAN VS NOISY ACTUAL CHANGE
# ============================================================

print("\n" + "=" * 80)
print("CLEAN / NOISY CHANGE CHECK")
print("=" * 80)

pairs_with_no_change = 0

pair_count = 0


for source_id, rows in (
    source_groups.items()
):

    clean = next(
        (
            row
            for row in rows
            if row.get("label") == 0
        ),
        None
    )

    if clean is None:
        continue

    noisies = [
        row
        for row in rows
        if row.get("label") == 1
    ]

    for noisy in noisies:

        pair_count += 1

        same_title = (
            clean.get(
                "title_segmented",
                ""
            )
            ==
            noisy.get(
                "title_segmented",
                ""
            )
        )

        same_abstract = (
            clean.get(
                "abstract_segmented",
                ""
            )
            ==
            noisy.get(
                "abstract_segmented",
                ""
            )
        )

        same_article = (
            clean.get(
                "article_segmented",
                ""
            )
            ==
            noisy.get(
                "article_segmented",
                ""
            )
        )

        if (
            same_title
            and same_abstract
            and same_article
        ):

            pairs_with_no_change += 1


print(
    f"Clean/noisy pairs checked: "
    f"{pair_count:,}"
)

print(
    "Clean/noisy pairs with no actual "
    f"text change: {pairs_with_no_change:,}"
)


# ============================================================
# ARTICLE LENGTH DIFFERENCE
# ============================================================

print("\n" + "=" * 80)
print("ARTICLE LENGTH DIFFERENCE")
print("=" * 80)

article_length_differences = []


for source_id, rows in (
    source_groups.items()
):

    clean = next(
        (
            row
            for row in rows
            if row.get("label") == 0
        ),
        None
    )

    if clean is None:
        continue

    clean_text = clean.get(
        "article_segmented",
        ""
    )

    for noisy in rows:

        if noisy.get("label") != 1:
            continue

        noisy_text = noisy.get(
            "article_segmented",
            ""
        )

        article_length_differences.append(
            len(noisy_text)
            - len(clean_text)
        )


if article_length_differences:

    print(
        f"Mean delta   : "
        f"{statistics.mean(article_length_differences):.3f}"
    )

    print(
        f"Median delta : "
        f"{statistics.median(article_length_differences):.3f}"
    )

    print(
        f"Min delta    : "
        f"{min(article_length_differences)}"
    )

    print(
        f"Max delta    : "
        f"{max(article_length_differences)}"
    )


# ============================================================
# UNDERSCORE CHECK
# ============================================================

print("\n" + "=" * 80)
print("UNDERSCORE CHECK")
print("=" * 80)

underscore_changed_pairs = 0

underscore_changed_by_field = Counter()


for source_id, rows in (
    source_groups.items()
):

    clean = next(
        (
            row
            for row in rows
            if row.get("label") == 0
        ),
        None
    )

    if clean is None:
        continue

    for noisy in rows:

        if noisy.get("label") != 1:
            continue

        pair_changed = False

        for field in [
            "title",
            "abstract",
            "article",
        ]:

            key = (
                f"{field}_segmented"
            )

            clean_count = (
                clean.get(
                    key,
                    ""
                ).count("_")
            )

            noisy_count = (
                noisy.get(
                    key,
                    ""
                ).count("_")
            )

            if clean_count != noisy_count:

                pair_changed = True

                underscore_changed_by_field[
                    field
                ] += 1

        if pair_changed:

            underscore_changed_pairs += 1


print(
    f"Pairs with changed '_' count: "
    f"{underscore_changed_pairs:,}"
)

for field, count in (
    underscore_changed_by_field.items()
):

    print(
        f"  {field:10s}: "
        f"{count:,}"
    )


# ============================================================
# UNICODE NFC CHECK
# ============================================================

print("\n" + "=" * 80)
print("UNICODE NFC CHECK")
print("=" * 80)

non_nfc_fields = []

non_nfc_error_tokens = []


for row in records:

    for key in [
        "title_segmented",
        "abstract_segmented",
        "article_segmented",
        "title_natural",
        "abstract_natural",
        "article_natural",
    ]:

        text = row.get(
            key,
            ""
        )

        if (
            text
            != unicodedata.normalize(
                "NFC",
                text
            )
        ):

            non_nfc_fields.append({
                "variant_id":
                    row.get(
                        "variant_id"
                    ),

                "field":
                    key,
            })


for row in noisy_records:

    for error in row.get(
        "errors",
        []
    ):

        for key in [
            "original",
            "corrupted",
        ]:

            text = error.get(
                key,
                ""
            )

            if (
                text
                != unicodedata.normalize(
                    "NFC",
                    text
                )
            ):

                non_nfc_error_tokens.append({
                    "variant_id":
                        row.get(
                            "variant_id"
                        ),

                    "key":
                        key,

                    "value":
                        text,
                })


print(
    f"Non-NFC text fields: "
    f"{len(non_nfc_fields):,}"
)

print(
    f"Non-NFC error tokens: "
    f"{len(non_nfc_error_tokens):,}"
)


if non_nfc_error_tokens:

    print(
        "\nExamples:"
    )

    for item in (
        non_nfc_error_tokens[:10]
    ):

        print(item)


# ============================================================
# OFFSET VALIDATION
# ============================================================

print("\n" + "=" * 80)
print("ERROR OFFSET VALIDATION")
print("=" * 80)

clean_offset_errors = []

noisy_offset_errors = []

missing_offset_metadata = 0


for source_id, rows in (
    source_groups.items()
):

    clean = next(
        (
            row
            for row in rows
            if row.get("label") == 0
        ),
        None
    )

    if clean is None:
        continue

    noisies = [
        row
        for row in rows
        if row.get("label") == 1
    ]

    for noisy in noisies:

        for error in noisy.get(
            "errors",
            []
        ):

            field = error.get(
                "field"
            )

            if field not in {
                "title",
                "abstract",
                "article",
            }:

                continue

            clean_key = (
                f"{field}_segmented"
            )

            noisy_key = clean_key

            original = error.get(
                "original",
                ""
            )

            corrupted = error.get(
                "corrupted",
                ""
            )

            required_keys = [
                "start",
                "end",
                "noisy_start",
                "noisy_end",
            ]

            if not all(
                key in error
                for key in required_keys
            ):

                missing_offset_metadata += 1

                continue

            start = int(
                error["start"]
            )

            end = int(
                error["end"]
            )

            noisy_start = int(
                error["noisy_start"]
            )

            noisy_end = int(
                error["noisy_end"]
            )

            clean_text = clean.get(
                clean_key,
                ""
            )

            noisy_text = noisy.get(
                noisy_key,
                ""
            )

            # ----------------------------------------------
            # Check original span
            # ----------------------------------------------

            clean_span = (
                clean_text[
                    start:end
                ]
            )

            if clean_span != original:

                clean_offset_errors.append({
                    "variant_id":
                        noisy.get(
                            "variant_id"
                        ),

                    "field":
                        field,

                    "expected":
                        original,

                    "actual":
                        clean_span,

                    "start":
                        start,

                    "end":
                        end,
                })

            # ----------------------------------------------
            # Check corrupted span
            # ----------------------------------------------

            noisy_span = (
                noisy_text[
                    noisy_start:
                    noisy_end
                ]
            )

            if noisy_span != corrupted:

                noisy_offset_errors.append({
                    "variant_id":
                        noisy.get(
                            "variant_id"
                        ),

                    "field":
                        field,

                    "expected":
                        corrupted,

                    "actual":
                        noisy_span,

                    "start":
                        noisy_start,

                    "end":
                        noisy_end,
                })


print(
    f"Missing offset metadata: "
    f"{missing_offset_metadata:,}"
)

print(
    f"Invalid clean offsets: "
    f"{len(clean_offset_errors):,}"
)

print(
    f"Invalid noisy offsets: "
    f"{len(noisy_offset_errors):,}"
)


if clean_offset_errors:

    print(
        "\nClean offset examples:"
    )

    for item in (
        clean_offset_errors[:5]
    ):

        print(item)


if noisy_offset_errors:

    print(
        "\nNoisy offset examples:"
    )

    for item in (
        noisy_offset_errors[:5]
    ):

        print(item)


# ============================================================
# GENERATION METHOD
# ============================================================

print("\n" + "=" * 80)
print("GENERATION METHOD")
print("=" * 80)

generation_method_counter = Counter(
    row.get(
        "generation_method",
        "unknown"
    )
    for row in records
)


for method, count in (
    generation_method_counter.most_common()
):

    print(
        f"  {method:20s}: "
        f"{count:,}"
    )


# ============================================================
# MINED METADATA CHECK
# ============================================================

print("\n" + "=" * 80)
print("MINED METADATA CHECK")
print("=" * 80)

mined_errors = []

mined_missing_frequency = 0


for row in noisy_records:

    for error in row.get(
        "errors",
        []
    ):

        if (
            error.get(
                "error_source"
            )
            != "mined"
        ):
            continue

        mined_errors.append(
            error
        )

        if (
            "pattern_frequency"
            not in error
        ):

            mined_missing_frequency += 1


print(
    f"Mined errors: "
    f"{len(mined_errors):,}"
)

print(
    "Mined errors missing "
    f"pattern_frequency: "
    f"{mined_missing_frequency:,}"
)


if mined_errors:

    frequencies = [
        int(
            error.get(
                "pattern_frequency",
                0
            )
        )
        for error in mined_errors
    ]

    print(
        f"Pattern frequency min    : "
        f"{min(frequencies):,}"
    )

    print(
        f"Pattern frequency max    : "
        f"{max(frequencies):,}"
    )

    print(
        f"Pattern frequency median : "
        f"{statistics.median(frequencies):,.1f}"
    )


# ============================================================
# ERROR EXAMPLES BY SOURCE + TYPE
# ============================================================

print("\n" + "=" * 80)
print("ERROR EXAMPLES BY SOURCE")
print("=" * 80)

examples_by_source_type = defaultdict(
    list
)


for row in noisy_records:

    for error in row.get(
        "errors",
        []
    ):

        source = error.get(
            "error_source",
            "unknown"
        )

        error_type = error.get(
            "error_type",
            "unknown"
        )

        key = (
            source,
            error_type
        )

        if (
            len(
                examples_by_source_type[
                    key
                ]
            )
            >= EXAMPLES_PER_GROUP
        ):

            continue

        example = (
            error.get(
                "original",
                ""
            ),
            error.get(
                "corrupted",
                ""
            )
        )

        # Tránh duplicate example.
        if (
            example
            not in
            examples_by_source_type[
                key
            ]
        ):

            examples_by_source_type[
                key
            ].append(
                example
            )


# ------------------------------------------------------------
# Print mined first
# ------------------------------------------------------------

source_order = [
    "mined",
    "rule_based",
    "unknown",
]


all_sources = set(
    source
    for source, _
    in examples_by_source_type.keys()
)


ordered_sources = []

for source in source_order:

    if source in all_sources:

        ordered_sources.append(
            source
        )


for source in sorted(
    all_sources
):

    if (
        source
        not in ordered_sources
    ):

        ordered_sources.append(
            source
        )


for source in ordered_sources:

    print("\n" + "#" * 80)

    print(
        f"SOURCE: {source.upper()}"
    )

    print("#" * 80)

    error_types = sorted(
        error_type
        for (
            item_source,
            error_type
        )
        in examples_by_source_type.keys()
        if item_source == source
    )

    for error_type in error_types:

        print(
            f"\n[{source} / "
            f"{error_type}]"
        )

        examples = (
            examples_by_source_type[
                (
                    source,
                    error_type
                )
            ]
        )

        for (
            original,
            corrupted
        ) in examples:

            print(
                f"  {original:20s}"
                f" -> "
                f"{corrupted}"
            )


# ============================================================
# VALIDATION SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)


checks = {
    "duplicate_variant_ids":
        len(
            duplicate_variant_ids
        ),

    "invalid_source_groups":
        len(
            invalid_source_groups
        ),

    "error_count_mismatch":
        len(
            error_count_mismatch
        ),

    "unchanged_errors":
        len(
            unchanged_errors
        ),

    "empty_original":
        len(
            empty_original
        ),

    "empty_corrupted":
        len(
            empty_corrupted
        ),

    "pairs_with_no_change":
        pairs_with_no_change,

    "underscore_changed_pairs":
        underscore_changed_pairs,

    "non_nfc_fields":
        len(
            non_nfc_fields
        ),

    "non_nfc_error_tokens":
        len(
            non_nfc_error_tokens
        ),

    "missing_offset_metadata":
        missing_offset_metadata,

    "invalid_clean_offsets":
        len(
            clean_offset_errors
        ),

    "invalid_noisy_offsets":
        len(
            noisy_offset_errors
        ),
}


all_structural_checks_passed = all(
    value == 0
    for value in checks.values()
)


for name, value in checks.items():

    status = (
        "OK"
        if value == 0
        else "CHECK"
    )

    print(
        f"{status:7s} | "
        f"{name:30s} | "
        f"{value:,}"
    )


print("\n" + "-" * 80)

if all_structural_checks_passed:

    print(
        "STRUCTURAL VALIDATION: PASSED"
    )

else:

    print(
        "STRUCTURAL VALIDATION: "
        "REQUIRES REVIEW"
    )


print("\n" + "=" * 80)
print("VALIDATION FINISHED")
print("=" * 80)