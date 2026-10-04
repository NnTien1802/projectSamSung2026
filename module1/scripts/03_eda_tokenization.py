from pathlib import Path
from collections import Counter
import json
import re

from datasets import load_from_disk


# ============================================================
# 1. PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "vietnews"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

INTERIM_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = INTERIM_DIR / "vietnews_tokenization_eda.json"


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 80)
print("VIETNEWS - EDA V2")
print("TOKENIZATION / UNDERSCORE / GUID ANALYSIS")
print("=" * 80)

print("\nLoading dataset...")

ds = load_from_disk(str(RAW_DIR))

print(ds)


# ============================================================
# 3. COMBINE SPLITS
# ============================================================

records = []

for split_name in ds.keys():

    print(
        f"\nReading {split_name}: "
        f"{len(ds[split_name]):,} articles"
    )

    for row in ds[split_name]:

        records.append({
            "guid": row["guid"],
            "title": row["title"] or "",
            "abstract": row["abstract"] or "",
            "article": row["article"] or "",
            "original_split": split_name,
        })


print(
    f"\nTotal records: {len(records):,}"
)


# ============================================================
# 4. UNDERSCORE BASIC ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("1. UNDERSCORE BASIC ANALYSIS")
print("=" * 80)


underscore_total = 0
articles_with_underscore = 0
max_underscore = 0

for row in records:

    text = row["article"]

    count = text.count("_")

    underscore_total += count

    if count > 0:
        articles_with_underscore += 1

    max_underscore = max(
        max_underscore,
        count
    )


print(
    f"Articles with underscore: "
    f"{articles_with_underscore:,} / {len(records):,}"
)

print(
    f"Total underscores: "
    f"{underscore_total:,}"
)

print(
    f"Average/article: "
    f"{underscore_total / len(records):.2f}"
)

print(
    f"Maximum/article: "
    f"{max_underscore:,}"
)


# ============================================================
# 5. CHECK UNDERSCORE PATTERNS
# ============================================================

print("\n" + "=" * 80)
print("2. UNDERSCORE PATTERNS")
print("=" * 80)


pattern_counter = Counter()

pattern_examples = {}

for row in records:

    text = row["article"]

    # Capture a token around underscore.
    matches = re.findall(
        r"\S+_\S+",
        text
    )

    for match in matches:

        # Remove punctuation from the outer edges
        cleaned = match.strip(
            ".,!?;:()[]{}\"'"
        )

        pattern_counter[cleaned] += 1

        if cleaned not in pattern_examples:

            pattern_examples[cleaned] = text[
                max(
                    0,
                    text.find(match) - 80
                ):
                text.find(match) + len(match) + 80
            ]


print(
    f"\nUnique underscore tokens: "
    f"{len(pattern_counter):,}"
)

print("\nTop 100 underscore tokens:")

for token, count in pattern_counter.most_common(100):

    print(
        f"{count:8,} | {token}"
    )


# ============================================================
# 6. CHECK MULTIPLE / EDGE UNDERSCORES
# ============================================================

print("\n" + "=" * 80)
print("3. ABNORMAL UNDERSCORE PATTERNS")
print("=" * 80)


multiple_underscore = 0
leading_underscore = 0
trailing_underscore = 0
space_underscore = 0
underscore_space = 0

multiple_examples = []
edge_examples = []


for row in records:

    text = row["article"]

    if "__" in text:

        multiple_underscore += 1

        if len(multiple_examples) < 10:
            multiple_examples.append(text[:500])

    if re.search(r"(^|\s)_", text):

        leading_underscore += 1

        if len(edge_examples) < 10:
            edge_examples.append(text[:500])

    if re.search(r"_(\s|$)", text):

        trailing_underscore += 1

    if " _" in text:
        space_underscore += 1

    if "_ " in text:
        underscore_space += 1


print(
    f"Articles containing '__': "
    f"{multiple_underscore:,}"
)

print(
    f"Articles containing ' _': "
    f"{space_underscore:,}"
)

print(
    f"Articles containing '_ ': "
    f"{underscore_space:,}"
)

print(
    f"Articles with underscore at token start: "
    f"{leading_underscore:,}"
)

print(
    f"Articles with underscore at token end: "
    f"{trailing_underscore:,}"
)


if multiple_examples:

    print("\nExamples containing '__':")

    for example in multiple_examples:
        print("-" * 60)
        print(example)


# ============================================================
# 7. TEST SIMPLE REPLACEMENT
# ============================================================

print("\n" + "=" * 80)
print("4. TEST '_' -> ' '")
print("=" * 80)


replacement_differences = 0
replacement_examples = []

for row in records[:1000]:

    text = row["article"]

    converted = text.replace("_", " ")

    if converted != text:

        replacement_differences += 1

        if len(replacement_examples) < 5:

            replacement_examples.append({
                "original": text[:500],
                "converted": converted[:500],
            })


print(
    "Among first 1,000 articles:"
)

print(
    f"Articles changed by replacement: "
    f"{replacement_differences:,}"
)


print("\nExamples:")

for example in replacement_examples:

    print("-" * 80)

    print("ORIGINAL:")
    print(example["original"])

    print("\nCONVERTED:")
    print(example["converted"])


# ============================================================
# 8. CHECK TITLE / ABSTRACT / ARTICLE CONSISTENCY
# ============================================================

print("\n" + "=" * 80)
print("5. TITLE / ABSTRACT / ARTICLE UNDERSCORE")
print("=" * 80)


title_underscore = 0
abstract_underscore = 0
article_underscore = 0

for row in records:

    if "_" in row["title"]:
        title_underscore += 1

    if "_" in row["abstract"]:
        abstract_underscore += 1

    if "_" in row["article"]:
        article_underscore += 1


print(
    f"Title with _: "
    f"{title_underscore:,} / {len(records):,}"
)

print(
    f"Abstract with _: "
    f"{abstract_underscore:,} / {len(records):,}"
)

print(
    f"Article with _: "
    f"{article_underscore:,} / {len(records):,}"
)


# ============================================================
# 9. GUID ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("6. GUID ANALYSIS")
print("=" * 80)


guid_counter = Counter(
    row["guid"]
    for row in records
)

duplicate_guids = {
    guid: count
    for guid, count in guid_counter.items()
    if count > 1
}


print(
    f"Total records: "
    f"{len(records):,}"
)

print(
    f"Unique GUIDs: "
    f"{len(guid_counter):,}"
)

print(
    f"GUIDs occurring more than once: "
    f"{len(duplicate_guids):,}"
)

print(
    f"Records belonging to duplicated GUIDs: "
    f"{sum(duplicate_guids.values()):,}"
)


print("\nTop duplicated GUIDs:")

for guid, count in sorted(
    duplicate_guids.items(),
    key=lambda x: x[1],
    reverse=True
)[:30]:

    print(
        f"GUID {guid}: "
        f"{count} records"
    )


# ============================================================
# 10. CHECK WHETHER DUPLICATE GUIDS HAVE DIFFERENT ARTICLES
# ============================================================

print("\n" + "=" * 80)
print("7. DUPLICATE GUID CONTENT CHECK")
print("=" * 80)


guid_articles = {}

for row in records:

    guid = row["guid"]
    article = row["article"]

    if guid not in guid_articles:
        guid_articles[guid] = set()

    guid_articles[guid].add(article)


same_guid_different_article = 0
same_guid_same_article = 0

examples_guid_conflict = []


for guid, articles in guid_articles.items():

    if len(articles) > 1:

        same_guid_different_article += 1

        if len(examples_guid_conflict) < 10:

            examples_guid_conflict.append({
                "guid": guid,
                "num_articles": len(articles),
                "articles": list(articles)[:2],
            })

    else:

        if guid in duplicate_guids:
            same_guid_same_article += 1


print(
    "Duplicated GUIDs with different article text:",
    f"{same_guid_different_article:,}"
)

print(
    "Duplicated GUIDs with identical article text:",
    f"{same_guid_same_article:,}"
)


if examples_guid_conflict:

    print("\nExamples of same GUID with different articles:")

    for example in examples_guid_conflict:

        print("-" * 80)

        print(
            f"GUID: {example['guid']}"
        )

        print(
            f"Number of articles: "
            f"{example['num_articles']}"
        )

        print("\nARTICLE A:")
        print(example["articles"][0][:500])

        print("\nARTICLE B:")
        print(example["articles"][1][:500])


# ============================================================
# 11. ARTICLE HASH-LIKE ID CHECK
# ============================================================

print("\n" + "=" * 80)
print("8. ARTICLE UNIQUENESS")
print("=" * 80)


article_counter = Counter(
    row["article"]
    for row in records
)

duplicate_article_count = sum(
    1
    for count in article_counter.values()
    if count > 1
)

print(
    f"Unique article texts: "
    f"{len(article_counter):,}"
)

print(
    f"Duplicated article texts: "
    f"{duplicate_article_count:,}"
)

print(
    f"Duplicate article rows: "
    f"{len(records) - len(article_counter):,}"
)


# ============================================================
# 12. CHECK SPLIT OVERLAP
# ============================================================

print("\n" + "=" * 80)
print("9. ORIGINAL SPLIT OVERLAP")
print("=" * 80)


guid_splits = {}

for row in records:

    guid = row["guid"]

    if guid not in guid_splits:
        guid_splits[guid] = set()

    guid_splits[guid].add(
        row["original_split"]
    )


cross_split_guid = {
    guid: splits
    for guid, splits in guid_splits.items()
    if len(splits) > 1
}


print(
    f"GUIDs appearing in multiple original splits: "
    f"{len(cross_split_guid):,}"
)


if cross_split_guid:

    print("\nExamples:")

    for guid, splits in list(
        cross_split_guid.items()
    )[:20]:

        print(
            f"GUID {guid}: "
            f"{sorted(splits)}"
        )


# ============================================================
# 13. SAMPLE BEFORE / AFTER REPLACEMENT
# ============================================================

print("\n" + "=" * 80)
print("10. MANUAL VISUAL CHECK")
print("=" * 80)


for i in range(10):

    row = records[i]

    original = row["article"]

    converted = original.replace(
        "_",
        " "
    )

    print("\n" + "-" * 80)

    print(
        f"ARTICLE {i + 1}"
    )

    print("\nBEFORE:")
    print(original[:700])

    print("\nAFTER '_' -> ' ':")
    print(converted[:700])


# ============================================================
# 14. SAVE REPORT
# ============================================================

report = {

    "dataset": "nam194/vietnews",

    "total_records": len(records),

    "underscore": {

        "articles_with_underscore":
            articles_with_underscore,

        "percentage":
            articles_with_underscore
            / len(records) * 100,

        "total":
            underscore_total,

        "average_per_article":
            underscore_total / len(records),

        "maximum_per_article":
            max_underscore,

        "unique_underscore_tokens":
            len(pattern_counter),

        "multiple_underscore_articles":
            multiple_underscore,

        "space_underscore_articles":
            space_underscore,

        "underscore_space_articles":
            underscore_space,

        "leading_underscore_articles":
            leading_underscore,

        "trailing_underscore_articles":
            trailing_underscore,
    },

    "guid": {

        "total_records":
            len(records),

        "unique_guids":
            len(guid_counter),

        "duplicated_guid_values":
            len(duplicate_guids),

        "records_with_duplicated_guid":
            sum(duplicate_guids.values()),

        "duplicated_guid_different_articles":
            same_guid_different_article,

        "duplicated_guid_same_articles":
            same_guid_same_article,

        "cross_split_guids":
            len(cross_split_guid),
    },

    "article_uniqueness": {

        "unique_articles":
            len(article_counter),

        "duplicated_article_values":
            duplicate_article_count,

        "duplicate_article_rows":
            len(records) - len(article_counter),
    },
}


with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        ensure_ascii=False,
        indent=2
    )


print("\n" + "=" * 80)
print("EDA V2 FINISHED")
print("=" * 80)

print(
    f"\nReport saved to:\n{REPORT_PATH}"
)

print(
    "\nRAW DATA WAS NOT MODIFIED."
)

print(
    "NO DATASET SPLIT WAS CREATED."
)

print(
    "NO SPELLING ERRORS WERE GENERATED."
)