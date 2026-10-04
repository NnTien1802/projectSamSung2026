from pathlib import Path
from collections import Counter
import json
import re
import unicodedata

import numpy as np
import pandas as pd
from datasets import load_from_disk


# ============================================================
# 1. PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "vietnews"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

INTERIM_DIR.mkdir(parents=True, exist_ok=True)

REPORT_PATH = INTERIM_DIR / "vietnews_eda_report.json"
LENGTH_CSV = INTERIM_DIR / "vietnews_article_lengths.csv"


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 80)
print("VIETNEWS - EDA V1")
print("=" * 80)

print(f"\nProject root : {PROJECT_ROOT}")
print(f"Dataset path : {RAW_DIR}")

print("\nLoading dataset...")

ds = load_from_disk(str(RAW_DIR))

print("\nDataset loaded successfully:")
print(ds)


# ============================================================
# 3. COMBINE ORIGINAL SPLITS FOR EDA ONLY
# ============================================================

print("\n" + "=" * 80)
print("COMBINING ORIGINAL VIETNEWS SPLITS FOR EDA")
print("=" * 80)

frames = []

for split_name in ds.keys():
    print(f"\nReading split: {split_name}")
    print(f"Rows: {len(ds[split_name]):,}")

    df_split = ds[split_name].to_pandas()

    # Keep track of original split.
    df_split["original_split"] = split_name

    frames.append(df_split)

df = pd.concat(frames, ignore_index=True)

print(f"\nTotal articles for EDA: {len(df):,}")


# ============================================================
# 4. BASIC SCHEMA
# ============================================================

print("\n" + "=" * 80)
print("SCHEMA")
print("=" * 80)

print(df.dtypes)

print("\nColumns:")
for column in df.columns:
    print(f" - {column}")


# ============================================================
# 5. MISSING VALUES
# ============================================================

print("\n" + "=" * 80)
print("MISSING VALUES")
print("=" * 80)

missing = df.isna().sum()

print(missing)

missing_percent = (
    missing / len(df) * 100
).round(4)

print("\nMissing percentage:")
print(missing_percent)


# ============================================================
# 6. BASIC TEXT STATISTICS
# ============================================================

print("\n" + "=" * 80)
print("ARTICLE LENGTH STATISTICS")
print("=" * 80)


def safe_text(value):
    if value is None:
        return ""
    return str(value)


df["title_text"] = df["title"].apply(safe_text)
df["abstract_text"] = df["abstract"].apply(safe_text)
df["article_text"] = df["article"].apply(safe_text)


# Character count
df["title_chars"] = df["title_text"].str.len()
df["abstract_chars"] = df["abstract_text"].str.len()
df["article_chars"] = df["article_text"].str.len()


# Word count using whitespace
df["title_words"] = df["title_text"].str.split().str.len()
df["abstract_words"] = df["abstract_text"].str.split().str.len()
df["article_words"] = df["article_text"].str.split().str.len()


def statistics(series):
    return {
        "count": int(series.count()),
        "min": int(series.min()),
        "max": int(series.max()),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std()),
        "p01": float(series.quantile(0.01)),
        "p05": float(series.quantile(0.05)),
        "p10": float(series.quantile(0.10)),
        "p25": float(series.quantile(0.25)),
        "p50": float(series.quantile(0.50)),
        "p75": float(series.quantile(0.75)),
        "p90": float(series.quantile(0.90)),
        "p95": float(series.quantile(0.95)),
        "p99": float(series.quantile(0.99)),
    }


length_stats = {
    "article_chars": statistics(df["article_chars"]),
    "article_words": statistics(df["article_words"]),
    "title_chars": statistics(df["title_chars"]),
    "title_words": statistics(df["title_words"]),
    "abstract_chars": statistics(df["abstract_chars"]),
    "abstract_words": statistics(df["abstract_words"]),
}

print("\nArticle characters:")
print(pd.Series(length_stats["article_chars"]))

print("\nArticle words:")
print(pd.Series(length_stats["article_words"]))


# ============================================================
# 7. EMPTY / VERY SHORT ARTICLES
# ============================================================

print("\n" + "=" * 80)
print("EMPTY / SHORT ARTICLES")
print("=" * 80)

empty_article = (df["article_text"].str.strip() == "")

short_thresholds = [50, 100, 200, 500]

short_counts = {}

print(f"\nEmpty articles: {empty_article.sum():,}")

for threshold in short_thresholds:

    count = (df["article_chars"] < threshold).sum()

    short_counts[str(threshold)] = int(count)

    print(
        f"Articles shorter than {threshold} characters: "
        f"{count:,}"
    )


# ============================================================
# 8. DUPLICATE ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("DUPLICATE ANALYSIS")
print("=" * 80)

duplicate_guid = df["guid"].duplicated(keep=False)
duplicate_article = df["article_text"].duplicated(keep=False)
duplicate_title = df["title_text"].duplicated(keep=False)

print(
    f"Duplicate GUID rows: "
    f"{duplicate_guid.sum():,}"
)

print(
    f"Duplicate article rows: "
    f"{duplicate_article.sum():,}"
)

print(
    f"Duplicate title rows: "
    f"{duplicate_title.sum():,}"
)

print(
    f"Unique GUIDs: "
    f"{df['guid'].nunique():,}"
)

print(
    f"Unique articles: "
    f"{df['article_text'].nunique():,}"
)


# ============================================================
# 9. UNDERSCORE ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("UNDERSCORE / TOKENIZATION ANALYSIS")
print("=" * 80)

df["underscore_count"] = df["article_text"].str.count("_")

articles_with_underscore = (
    df["underscore_count"] > 0
)

print(
    f"Articles containing '_': "
    f"{articles_with_underscore.sum():,} / {len(df):,}"
)

print(
    "Percentage:",
    round(
        articles_with_underscore.mean() * 100,
        4
    ),
    "%"
)

print(
    "\nTotal '_' characters:",
    int(df["underscore_count"].sum())
)

print(
    "Average '_' per article:",
    round(
        df["underscore_count"].mean(),
        4
    )
)

print(
    "Maximum '_' in one article:",
    int(df["underscore_count"].max())
)


# ============================================================
# 10. SPECIAL CHARACTER ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("SPECIAL CHARACTER ANALYSIS")
print("=" * 80)


def count_pattern(text, pattern):
    return len(re.findall(pattern, text))


patterns = {
    "newline": r"\n",
    "carriage_return": r"\r",
    "tab": r"\t",
    "underscore": r"_",
    "multiple_spaces": r" {2,}",
    "url": r"https?://\S+|www\.\S+",
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "html_like": r"<[^>]+>",
}


pattern_results = {}

for name, pattern in patterns.items():

    print(f"\nAnalyzing: {name}")

    counts = df["article_text"].apply(
        lambda x: count_pattern(x, pattern)
    )

    pattern_results[name] = {
        "total_occurrences": int(counts.sum()),
        "articles_affected": int((counts > 0).sum()),
        "max_per_article": int(counts.max()),
    }

    print(
        "Total:",
        pattern_results[name]["total_occurrences"]
    )

    print(
        "Articles affected:",
        pattern_results[name]["articles_affected"]
    )

    print(
        "Max/article:",
        pattern_results[name]["max_per_article"]
    )


# ============================================================
# 11. UNICODE ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("UNICODE ANALYSIS")
print("=" * 80)


unicode_forms = {
    "NFC": 0,
    "NFD": 0,
    "NFKC": 0,
    "NFKD": 0,
}

for text in df["article_text"]:

    for form in unicode_forms:

        if unicodedata.normalize(form, text) == text:
            unicode_forms[form] += 1


print("\nArticles already normalized:")
for form, count in unicode_forms.items():

    print(
        f"{form}: "
        f"{count:,} "
        f"({count / len(df) * 100:.2f}%)"
    )


# ============================================================
# 12. CHARACTER DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("CHARACTER DISTRIBUTION")
print("=" * 80)

all_text = "".join(df["article_text"].tolist())

counter = Counter(all_text)

print("\nMost common 50 characters:")

for char, count in counter.most_common(50):

    printable = repr(char)

    print(
        f"{printable:>10} : {count:,}"
    )


# ============================================================
# 13. DIGIT / LETTER STATISTICS
# ============================================================

print("\n" + "=" * 80)
print("TEXT COMPOSITION")
print("=" * 80)


def count_digits(text):
    return sum(char.isdigit() for char in text)


def count_letters(text):
    return sum(char.isalpha() for char in text)


def count_spaces(text):
    return sum(char.isspace() for char in text)


df["digit_count"] = df["article_text"].apply(count_digits)
df["letter_count"] = df["article_text"].apply(count_letters)
df["space_count"] = df["article_text"].apply(count_spaces)


print(
    "Total letters:",
    f"{df['letter_count'].sum():,}"
)

print(
    "Total digits:",
    f"{df['digit_count'].sum():,}"
)

print(
    "Total whitespace:",
    f"{df['space_count'].sum():,}"
)


# ============================================================
# 14. ORIGINAL SPLIT DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("ORIGINAL VIETNEWS SPLITS")
print("=" * 80)

split_counts = df["original_split"].value_counts()

print(split_counts)

print(
    "\nPercentages:"
)

print(
    (split_counts / len(df) * 100).round(2)
)


# ============================================================
# 15. SAMPLE ARTICLES
# ============================================================

print("\n" + "=" * 80)
print("SAMPLE ARTICLES")
print("=" * 80)

sample_indices = [0, 1, 2, 3, 4]

for i in sample_indices:

    row = df.iloc[i]

    print("\n" + "-" * 80)

    print(f"GUID: {row['guid']}")
    print(f"Original split: {row['original_split']}")

    print(
        f"\nTITLE:\n{row['title_text']}"
    )

    print(
        f"\nABSTRACT:\n{row['abstract_text']}"
    )

    article_preview = row["article_text"][:1000]

    print(
        f"\nARTICLE PREVIEW (first 1000 chars):\n"
        f"{article_preview}"
    )


# ============================================================
# 16. SAVE ARTICLE LENGTH DATA
# ============================================================

print("\n" + "=" * 80)
print("SAVING EDA DATA")
print("=" * 80)

length_df = df[
    [
        "guid",
        "original_split",
        "article_chars",
        "article_words",
        "title_chars",
        "title_words",
        "abstract_chars",
        "abstract_words",
        "underscore_count",
        "digit_count",
        "letter_count",
        "space_count",
    ]
].copy()

length_df.to_csv(
    LENGTH_CSV,
    index=False,
    encoding="utf-8-sig"
)

print(
    f"\nSaved article statistics to:\n"
    f"{LENGTH_CSV}"
)


# ============================================================
# 17. SAVE JSON REPORT
# ============================================================

report = {
    "dataset": "nam194/vietnews",

    "total_articles": int(len(df)),

    "splits": {
        split: int(count)
        for split, count in split_counts.items()
    },

    "columns": [
        str(column)
        for column in df.columns
    ],

    "missing_values": {
        str(column): int(value)
        for column, value in missing.items()
    },

    "missing_percentage": {
        str(column): float(value)
        for column, value in missing_percent.items()
    },

    "length_statistics": length_stats,

    "short_articles": short_counts,

    "duplicates": {
        "duplicate_guid_rows": int(
            duplicate_guid.sum()
        ),
        "duplicate_article_rows": int(
            duplicate_article.sum()
        ),
        "duplicate_title_rows": int(
            duplicate_title.sum()
        ),
        "unique_guid": int(
            df["guid"].nunique()
        ),
        "unique_articles": int(
            df["article_text"].nunique()
        ),
    },

    "underscore_analysis": {
        "articles_with_underscore": int(
            articles_with_underscore.sum()
        ),
        "percentage_articles_with_underscore": float(
            articles_with_underscore.mean() * 100
        ),
        "total_underscore_characters": int(
            df["underscore_count"].sum()
        ),
        "average_underscore_per_article": float(
            df["underscore_count"].mean()
        ),
        "max_underscore_per_article": int(
            df["underscore_count"].max()
        ),
    },

    "special_characters": pattern_results,

    "unicode": unicode_forms,

    "text_composition": {
        "total_letters": int(
            df["letter_count"].sum()
        ),
        "total_digits": int(
            df["digit_count"].sum()
        ),
        "total_whitespace": int(
            df["space_count"].sum()
        ),
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


print(
    f"Saved JSON report to:\n"
    f"{REPORT_PATH}"
)


# ============================================================
# 18. FINISH
# ============================================================

print("\n" + "=" * 80)
print("EDA V1 FINISHED")
print("=" * 80)

print("\nIMPORTANT:")
print("No raw data was modified.")
print("No train/validation/test split was created.")
print("No spelling errors were generated.")

print("\nGenerated files:")

print(
    f" - {LENGTH_CSV}"
)

print(
    f" - {REPORT_PATH}"
)