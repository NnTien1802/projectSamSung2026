from pathlib import Path
from collections import defaultdict
import hashlib
import unicodedata

from datasets import load_from_disk


PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIETNEWS_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "vietnews"
)


TARGET_SOURCE_ID = "2efef5448c433a1a4c57"


def normalize_text(text):

    if text is None:
        return ""

    return unicodedata.normalize(
        "NFC",
        str(text)
    )


def create_source_id(article):

    article = normalize_text(
        article
    )

    return hashlib.sha1(
        article.encode("utf-8")
    ).hexdigest()[:20]


def sha256_text(text):

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def main():

    print("=" * 80)
    print("INSPECT DUPLICATE SOURCE ID")
    print("=" * 80)

    dataset = load_from_disk(
        str(VIETNEWS_DIR)
    )

    matches = []

    for split_name in dataset.keys():

        for index, row in enumerate(
            dataset[split_name]
        ):

            raw_article = str(
                row["article"]
            )

            normalized_article = (
                normalize_text(
                    raw_article
                )
            )

            source_id = (
                create_source_id(
                    raw_article
                )
            )

            if (
                source_id
                != TARGET_SOURCE_ID
            ):
                continue

            matches.append({
                "split":
                    split_name,

                "index":
                    index,

                "guid":
                    row["guid"],

                "title":
                    row["title"],

                "abstract":
                    row["abstract"],

                "raw_article":
                    raw_article,

                "normalized_article":
                    normalized_article,

                "raw_sha256":
                    sha256_text(
                        raw_article
                    ),

                "normalized_sha256":
                    sha256_text(
                        normalized_article
                    ),
            })

    print(
        f"\nMatches: "
        f"{len(matches)}"
    )

    for i, item in enumerate(
        matches,
        start=1
    ):

        print("\n" + "=" * 80)

        print(
            f"MATCH {i}"
        )

        print("=" * 80)

        print(
            f"split: {item['split']}"
        )

        print(
            f"index: {item['index']}"
        )

        print(
            f"guid: {item['guid']}"
        )

        print(
            f"raw sha256:"
            f"\n{item['raw_sha256']}"
        )

        print(
            f"normalized sha256:"
            f"\n{item['normalized_sha256']}"
        )

        print(
            f"\ntitle:\n"
            f"{item['title']}"
        )

        print(
            "\nARTICLE PREFIX:"
        )

        print(
            item[
                "normalized_article"
            ][:1000]
        )

    # ========================================================
    # PAIRWISE COMPARISON
    # ========================================================

    if len(matches) >= 2:

        print("\n" + "=" * 80)
        print("PAIRWISE COMPARISON")
        print("=" * 80)

        for i in range(
            len(matches)
        ):

            for j in range(
                i + 1,
                len(matches)
            ):

                a = matches[i]
                b = matches[j]

                print(
                    f"\nMatch {i + 1} "
                    f"vs "
                    f"Match {j + 1}"
                )

                print(
                    "Raw article identical: "
                    f"{a['raw_article'] == b['raw_article']}"
                )

                print(
                    "NFC article identical: "
                    f"{a['normalized_article'] == b['normalized_article']}"
                )

                print(
                    "Title identical: "
                    f"{a['title'] == b['title']}"
                )

                print(
                    "Abstract identical: "
                    f"{a['abstract'] == b['abstract']}"
                )

                print(
                    "GUID identical: "
                    f"{a['guid'] == b['guid']}"
                )

                print(
                    "Same source split: "
                    f"{a['split'] == b['split']}"
                )


if __name__ == "__main__":
    main()