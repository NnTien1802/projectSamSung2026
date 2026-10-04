from pathlib import Path
from datetime import datetime
import hashlib
import json
import urllib.request

from huggingface_hub import snapshot_download


# ============================================================
# PATHS
# ============================================================

MODULE_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = MODULE_ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw"

CLICKBAIT_RAW = RAW_ROOT / "clickbait"
CONSISTENCY_RAW = RAW_ROOT / "consistency"

VICLICKBAIT_DIR = (
    CLICKBAIT_RAW
    / "viclickbait_2025"
)

VIANLI_DIR = (
    CONSISTENCY_RAW
    / "vianli"
)

VIET_ONLINE_NEWS_DIR = (
    CONSISTENCY_RAW
    / "viet_online_news"
)

MANIFEST_PATH = (
    RAW_ROOT
    / "download_manifest.json"
)


# ============================================================
# DATASET SOURCES
# ============================================================

VICLICKBAIT_BASE_URL = (
    "https://raw.githubusercontent.com/"
    "blanatole/ViClickbait-2025/main"
)

VICLICKBAIT_FILES = [
    "clickbait_dataset_vietnamese.csv",
    "clickbait_dataset_vietnamese.jsonl",
    "README.md",
]

VIANLI_REPO = "uitnlp/ViANLI"

VIET_ONLINE_NEWS_REPO = (
    "VLUS06/VietOnlineNews"
)


# ============================================================
# HELPERS
# ============================================================

def ensure_directories():
    """Create required Module 2 raw directories."""

    directories = [
        VICLICKBAIT_DIR,
        VIANLI_DIR,
        VIET_ONLINE_NEWS_DIR,
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def human_size(size_bytes):
    """Convert byte count to human-readable text."""

    size = float(size_bytes)

    for unit in [
        "B",
        "KB",
        "MB",
        "GB",
        "TB",
    ]:
        if size < 1024:
            return f"{size:.2f} {unit}"

        size /= 1024

    return f"{size:.2f} PB"


def directory_size(directory):
    """Calculate total size of files in a directory."""

    total = 0

    if not directory.exists():
        return total

    for path in directory.rglob("*"):
        if path.is_file():
            total += path.stat().st_size

    return total


def sha256_file(path):
    """Calculate SHA256 for a file."""

    hasher = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)

            if not block:
                break

            hasher.update(block)

    return hasher.hexdigest()


# ============================================================
# VICLICKBAIT
# ============================================================

def download_file(url, destination):
    """
    Download one file.

    Existing non-empty files are not downloaded again.
    Temporary .part file protects against incomplete downloads.
    """

    if (
        destination.exists()
        and destination.stat().st_size > 0
    ):
        print(
            f"[SKIP] {destination.name} "
            f"already exists."
        )
        return

    temp_path = destination.with_suffix(
        destination.suffix + ".part"
    )

    print(
        f"[DOWNLOAD] {destination.name}"
    )

    try:
        urllib.request.urlretrieve(
            url,
            temp_path,
        )

        temp_path.replace(
            destination
        )

    except Exception:
        if temp_path.exists():
            temp_path.unlink()

        raise


def download_viclickbait():
    print()
    print("=" * 80)
    print("ViClickbait-2025")
    print("=" * 80)

    for filename in VICLICKBAIT_FILES:

        url = (
            f"{VICLICKBAIT_BASE_URL}/"
            f"{filename}"
        )

        destination = (
            VICLICKBAIT_DIR
            / filename
        )

        download_file(
            url=url,
            destination=destination,
        )

        print(
            f"  {filename}"
            f" -> "
            f"{human_size(destination.stat().st_size)}"
        )


# ============================================================
# HUGGING FACE
# ============================================================

def download_hf_dataset(
    name,
    repo_id,
    destination,
):
    print()
    print("=" * 80)
    print(name)
    print("=" * 80)

    print(
        f"Repository : {repo_id}"
    )

    print(
        f"Destination: {destination}"
    )

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(destination),
    )

    size = directory_size(
        destination
    )

    print(
        f"Downloaded size: "
        f"{human_size(size)}"
    )


# ============================================================
# MANIFEST
# ============================================================

def build_manifest():
    """
    Store provenance information for Module 2 raw data.

    Raw files themselves are never modified here.
    """

    manifest = {
        "module": (
            "module2_headline_integrity"
        ),

        "created_at": (
            datetime.now()
            .astimezone()
            .isoformat()
        ),

        "datasets": {

            "viclickbait_2025": {
                "task": "clickbait",
                "source_type": "github",
                "source": (
                    "blanatole/"
                    "ViClickbait-2025"
                ),
                "local_path": str(
                    VICLICKBAIT_DIR
                    .relative_to(MODULE_ROOT)
                ),
                "size_bytes": (
                    directory_size(
                        VICLICKBAIT_DIR
                    )
                ),
            },

            "vianli": {
                "task": "consistency_auxiliary",
                "source_type": (
                    "huggingface"
                ),
                "source": VIANLI_REPO,
                "local_path": str(
                    VIANLI_DIR
                    .relative_to(MODULE_ROOT)
                ),
                "size_bytes": (
                    directory_size(
                        VIANLI_DIR
                    )
                ),
            },

            "viet_online_news": {
                "task": (
                    "consistency_corpus"
                ),
                "source_type": (
                    "huggingface"
                ),
                "source": (
                    VIET_ONLINE_NEWS_REPO
                ),
                "local_path": str(
                    VIET_ONLINE_NEWS_DIR
                    .relative_to(MODULE_ROOT)
                ),
                "size_bytes": (
                    directory_size(
                        VIET_ONLINE_NEWS_DIR
                    )
                ),
            },
        },
    }

    # Add hashes for the small ViClickbait source files.
    manifest[
        "datasets"
    ][
        "viclickbait_2025"
    ][
        "files"
    ] = []

    for path in sorted(
        VICLICKBAIT_DIR.iterdir()
    ):
        if not path.is_file():
            continue

        manifest[
            "datasets"
        ][
            "viclickbait_2025"
        ][
            "files"
        ].append(
            {
                "name": path.name,
                "size_bytes": (
                    path.stat().st_size
                ),
                "sha256": (
                    sha256_file(path)
                ),
            }
        )

    with MANIFEST_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# SUMMARY
# ============================================================

def print_summary():

    datasets = [
        (
            "ViClickbait-2025",
            VICLICKBAIT_DIR,
        ),
        (
            "ViANLI",
            VIANLI_DIR,
        ),
        (
            "VietOnlineNews",
            VIET_ONLINE_NEWS_DIR,
        ),
    ]

    print()
    print("=" * 80)
    print("MODULE 2 RAW DATA SUMMARY")
    print("=" * 80)

    total = 0

    for name, path in datasets:

        size = directory_size(path)
        total += size

        print()
        print(name)
        print(f"  Path: {path}")
        print(
            f"  Size: "
            f"{human_size(size)}"
        )

    print()
    print(
        f"Total: "
        f"{human_size(total)}"
    )

    print()
    print(
        f"Manifest: "
        f"{MANIFEST_PATH}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("MODULE 2 - HEADLINE INTEGRITY")
    print("STEP 01 - DOWNLOAD RAW DATA")
    print("=" * 80)

    print()
    print(
        f"Module root:\n"
        f"{MODULE_ROOT}"
    )

    ensure_directories()

    # --------------------------------------------------------
    # TASK 2A - CLICKBAIT
    # --------------------------------------------------------

    download_viclickbait()

    # --------------------------------------------------------
    # TASK 2B - CONSISTENCY
    # Auxiliary NLI dataset
    # --------------------------------------------------------

    download_hf_dataset(
        name="ViANLI",
        repo_id=VIANLI_REPO,
        destination=VIANLI_DIR,
    )

    # --------------------------------------------------------
    # TASK 2B - CONSISTENCY
    # Full-news corpus
    # --------------------------------------------------------

    download_hf_dataset(
        name="VietOnlineNews",
        repo_id=VIET_ONLINE_NEWS_REPO,
        destination=VIET_ONLINE_NEWS_DIR,
    )

    build_manifest()

    print_summary()

    print()
    print("=" * 80)
    print("STEP 01 FINISHED")
    print("=" * 80)


if __name__ == "__main__":
    main()