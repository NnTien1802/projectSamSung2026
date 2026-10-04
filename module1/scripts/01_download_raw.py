from pathlib import Path
from datasets import load_dataset


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

RAW_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATASETS
# ============================================================

DATASETS = {
    "vietnews": {
        "name": "nam194/vietnews",
    },
    "vsec": {
        "name": "nguyenthanhasia/vsec-vietnamese-spell-correction",
    },
    "vi_spelling_correction": {
        "name": "coung21/vi-spelling-correction",
    },
}


# ============================================================
# DOWNLOAD
# ============================================================

for dataset_key, config in DATASETS.items():

    print("\n" + "=" * 70)
    print(f"Downloading: {dataset_key}")
    print(f"HuggingFace: {config['name']}")
    print("=" * 70)

    output_dir = RAW_DIR / dataset_key

    if output_dir.exists():
        print(f"Directory already exists: {output_dir}")
        print("Skipping...")
        continue

    try:
        dataset = load_dataset(config["name"])

        print("\nDataset loaded successfully.")
        print(dataset)

        dataset.save_to_disk(str(output_dir))

        print(f"\nSaved to:")
        print(output_dir)

    except Exception as e:
        print("\nERROR")
        print(type(e).__name__)
        print(e)


print("\n" + "=" * 70)
print("DATA COLLECTION FINISHED")
print("=" * 70)