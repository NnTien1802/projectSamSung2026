"""Install the API-only M1 experimental PyVi patch into projectSamSung V1.

Default target is C:\\projectSamSung_v1. Refuses to touch unknown/modified
API sources or existing new-file destinations. Backups precede all writes.
Run with the V1 .venv Python; no dependencies beyond the stdlib needed.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from datetime import datetime
from pathlib import Path

EXPECTED_BASELINE = {
    "system/backend/app/api/main.py": "59f14199c612f127af165d5cd4f7aee3085361d7fac8834276238c37ac58d2a7",
    "system/backend/app/api/schemas.py": "db06d2bc20c30602fe5524258650a9bacd3687402559313b08abc914d0a03829",
    "system/backend/app/api/service.py": "4c9049f3696cbd894fd75456c6a651c44f06f1b4ca1ad59b7c9d43b2f15c472d",
}
EXTRAS = [
    "system/backend/app/api/experimental_segmenter.py",
    "system/backend/tests/test_api_m1_pyvi_v1.py",
    "system/scripts/16_smoke_m1_pyvi_http.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(r"C:\projectSamSung_v1"))
    parser.add_argument("--dry-run", action="store_true", help="Only verify destination; do not write any files")
    args = parser.parse_args()
    root = args.project.resolve()
    overlay = Path(__file__).resolve().parent / "overlay"

    problems = []
    for relative, expected in EXPECTED_BASELINE.items():
        original = root / relative
        replacement = overlay / relative
        if not original.is_file():
            problems.append(f"Missing baseline file: {original}")
        elif sha256(original) != expected:
            problems.append(f"Baseline API file differs; refusing overwrite: {original}")
        if not replacement.is_file():
            problems.append(f"Patch is incomplete: {replacement}")
    for relative in EXTRAS:
        destination = root / relative
        if destination.exists():
            problems.append(f"New destination already exists; refusing overwrite: {destination}")
        if not (overlay / relative).is_file():
            problems.append(f"Patch is incomplete: {overlay / relative}")
    if problems:
        print("PATCH NOT INSTALLED; no files changed:")
        for item in problems:
            print("  -", item)
        print("If your API differs from the original V1, do NOT force install; review those three files first.")
        return 2

    if args.dry_run:
        print("DRY RUN PASS: baseline API files match and new paths are free; nothing changed.")
        return 0

    backup = root / "system/reports/api_patch_backups" / datetime.now().strftime("pre_pyvi_v1_%Y%m%d_%H%M%S_%f")
    backup.mkdir(parents=True, exist_ok=False)
    created = []
    overwritten = []
    try:
        # Make complete backups before the first overlay write.
        for relative in EXPECTED_BASELINE:
            original = root / relative
            dest = backup / Path(relative).name
            shutil.copy2(original, dest)
            if sha256(dest) != EXPECTED_BASELINE[relative]:
                raise RuntimeError(f"Backup verification failed: {original}")
        for relative in EXPECTED_BASELINE:
            dest = root / relative
            shutil.copy2(overlay / relative, dest)
            overwritten.append(relative)
        for relative in EXTRAS:
            dest = root / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Preflight ensures these paths did not previously exist.
            shutil.copy2(overlay / relative, dest)
            created.append(relative)
    except Exception:
        for relative in overwritten:
            shutil.copy2(backup / Path(relative).name, root / relative)
        for relative in created:
            (root / relative).unlink(missing_ok=True)
        print("Installation failed; restored all prior baseline API files.")
        raise

    print("PATCH INSTALLED: opt-in PyVi mode added to API V1 without modifying frozen model artifacts.")
    print("BACKUP:", backup)
    print("NEXT: run .\\.venv\\Scripts\\python.exe -m unittest discover -s system\\backend\\tests -v")
    return 0


if __name__ == "__main__":
    sys.exit(main())
