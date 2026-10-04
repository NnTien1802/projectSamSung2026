from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_SOURCE = Path(r"C:\1_projectSamSung")
DEFAULT_DEST = Path(r"C:\projectSamSung_v1")

# V1 keeps the *existing runtime-relative layout* so the frozen adapters and
# release manifest continue to work. Research/training data and the old
# plagiarism branch are intentionally not copied.
COPY_PATHS = [
    # Module 1 runtime + fixed regression source/data.
    Path("models/module1_phobert_mil_full"),
    Path("results/module1_phobert_mil_full"),
    Path("scripts/16_evaluate_phobert_test.py"),
    Path("data/processed/vn_newsspell_v1/validation.jsonl"),
    Path("data/processed/vn_newsspell_v1/split_summary.json"),

    # Former Module 2 runtime, now exposed logically as M2 Clickbait + M3 Consistency.
    Path("module2_headline_integrity/inference"),
    Path("module2_headline_integrity/models/clickbait/mbert_title_only_v1"),
    Path("module2_headline_integrity/models/consistency/baseline_title_description_mbert"),
    Path("module2_headline_integrity/models/consistency/full_article_max_mil_mbert"),
    Path("module2_headline_integrity/results"),
    Path("module2_headline_integrity/data/processed"),

    # Integration code, release metadata, regression/GPU reports.
    Path("system/backend"),
    Path("system/release"),
    Path("system/scripts"),
    Path("system/reports"),
]

EXCLUDED_NAMES = {
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".cache",
}


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(block_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def iter_files(path: Path):
    if path.is_file():
        yield path
        return
    if not path.exists():
        return
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_NAMES and not (Path(root) / d).is_symlink()]
        for name in files:
            p = Path(root) / name
            if p.is_symlink():
                continue
            yield p


def load_manifest_paths(manifest_path: Path) -> list[Path]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    found: list[str] = []

    def walk(obj):
        if isinstance(obj, dict):
            p = obj.get("path")
            if isinstance(p, str) and p.strip():
                found.append(p)
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)

    # Prefer the canonical files list, but tolerate a different compatible schema.
    if isinstance(data, dict) and "files" in data:
        walk(data["files"])
    else:
        walk(data)

    unique = []
    seen = set()
    for raw in found:
        norm = raw.replace("\\", "/").lstrip("./")
        if norm and norm not in seen:
            seen.add(norm)
            unique.append(Path(norm))
    if not unique:
        raise RuntimeError(f"No protected file paths found in manifest: {manifest_path}")
    return unique


def build_copy_set(source: Path) -> tuple[list[tuple[Path, Path]], list[Path]]:
    missing_required: list[Path] = []
    files: dict[str, tuple[Path, Path]] = {}

    def add_path(rel: Path, required: bool = True):
        src = source / rel
        if not src.exists():
            if required:
                missing_required.append(rel)
            return
        if src.is_file():
            files[rel.as_posix().lower()] = (src, rel)
        else:
            for f in iter_files(src):
                r = f.relative_to(source)
                files[r.as_posix().lower()] = (f, r)

    for rel in COPY_PATHS:
        # split_summary is useful documentation but may be absent in a trimmed source.
        add_path(rel, required=(rel.name != "split_summary.json"))

    # Guarantee every release-locked artifact is included, regardless of the
    # hand-maintained V1 copy list above.
    manifest = source / "system/release/model_lock_manifest_v1.json"
    if not manifest.is_file():
        missing_required.append(Path("system/release/model_lock_manifest_v1.json"))
        locked = []
    else:
        locked = load_manifest_paths(manifest)
        for rel in locked:
            add_path(rel, required=True)

    ordered = sorted(files.values(), key=lambda x: x[1].as_posix().lower())
    return ordered, missing_required


def human_gb(n: int) -> str:
    return f"{n / 1_000_000_000:.2f} GB"


def ensure_destination_safe(source: Path, dest: Path, resume: bool):
    if source.resolve() == dest.resolve():
        raise RuntimeError("Destination must differ from source.")
    try:
        dest.resolve().relative_to(source.resolve())
        raise RuntimeError("Destination must not be inside the source project.")
    except ValueError:
        pass
    if dest.exists() and any(dest.iterdir()) and not resume:
        raise RuntimeError(
            f"Destination is not empty: {dest}\n"
            "Use a new empty directory, or --resume only for a previously interrupted V1 copy."
        )


def write_scaffold(dest: Path, source: Path, report: dict):
    (dest / "frontend").mkdir(parents=True, exist_ok=True)
    (dest / "docs").mkdir(parents=True, exist_ok=True)
    (dest / "frontend/.gitkeep").touch(exist_ok=True)
    (dest / "docs/.gitkeep").touch(exist_ok=True)

    readme = f"""# projectSamSung V1\n\nDeployment/integration workspace created from `{source}`.\n\n## Scope\n\n- Module 1: Vietnamese document-level spelling/textual-error detection (PhoBERT MIL).\n- Module 2: clickbait headline classification (mBERT title-only).\n- Module 3: headline-content consistency (description + full-article MAX-MIL fusion).\n- The old `module3_plagiarism_detection` research branch is intentionally NOT included.\n\n## Compatibility rule\n\nThe original runtime-relative paths are preserved so the frozen release manifest,\nM1 adapter, and original headline pipeline can be regression-tested before any\npath migration. Do not move locked artifacts until a separate release migration.\n\n## Next verification commands\n\nRun with the Python environment from the research project if this V1 directory has\nno local `.venv`:\n\n```cmd\nC:\\1_projectSamSung\\.venv\\Scripts\\python.exe system\\scripts\\01_model_release.py verify\nC:\\1_projectSamSung\\.venv\\Scripts\\python.exe -m unittest discover -s system\\backend\\tests -v\nC:\\1_projectSamSung\\.venv\\Scripts\\python.exe system\\scripts\\05_smoke_test_adapters.py --device cuda\nC:\\1_projectSamSung\\.venv\\Scripts\\python.exe system\\scripts\\06_regression_test_module1.py --device cuda\nC:\\1_projectSamSung\\.venv\\Scripts\\python.exe module2_headline_integrity\\inference\\04_regression_check_unified_pipeline.py\n```\n\nAfter those pass, build FastAPI and React in this V1 workspace.\n"""
    (dest / "README_V1.md").write_text(readme, encoding="utf-8")

    gitignore = """.venv/\n__pycache__/\n*.py[cod]\n.pytest_cache/\n.mypy_cache/\n.ruff_cache/\n.cache/\nnode_modules/\ndist/\nbuild/\n.env\n*.log\n"""
    (dest / ".gitignore").write_text(gitignore, encoding="utf-8")

    (dest / "PROJECT_V1_CREATE_REPORT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Create a trimmed projectSamSung V1 deployment workspace without modifying the research project."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--execute", action="store_true", help="Actually copy files. Default is dry-run.")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted copy into an existing V1 directory.")
    parser.add_argument("--skip-all-hash", action="store_true", help="Verify copied non-locked files by size only; locked files are always SHA-256 checked.")
    args = parser.parse_args()

    source = args.source.resolve()
    dest = args.dest.resolve()

    if not source.is_dir():
        raise FileNotFoundError(f"Source project does not exist: {source}")
    ensure_destination_safe(source, dest, args.resume)

    copy_files, missing = build_copy_set(source)
    if missing:
        print("ERROR: required V1 source paths are missing:")
        for p in missing:
            print(f"  - {p}")
        raise SystemExit(2)

    total_bytes = sum(src.stat().st_size for src, _ in copy_files)
    manifest_path = source / "system/release/model_lock_manifest_v1.json"
    locked_paths = {p.as_posix().lower() for p in load_manifest_paths(manifest_path)}

    groups: dict[str, int] = {}
    for src, rel in copy_files:
        key = rel.parts[0] if rel.parts else "root"
        groups[key] = groups.get(key, 0) + src.stat().st_size

    print(f"SOURCE      : {source}")
    print(f"DESTINATION : {dest}")
    print(f"FILES       : {len(copy_files)}")
    print(f"TOTAL       : {human_gb(total_bytes)} (decimal)")
    print(f"LOCKED      : {len(locked_paths)} manifest paths included")
    print("SCOPE       : runtime + regression evidence; old plagiarism branch excluded")
    for key, n in sorted(groups.items()):
        print(f"  {key:28s}: {human_gb(n)}")

    try:
        usage = shutil.disk_usage(dest.anchor or str(dest))
        print(f"DEST FREE   : {human_gb(usage.free)}")
        # 2 GB operational headroom in addition to the copy size.
        if usage.free < total_bytes + 2_000_000_000:
            raise RuntimeError(
                f"Not enough free space. Need copy size + 2 GB headroom; free={human_gb(usage.free)}"
            )
    except FileNotFoundError:
        # On Windows the drive should exist; this also catches typos like D: on a machine without D:.
        raise RuntimeError(f"Destination drive/path is not available: {dest}")

    if not args.execute:
        print("\nDRY RUN: nothing copied.")
        print("Run again with --execute after reviewing the plan.")
        return

    dest.mkdir(parents=True, exist_ok=True)
    copied_bytes = 0
    copied_count = 0
    reused_count = 0
    failures = []

    print("\nCOPY + VERIFY:")
    for idx, (src, rel) in enumerate(copy_files, start=1):
        dst = dest / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        size = src.stat().st_size
        rel_key = rel.as_posix().lower()

        reuse = False
        if args.resume and dst.is_file() and dst.stat().st_size == size:
            if rel_key in locked_paths or not args.skip_all_hash:
                reuse = sha256_file(src) == sha256_file(dst)
            else:
                reuse = True

        if not reuse:
            shutil.copy2(src, dst)
            if dst.stat().st_size != size:
                failures.append({"path": rel.as_posix(), "reason": "size_mismatch"})
                continue
            must_hash = rel_key in locked_paths or not args.skip_all_hash
            if must_hash:
                src_hash = sha256_file(src)
                dst_hash = sha256_file(dst)
                if src_hash != dst_hash:
                    failures.append({"path": rel.as_posix(), "reason": "sha256_mismatch"})
                    continue
            copied_count += 1
        else:
            reused_count += 1

        copied_bytes += size
        if idx == 1 or idx % 20 == 0 or idx == len(copy_files) or size >= 500_000_000:
            print(f"  {idx:4d}/{len(copy_files)} files | {human_gb(copied_bytes)}/{human_gb(total_bytes)} | {rel}")

    # Explicitly verify the source manifest's SHA-256 values against the V1 copy.
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_hash_by_path: dict[str, str] = {}

    def collect_hashes(obj):
        if isinstance(obj, dict):
            p = obj.get("path")
            h = obj.get("sha256")
            if isinstance(p, str) and isinstance(h, str):
                expected_hash_by_path[p.replace("\\", "/").lstrip("./").lower()] = h.lower()
            for v in obj.values():
                collect_hashes(v)
        elif isinstance(obj, list):
            for v in obj:
                collect_hashes(v)

    collect_hashes(manifest_data.get("files", manifest_data) if isinstance(manifest_data, dict) else manifest_data)
    locked_verified = 0
    locked_failures = []
    for rel_lower, expected in expected_hash_by_path.items():
        dst = dest / Path(rel_lower)
        if not dst.is_file():
            locked_failures.append({"path": rel_lower, "reason": "missing"})
            continue
        actual = sha256_file(dst).lower()
        if actual != expected:
            locked_failures.append({"path": rel_lower, "reason": "manifest_sha256_mismatch"})
        else:
            locked_verified += 1

    failures.extend(locked_failures)

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(source),
        "destination": str(dest),
        "scope": "V1 runtime + regression evidence; excludes old plagiarism research branch and training-scale datasets",
        "file_count": len(copy_files),
        "total_bytes": total_bytes,
        "copied_count": copied_count,
        "reused_count": reused_count,
        "locked_manifest_path_count": len(locked_paths),
        "locked_hashes_verified": locked_verified,
        "failures": failures,
        "status": "PASS" if not failures and locked_verified == len(expected_hash_by_path) else "FAIL",
    }
    write_scaffold(dest, source, report)

    print("\nV1 CREATE SUMMARY")
    print(f"status                  : {report['status']}")
    print(f"files in copy plan      : {len(copy_files)}")
    print(f"newly copied            : {copied_count}")
    print(f"reused (--resume)       : {reused_count}")
    print(f"locked hashes verified  : {locked_verified}/{len(expected_hash_by_path)}")
    print(f"report                  : {dest / 'PROJECT_V1_CREATE_REPORT.json'}")
    print(f"readme                  : {dest / 'README_V1.md'}")

    if report["status"] != "PASS":
        print("FAILURES:")
        for item in failures[:20]:
            print(f"  - {item}")
        raise SystemExit(3)

    print("\nPROJECT V1 CREATED AND COPY-INTEGRITY VERIFIED.")
    print("Original research project was not modified.")
    print("Next: cd /d to the V1 directory and run release + adapter + regression checks before API work.")


if __name__ == "__main__":
    main()
