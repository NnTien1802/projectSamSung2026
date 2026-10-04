#!/usr/bin/env python3
"""Backup Module 1 research files without moving or deleting originals.

Place at: C:\\1_projectSamSung\\system\\scripts\\03_backup_module1.py
From project root:
  python system\\scripts\\03_backup_module1.py --dest D:\\projectSamSung_module1_backup_v1
  python system\\scripts\\03_backup_module1.py --dest D:\\projectSamSung_module1_backup_v1 --execute

The dry run is the default. Execute only when a separate destination has enough
space. The backup copies all four top-level Module 1 roots: data, models,
scripts, results, as well as system/release. It SHA256-verifies EVERY copied
file and checks frozen Module 1 artifacts against the original release manifest.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2]
ROOT_DIRS = ("data", "models", "scripts", "results", "system/release")
HASH_BLOCK = 8 * 1024 * 1024


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(HASH_BLOCK)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def all_files(root: Path):
    result = []
    for root_name in ROOT_DIRS:
        p = root / root_name
        if not p.is_dir():
            raise FileNotFoundError(f"Required Module 1 directory missing: {p}")
        for file in p.rglob('*'):
            if file.is_symlink():
                raise RuntimeError(f"Symlink found (review manually): {file}")
            if file.is_file():
                result.append(file)
    return sorted(set(result))


def read_frozen_m1(root: Path):
    mpath = root / 'system/release/model_lock_manifest_v1.json'
    manifest = json.loads(mpath.read_text(encoding='utf-8'))
    if manifest.get('release_id') != 'news_quality_models_v1':
        raise RuntimeError('Unexpected release_id; inspect current manifest before backup.')
    records = [r for r in manifest.get('files', [])
               if any(str(role).startswith('spelling.') for role in r.get('roles', []))]
    if not records:
        raise RuntimeError('Manifest contains no spelling artifact entries; cannot prove M1 lock.')
    for record in records:
        rel = Path(record['path'])
        p = (root / rel).resolve()
        if not is_within(p, root.resolve()) or not p.is_file():
            raise RuntimeError(f'Frozen Module 1 artifact missing/unsafe: {rel}')
        if digest(p) != record['sha256']:
            raise RuntimeError(f'Frozen Module 1 artifact has changed: {rel}')
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dest', type=Path, required=True, help='Separate backup directory, outside project root')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true', help='Copy files, verify SHA-256, write report')
    mode.add_argument('--dry-run', action='store_true', help='Print plan without copying (default)')
    args = parser.parse_args()
    source = SOURCE_ROOT.resolve()
    destination = args.dest.expanduser().resolve()
    if is_within(destination, source) or is_within(source, destination):
        parser.error('Destination must be outside the source project and cannot be its parent.')
    if destination.exists() and any(destination.iterdir()):
        parser.error('Destination exists and is NOT EMPTY. Choose a new backup directory; no overwriting.')
    locked = read_frozen_m1(source)
    files = all_files(source)
    sizes = {name: 0 for name in ROOT_DIRS}
    for file in files:
        relative = file.relative_to(source)
        for name in ROOT_DIRS:
            if is_within(relative, Path(name)):
                sizes[name] += file.stat().st_size
                break
    required_bytes = sum(f.stat().st_size for f in files)
    print(f'SOURCE       : {source}')
    print(f'DESTINATION  : {destination}')
    print(f'FILES        : {len(files):,}')
    print(f'TOTAL        : {required_bytes / 1e9:.2f} GB (decimal)')
    print(f'LOCKED M1    : {len(locked)} files verified against source release manifest')
    for group, size in sizes.items():
        print(f'  {group:<16} : {size / 1e9:.3f} GB')
    if not args.execute:
        print('DRY RUN: originals unchanged; no backup written.')
        print('Run again with --execute after selecting a separate destination with sufficient space.')
        return 0
    ancestor = destination
    while not ancestor.exists():
        ancestor = ancestor.parent
    free_bytes = shutil.disk_usage(ancestor).free
    # A little headroom for report/filesystem metadata; do not insist on 10GB extra.
    reserve = max(512 * 1024 * 1024, required_bytes // 20)
    if free_bytes < required_bytes + reserve:
        raise RuntimeError(f'Not enough free space: {free_bytes / 1e9:.2f} GB free, '
                           f'need at least {(required_bytes + reserve) / 1e9:.2f} GB.')
    destination.mkdir(parents=True, exist_ok=True)
    copied_records = []
    copied_total = 0
    print('\nCOPY + VERIFY (source -> destination):')
    for index, original in enumerate(files, 1):
        relative = original.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise RuntimeError(f'Refusing to overwrite existing target: {target}')
        expected_hash = digest(original)
        shutil.copy2(original, target)
        actual_hash = digest(target)
        if expected_hash != actual_hash:
            raise RuntimeError(f'COPY HASH MISMATCH: {relative}; backup incomplete.')
        copied_records.append({
            'path': relative.as_posix(),
            'size_bytes': original.stat().st_size,
            'sha256': expected_hash
        })
        copied_total += original.stat().st_size
        if index % 20 == 0 or index == len(files):
            print(f'  {index:,}/{len(files):,} files | {copied_total / 1e9:.2f}/{required_bytes / 1e9:.2f} GB')
    for r in locked:
        rel = r['path']
        target = destination / rel
        if not target.is_file() or digest(target) != r['sha256']:
            raise RuntimeError(f'BACKUP FROZEN M1 HASH MISMATCH: {rel}')
    report = {
        'status': 'VERIFIED',
        'backup_name': 'projectSamSung_module1_research_v1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source_project_root': str(source),
        'destination': str(destination),
        'policy': 'COPY ONLY; originals unchanged; all copied files SHA256 verified',
        'module1_roots_copied': list(ROOT_DIRS),
        'release_id': 'news_quality_models_v1',
        'module1_frozen_artifacts_verified': [r['path'] for r in locked],
        'file_count': len(copied_records),
        'total_size_bytes': copied_total,
        'files': copied_records
    }
    report_file = destination / 'MODULE1_BACKUP_REPORT.json'
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'\nBACKUP VERIFIED: {len(copied_records)} files; {copied_total / 1e9:.2f} GB')
    print(f'REPORT: {report_file}')
    print('Do not move/delete originals until all three release regression tests pass after any project reorganization.')
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f'BACKUP FAILED: {error}', file=sys.stderr)
        sys.exit(1)
