#!/usr/bin/env python3
"""Backup old Module 2 (clickbait + headline consistency), BEFORE splitting logical modules.

Place this file at C:\\1_projectSamSung\\system\\scripts\\04_backup_module2_before_split.py
Run from C:\\1_projectSamSung:
    python system\\scripts\\04_backup_module2_before_split.py --dest E:\\projectSamSung_module2_original_backup_v1
    python system\\scripts\\04_backup_module2_before_split.py --dest E:\\projectSamSung_module2_original_backup_v1 --execute

Copy-only. Preserves the entire old Module 2 directory and a snapshot of
system/release (manifest, config, environment, tokenizer). It does not copy
other modules or claim the backup contains the full 3-model release.
Every copied file is SHA-256 verified; original frozen Module 2 paths are
compared against the project's release manifest before and after copying.
For safety it refuses nonempty destinations, symlinks and changes to locked
source files. Outputs a JSON verification report.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2]
ROOT_DIRS = ('module2_headline_integrity', 'system/release')
LOCK_MANIFEST = 'system/release/model_lock_manifest_v1.json'
EXPECTED_LOCKED = {
    'module2_headline_integrity/inference/headline_integrity_pipeline.py',
    'module2_headline_integrity/models/clickbait/mbert_title_only_v1/best_model.pt',
    'module2_headline_integrity/models/consistency/baseline_title_description_mbert/best_model.pt',
    'module2_headline_integrity/models/consistency/full_article_max_mil_mbert/best_model.pt',
}
SKIP_DIRS = {'__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache', '.cache'}
BLOCK = 8 * 1024 * 1024


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(BLOCK), b''):
            h.update(block)
    return h.hexdigest()


def within(path, root):
    return path == root or root in path.parents


def get_files(root):
    files = []
    for rel in ROOT_DIRS:
        top = root / rel
        if not top.is_dir() or top.is_symlink():
            raise RuntimeError(f'Missing or symbolic-link source folder: {top}')
        for folder, dirs, names in os.walk(top, followlinks=False):
            base = Path(folder)
            for name in dirs:
                if (base / name).is_symlink():
                    raise RuntimeError(f'Symlink folder requires manual review: {base / name}')
            dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
            for name in sorted(names):
                f = base / name
                if f.is_symlink():
                    raise RuntimeError(f'Symlink file requires manual review: {f}')
                if f.is_file():
                    files.append(f)
    return sorted(files)


def get_locked_module2(root):
    manifest_path = root / LOCK_MANIFEST
    if not manifest_path.is_file():
        raise RuntimeError(f'Release manifest missing: {manifest_path}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('release_id') != 'news_quality_models_v1':
        raise RuntimeError('Unexpected release ID; inspect manifest before proceeding.')
    records = {}
    for row in manifest.get('files', []):
        rel = str(row.get('path', '')).replace('\\', '/')
        if rel.startswith('module2_headline_integrity/'):
            if not row.get('sha256'):
                raise RuntimeError(f'Missing SHA-256 in release manifest: {rel}')
            records[rel] = row['sha256']
    missing = EXPECTED_LOCKED - set(records)
    if missing:
        raise RuntimeError('Expected frozen Module 2 artifacts missing from manifest: ' + ', '.join(sorted(missing)))
    for rel, expected in records.items():
        file = (root / rel).resolve()
        if not within(file, root.resolve()) or not file.is_file():
            raise RuntimeError(f'Missing/unsafe locked file: {rel}')
        if sha256(file) != expected:
            raise RuntimeError(f'Frozen source changed: {rel}')
    return manifest['release_id'], records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dest', required=True, type=Path,
                   help='NEW empty directory, ideally on a separate disk (e.g. E:)')
    p.add_argument('--execute', action='store_true', help='Actually copy and SHA-256 verify; default is dry-run')
    args = p.parse_args()
    source = SOURCE_ROOT.resolve()
    dest = args.dest.expanduser().resolve()
    if within(dest, source) or within(source, dest):
        p.error('Destination must not be inside source project or contain source project.')
    if not dest.parent.exists():
        p.error(f'Destination parent does not exist or drive unavailable: {dest.parent}')
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        p.error('Destination exists and is not an empty directory. No overwriting.')
    release_id, locks = get_locked_module2(source)
    files = get_files(source)
    paths = {f.relative_to(source).as_posix() for f in files}
    for rel in locks:
        if rel not in paths:
            raise RuntimeError(f'Locked file excluded from backup: {rel}')
    sizes = {key: 0 for key in ROOT_DIRS}
    for f in files:
        rel = f.relative_to(source).as_posix()
        sizes[next(key for key in ROOT_DIRS if rel.startswith(key + '/'))] += f.stat().st_size
    total = sum(sizes.values())
    print(f'SOURCE         : {source}', flush=True)
    print(f'DESTINATION    : {dest}', flush=True)
    print(f'FILES          : {len(files):,}', flush=True)
    print(f'TOTAL          : {total/1e9:.2f} GB (decimal)', flush=True)
    print(f'FROZEN MODULE2 : {len(locks)} artifact(s) verified against source release manifest', flush=True)
    for rel, n in sizes.items():
        print(f'  {rel:<28}: {n/1e9:.3f} GB', flush=True)
    if not args.execute:
        print('DRY RUN: nothing copied or changed.', flush=True)
        print('When ready, repeat with --execute.', flush=True)
        return 0
    # Check space before creating any output. 5%+512MiB buffer.
    free = shutil.disk_usage(dest.parent).free
    reserve = max(512 * 1024 * 1024, int(total * .05))
    if free < total + reserve:
        raise RuntimeError(f'Insufficient destination space: {free/1e9:.2f} GB free; require {(total+reserve)/1e9:.2f} GB.')
    dest.mkdir(parents=True, exist_ok=True)
    completed = 0
    records = []
    print('\nCOPY + SHA256 VERIFY (per-file progress):', flush=True)
    for index, src in enumerate(files, start=1):
        rel = src.relative_to(source)
        target = dest / rel
        if target.exists():
            raise RuntimeError(f'Refusing to overwrite existing target: {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        size_before = src.stat().st_size
        print(f'[{index}/{len(files)}] {rel.as_posix()} ({size_before/1e6:.1f} MB) ... ', end='', flush=True)
        expected = sha256(src)
        if rel.as_posix() in locks and expected != locks[rel.as_posix()]:
            raise RuntimeError(f'Frozen file changed during backup: {rel}')
        shutil.copy2(src, target)
        actual = sha256(target)
        if actual != expected or target.stat().st_size != size_before:
            raise RuntimeError(f'HASH/SIZE MISMATCH at {rel}: backup incomplete; do not delete originals.')
        records.append({'path': rel.as_posix(), 'size_bytes': size_before, 'sha256': expected})
        completed += size_before
        print(f'OK | {completed/1e9:.2f}/{total/1e9:.2f} GB', flush=True)
    for rel, expected in locks.items():
        if sha256(dest / rel) != expected:
            raise RuntimeError(f'Frozen Module 2 backup has changed: {rel}')
    # Do not claim that standalone verify of global manifest will pass in this partial backup.
    report = {
        'status': 'VERIFIED',
        'scope': 'Old Module 2 full tree + system/release snapshot; not a standalone full-model release',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'source': str(source), 'destination': str(dest),
        'release_id': release_id,
        'policy': 'COPY ONLY; source unchanged; each copied file checked with SHA-256',
        'frozen_module2_paths_verified': sorted(locks),
        'file_count': len(records), 'total_size_bytes': completed,
        'files': records,
    }
    report_path = dest / 'MODULE2_ORIGINAL_BACKUP_REPORT.json'
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(f'\nBACKUP VERIFIED: {len(records)} files; {completed/1e9:.2f} GB', flush=True)
    print(f'REPORT: {report_path}', flush=True)
    print('Keep original Module 2 unchanged until new M2/M3 adapters pass regression.', flush=True)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (RuntimeError, OSError, ValueError) as exc:
        print(f'BACKUP FAILED: {exc}', file=sys.stderr, flush=True)
        sys.exit(1)
