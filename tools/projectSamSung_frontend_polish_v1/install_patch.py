#!/usr/bin/env python3
"""Safely apply UI-only polish to the original projectSamSung React V1.

Does not install packages, touch back-end/.venv, or modify frozen model artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from datetime import datetime
from pathlib import Path

BASE_SHA256 = {
    'frontend/src/App.tsx': '0c977bf160272f02cbe282c44c5e91f253e0a821fdb04b13357059ce8cfbb934',
    'frontend/src/api.ts': '3e587c9090ea89b68938b6d4837e178f0ffd9ebddb0c3119502735677d48c2ee',
    'frontend/src/style.css': '4ff62d0e7addf2417c8cff650d2c1ef84e4007d604ef3c215a0b89edaa63de09',
}

PATCH_FILES = (
    'frontend/src/App.tsx',
    'frontend/src/api.ts',
    'frontend/src/style.css',
    'frontend/src/demo_cases.ts',
    'frontend/src/presentation.ts',
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description='Install React V1 UI polish without touching frozen models')
    parser.add_argument('--project', type=Path, required=True, help='C:\\projectSamSung_v1')
    parser.add_argument('--dry-run', action='store_true', help='Validate hashes without copying any files')
    args = parser.parse_args()
    root = args.project.resolve()
    overlay = Path(__file__).resolve().parent / 'overlay'

    if not (root / 'frontend' / 'package.json').is_file():
        raise SystemExit('ABORT: frontend/package.json missing. Give --project the project root, not its frontend directory.')
    for rel in PATCH_FILES:
        if not (overlay / rel).is_file():
            raise SystemExit(f'ABORT: patch ZIP incomplete: {rel}')

    unchanged = True
    for rel in PATCH_FILES:
        target = root / rel
        replacement = overlay / rel
        if not target.is_file():
            if rel in BASE_SHA256:
                raise SystemExit(f'ABORT: base file missing: {rel}')
            unchanged = False
            continue
        current_hash, patch_hash = sha256(target), sha256(replacement)
        if current_hash == patch_hash:
            continue
        unchanged = False
        if rel in BASE_SHA256 and current_hash == BASE_SHA256[rel]:
            continue
        raise SystemExit(
            f'ABORT: {rel} differs from original React V1 (or partially changed). '
            'No files were modified. Back up your customized file before applying this patch.'
        )
    if unchanged:
        print('ALREADY INSTALLED: all 5 UI files match this patch. No changes needed.')
        return
    if args.dry_run:
        print('DRY RUN PASS: expected React V1 source verified. No changes made.')
        return

    backup = root / 'system' / 'reports' / 'frontend_patch_backups' / (
        'pre_polish_v1_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    )
    for rel in BASE_SHA256:
        target = root / rel
        if target.is_file():
            destination = backup / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, destination)
    for rel in PATCH_FILES:
        source = overlay / rel
        destination = root / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if sha256(destination) != sha256(source):
            raise RuntimeError(f'Verification failed after copying {rel}')
    print('FRONTEND POLISH INSTALLED: Vietnamese warnings, local-API notices, sample scenarios.')
    print('BACKUP:', backup)
    print('UNCHANGED: system/backend, .venv, checkpoints, thresholds, source research project.')
    print('NEXT: cd frontend && npm run build && npm run dev')


if __name__ == '__main__':
    main()
