#!/usr/bin/env python3
"""Read-only project inventory for projectSamSung.

Place at <project>/system/scripts/02_audit_project_structure.py and run from
<project> with: python system/scripts/02_audit_project_structure.py

This script never deletes or moves project files. It only writes reports to
system/reports/project_inventory_v1. SHA-256 values are read from the existing
lock manifest; checkpoint files are not rehashed here.
"""
from __future__ import annotations
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'system' / 'reports' / 'project_inventory_v1'
LOCK = ROOT / 'system' / 'release' / 'model_lock_manifest_v1.json'
SKIP = {'.git', '.venv', 'venv', '__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache',
        '.tox', 'node_modules', '.next', '.cache', 'dist', 'build'}
MAX_TREE_DEPTH = 4
MAX_TREE_ITEMS_PER_DIR = 75


def human(size):
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if size < 1024 or unit == 'TB':
            return f'{size:.1f} {unit}'
        size /= 1024


def relative(path):
    return path.relative_to(ROOT).as_posix()


def classify(path, protected):
    p = path.as_posix()
    if p in protected:
        return 'LOCKED_KEEP', 'SHA-256 manifest: do not edit/move/delete'
    if p.startswith('system/release/'):
        return 'KEEP', 'release records/tokenizers; inspect references'
    if p == 'system/scripts/01_model_release.py':
        return 'KEEP', 'release creation and verification'
    if p.startswith('module2_headline_integrity/inference/'):
        return 'KEEP', 'frozen unified pipeline/regression provenance'
    if p.startswith('module3_plagiarism_detection/'):
        return 'ARCHIVE_REVIEW', 'paused plagiarism research; preserve until archived/backed up'
    if p.startswith(('data/raw/', 'data/processed/')) or '/data/raw/' in p or '/data/processed/' in p:
        return 'KEEP_RESEARCH', 'raw/final datasets; retain for reproducibility or archive'
    if p.startswith(('models/', 'module2_headline_integrity/models/')):
        return 'REVIEW_MODEL', 'non-release checkpoint; keep if research provenance is needed'
    if p.startswith('results/') or '/results/' in p:
        return 'REVIEW_RESULTS', 'keep official metrics/test reports; review diagnostic outputs'
    if '/data/interim/' in p or p.startswith('data/interim/'):
        return 'REVIEW_INTERIM', 'derived data: verify dependencies and backups before removal'
    if p.endswith(('.log', '.tmp', '.bak')):
        return 'CLEANUP_CANDIDATE', 'inspect before removing; may contain unique run logs'
    return 'REVIEW', 'inspect use/dependencies; no automatic deletion'


def main():
    if not LOCK.is_file():
        raise SystemExit(f'Missing lock manifest: {LOCK}. Do not reorganize before locating it.')
    manifest = json.loads(LOCK.read_text(encoding='utf-8'))
    protected = {item['path'].replace('\\', '/') for item in manifest.get('files', [])}
    protected.add('system/release/model_lock_manifest_v1.json')
    protected.add('system/release/environment_v1.json')
    protected.add('system/release/model_lock_config.json')
    protected.add('system/release/model_inventory_v1.json')
    records = []
    excluded_counts = Counter()
    errors = []
    dir_totals = defaultdict(lambda: [0, 0])
    present_protected = set()
    for base, dirs, filenames in os.walk(ROOT, topdown=True, followlinks=False):
        base = Path(base)
        allowed = []
        for d in sorted(dirs):
            full = base / d
            if d in SKIP or full == ROOT / 'system' / 'reports':
                excluded_counts[d] += 1
            elif full.is_symlink():
                excluded_counts['symlink_directory'] += 1
            else:
                allowed.append(d)
        dirs[:] = allowed
        for name in filenames:
            file = base / name
            if file.is_symlink():
                excluded_counts['symlink_file'] += 1
                continue
            try:
                size = file.stat().st_size
            except OSError as exc:
                errors.append(f'{file}: {exc}')
                continue
            p = relative(file)
            status, reason = classify(Path(p), protected)
            records.append(dict(path=p, size_bytes=size, status=status, reason=reason,
                                suffix=file.suffix.lower(), root_dir=p.split('/')[0]))
            if p in protected:
                present_protected.add(p)
            top = p.split('/')[0]
            dir_totals[top][0] += 1
            dir_totals[top][1] += size
    records.sort(key=lambda x: x['path'])
    missing = sorted(protected - present_protected)
    # Compact tree from recorded file paths, including directory nodes.
    tree = {}
    for item in records:
        parts = item['path'].split('/')
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part + '/', {})
        node[parts[-1]] = None
    lines = [f'PROJECT ROOT: {ROOT}', 'TREE (depth <= 4, <= 75 children per directory):']
    def write_tree(node, prefix='', depth=0):
        if depth >= MAX_TREE_DEPTH:
            lines.append(prefix + '  ... (depth limit)')
            return
        items = sorted(node.items(), key=lambda it: (not it[0].endswith('/'), it[0].casefold()))
        for i, (name, child) in enumerate(items[:MAX_TREE_ITEMS_PER_DIR]):
            lines.append(prefix + name)
            if child is not None:
                write_tree(child, prefix + '  ', depth + 1)
        if len(items) > MAX_TREE_ITEMS_PER_DIR:
            lines.append(prefix + f'... ({len(items)-MAX_TREE_ITEMS_PER_DIR} more entries)')
    write_tree(tree)
    summary = {
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'project_root': str(ROOT),
        'file_count': len(records),
        'total_scanned_size_bytes': sum(r['size_bytes'] for r in records),
        'release_id': manifest.get('release_id'),
        'locked_manifest_files_expected': len(manifest.get('files', [])),
        'locked_manifest_files_missing': sorted(p for p in (set(item['path'].replace('\\','/') for item in manifest.get('files', [])) - present_protected)),
        'missing_other_protected_paths': sorted(set(missing) - set(item['path'].replace('\\','/') for item in manifest.get('files', []))),
        'status_count': dict(Counter(r['status'] for r in records)),
        'top_level': [{'directory': k, 'files': v[0], 'size_bytes': v[1]}
                      for k, v in sorted(dir_totals.items(), key=lambda kv: -kv[1][1])],
        'largest_files': [{k: r[k] for k in ('path', 'size_bytes', 'status')} for r in
                          sorted(records, key=lambda x: -x['size_bytes'])[:70]],
        'excluded_directory_count': dict(excluded_counts),
        'scan_errors': errors[:100],
        'notice': 'Classification is preliminary. KEEP/ARCHIVE_REVIEW/CLEANUP_CANDIDATE are not deletion permissions. No data was deleted or moved.'
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'tree.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    with (OUT / 'files.csv').open('w', newline='', encoding='utf-8-sig') as fp:
        writer = csv.DictWriter(fp, fieldnames=['path', 'size_bytes', 'status', 'reason', 'suffix', 'root_dir'])
        writer.writeheader()
        writer.writerows(records)
    print('READ-ONLY INVENTORY COMPLETE')
    print('Project root:', ROOT)
    print('Files:', len(records), '| Approx size:', human(summary['total_scanned_size_bytes']))
    print('Locked files expected:', summary['locked_manifest_files_expected'])
    print('Missing locked files:', len(summary['locked_manifest_files_missing']))
    print('Main folders:')
    for item in summary['top_level'][:20]:
        print(' ', item['directory'], '|', item['files'], 'files |', human(item['size_bytes']))
    print('Reports:', OUT)
    print('Send tree.txt, summary.json and files.csv for a per-file keep/archive/review plan.')
    if errors or summary['locked_manifest_files_missing']:
        print('WARNING: scan errors or missing locked files; inspect summary.json.')


if __name__ == '__main__':
    main()
