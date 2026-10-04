"""Read-only audit before connecting raw Vietnamese input to frozen M1.

From C:\\projectSamSung_v1:
  python system\\scripts\\10_audit_m1_segmentation.py --research-root C:\\1_projectSamSung

No model loading, inference, training, dataset editing, package installation or test split access.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / 'data' / 'processed' / 'vn_newsspell_v1' / 'validation.jsonl'
REPORT = ROOT / 'system' / 'reports' / 'm1_preprocessing_audit_v1' / 'audit.json'
RAW_NAMES = ('article_raw', 'raw_article', 'original_article', 'article', 'content', 'text', 'article_text')
SOURCE_PATTERN = re.compile(r'article_segmented|word_tokenize|underthesea|pyvi|vncorenlp|rdrsegmenter|word[ _-]?segment', re.I)


def normalize(value):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', value)).strip()


def read_records(path, limit):
    result = []
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError('Validation rows must be JSON objects')
            result.append(item)
            if len(result) >= limit:
                break
    if not result:
        raise ValueError('Validation file is empty')
    return result


def inspect_source(root, maximum_hits=35):
    if not root or not root.is_dir():
        return []
    hits = []
    # The original M1 dataset builders live under top-level scripts/.
    scripts = root / 'scripts'
    if not scripts.exists():
        return hits
    for path in sorted(scripts.glob('*.py')):
        if path.stat().st_size > 1_000_000:
            continue
        for i, line in enumerate(path.read_text(encoding='utf-8', errors='replace').splitlines(), start=1):
            if SOURCE_PATTERN.search(line):
                hits.append({'file': str(path.relative_to(root)), 'line': i, 'snippet': line.strip()[:150]})
                if len(hits) >= maximum_hits:
                    return hits
    return hits


def audit(data_path, sample_limit, source_root):
    rows = read_records(data_path, sample_limit)
    segmented = [r['article_segmented'] for r in rows if isinstance(r.get('article_segmented'), str) and r['article_segmented'].strip()]
    if not segmented:
        raise RuntimeError('No nonempty article_segmented field in inspected validation rows')
    lengths = [len(value) for value in segmented]
    whitespace_words = [len(value.split()) for value in segmented]
    joined_words = [sum('_' in token for token in value.split()) for value in segmented]
    observed_fields = sorted({field for row in rows for field in row})
    raw_analysis = {}
    for candidate in RAW_NAMES:
        present = [r for r in rows if isinstance(r.get(candidate), str) and r[candidate].strip() and isinstance(r.get('article_segmented'), str)]
        if not present:
            continue
        matched = sum(normalize(r[candidate]) == normalize(r['article_segmented'].replace('_', ' ')) for r in present)
        raw_analysis[candidate] = {'paired_rows': len(present), 'exact_match_after_unjoining': matched, 'fraction': round(matched / len(present), 4)}
    optional_packages = {pkg: importlib.util.find_spec(pkg) is not None for pkg in ['underthesea', 'pyvi', 'py_vncorenlp', 'vncorenlp', 'transformers']}
    return {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'project_root': str(ROOT),
        'validation_path': str(data_path),
        'validation_rows_inspected': len(rows),
        'rows_with_segmented_text': len(segmented),
        'field_names': observed_fields,
        'segmented_character_length': {'min': min(lengths), 'median': statistics.median(lengths), 'max': max(lengths)},
        'space_delimited_units': {'median': statistics.median(whitespace_words), 'max': max(whitespace_words)},
        'underscore_joined_units': {'median': statistics.median(joined_words), 'max': max(joined_words)},
        'possible_parallel_raw_fields': raw_analysis,
        'available_packages': optional_packages,
        'research_source_root': str(source_root) if source_root else None,
        'possible_preprocessing_code_matches': inspect_source(source_root),
        'limitations': [
            'An underscore is a word-segmentation convention, not proof of a specific tokenizer.',
            'Matching text after underscore removal does not establish identical segmentation.',
            'This is a schema and source audit only. Do not auto-segment production requests until parity is validated.',
            'No model checkpoint, threshold, or test split was accessed.',
        ],
    }


def main():
    parser = argparse.ArgumentParser(description='Inspect frozen M1 validation format and original research preprocessing code, read-only')
    parser.add_argument('--validation', type=Path, default=VALIDATION)
    parser.add_argument('--research-root', type=Path, default=None, help='Optional original research root, e.g. C:\\1_projectSamSung')
    parser.add_argument('--sample-limit', type=int, default=50)
    parser.add_argument('--report', type=Path, default=REPORT)
    args = parser.parse_args()
    if args.sample_limit < 1 or args.sample_limit > 500:
        parser.error('--sample-limit must be 1..500')
    result = audit(args.validation, args.sample_limit, args.research_root)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('M1 SEGMENTATION AUDIT — read-only, validation only')
    print('Inspected:', result['validation_rows_inspected'], 'rows')
    print('Fields:', ', '.join(result['field_names']))
    print('Joined units median:', result['underscore_joined_units']['median'])
    print('Parallel raw candidates:', result['possible_parallel_raw_fields'])
    print('Installed segmenters:', {k:v for k,v in result['available_packages'].items() if k!='transformers'})
    print('Original M1 script matches:', len(result['possible_preprocessing_code_matches']))
    for hit in result['possible_preprocessing_code_matches'][:15]:
        print('  {}:{}: {}'.format(hit['file'], hit['line'], hit['snippet']))
    print('REPORT:', args.report)
    print('Next: choose segmentation method from actual source evidence, then test parity; no auto-segmentation added yet.')


if __name__ == '__main__':
    main()
