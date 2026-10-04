r"""Trace M1's *observed* natural<->segmented text pairs and original builder code.

Read-only, validation-only. No model loading, no test, no training, no installs.
Run from C:\projectSamSung_v1:
  python system\scripts\11_trace_m1_segmentation_origin.py --research-root C:\1_projectSamSung
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VALIDATION = ROOT / 'data/processed/vn_newsspell_v1/validation.jsonl'
DEFAULT_REPORT = ROOT / 'system/reports/m1_preprocessing_audit_v1/origin_trace_v1.json'
SOURCE_FILES = [
    'scripts/04_generate_spelling_dataset.py',
    'scripts/04_generate_spelling_dataset_v02.py',
    'scripts/09_generate_hybrid_spelling_dataset.py',
]
# Show nearby source to distinguish a copied, already-segmented input from a
# call to a particular segmentation algorithm. Text matches alone cannot prove it.
SOURCE_KEYS = re.compile(
    r'article_natural|article_segmented|word_tokenize|underthesea|pyvi|vncorenlp|rdrsegmenter|segment(?:ed|ation|er)?|\.replace\(',
    re.IGNORECASE,
)


def normalized(text: str) -> str:
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', text)).strip()


def load_validation(path: Path, max_per_label: int, scan_limit: int):
    sample = []
    counts: Counter[int] = Counter()
    seen = 0
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            seen += 1
            if not isinstance(row, dict):
                continue
            raw = row.get('article_natural')
            seg = row.get('article_segmented')
            if isinstance(raw, str) and raw.strip() and isinstance(seg, str) and seg.strip():
                try:
                    label = int(row.get('label', -1))
                except (TypeError, ValueError):
                    label = -1
                if counts[label] < max_per_label:
                    sample.append({'label': label, 'natural': raw, 'segmented': seg,
                                   'variant_id': row.get('variant_id')})
                    counts[label] += 1
            if seen >= scan_limit or (counts[0] >= max_per_label and counts[1] >= max_per_label):
                break
    return sample, seen, dict(counts)


def builder_evidence(research_root: Path, radius: int, max_windows: int):
    result = []
    for rel in SOURCE_FILES:
        p = research_root / rel
        if not p.is_file():
            result.append({'file': rel, 'exists': False, 'windows': []})
            continue
        lines = p.read_text(encoding='utf-8', errors='replace').splitlines()
        # Group neighboring hits so we do not print the same source many times.
        match_indices = [i for i, line in enumerate(lines) if SOURCE_KEYS.search(line)]
        windows = []
        for i in match_indices:
            lo = max(0, i - radius)
            hi = min(len(lines), i + radius + 1)
            if windows and lo <= windows[-1]['end_index'] + 1:
                windows[-1]['end_index'] = max(hi, windows[-1]['end_index'])
            else:
                windows.append({'start_index': lo, 'end_index': hi})
        total_windows = len(windows)
        formatted = []
        for win in windows[:max_windows]:
            lo = win['start_index']
            hi = win['end_index']
            formatted.append({'start_line': lo + 1, 'end_line': hi,
                              'source': '\n'.join(f'{k+1:5}: {lines[k]}' for k in range(lo, hi))})
        result.append({'file': rel, 'exists': True, 'total_matching_lines': len(match_indices),
                       'total_context_windows': total_windows, 'shown_windows': len(formatted),
                       'windows': formatted})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root', type=Path, required=True)
    parser.add_argument('--validation', type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument('--report', type=Path, default=DEFAULT_REPORT)
    parser.add_argument('--per-label', type=int, default=30, help='Up to 30 validation records per label')
    parser.add_argument('--scan-limit', type=int, default=15000)
    parser.add_argument('--context-lines', type=int, default=3)
    parser.add_argument('--max-windows', type=int, default=25)
    args = parser.parse_args()
    if not (1 <= args.per_label <= 250 and 1 <= args.scan_limit <= 200000):
        parser.error('per-label must be 1..250; scan-limit must be 1..200000')
    if not (0 <= args.context_lines <= 12 and 1 <= args.max_windows <= 100):
        parser.error('context-lines must be 0..12; max-windows 1..100')
    rows, scanned, labels = load_validation(args.validation, args.per_label, args.scan_limit)
    if not rows:
        raise RuntimeError('No paired nonempty article_natural / article_segmented records found')

    same_raw_as_unjoined = 0
    same_as_segmented = 0
    joined = []
    mismatches = []
    for row in rows:
        raw = normalized(row['natural'])
        seg = normalized(row['segmented'])
        unjoined = normalized(seg.replace('_', ' '))
        if raw == unjoined:
            same_raw_as_unjoined += 1
        elif len(mismatches) < 5:
            # Only excerpts; report does not export whole articles.
            mismatches.append({'variant_id': row['variant_id'], 'label': row['label'],
                               'natural_excerpt': raw[:180], 'unjoined_excerpt': unjoined[:180]})
        if raw == seg:
            same_as_segmented += 1
        joined.append(sum('_' in token for token in seg.split()))
    evidence = builder_evidence(args.research_root, args.context_lines, args.max_windows)
    result = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'validation_path': str(args.validation), 'validation_rows_scanned': scanned,
        'paired_rows': len(rows), 'label_counts': labels,
        'natural_equals_segmented_unjoined': same_raw_as_unjoined,
        'natural_equals_segmented_unjoined_rate': round(same_raw_as_unjoined / len(rows), 4),
        'natural_equals_segmented_original': same_as_segmented,
        'underscore_joined_units_median': statistics.median(joined),
        'mismatch_excerpts': mismatches,
        'builder_evidence': evidence,
        'limitations': [
            'Exact match after underscore removal shows paired text alignment but DOES NOT identify the algorithm or tokenizer version.',
            'This script does not install a segmenter or claim preprocessing parity.',
            'The original scripts are read, never executed; model weights and test data are not accessed.',
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

    print('M1 ORIGIN TRACE — read-only / validation-only')
    print(f'Paired rows: {len(rows)} / scanned {scanned}; labels {labels}')
    print(f'article_natural == article_segmented.replace("_", " "): {same_raw_as_unjoined}/{len(rows)}'
          f' ({result["natural_equals_segmented_unjoined_rate"]:.1%})')
    print(f'article_natural == original article_segmented: {same_as_segmented}/{len(rows)}')
    print(f'Underscore-joined units median: {result["underscore_joined_units_median"]}')
    for item in evidence:
        print(f'SOURCE: {item["file"]} | exists={item["exists"]}'
              f' | matching lines={item.get("total_matching_lines", 0)}'
              f' | context windows={item.get("total_context_windows", 0)}')
        # Short relevant excerpt to paste into chat; full source windows in JSON report.
        for win in item['windows'][:3]:
            print(f'  lines {win["start_line"]}-{win["end_line"]}:')
            for line in win['source'].splitlines()[:14]:
                print('   ', line)
    if mismatches:
        print('WARN: natural/segmented discrepancies found. Sample excerpts saved in report.')
    print('REPORT:', args.report)
    print('NEXT: inspect original builders, then choose a candidate segmenter and run parity comparison.')


if __name__ == '__main__':
    main()
