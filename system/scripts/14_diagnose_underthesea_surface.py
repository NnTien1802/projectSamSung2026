"""Diagnose why Underthesea outputs fail visible-surface validation for M1.

Run inside C:\\projectSamSung_v1 with the project's .venv:
  .\\.venv\\Scripts\\python.exe system\\scripts\\14_diagnose_underthesea_surface.py

Read-only: validation set only. No GPU/model loading, no checkpoint edits,
no threshold changes and no test-set access. Writes only a NEW JSON report.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / 'data' / 'processed' / 'vn_newsspell_v1' / 'validation.jsonl'
REPORT = ROOT / 'system' / 'reports' / 'm1_preprocessing_audit_v1' / 'underthesea_surface_diagnostic_v1.json'


def visible(text: str) -> str:
    """Match the same NFC + spacing/underscore removal as step 12/13."""
    return re.sub(r'[\s_]+', '', unicodedata.normalize('NFC', text))


def char_info(text: str) -> list[str]:
    return [f'U+{ord(c):04X} {unicodedata.name(c, "UNNAMED")}' for c in text[:12]]


def describe_difference(natural: str, predicted: str, context: int = 35) -> dict:
    original = visible(natural)
    candidate = visible(predicted)
    if original == candidate:
        return {'status': 'surface_preserved'}
    # SequenceMatcher with autojunk=True prevents quadratic work on long articles.
    opcodes = difflib.SequenceMatcher(a=original, b=candidate, autojunk=True).get_opcodes()
    changes = []
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == 'equal':
            continue
        before, after = original[i1:i2], candidate[j1:j2]
        changes.append({
            'operation': tag,
            'original_fragment': before[:60],
            'candidate_fragment': after[:60],
            'original_codepoints': char_info(before),
            'candidate_codepoints': char_info(after),
            'original_context': original[max(0, i1-context):min(len(original), i2+context)],
            'candidate_context': candidate[max(0, j1-context):min(len(candidate), j2+context)],
        })
    return {
        'status': 'surface_changed',
        'original_visible_length': len(original),
        'candidate_visible_length': len(candidate),
        'change_count': len(changes),
        'first_changes': changes[:3],
        'patterns': [(x['operation'], x['original_fragment'], x['candidate_fragment']) for x in changes[:50]],
    }


def read_pairs(path: Path, per_label: int, scan_limit: int):
    labels = Counter()
    rows = []
    scanned = 0
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            scanned += 1
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            try:
                label = int(obj.get('label'))
            except (TypeError, ValueError):
                continue
            natural = obj.get('article_natural')
            reference = obj.get('article_segmented')
            if (label in (0, 1) and labels[label] < per_label
                    and isinstance(natural, str) and natural.strip()
                    and isinstance(reference, str) and reference.strip()):
                if visible(natural) != visible(reference):
                    raise ValueError(f"Invalid original pair: {obj.get('variant_id', '')}")
                rows.append({
                    'label': label,
                    'variant_id': str(obj.get('variant_id', '')),
                    'natural': natural,
                })
                labels[label] += 1
            if (labels[0] == per_label and labels[1] == per_label) or scanned >= scan_limit:
                break
    if labels[0] != per_label or labels[1] != per_label:
        raise RuntimeError(f'Insufficient validation pairs: {dict(labels)}')
    return rows, scanned


def diagnose(rows, fn):
    statuses = Counter()
    by_label = {'0': Counter(), '1': Counter()}
    patterns = Counter()
    examples = {'0': [], '1': []}
    for index, row in enumerate(rows, 1):
        predicted = fn(row['natural'])
        if not isinstance(predicted, str):
            raise TypeError(f'Underthesea returned {type(predicted).__name__} instead of str')
        diagnostic = describe_difference(row['natural'], predicted)
        status = diagnostic['status']
        statuses[status] += 1
        by_label[str(row['label'])][status] += 1
        if status == 'surface_changed':
            for tag, before, after in diagnostic['patterns']:
                patterns[(tag, before, after)] += 1
            if len(examples[str(row['label'])]) < 6:
                examples[str(row['label'])].append({
                    'variant_id': row['variant_id'],
                    'label': row['label'],
                    'original_visible_length': diagnostic['original_visible_length'],
                    'candidate_visible_length': diagnostic['candidate_visible_length'],
                    'change_count': diagnostic['change_count'],
                    'first_changes': diagnostic['first_changes'],
                })
        if index % 10 == 0 or index == len(rows):
            print(f'  diagnosed {index}/{len(rows)}', flush=True)
    common_patterns = [
        {'operation': tag, 'original_fragment': before, 'candidate_fragment': after, 'count': n}
        for (tag, before, after), n in patterns.most_common(20)
    ]
    return {
        'sample_count': len(rows),
        'overall': dict(statuses),
        'by_label': {label: dict(counts) for label, counts in by_label.items()},
        'common_changes': common_patterns,
        'examples_by_label': examples,
        'notes': [
            'Visible surface means NFC Unicode with underscores and whitespace removed, identical to steps 12/13.',
            'This diagnoses character changes but does not correct or infer the original word boundaries.',
            'No model inference or test data was used.',
        ],
    }


def self_test():
    assert describe_difference('hội đồng quản trị', 'hội_đồng quản_trị')['status'] == 'surface_preserved'
    diff = describe_difference('Hôm nay trời đẹp.', 'Hôm nay trời đep.')
    assert diff['status'] == 'surface_changed' and diff['first_changes']
    assert diff['first_changes'][0]['original_fragment'] == 'ẹ'
    assert diff['first_changes'][0]['candidate_fragment'] == 'e'
    rows = [{'label': 0, 'variant_id': 'a', 'natural': 'hội đồng quản trị'},
            {'label': 1, 'variant_id': 'b', 'natural': 'Hôm nay trời đẹp.'}]
    output = diagnose(rows, lambda s: 'hội_đồng quản_trị' if s.startswith('hội') else 'Hôm nay trời đep.')
    assert output['overall'] == {'surface_preserved': 1, 'surface_changed': 1}
    assert output['by_label']['0']['surface_preserved'] == 1
    assert output['by_label']['1']['surface_changed'] == 1
    print('SELF-TEST PASS: same-surface joins, changed diacritics, label aggregation')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation', type=Path, default=VALIDATION)
    parser.add_argument('--report', type=Path, default=REPORT)
    parser.add_argument('--per-label', type=int, default=30)
    parser.add_argument('--scan-limit', type=int, default=20000)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not (1 <= args.per_label <= 250 and args.scan_limit >= args.per_label * 2):
        parser.error('per-label must be 1..250; scan-limit must be at least twice per-label')
    from underthesea import word_tokenize
    rows, scanned = read_pairs(args.validation, args.per_label, args.scan_limit)
    print(f'UNDERTHESEA SURFACE DIAGNOSTIC — validation only, scanned={scanned}, samples={len(rows)}', flush=True)
    result = diagnose(rows, lambda text: word_tokenize(text, format='text'))
    output = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'validation_path': str(args.validation),
        'scanned': scanned,
        **result,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Preserved surface:', result['overall'].get('surface_preserved', 0), '/', len(rows))
    print('Changed surface:', result['overall'].get('surface_changed', 0), '/', len(rows))
    print('Common changes:')
    for p in result['common_changes'][:8]:
        print(f"  {p['count']}x {p['operation']}: {p['original_fragment']!r} -> {p['candidate_fragment']!r}")
    print('REPORT:', args.report)


if __name__ == '__main__':
    main()
