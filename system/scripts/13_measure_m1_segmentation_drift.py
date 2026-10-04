"""Measure effect of word-segmentation mismatch on frozen M1 predictions.

Run from C:\\projectSamSung_v1 using the proven GPU model environment:
  python system\\scripts\\13_measure_m1_segmentation_drift.py --self-test
  python system\\scripts\\13_measure_m1_segmentation_drift.py --device cuda --candidates pyvi

Validation-only, read-only for all model artifacts. Does not auto-choose or
integrate a segmenter; no model retraining or threshold search. Keeps the
original 0.57 model threshold. Loads the frozen M1 adapter only once.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import statistics
import sys
import time
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / 'data' / 'processed' / 'vn_newsspell_v1' / 'validation.jsonl'
REPORT = ROOT / 'system' / 'reports' / 'm1_preprocessing_audit_v1' / 'prediction_drift_v1.json'
THRESHOLD = 0.57  # existing frozen threshold; never tuned here


def canonical_surface(text: str) -> str:
    return re.sub(r'[\s_]+', '', unicodedata.normalize('NFC', text))


def read_pairs(path: Path, per_label: int = 30, scan_limit: int = 20000):
    rows = []
    label_counts = Counter()
    scanned = 0
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            scanned += 1
            row = json.loads(line)
            try:
                label = int(row['label'])
            except (ValueError, TypeError, KeyError):
                continue
            natural, segmented = row.get('article_natural'), row.get('article_segmented')
            if label in (0, 1) and label_counts[label] < per_label and isinstance(natural, str) and natural.strip() and isinstance(segmented, str) and segmented.strip():
                if canonical_surface(natural) != canonical_surface(segmented):
                    raise RuntimeError(f"Invalid validation pair: {row.get('variant_id', 'unknown')}")
                rows.append({'label': label, 'variant_id': str(row.get('variant_id', '')),
                             'natural': natural, 'segmented': segmented})
                label_counts[label] += 1
            if (label_counts[0] >= per_label and label_counts[1] >= per_label) or scanned >= scan_limit:
                break
    if label_counts[0] != per_label or label_counts[1] != per_label:
        raise RuntimeError(f'Insufficient validation pairs: {dict(label_counts)}')
    return rows, scanned


def segmenter(name):
    if importlib.util.find_spec(name) is None:
        raise ImportError(f'{name} is not installed in this Python environment')
    if name == 'pyvi':
        from pyvi import ViTokenizer
        return ViTokenizer.tokenize
    if name == 'underthesea':
        from underthesea import word_tokenize
        return lambda s: word_tokenize(s, format='text')
    raise ValueError(name)


def summarize(rows):
    if not rows:
        return {'n': 0}
    differences = [r['absolute_probability_difference'] for r in rows]
    sorted_diff = sorted(differences)
    p95_i = math.ceil(len(sorted_diff) * 0.95) - 1
    return {
        'n': len(rows),
        'prediction_flips': sum(r['prediction_changed'] for r in rows),
        'reference_predicted_error': sum(r['reference_prediction'] == 1 for r in rows),
        'candidate_predicted_error': sum(r['candidate_prediction'] == 1 for r in rows),
        'mean_absolute_probability_difference': round(statistics.mean(differences), 6),
        'p95_absolute_probability_difference': round(sorted_diff[p95_i], 6),
        'max_absolute_probability_difference': round(max(differences), 6),
        'chunk_count_changes': sum(r['chunk_count_changed'] for r in rows),
        'changed_predictions_near_frozen_threshold': sum(
            r['prediction_changed'] and min(abs(r['reference_probability_error']-THRESHOLD),
                                            abs(r['candidate_probability_error']-THRESHOLD)) <= 0.05
            for r in rows
        ),
    }


def self_test():
    assert canonical_surface('người dân') == canonical_surface('người_dân')
    assert canonical_surface('quận Bình-Tân') == canonical_surface('quận Bình - Tân')
    assert canonical_surface('của') != canonical_surface('cua')
    rows = [
        {'absolute_probability_difference': 0.12, 'prediction_changed': True,
         'reference_prediction': 0, 'candidate_prediction': 1,
         'reference_probability_error': 0.52, 'candidate_probability_error': 0.64,
         'chunk_count_changed': False},
        {'absolute_probability_difference': 0.02, 'prediction_changed': False,
         'reference_prediction': 1, 'candidate_prediction': 1,
         'reference_probability_error': 0.80, 'candidate_probability_error': 0.78,
         'chunk_count_changed': True},
    ]
    result = summarize(rows)
    assert result['n'] == 2 and result['prediction_flips'] == 1
    assert result['chunk_count_changes'] == 1
    assert result['max_absolute_probability_difference'] == 0.12
    assert result['changed_predictions_near_frozen_threshold'] == 1
    print('SELF-TEST PASS: surface validation, probability drift and flip aggregation')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    ap.add_argument('--candidates', nargs='+', choices=['pyvi', 'underthesea'], default=['pyvi'])
    ap.add_argument('--per-label', type=int, default=30)
    ap.add_argument('--scan-limit', type=int, default=20000)
    ap.add_argument('--validation', type=Path, default=VALIDATION)
    ap.add_argument('--report', type=Path, default=REPORT)
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if not (1 <= args.per_label <= 250 and args.scan_limit >= args.per_label * 2):
        ap.error('--per-label must be 1..250; scan-limit must be at least twice per-label')

    candidates = list(dict.fromkeys(args.candidates))
    fns = {name: segmenter(name) for name in candidates}  # fail before GPU load
    samples, scanned = read_pairs(args.validation, args.per_label, args.scan_limit)
    print(f'M1 SEGMENTATION PREDICTION DRIFT — validation only; {len(samples)} paired samples; scanned={scanned}', flush=True)
    print('Frozen M1 threshold:', THRESHOLD, '| Device:', args.device, '| Candidates:', candidates, flush=True)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from system.backend.app.services.model_registry import ModelRegistry
    registry = ModelRegistry(device=args.device)
    adapter = registry.spelling_adapter()

    report = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'validation_path': str(args.validation), 'scanned': scanned,
        'paired_sample_count': len(samples), 'per_label': args.per_label,
        'device': args.device, 'locked_threshold': THRESHOLD,
        'notes': [
            'Resource and compatibility experiment on a small validation sample; not an official accuracy benchmark.',
            'Longer input or different segmentation can change selected chunks as well as model probabilities.',
            'No training, threshold search, test access, modification to checkpoint or API integration.',
            'Natural and candidate texts must preserve all visible Unicode characters; spaces/underscores ignored in surface check.',
        ],
        'candidates': []
    }

    # Cache original-reference predictions across candidates. No mutation of input rows.
    print('\nFrozen segmented REFERENCE inference:', flush=True)
    reference = []
    for i, row in enumerate(samples, 1):
        result = adapter.predict(row['segmented'])
        reference.append(result)
        if i % 10 == 0 or i == len(samples):
            print(f'  reference {i}/{len(samples)}', flush=True)

    for name, fn in fns.items():
        print(f'\nCANDIDATE {name}:', flush=True)
        records = []
        refused = []
        elapsed = []
        for i, (row, ref) in enumerate(zip(samples, reference), 1):
            started = time.perf_counter()
            candidate_text = fn(row['natural'])
            elapsed.append(time.perf_counter() - started)
            if not isinstance(candidate_text, str):
                raise TypeError(f'{name}: expected segmenter to return str')
            if canonical_surface(candidate_text) != canonical_surface(row['natural']):
                refused.append({'variant_id': row['variant_id'], 'reason': 'candidate_changed_visible_characters'})
                continue
            pred = adapter.predict(candidate_text)
            ref_prob = float(ref['probability_error'])
            cand_prob = float(pred['probability_error'])
            records.append({
                'variant_id': row['variant_id'], 'label': row['label'],
                'reference_probability_error': round(ref_prob, 7),
                'candidate_probability_error': round(cand_prob, 7),
                'absolute_probability_difference': abs(ref_prob - cand_prob),
                'reference_prediction': int(ref['prediction']),
                'candidate_prediction': int(pred['prediction']),
                'prediction_changed': int(ref['prediction']) != int(pred['prediction']),
                'reference_chunks': int(ref['chunk_count']),
                'candidate_chunks': int(pred['chunk_count']),
                'chunk_count_changed': int(ref['chunk_count']) != int(pred['chunk_count']),
                'reference_excerpt': row['segmented'][:180],
                'candidate_excerpt': candidate_text[:180],
            })
            if i % 10 == 0 or i == len(samples):
                print(f'  {name} {i}/{len(samples)}', flush=True)
        overall = summarize(records)
        by_label = {str(label): summarize([r for r in records if r['label'] == label]) for label in (0, 1)}
        largest_drifts = sorted(records, key=lambda r: r['absolute_probability_difference'], reverse=True)[:10]
        flips = [r for r in records if r['prediction_changed']][:20]
        report['candidates'].append({
            'candidate': name, 'overall': overall, 'by_label': by_label,
            'refused_surface_change': len(refused), 'refused_examples': refused[:10],
            'mean_segmentation_seconds_per_document': round(statistics.mean(elapsed), 4) if elapsed else None,
            'largest_drift_examples': largest_drifts, 'prediction_flip_examples': flips,
        })
        print('  valid:', overall['n'], '/', len(samples), '| surface refusals:', len(refused), flush=True)
        print('  prediction flips:', overall.get('prediction_flips'), '| chunk changes:', overall.get('chunk_count_changes'), flush=True)
        print('  mean/max |delta p|:', overall.get('mean_absolute_probability_difference'), '/', overall.get('max_absolute_probability_difference'), flush=True)
        print('  by-label:', {k: {'n': v['n'], 'flips': v.get('prediction_flips')} for k, v in by_label.items()}, flush=True)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nREPORT:', args.report)
    print('Not auto-integrated; inspect flips and confidence drift before enabling raw-text M1 in API.')


if __name__ == '__main__':
    main()
