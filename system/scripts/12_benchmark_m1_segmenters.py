"""Compare candidate Vietnamese word segmenters against frozen M1 validation pairs.

Run from C:\\projectSamSung_v1, in the existing model environment:
  python system\\scripts\\12_benchmark_m1_segmenters.py --self-test
  python system\\scripts\\12_benchmark_m1_segmenters.py --candidates pyvi
  python system\\scripts\\12_benchmark_m1_segmenters.py --candidates pyvi underthesea

READ ONLY: validation data, segmenter inference and a new JSON report only.
No model load, no training, no test data, no frozen-file writes. The dataset
builders retain a pre-segmented source and derive natural text by removing '_';
this experiment does not assume the original upstream segmenter is known.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
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
REPORT = ROOT / 'system' / 'reports' / 'm1_preprocessing_audit_v1' / 'segmenter_benchmark_v1.json'


def nfc(text: str) -> str:
    return unicodedata.normalize('NFC', text)


def canonical_surface(text: str) -> str:
    """Ignore segmentation underscores and spacing, preserving all other chars."""
    return re.sub(r'[\s_]+', '', nfc(text))


def join_boundaries(text: str) -> set[int]:
    """Visible-character offsets of '_' joins (works despite punctuation spacing)."""
    current = 0
    positions: set[int] = set()
    for character in nfc(text):
        if character == '_':
            positions.add(current)
        elif not character.isspace():
            current += 1
    return positions


def natural_boundaries(text: str) -> set[int]:
    """Only existing inter-syllable gaps in the source are scorable joins."""
    source = nfc(text)
    positions = set()
    current = 0
    pending_space = False
    for character in source:
        if character.isspace():
            if current:
                pending_space = True
        else:
            if pending_space:
                positions.add(current)
                pending_space = False
            current += 1
    return positions


def counts(raw: str, reference: str, candidate: str) -> dict:
    """Compare joined *original-space* boundaries, not punctuation token offsets."""
    natural = canonical_surface(raw)
    ref_surface = canonical_surface(reference)
    cand_surface = canonical_surface(candidate)
    if natural != ref_surface:
        return {'status': 'invalid_reference_surface'}
    if natural != cand_surface:
        return {'status': 'candidate_surface_changed'}
    valid_spaces = natural_boundaries(raw)
    expected = join_boundaries(reference)
    predicted = join_boundaries(candidate)
    # Unscorable joins within originally unspaced punctuation are tracked,
    # not silently counted as correct.
    unexpected_ref = expected - valid_spaces
    unexpected_pred = predicted - valid_spaces
    expected &= valid_spaces
    predicted &= valid_spaces
    return {
        'status': 'aligned', 'tp': len(expected & predicted),
        'fp': len(predicted - expected), 'fn': len(expected - predicted),
        'reference_joins': len(expected), 'candidate_joins': len(predicted),
        'candidate_non_original_joins': len(unexpected_pred),
        'reference_non_original_joins': len(unexpected_ref),
        'all_join_boundaries_exact': expected == predicted and not unexpected_pred,
        'segmented_text_exact': re.sub(r'\s+', ' ', nfc(candidate)).strip()
            == re.sub(r'\s+', ' ', nfc(reference)).strip(),
    }


def prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {'precision': round(precision, 5), 'recall': round(recall, 5), 'f1': round(f1, 5)}


def read_validation(path: Path, per_label: int, scan_limit: int):
    samples = []
    labels = Counter()
    scanned = 0
    with path.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            scanned += 1
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            try:
                label = int(row.get('label', -1))
            except (TypeError, ValueError):
                continue
            raw = row.get('article_natural')
            ref = row.get('article_segmented')
            if label in (0, 1) and labels[label] < per_label and isinstance(raw, str) and raw.strip() and isinstance(ref, str) and ref.strip():
                samples.append({'label': label, 'natural': raw, 'reference': ref, 'variant_id': str(row.get('variant_id', ''))})
                labels[label] += 1
            if (labels[0] >= per_label and labels[1] >= per_label) or scanned >= scan_limit:
                break
    if not samples:
        raise RuntimeError('No paired validation article_natural/article_segmented texts found')
    return samples, scanned, dict(labels)


def get_segmenter(name: str):
    if name == 'pyvi':
        if importlib.util.find_spec('pyvi') is None:
            raise ImportError('pyvi not installed (python -m pip install pyvi)')
        from pyvi import ViTokenizer
        return ViTokenizer.tokenize
    if name == 'underthesea':
        if importlib.util.find_spec('underthesea') is None:
            raise ImportError('underthesea not installed (python -m pip install underthesea)')
        from underthesea import word_tokenize
        return lambda text: word_tokenize(text, format='text')
    raise ValueError(f'Unknown segmenter {name}')


def score_one(samples: list[dict], name: str, func):
    aggregate = Counter()
    by_label = {0: Counter(), 1: Counter()}
    elapsed = []
    excerpts = []
    errors = []
    for i, row in enumerate(samples, start=1):
        try:
            start = time.perf_counter()
            predicted = func(row['natural'])
            elapsed.append(time.perf_counter() - start)
            if not isinstance(predicted, str):
                raise TypeError(f'Expected str output; got {type(predicted).__name__}')
            stat = counts(row['natural'], row['reference'], predicted)
            aggregate[stat['status']] += 1
            by_label[row['label']][stat['status']] += 1
            if stat['status'] == 'aligned':
                for key in ('tp', 'fp', 'fn', 'reference_joins', 'candidate_joins', 'segmented_text_exact',
                            'all_join_boundaries_exact', 'candidate_non_original_joins'):
                    aggregate[key] += int(stat[key])
                    by_label[row['label']][key] += int(stat[key])
                if not stat['all_join_boundaries_exact'] and len(excerpts) < 5:
                    excerpts.append({'variant_id': row['variant_id'], 'label': row['label'],
                                     'reference_excerpt': row['reference'][:180],
                                     'candidate_excerpt': predicted[:180]})
            elif len(excerpts) < 5:
                excerpts.append({'variant_id': row['variant_id'], 'label': row['label'],
                                 'status': stat['status'], 'reference_excerpt': row['reference'][:180],
                                 'candidate_excerpt': predicted[:180]})
        except Exception as exc:
            aggregate['exception'] += 1
            by_label[row['label']]['exception'] += 1
            if len(errors) < 5:
                errors.append({'variant_id': row['variant_id'], 'error': str(exc)[:250]})
        if i % 10 == 0 or i == len(samples):
            print(f'  {name}: {i}/{len(samples)} samples', flush=True)

    def summarize(counter):
        tp, fp, fn = counter['tp'], counter['fp'], counter['fn']
        aligned = counter['aligned']
        return {
            'aligned_samples': aligned,
            'unalignable_candidate_surface': counter['candidate_surface_changed'],
            'invalid_reference_surface': counter['invalid_reference_surface'],
            'exceptions': counter['exception'],
            'original_space_join_prf': prf(tp, fp, fn),
            'reference_join_count': counter['reference_joins'],
            'predicted_join_count': counter['candidate_joins'],
            'exact_join_boundary_rate_aligned': round(counter['all_join_boundaries_exact'] / aligned, 4) if aligned else None,
            'exact_segmented_text_rate_aligned': round(counter['segmented_text_exact'] / aligned, 4) if aligned else None,
            'extra_joins_not_at_original_spaces': counter['candidate_non_original_joins'],
        }

    return {'candidate': name, 'overall': summarize(aggregate),
            'by_label': {str(label): summarize(c) for label, c in by_label.items()},
            'mean_seconds_per_document': round(statistics.mean(elapsed), 4) if elapsed else None,
            'mismatch_examples': excerpts, 'runtime_errors': errors}


def self_test():
    raw = 'người dân đi học'
    ref = 'người_dân đi_học'
    assert counts(raw, ref, ref)['all_join_boundaries_exact']
    assert counts(raw, ref, ref)['tp'] == 2
    wrong = counts(raw, ref, 'người dân đi_học')
    assert wrong['tp'] == 1 and wrong['fn'] == 1 and wrong['fp'] == 0
    assert counts(raw, ref, 'người_dân đi làm')['status'] == 'candidate_surface_changed'
    assert counts('đất nước hôm nay', 'đất_nước hôm_nay', 'đất_nước hôm_nay')['segmented_text_exact']
    assert counts('đất nước, hôm nay.', 'đất_nước, hôm_nay.', 'đất_nước , hôm_nay .')['all_join_boundaries_exact']
    assert counts(raw, ref, 'ngườidân đi_học')['status'] == 'aligned'  # altered whitespace loses an existing join
    print('SELF-TEST PASS: boundary scoring, mismatch and punctuation spacing')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidates', nargs='+', choices=['pyvi', 'underthesea'], default=['pyvi'])
    ap.add_argument('--per-label', type=int, default=30)
    ap.add_argument('--scan-limit', type=int, default=20000)
    ap.add_argument('--validation', type=Path, default=VALIDATION)
    ap.add_argument('--report', type=Path, default=REPORT)
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not (1 <= args.per_label <= 250 and args.scan_limit >= args.per_label * 2):
        ap.error('--per-label 1..250; --scan-limit must be at least twice per-label')
    samples, scanned, distribution = read_validation(args.validation, args.per_label, args.scan_limit)
    print('M1 SEGMENTER BENCHMARK — validation-only, read-only')
    print(f'Samples: {len(samples)} of {scanned} scanned, labels: {distribution}')
    output = {'created_utc': datetime.now(timezone.utc).isoformat(), 'validation_path': str(args.validation),
              'scanned': scanned, 'sample_count': len(samples), 'labels': distribution, 'candidate_results': [],
              'limits': [
                  'Exact join parity to an upstream-segmented dataset is not guaranteed by a high average boundary F1.',
                  'No test data, checkpoint, training, threshold search or numerical model inference is involved.',
                  'Only examples with identical Unicode surface after removing spaces and underscores are scored.',
                  'Sources may contain errors; both labels are displayed separately.',
              ]}
    for name in dict.fromkeys(args.candidates):
        print(f'\nCANDIDATE: {name}')
        try:
            fn = get_segmenter(name)
        except (ImportError, Exception) as exc:
            print('  UNAVAILABLE:', exc)
            output['candidate_results'].append({'candidate': name, 'status': 'UNAVAILABLE', 'reason': str(exc)})
            continue
        result = score_one(samples, name, fn)
        result['status'] = 'MEASURED'
        output['candidate_results'].append(result)
        o = result['overall']
        print('  aligned:', o['aligned_samples'], '/', len(samples))
        print('  join boundary precision/recall/F1:', o['original_space_join_prf'])
        print('  exact original-space boundary rate:', o['exact_join_boundary_rate_aligned'])
        print('  exact segmented text rate:', o['exact_segmented_text_rate_aligned'])
        print('  candidate surface mismatches:', o['unalignable_candidate_surface'])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\nREPORT:', args.report)
    print('The benchmark measures compatibility; no candidate is auto-selected or integrated.')
    return 0 if any(x.get('status') == 'MEASURED' for x in output['candidate_results']) else 2


if __name__ == '__main__':
    sys.exit(main())
