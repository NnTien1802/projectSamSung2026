"""Evaluation-only: restore original Vietnamese surface after Underthesea word segmentation.

Place in C:\\projectSamSung_v1\\system\\scripts\\ and run using project V1's .venv:
  .\\.venv\\Scripts\\python.exe system\\scripts\\15_evaluate_underthesea_surface_restored.py --self-test
  .\\.venv\\Scripts\\python.exe system\\scripts\\15_evaluate_underthesea_surface_restored.py
  .\\.venv\\Scripts\\python.exe system\\scripts\\15_evaluate_underthesea_surface_restored.py --with-inference --device cuda

Reads ONLY 60 paired validation examples; never writes/changes checkpoints, dataset,
threshold, locked files or the API. Writes a NEW report filename independent of
previous PyVi and Underthesea reports. Does not auto-enable preprocessing.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics
import sys
import time
import unicodedata as ud

ROOT = Path(__file__).resolve().parents[2]
VALIDATION = ROOT / 'data' / 'processed' / 'vn_newsspell_v1' / 'validation.jsonl'
REPORT = ROOT / 'system' / 'reports' / 'm1_preprocessing_audit_v1' / 'underthesea_surface_restored_v1.json'
THRESHOLD = 0.57
TONE_MARKS = frozenset(('\u0300', '\u0301', '\u0303', '\u0309', '\u0323'))


def _base_and_tones(ch: str) -> tuple[str, tuple[str, ...]]:
    decomposed = ud.normalize('NFD', ch)
    tones = tuple(x for x in decomposed if x in TONE_MARKS)
    base = ud.normalize('NFC', ''.join(x for x in decomposed if x not in TONE_MARKS))
    return base, tones


def _pieces(text: str):
    """Retain character offsets for tone-insensitive alignment, not for model text."""
    result = []
    for ch in ud.normalize('NFC', text):
        if ch.isspace() or ch == '_':
            continue
        base, tones = _base_and_tones(ch)
        if len(base) != 1:  # unsupported exotic Unicode expansion; refuse rather than guess
            raise ValueError('unsupported_unicode_expansion')
        result.append((base, tones))
    return result


def restore_original_surface(raw: str, segmented: str) -> tuple[str | None, str]:
    """Project Underthesea's underscore joins onto ORIGINAL source characters.

    A candidate is accepted only when (1) tone-insensitive character sequences
    match exactly, (2) tone MARKS match within every source whitespace-token,
    (3) every proposed join maps to a single existing ordinary source space.
    The raw document supplies every character; Underthesea supplies only joins.
    """
    raw_nfc = ud.normalize('NFC', raw)
    candidate_nfc = ud.normalize('NFC', segmented)
    if '_' in raw_nfc:
        return None, 'raw_contains_underscore'
    try:
        src = _pieces(raw_nfc)
        dst = _pieces(candidate_nfc)
    except ValueError:
        return None, 'unsupported_unicode_expansion'
    if ''.join(c for c, _ in src) != ''.join(c for c, _ in dst):
        return None, 'non_tone_character_changed'
    if not src:
        return None, 'empty_source'

    # Original whitespace-token spans (separately guard tone MARK identity).
    spans = []
    cursor = 0
    for match in re.finditer(r'\S+', raw_nfc):
        token = match.group()
        token_pieces = _pieces(token)
        end = cursor + len(token_pieces)
        spans.append((cursor, end, Counter(t for _, ts in token_pieces for t in ts)))
        cursor = end
    for start, end, src_tones in spans:
        dst_tones = Counter(t for _, ts in dst[start:end] for t in ts)
        if src_tones != dst_tones:
            return None, 'tone_marks_changed_or_moved_across_source_tokens'

    joins = set()
    cursor = 0
    for ch in candidate_nfc:
        if ch == '_':
            joins.add(cursor)
        elif not ch.isspace():
            cursor += 1  # one piece per NFC character, enforced above
    if not joins:
        # Pure no-join candidate is legal; return exact source.
        return raw_nfc, 'restored'

    # Project onto existing single-space positions, without normalizing source
    # punctuation or changing pre-existing spaces (unlike naive word_tokenize).
    gap_at = {}
    visible = 0
    i = 0
    while i < len(raw_nfc):
        ch = raw_nfc[i]
        if ch.isspace():
            j = i + 1
            while j < len(raw_nfc) and raw_nfc[j].isspace():
                j += 1
            if i > 0 and j < len(raw_nfc) and raw_nfc[i:j] == ' ':
                gap_at[visible] = i
            i = j
        else:
            visible += 1
            i += 1
    if not joins.issubset(gap_at):
        return None, 'candidate_join_not_on_single_source_space'
    indices = {gap_at[x] for x in joins}
    restored = ''.join('_' if i in indices else ch for i, ch in enumerate(raw_nfc))
    if restored.replace('_', ' ') != raw_nfc:
        return None, 'restoration_invariant_failed'
    return restored, 'restored'


def read_pairs(path: Path, per_label: int, scan_limit: int):
    results, labels, scanned = [], Counter(), 0
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            scanned += 1
            item = json.loads(line)
            try:
                label = int(item['label'])
            except (TypeError, ValueError, KeyError):
                continue
            natural, reference = item.get('article_natural'), item.get('article_segmented')
            if label in (0, 1) and labels[label] < per_label and isinstance(natural, str) and isinstance(reference, str) and natural.strip() and reference.strip():
                if re.sub(r'[\s_]+', '', ud.normalize('NFC', natural)) != re.sub(r'[\s_]+', '', ud.normalize('NFC', reference)):
                    raise RuntimeError('Invalid original validation pair: ' + str(item.get('variant_id')))
                results.append({'label': label, 'variant_id': str(item.get('variant_id', '')), 'natural': natural, 'reference': reference})
                labels[label] += 1
            if labels[0] >= per_label and labels[1] >= per_label:
                break
            if scanned >= scan_limit:
                break
    if labels[0] != per_label or labels[1] != per_label:
        raise RuntimeError('Insufficient validation pairs: ' + str(dict(labels)))
    return results, scanned


def join_positions(text: str) -> set[int]:
    positions = set()
    current = 0
    for ch in ud.normalize('NFC', text):
        if ch == '_':
            positions.add(current)
        elif not ch.isspace():
            current += 1
    return positions


def valid_gaps(text: str) -> set[int]:
    gaps, current, pending = set(), 0, False
    for ch in ud.normalize('NFC', text):
        if ch.isspace():
            pending = current > 0
        else:
            if pending:
                gaps.add(current)
                pending = False
            current += 1
    return gaps


def metrics(items):
    if not items:
        return {'n': 0}
    tp = sum(x['tp'] for x in items)
    fp = sum(x['fp'] for x in items)
    fn = sum(x['fn'] for x in items)
    precision = tp / (tp + fp) if tp + fp else (1 if fn == 0 else 0)
    recall = tp / (tp + fn) if tp + fn else 1
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {'n': len(items), 'precision': round(precision, 5), 'recall': round(recall, 5),
            'f1': round(f1, 5), 'exact_join_boundaries': sum(x['exact'] for x in items),
            'exact_segmented_text': sum(x['text_exact'] for x in items),
            'unexpected_joins': sum(x['unexpected'] for x in items)}


def drift_summary(items):
    if not items:
        return {'n': 0}
    differences = sorted(x['absolute_probability_difference'] for x in items)
    return {'n': len(items), 'prediction_flips': sum(x['prediction_changed'] for x in items),
            'mean_absolute_probability_difference': round(statistics.mean(differences), 6),
            'p95_absolute_probability_difference': round(differences[math.ceil(.95 * len(differences)) - 1], 6),
            'max_absolute_probability_difference': round(max(differences), 6),
            'chunk_count_changes': sum(x['chunk_count_changed'] for x in items)}


def self_test():
    examples = [
        ('Bộ trưởng ủy ban công bố.', 'Bộ_trưởng uỷ_ban công_bố.', 'Bộ_trưởng ủy_ban công_bố.'),
        ('Họ vẫn khỏe mạnh hôm nay.', 'Họ vẫn khoẻ_mạnh hôm_nay.', 'Họ vẫn khỏe_mạnh hôm_nay.'),
        ('Ngày 19/7 , người dân vui.', 'Ngày 19 / 7 , người_dân vui .', 'Ngày 19/7 , người_dân vui.'),
        ('hội đồng quản trị', 'hội_đồng quản_trị', 'hội_đồng quản_trị'),
        ('trời đẹp', 'trời đẹp', 'trời đẹp'),
    ]
    for source, candidate, expected in examples:
        actual, status = restore_original_surface(source, candidate)
        assert (actual, status) == (expected, 'restored'), (source, actual, status)
        assert actual.replace('_', ' ') == source
    assert restore_original_surface('má ăn cơm', 'mà ăn cơm')[0] is None
    assert restore_original_surface('người dân', 'ngườì_dân')[0] is None
    assert restore_original_surface('người dân', 'người_dâng')[0] is None
    assert restore_original_surface('người dân', 'ngườ_i dân')[0] is None
    assert metrics([{'tp': 2, 'fp': 0, 'fn': 1, 'exact': False, 'text_exact': False, 'unexpected': 0}])['recall'] == round(2/3, 5)
    print('SELF-TEST PASS: tone placement projection, original punctuation, join safety and edit rejection')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--with-inference', action='store_true', help='Also evaluate frozen M1 predictions on restored texts')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--validation', type=Path, default=VALIDATION)
    parser.add_argument('--report', type=Path, default=REPORT)
    parser.add_argument('--per-label', type=int, default=30)
    parser.add_argument('--scan-limit', type=int, default=20000)
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if not 1 <= args.per_label <= 250 or args.scan_limit < args.per_label * 2:
        parser.error('Invalid --per-label/--scan-limit')

    from underthesea import word_tokenize
    pairs, scanned = read_pairs(args.validation, args.per_label, args.scan_limit)
    report = {'created_utc': datetime.now(timezone.utc).isoformat(), 'validation_path': str(args.validation),
              'sample_count': len(pairs), 'scanned': scanned, 'candidate': 'underthesea_original_surface_restored',
              'threshold_unchanged': THRESHOLD,
              'limitations': ['60 selected validation rows, not official accuracy or generalization.',
                              'Tone-insensitive alignment is only accepted if tone-mark identities match per original whitespace token.',
                              'Underlying segmenter remains different from the source segmenter.',
                              'This script does not change API, model, checkpoint, thresholds or validation rows.']}
    accepted, rejected, stats = [], [], Counter()
    times = []
    print(f'UNDERTHESEA RESTORATION — validation {len(pairs)} examples, inference={args.with_inference}', flush=True)
    for i, row in enumerate(pairs, 1):
        start = time.perf_counter()
        candidate = word_tokenize(row['natural'], format='text')
        restored, reason = restore_original_surface(row['natural'], candidate)
        times.append(time.perf_counter() - start)
        stats[reason] += 1
        if restored is None:
            rejected.append({'variant_id': row['variant_id'], 'label': row['label'], 'reason': reason,
                             'source_excerpt': row['natural'][:160], 'candidate_excerpt': candidate[:160]})
        else:
            original = join_positions(row['reference']) & valid_gaps(row['natural'])
            proposed = join_positions(restored) & valid_gaps(row['natural'])
            unexpected = join_positions(restored) - valid_gaps(row['natural'])
            current = {'variant_id': row['variant_id'], 'label': row['label'],
                       'reference': row['reference'], 'restored': restored, 'natural': row['natural'],
                       'tp': len(original & proposed), 'fp': len(proposed - original),
                       'fn': len(original - proposed), 'unexpected': len(unexpected),
                       'exact': original == proposed and not unexpected,
                       'text_exact': re.sub(r'\s+', ' ', ud.normalize('NFC', restored)).strip()
                                     == re.sub(r'\s+', ' ', ud.normalize('NFC', row['reference'])).strip()}
            accepted.append(current)
        if i % 10 == 0 or i == len(pairs):
            print(f'  restored {i}/{len(pairs)}', flush=True)
    report['restoration'] = {'accepted': len(accepted), 'rejected': len(rejected), 'reasons': dict(stats),
                             'rejected_examples': rejected[:10],
                             'mean_segmentation_seconds_per_document': round(statistics.mean(times), 4),
                             'overall': metrics(accepted),
                             'by_label': {str(label): metrics([x for x in accepted if x['label'] == label]) for label in (0, 1)},
                             'mismatch_examples': [{'variant_id': x['variant_id'], 'reference_excerpt': x['reference'][:180],
                                                   'restored_excerpt': x['restored'][:180]} for x in accepted if not x['exact']][:10]}
    print('  surface restored:', len(accepted), '/', len(pairs), '| rejected:', len(rejected), flush=True)
    print('  rejection reasons:', {k: v for k, v in stats.items() if k != 'restored'}, flush=True)
    print('  join boundary P/R/F1:', {k: report['restoration']['overall'].get(k) for k in ('precision', 'recall', 'f1')}, flush=True)

    if args.with_inference and accepted:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from system.backend.app.services.model_registry import ModelRegistry
        adapter = ModelRegistry(device=args.device).spelling_adapter()
        records = []
        print('  Frozen M1 inference on original/restored pairs:', flush=True)
        for i, row in enumerate(accepted, 1):
            reference = adapter.predict(row['reference'])
            candidate = adapter.predict(row['restored'])
            a = float(reference['probability_error'])
            b = float(candidate['probability_error'])
            records.append({'variant_id': row['variant_id'], 'label': row['label'],
                            'reference_probability_error': round(a, 7),
                            'candidate_probability_error': round(b, 7),
                            'absolute_probability_difference': abs(a - b),
                            'reference_prediction': int(reference['prediction']),
                            'candidate_prediction': int(candidate['prediction']),
                            'prediction_changed': int(reference['prediction']) != int(candidate['prediction']),
                            'reference_chunks': int(reference['chunk_count']),
                            'candidate_chunks': int(candidate['chunk_count']),
                            'chunk_count_changed': int(reference['chunk_count']) != int(candidate['chunk_count'])})
            if i % 10 == 0 or i == len(accepted):
                print(f'    inference {i}/{len(accepted)}', flush=True)
        report['inference'] = {'device': args.device, 'overall': drift_summary(records),
                               'by_label': {str(l): drift_summary([x for x in records if x['label'] == l]) for l in (0, 1)},
                               'largest_drifts': sorted(records, key=lambda x: x['absolute_probability_difference'], reverse=True)[:10],
                               'prediction_flips': [x for x in records if x['prediction_changed']][:20]}
        print('  flips:', report['inference']['overall']['prediction_flips'],
              '| mean/max |delta p|:', report['inference']['overall']['mean_absolute_probability_difference'],
              '/', report['inference']['overall']['max_absolute_probability_difference'], flush=True)
    elif args.with_inference:
        report['inference'] = {'skipped': 'No valid restored samples; GPU not loaded'}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print('REPORT:', args.report, flush=True)
    print('No automatic integration, training, threshold tuning, or test data access.', flush=True)


if __name__ == '__main__':
    main()
