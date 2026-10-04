"""ProjectSamSung V1: read-only, isolated CUDA/CPU profiling of frozen M1/M2/M3 adapters.

Run from project root:
  python system/scripts/07_profile_gpu_memory.py --device cuda
  python system/scripts/07_profile_gpu_memory.py --device cuda --mode combined

Default runs M1 and shared-headline in SEPARATE subprocesses, then combined. Each
stage writes its own JSON. No training, checkpoint writes, threshold changes or
changes to locked sources. On CUDA OOM the combined stage reports failure, not PASS.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / 'system' / 'reports' / 'gpu_profile_v1'


def read_inputs():
    validation = ROOT / 'data' / 'processed' / 'vn_newsspell_v1' / 'validation.jsonl'
    headline = ROOT / 'module2' / 'inference' / 'sample_input.json'
    with validation.open(encoding='utf-8') as f:
        first = next((json.loads(line) for line in f if line.strip()), None)
    if not first or not str(first.get('article_segmented', '')).strip():
        raise RuntimeError('First M1 validation row missing article_segmented')
    m1_short = str(first['article_segmented']).strip()
    # Real segmented validation data, not a raw-unsegmented synthetic example.
    # Repeat an existing sample to test a longer document without reading the
    # whole dataset; no new training/quality claims are made from this sample.
    m1_long = ' '.join([m1_short] * max(1, min(8, 5000 // max(1, len(m1_short)))))
    h = json.loads(headline.read_text(encoding='utf-8'))
    title = str(h['title']).strip()
    desc = str(h.get('lead', h.get('description', '')) or '').strip()
    article = str(h['content']).strip()
    if not title or not article:
        raise ValueError('Headline sample must have title and content')
    long_article = '\n'.join([article] * max(2, min(8, 5000 // max(1, len(article)))))
    return {'m1_short': m1_short, 'm1_long': m1_long,
            'title': title, 'description': desc,
            'article_short': article, 'article_long': long_article}


def ram_mb():
    try:
        import psutil
        return round(psutil.Process(os.getpid()).memory_info().rss / 1e6, 2)
    except ImportError:
        return None


def cuda_sample(torch, device):
    if device.type != 'cuda':
        return {'allocated_mb': None, 'reserved_mb': None,
                'peak_allocated_mb': None, 'free_vram_mb': None,
                'total_vram_mb': None}
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    return {
        'allocated_mb': round(torch.cuda.memory_allocated(device) / 1e6, 2),
        'reserved_mb': round(torch.cuda.memory_reserved(device) / 1e6, 2),
        'peak_allocated_mb': round(torch.cuda.max_memory_allocated(device) / 1e6, 2),
        'free_vram_mb': round(free / 1e6, 2),
        'total_vram_mb': round(total / 1e6, 2),
    }


def snapshot(torch, device):
    return {'gpu': cuda_sample(torch, device), 'process_ram_mb': ram_mb()}


def measure(torch, device, name, func, repeats):
    # Warm up once; only measured calls are included in timed stats.
    func()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = func()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        times.append((time.perf_counter() - start) * 1000)
    detail = {'name': name, 'repeat_count': repeats,
              'latency_mean_ms': round(statistics.mean(times), 2),
              'latency_min_ms': round(min(times), 2),
              'latency_max_ms': round(max(times), 2),
              'memory_after_calls': snapshot(torch, device)}
    if isinstance(result, dict):
        detail['prediction'] = result.get('prediction')
        if 'chunk_count' in result:
            detail['chunk_count'] = result['chunk_count']
        else:
            detail['chunk_count'] = result.get('branches', {}).get('max_mil', {}).get('chunk_count')
    print('  {}: mean {:.2f} ms; GPU peak {} MB; free GPU {} MB'.format(
        name, detail['latency_mean_ms'],
        detail['memory_after_calls']['gpu']['peak_allocated_mb'],
        detail['memory_after_calls']['gpu']['free_vram_mb']), flush=True)
    return detail


def run_stage(stage, device_choice, repeats, output):
    import torch
    ROOT_str = str(ROOT)
    if ROOT_str not in sys.path:
        sys.path.insert(0, ROOT_str)
    from system.backend.app.services.model_registry import ModelRegistry

    if device_choice == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but torch.cuda.is_available() is false')
    device = torch.device(device_choice)
    data = read_inputs()
    registry = ModelRegistry(device=device_choice)
    record = {'stage': stage, 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
              'device': device_choice, 'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
              'notes': ['torch peak tracks allocations through this process only; free VRAM also reflects other processes',
                        'warm-up excluded from timing; CUDA synchronized after calls',
                        'long inputs derived by repeating validation/demo text; resource testing only'],
              'initial': snapshot(torch, device), 'loads': [], 'calls': [], 'status': 'RUNNING'}
    output.parent.mkdir(parents=True, exist_ok=True)

    def load(name, factory):
        start = time.perf_counter()
        obj = factory()
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        stat = {'name': name, 'load_seconds': round(time.perf_counter()-start, 2),
                'memory_after_load': snapshot(torch, device)}
        record['loads'].append(stat)
        print('  loaded {} in {} s, allocated {} MB, free GPU {} MB'.format(
            name, stat['load_seconds'], stat['memory_after_load']['gpu']['allocated_mb'],
            stat['memory_after_load']['gpu']['free_vram_mb']), flush=True)
        return obj

    try:
        if stage in ('m1', 'combined'):
            m1 = load('M1_PhoBERT', registry.spelling_adapter)
            record['calls'].append(measure(torch, device, 'M1_short', lambda: m1.predict(data['m1_short']), repeats))
            record['calls'].append(measure(torch, device, 'M1_long', lambda: m1.predict(data['m1_long']), repeats))
        if stage in ('headline', 'combined'):
            clickbait, consistency = load('M2_M3_shared_mBERT_pipeline', registry.headline_adapters)
            record['calls'].append(measure(torch, device, 'M2_clickbait', lambda: clickbait.predict(data['title']), repeats))
            record['calls'].append(measure(torch, device, 'M3_short', lambda: consistency.predict(
                data['title'], data['description'], data['article_short']), repeats))
            record['calls'].append(measure(torch, device, 'M3_long', lambda: consistency.predict(
                data['title'], data['description'], data['article_long']), repeats))
        record['status'] = 'PASS'
    except (RuntimeError, MemoryError) as exc:
        # Even if torch OOM leaves limited CUDA memory available, write a JSON
        # report. Avoid calling CUDA memory APIs inside the exception handler.
        record['status'] = 'CUDA_OOM' if 'out of memory' in str(exc).lower() else 'FAIL'
        record['error'] = '{}: {}'.format(type(exc).__name__, str(exc))
        print('  STAGE {}: {}'.format(record['status'], record['error']), flush=True)
    except Exception as exc:
        record['status'] = 'FAIL'
        record['error'] = '{}: {}'.format(type(exc).__name__, str(exc))
        print(traceback.format_exc(), flush=True)
    finally:
        output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
        print('  stage report:', output, flush=True)
        # Process exits after the stage, releasing GPU memory to next stage.
        gc.collect()
    return record['status'] == 'PASS'


def main():
    parser = argparse.ArgumentParser(description='Measure isolated and combined GPU/RAM/latency for 3 frozen adapters')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--mode', choices=['all', 'm1', 'headline', 'combined'], default='all',
                        help='all runs 3 stages in separate child processes (default)')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--_stage', choices=['m1', 'headline', 'combined'], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.repeats < 1 or args.repeats > 20:
        parser.error('--repeats must be between 1 and 20')
    out = args.output_dir.resolve()
    if args._stage is not None:
        okay = run_stage(args._stage, args.device, args.repeats, out / (args._stage + '.json'))
        return 0 if okay else 1
    stages = ['m1', 'headline', 'combined'] if args.mode == 'all' else [args.mode]
    print('PROJECT ROOT:', ROOT, '\nDEVICE:', args.device, '\nSTAGES:', ', '.join(stages),
          '\nOUTPUT:', out, '\nNo training / no changes to locked files.', flush=True)
    out.mkdir(parents=True, exist_ok=True)
    summary = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'device': args.device, 'stages': {},
               'overall_status': 'PASS'}
    for stage in stages:
        print('\n==== STAGE {} (isolated subprocess) ===='.format(stage), flush=True)
        command = [sys.executable, str(Path(__file__).resolve()), '--device', args.device,
                   '--repeats', str(args.repeats), '--output-dir', str(out), '--_stage', stage]
        try:
            proc = subprocess.run(command, check=False)
            path = out / (stage + '.json')
            stage_report = json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
            status = stage_report.get('status', 'FAIL') if stage_report else 'FAIL'
            summary['stages'][stage] = {'status': status, 'exit_code': proc.returncode, 'report': str(path),
                                        'loads': stage_report.get('loads', []) if stage_report else [],
                                        'calls': stage_report.get('calls', []) if stage_report else []}
            if status != 'PASS' or proc.returncode != 0:
                summary['overall_status'] = 'FAIL'
        except Exception as exc:
            summary['overall_status'] = 'FAIL'
            summary['stages'][stage] = {'status': 'FAIL', 'error': str(exc)}
    summary_path = out / 'summary.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n==== GPU PROFILE SUMMARY ====')
    for key, value in summary['stages'].items():
        print('{}: {}'.format(key, value['status']))
    print('OVERALL:', summary['overall_status'])
    print('REPORT:', summary_path)
    if summary['overall_status'] != 'PASS':
        print('Do not deploy simultaneous resident models until failures are reviewed.')
    return 0 if summary['overall_status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
