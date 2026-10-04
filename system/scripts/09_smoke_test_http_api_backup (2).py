"""Real HTTP smoke test. Start the local API server in a separate terminal first."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]


def get_json(url):
    with urlopen(url, timeout=15) as response:
        return json.load(response)


def post_json(url, payload):
    data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
    with urlopen(Request(url, data=data, headers={'Content-Type': 'application/json'}, method='POST'), timeout=120) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    args = parser.parse_args()
    url = args.url.rstrip('/')
    if url not in ('http://127.0.0.1:8000', 'http://localhost:8000'):
        parser.error('This V1 smoke script only tests local port 8000')
    article_path = ROOT / 'data' / 'processed' / 'vn_newsspell_v1' / 'validation.jsonl'
    sample_path = ROOT / 'module2' / 'inference' / 'sample_input.json'
    with article_path.open(encoding='utf-8') as file:
        validation_row = json.loads(next(line for line in file if line.strip()))
    with sample_path.open(encoding='utf-8') as file:
        sample = json.load(file)
    payload = {'title': sample['title'],
               'description': sample.get('lead', sample.get('description', '')),
               'article_text': sample['content'],
               'article_segmented': validation_row['article_segmented']}
    health = get_json(url + '/api/v1/health')
    print('HEALTH:', health)
    if health.get('models_ready') is not True:
        raise RuntimeError('Models are not ready')
    results = post_json(url + '/api/v1/analyze', payload)
    for key, threshold in [('module1_spelling', .57), ('module2_clickbait', .61), ('module3_consistency', .45)]:
        entry = results[key]
        if entry['status'] != 'completed' or abs(entry['result']['threshold'] - threshold) > 1e-8:
            raise RuntimeError(f'HTTP smoke failure: {key}: {entry}')
        print(f'{key}: {entry["result"]["label"]} threshold={threshold}')
    print('HTTP SMOKE PASS: all three endpoints completed through the real API')
    print('NOTE: API routing check only, not a numerical regression or accuracy benchmark.')


if __name__ == '__main__':
    main()
