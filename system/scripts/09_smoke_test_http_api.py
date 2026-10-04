"""Real HTTP smoke test for refactored ProjectSamSung V1.

Start the local API server in a separate terminal first.

This script is intentionally independent from research datasets:
- M1 uses a fixed pre-segmented Vietnamese sample.
- M2/M3 use module2/inference/sample_input.json.
- It checks routing/status/locked thresholds only.
- It is NOT a numerical regression or accuracy benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]

# Fixed smoke input only. This is not a benchmark sample.
M1_SEGMENTED_SMOKE_TEXT = (
    "Người_dân đang quan_tâm đến tình_hình kinh_tế và các chính_sách mới . "
    "Bài_viết cung_cấp thông_tin tổng_quan về những thay_đổi trong thời_gian gần_đây ."
)


def get_json(url: str):
    with urlopen(url, timeout=15) as response:
        return json.load(response)


def post_json(url: str, payload: dict):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=120) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    url = args.url.rstrip("/")

    if url not in ("http://127.0.0.1:8000", "http://localhost:8000"):
        parser.error("This V1 smoke script only tests local port 8000")

    sample_path = ROOT / "module2" / "inference" / "sample_input.json"
    if not sample_path.exists():
        raise FileNotFoundError(f"Missing smoke sample: {sample_path}")

    with sample_path.open(encoding="utf-8") as file:
        sample = json.load(file)

    payload = {
        "title": sample["title"],
        "description": sample.get("lead", sample.get("description", "")),
        "article_text": sample["content"],
        "article_segmented": M1_SEGMENTED_SMOKE_TEXT,
    }

    health = get_json(url + "/api/v1/health")
    print("HEALTH:", health)

    if health.get("models_ready") is not True:
        raise RuntimeError("Models are not ready")

    results = post_json(url + "/api/v1/analyze", payload)

    expected = [
        ("module1_spelling", 0.57),
        ("module2_clickbait", 0.61),
        ("module3_consistency", 0.45),
    ]

    for key, threshold in expected:
        if key not in results:
            raise RuntimeError(
                f"HTTP smoke failure: missing result key: {key}; "
                f"available={list(results)}"
            )

        entry = results[key]

        if entry.get("status") != "completed":
            raise RuntimeError(
                f"HTTP smoke failure: {key} status={entry.get('status')}: {entry}"
            )

        result = entry.get("result") or {}

        actual_threshold = result.get("threshold")
        if actual_threshold is None:
            raise RuntimeError(
                f"HTTP smoke failure: {key} missing threshold: {entry}"
            )

        if abs(float(actual_threshold) - threshold) > 1e-8:
            raise RuntimeError(
                f"HTTP smoke failure: {key} threshold={actual_threshold}, "
                f"expected={threshold}"
            )

        print(
            f'{key}: label={result.get("label")} '
            f'probability={result.get("probability", result.get("score"))} '
            f'threshold={threshold}'
        )

    print("HTTP SMOKE PASS: all three modules completed through the real API")
    print("NOTE: routing/integration check only; not a numerical regression or accuracy benchmark.")


if __name__ == "__main__":
    main()
