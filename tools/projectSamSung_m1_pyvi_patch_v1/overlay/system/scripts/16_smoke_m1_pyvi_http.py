"""Local HTTP smoke for explicit experimental PyVi mode; no accuracy claims.

Use after starting FastAPI with the V1 project .venv and one GPU worker.
Uses a short fixed Vietnamese example; does not read or log user articles.
"""
from __future__ import annotations

import json
from urllib.request import Request, urlopen

URL = "http://127.0.0.1:8000"


def post_json(route: str, body: dict) -> dict:
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    with urlopen(Request(URL + route, data=data,
                         headers={"Content-Type": "application/json"}, method="POST"),
                 timeout=180) as resp:
        return json.load(resp)


def main() -> None:
    raw = "người dân đi học"
    # This validates new routing with a real M1 checkpoint, not segmentation parity.
    result = post_json("/api/v1/spelling/raw", {"article_text": raw})
    assert result["status"] == "completed", result
    assert result["input_mode"] == "pyvi_experimental", result
    assert "EXPERIMENTAL" in result["warning"], result
    assert abs(result["result"]["threshold"] - .57) < 1e-8, result
    print("M1 RAW HTTP SMOKE PASS; mode=pyvi_experimental; threshold=0.57")
    print("Experimental preprocessing is NOT equivalent to frozen reference segmentation.")


if __name__ == "__main__":
    main()
