"""Real checkpoint smoke test. Only validation sample for M1; no retraining/test tuning."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from system.backend.app.services.model_registry import ModelRegistry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--m1-validation-record", type=Path,
                        default=ROOT / "data/processed/vn_newsspell_v1/validation.jsonl")
    parser.add_argument("--headline-input", type=Path,
                        default=ROOT / "module2/inference/sample_input.json")
    args = parser.parse_args()
    registry = ModelRegistry(device=args.device)
    # Read ONE validation record only, not the locked test set.
    with args.m1_validation_record.open(encoding="utf-8") as stream:
        m1_record = json.loads(next(line for line in stream if line.strip()))
    m1_text = m1_record.get("article_segmented", "")
    if not m1_text:
        raise RuntimeError("Validation record has no article_segmented; parity input unavailable")
    spelling = registry.spelling_adapter()
    m1 = spelling.predict(m1_text)
    print("M1", json.dumps(m1, ensure_ascii=False))
    inputs = json.loads(args.headline_input.read_text(encoding="utf-8"))
    title = inputs["title"]
    desc = inputs.get("lead", inputs.get("description", ""))
    content = inputs["content"]
    clickbait, consistency = registry.headline_adapters()
    m2 = clickbait.predict(title)
    m3 = consistency.predict(title, desc, content)
    # Same frozen pipeline, same arguments => direct adapter parity.
    orig_m2 = clickbait.pipeline.predict_clickbait(title=title.strip())
    orig_m3 = consistency.pipeline.predict_consistency(title=title.strip(), lead=str(desc).strip(), content=content.strip())
    assert m2 == orig_m2, "M2 adapter differs from original call"
    assert {k:v for k,v in m3.items() if k != "warnings"} == orig_m3, "M3 adapter differs from original call"
    print("M2", json.dumps(m2, ensure_ascii=False))
    print("M3", json.dumps(m3, ensure_ascii=False))
    print("ADAPTER SMOKE PASS: M1 inference + M2/M3 direct routing parity")
    print("NOTE: M1 needs fixed-input numeric regression; this script is not a full benchmark.")


if __name__ == "__main__":
    main()
