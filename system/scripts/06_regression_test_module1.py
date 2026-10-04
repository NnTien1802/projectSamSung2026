"""Module 1 fixed-input numerical regression (no training, no threshold search).

Run from the project root:
  python system/scripts/06_regression_test_module1.py --device cuda --build-reference
  python system/scripts/06_regression_test_module1.py --device cuda

The first command freezes validation samples + ORIGINAL evaluation-script predictions.
Later runs refuse to rewrite the reference and compare the adapter against it.
Original checkpoint, tokenizer, locked pipeline and source evaluation scripts are read-only.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import heapq
import importlib.util
import json
import sys
from collections import Counter
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "system" / "reports" / "module1_regression_v1"
REFERENCE_PATH = REPORT_DIR / "reference_v1.json"
RESULT_PATH = REPORT_DIR / "regression_report.json"
DETAIL_PATH = REPORT_DIR / "regression_details.csv"
VALIDATION_PATH = ROOT / "data" / "processed" / "vn_newsspell_v1" / "validation.jsonl"
SOURCE_PATH = ROOT / "scripts" / "16_evaluate_phobert_test.py"
CHECKPOINT_PATH = ROOT / "models" / "module1_phobert_mil_full" / "best_model.pt"
THRESHOLD_PATH = ROOT / "results" / "module1_phobert_mil_full" / "threshold_tuning" / "best_threshold.json"
REFERENCE_VERSION = 1
THRESHOLD = 0.57
N_PER_GROUP = 6  # two classes × two lengths × six = 24 samples
LENGTH_SPLIT = 1000  # used only to diversify test cases; not a model input cutoff


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sample_key(row: dict) -> str:
    """Unique stable fingerprint; results do not retain article body."""
    payload = json.dumps(
        [row.get("source_id", ""), row.get("variant_id", ""), row["label"],
         row["article_segmented"]], ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def pick_samples(validation_path: Path, per_group: int = N_PER_GROUP) -> list[dict]:
    """Pick lowest stable hashes per class/length, regardless of JSONL line order."""
    heaps: dict[tuple[int, str], list] = {
        (label, kind): [] for label in (0, 1) for kind in ("short", "long")
    }
    with validation_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            label = int(row["label"])
            if label not in (0, 1):
                raise ValueError(f"Invalid label on line {line_number}: {label}")
            article = row.get("article_segmented")
            if not isinstance(article, str) or not article.strip():
                continue
            row["label"] = label
            key = sample_key(row)
            group = (label, "long" if len(article) >= LENGTH_SPLIT else "short")
            rank = int(key, 16)
            record = {"id": key, "label": label, "article_segmented": article,
                      "text_chars": len(article), "length_group": group[1]}
            # heap stores negative rank, so root is the largest selected hash
            heap = heaps[group]
            item = (-rank, key, record)
            if len(heap) < per_group:
                heapq.heappush(heap, item)
            elif item[0] > heap[0][0]:
                heapq.heapreplace(heap, item)
    selected = []
    for group in sorted(heaps):
        matches = sorted(heaps[group], key=lambda entry: entry[1])
        if not matches:
            raise RuntimeError(f"No validation samples available for group {group}")
        if len(matches) < per_group:
            print(f"WARNING: group {group} has only {len(matches)} available examples")
        selected.extend(record for _, _, record in matches)
    ids = [record["id"] for record in selected]
    if len(set(ids)) != len(ids):
        raise RuntimeError("Selected duplicate samples; inspect validation dataset")
    return selected


def import_original():
    spec = importlib.util.spec_from_file_location("_m1_locked_reference_eval", SOURCE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {SOURCE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_checkpoint(torch):
    try:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    cfg = (checkpoint["model_name"], checkpoint["text_field"],
           int(checkpoint["max_length"]), int(checkpoint["stride"]),
           int(checkpoint["max_chunks_per_document"]))
    if cfg != ("vinai/phobert-base-v2", "article_segmented", 256, 64, 3):
        raise RuntimeError(f"Unexpected frozen checkpoint config: {cfg}")
    threshold = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))["selected_threshold"]
    if abs(float(threshold) - THRESHOLD) > 1e-10:
        raise RuntimeError(f"Unexpected frozen threshold: {threshold}")
    return checkpoint


def baseline_predict(samples: list[dict], device_name: str) -> list[dict]:
    """Uses the original evaluation model and collator, *not* SpellingAdapter."""
    import torch
    from transformers import AutoTokenizer

    original = import_original()
    checkpoint = load_checkpoint(torch)
    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT_PATH.parent / "tokenizer", use_fast=False)
    collator = original.DocumentCollator(
        tokenizer, max_length=checkpoint["max_length"], stride=checkpoint["stride"],
        max_chunks=checkpoint["max_chunks_per_document"]
    )
    model = original.PhoBERTMIL(checkpoint["model_name"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    del checkpoint
    model.to(device_name)
    model.eval()
    predictions = []
    try:
        with torch.inference_mode():
            for index, record in enumerate(samples, 1):
                row = {"text": record["article_segmented"], "label": record["label"],
                       "source_id": "regression", "variant_id": record["id"],
                       "severity": "unknown", "error_count": 0,
                       "error_types": [], "error_sources": []}
                batch = collator([row])
                amp = (torch.amp.autocast(device_type="cuda", dtype=torch.float16)
                       if device_name == "cuda" else nullcontext())
                with amp:
                    logit = model(
                        input_ids=batch["input_ids"].to(device_name),
                        attention_mask=batch["attention_mask"].to(device_name),
                        doc_indices=batch["doc_indices"].to(device_name),
                        num_documents=1,
                    )
                probability = float(torch.sigmoid(logit).float().cpu()[0].item())
                predictions.append({
                    "id": record["id"], "label": record["label"],
                    "length_group": record["length_group"], "text_chars": record["text_chars"],
                    "probability_error": probability,
                    "prediction": int(probability >= THRESHOLD),
                    "chunk_count": int(batch["input_ids"].shape[0]),
                })
                print(f"  reference {index:02d}/{len(samples)} | "
                      f"p={probability:.6f} chunks={predictions[-1]['chunk_count']}", flush=True)
    finally:
        del model, collator, tokenizer
        gc.collect()
        if device_name == "cuda":
            torch.cuda.empty_cache()
    return predictions


def compare(predictions: list[dict], baseline: list[dict], tolerance: float) -> tuple[dict, list[dict]]:
    reference_by_id = {entry["id"]: entry for entry in baseline}
    if len(reference_by_id) != len(baseline) or len(predictions) != len(baseline):
        raise RuntimeError("Mismatch or duplicate sample IDs")
    rows = []
    for entry in predictions:
        original = reference_by_id[entry["id"]]
        deviation = abs(entry["probability_error"] - original["probability_error"])
        same_prediction = entry["prediction"] == original["prediction"]
        same_chunks = entry["chunk_count"] == original["chunk_count"]
        rows.append({
            "id": entry["id"], "label": original["label"],
            "length_group": original["length_group"], "text_chars": original["text_chars"],
            "reference_probability": original["probability_error"],
            "adapter_probability": entry["probability_error"],
            "absolute_error": deviation, "reference_prediction": original["prediction"],
            "adapter_prediction": entry["prediction"], "reference_chunks": original["chunk_count"],
            "adapter_chunks": entry["chunk_count"],
            "pass": deviation <= tolerance and same_prediction and same_chunks,
        })
    summary = {
        "sample_count": len(rows), "probability_tolerance": tolerance,
        "mean_absolute_error": sum(row["absolute_error"] for row in rows) / len(rows),
        "max_absolute_error": max(row["absolute_error"] for row in rows),
        "prediction_mismatches": sum(row["reference_prediction"] != row["adapter_prediction"] for row in rows),
        "chunk_count_mismatches": sum(row["reference_chunks"] != row["adapter_chunks"] for row in rows),
        "failed_samples": sum(not row["pass"] for row in rows),
        "result": "PASS" if all(row["pass"] for row in rows) else "FAIL",
    }
    return summary, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--build-reference", action="store_true",
                        help="Create immutable local fixed-input baseline only if missing")
    parser.add_argument("--tolerance", type=float, default=0.005)
    args = parser.parse_args()
    if not 0 < args.tolerance <= 0.01:
        parser.error("--tolerance must be >0 and <=0.01")
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable; use --device cpu")
    for path in (VALIDATION_PATH, SOURCE_PATH, CHECKPOINT_PATH, THRESHOLD_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.build_reference and REFERENCE_PATH.exists():
        raise FileExistsError(f"Reference already frozen: {REFERENCE_PATH}. Do not overwrite it.")
    if not args.build_reference and not REFERENCE_PATH.exists():
        raise FileNotFoundError(f"First run with --build-reference: {REFERENCE_PATH}")
    print("MODULE 1 REGRESSION — no training, validation-only, single-model GPU loading")
    print(f"Device: {args.device}", flush=True)
    samples = pick_samples(VALIDATION_PATH)
    print(f"Fixed samples: {len(samples)}; labels: {dict(Counter(r['label'] for r in samples))}")
    hashes = {"evaluation_source_sha256": sha256_file(SOURCE_PATH),
              "checkpoint_sha256": sha256_file(CHECKPOINT_PATH),
              "threshold_json_sha256": sha256_file(THRESHOLD_PATH),
              "validation_jsonl_sha256": sha256_file(VALIDATION_PATH)}
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    if args.build_reference:
        print("\nBuilding ORIGINAL evaluation-script reference (first run only)...", flush=True)
        original_predictions = baseline_predict(samples, args.device)
        frozen = {"version": REFERENCE_VERSION, "created_utc": datetime.now(timezone.utc).isoformat(),
                  "device": args.device, "threshold": THRESHOLD, "hashes": hashes,
                  "samples": original_predictions}
        # x = exclusive creation; will not overwrite existing reference
        with REFERENCE_PATH.open("x", encoding="utf-8") as out:
            json.dump(frozen, out, ensure_ascii=False, indent=2)
        print(f"REFERENCE CREATED: {REFERENCE_PATH}", flush=True)
    frozen = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    if frozen.get("version") != REFERENCE_VERSION or frozen.get("device") != args.device:
        raise RuntimeError("Reference version or device mismatch; do not mix CPU and CUDA references")
    if frozen.get("hashes") != hashes:
        raise RuntimeError("Dataset/evaluation/checkpoint/threshold SHA-256 differs from frozen reference")
    if frozen.get("threshold") != THRESHOLD:
        raise RuntimeError("Frozen reference threshold differs")
    expected_ids = [r["id"] for r in frozen["samples"]]
    if [r["id"] for r in samples] != expected_ids:
        raise RuntimeError("Fixed-sample IDs differ from reference (validation may have changed)")
    print("\nComparing NEW spelling adapter with frozen original reference...", flush=True)
    sys.path.insert(0, str(ROOT))
    from system.backend.app.adapters.spelling import SpellingAdapter
    adapter = SpellingAdapter(device=args.device)
    actual = []
    for index, record in enumerate(samples, 1):
        output = adapter.predict(record["article_segmented"])
        actual.append({"id": record["id"], "probability_error": output["probability_error"],
                       "prediction": output["prediction"], "chunk_count": output["chunk_count"]})
        print(f"  adapter   {index:02d}/{len(samples)} | p={output['probability_error']:.6f} "
              f"chunks={output['chunk_count']}", flush=True)
    summary, rows = compare(actual, frozen["samples"], args.tolerance)
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(), "device": args.device,
              "reference": str(REFERENCE_PATH.relative_to(ROOT)).replace('\\', '/'),
              "hashes": hashes, "threshold": THRESHOLD, **summary,
              "note": "Validation reference from original evaluation code, not the official held-out test. "
                      "Adapter reuses the same source architecture/collator; this is numeric routing parity, "
                      "not an independent implementation or accuracy benchmark."}
    with DETAIL_PATH.open("w", newline="", encoding="utf-8-sig") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    RESULT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nREGRESSION SUMMARY")
    for key in ("sample_count", "mean_absolute_error", "max_absolute_error",
                "prediction_mismatches", "chunk_count_mismatches", "failed_samples", "result"):
        print(f"{key}: {report[key]}")
    print(f"REPORT: {RESULT_PATH}")
    print(f"DETAIL: {DETAIL_PATH}")
    return 0 if summary["result"] == "PASS" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"REGRESSION ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(2)
