#!/usr/bin/env python3
"""Model release inventory, file-hash lock, and verification.

This script NEVER modifies source model/checkpoint files, changes thresholds,
loads test datasets, or retrains. Hash-lock is not a functional regression test.

Run from project root:
  python system/scripts/01_model_release.py inventory
  python system/scripts/01_model_release.py init
  # Edit system/release/model_lock_config.json with exact discovered paths.
  python system/scripts/01_model_release.py freeze
  python system/scripts/01_model_release.py verify
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve()
DEFAULT_ROOT = HERE.parents[2]  # <project>/system/scripts/01_model_release.py

EXPECTED_MODULE2_PIPELINE_SHA256 = (
    "ac3137209f676c2e1e279a967ab642b20b05c8cc64a1c0edb741be052045ace9"
)
CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".bin"}
EXCLUDED_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", "data", "results",
    ".cache", "site-packages", "dist", "build"
}


def rel(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Artifact must be inside project root: {path}") from exc


def resolve_inside(root: Path, path_str: str) -> Path:
    path = (root / path_str).resolve()
    rel(root, path)
    return path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)


def json_dump(obj) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


def get_version(module_name: str) -> str | None:
    try:
        m = __import__(module_name)
        return str(getattr(m, "__version__", "unknown"))
    except ImportError:
        return None


def discover(root: Path):
    # Only inspect known model/script areas, never the entire datasets tree.
    dirs = [root / "models", root / "module2_headline_integrity", root / "scripts"]
    found = {"checkpoint_candidates": [], "possible_config_and_thresholds": [], "inference_and_regression_scripts": []}
    count = 0
    for base in dirs:
        if not base.exists():
            continue
        for current, dirs_mut, files in os.walk(base):
            dirs_mut[:] = [d for d in dirs_mut if d not in EXCLUDED_DIRS and not d.startswith("checkpoint-")]
            for filename in files:
                p = Path(current) / filename
                name = filename.lower()
                if p.suffix.lower() in CHECKPOINT_SUFFIXES:
                    found["checkpoint_candidates"].append(rel(root, p))
                elif p.suffix.lower() == ".json" and any(k in name for k in ["threshold", "config", "tokenizer", "special_tokens"]):
                    found["possible_config_and_thresholds"].append(rel(root, p))
                elif p.suffix.lower() == ".py" and any(k in name for k in ["inference", "pipeline", "regression", "evaluate", "lock"]):
                    found["inference_and_regression_scripts"].append(rel(root, p))
                count += 1
    for kind in found:
        found[kind] = sorted(set(found[kind]))
    return found


def template():
    # Fill null fields from inventory: never guess Module 2 checkpoint locations.
    return {
        "release_id": "news_quality_models_v1",
        "notes": "Three logical modules, with the existing frozen Module 2 unified pipeline retained and wrapped, not edited.",
        "models": {
            "spelling": {
                "model_family": "vinai/phobert-base-v2; MIL",
                "checkpoint": "models/module1_phobert_mil_full/best_model.pt",
                "threshold_file": "results/module1_phobert_mil_full/threshold_tuning/best_threshold.json",
                "inference_entrypoint": None,
                "tokenizer_snapshot_or_revision": None,
                "locked_parameters": {"max_length": 256, "stride": 64, "max_chunks": 3, "threshold": 0.57},
                "test_reference": {"macro_f1": 0.914730, "accuracy": 0.914824, "test_n": 28764},
                "additional_files": []
            },
            "clickbait": {
                "model_family": "mBERT title-only",
                "checkpoint": None,
                "threshold_file": None,
                "inference_entrypoint": "module2_headline_integrity/inference/headline_integrity_pipeline.py",
                "tokenizer_snapshot_or_revision": None,
                "locked_parameters": {"threshold": 0.61},
                "test_reference": {"macro_f1": 0.798830, "accuracy": 0.832031, "test_n": 512},
                "additional_files": []
            },
            "headline_consistency": {
                "model_family": "mBERT description + full article MAX-MIL",
                "checkpoint": None,
                "threshold_file": None,
                "inference_entrypoint": "module2_headline_integrity/inference/headline_integrity_pipeline.py",
                "tokenizer_snapshot_or_revision": None,
                "locked_parameters": {"threshold": None},
                "test_reference": {"macro_f1": 0.939591, "accuracy": 0.939591},
                "additional_files": []
            }
        },
        "shared_files": [],
        "expected_sha256": {
            "module2_headline_integrity/inference/headline_integrity_pipeline.py": EXPECTED_MODULE2_PIPELINE_SHA256
        }
    }


def gather(root: Path, cfg: dict):
    problems = []
    file_to_roles: dict[str, set[str]] = {}
    must_have = ("spelling", "clickbait", "headline_consistency")
    for key in must_have:
        model = cfg.get("models", {}).get(key)
        if not isinstance(model, dict):
            problems.append(f"Missing model config: {key}")
            continue
        if not model.get("checkpoint"):
            problems.append(f"{key}.checkpoint is empty. Select from inventory.")
        if not model.get("tokenizer_snapshot_or_revision"):
            problems.append(f"{key}.tokenizer_snapshot_or_revision is empty. Pin exact version/snapshot.")
        if key == "headline_consistency" and model.get("locked_parameters", {}).get("threshold") is None:
            problems.append("headline_consistency.locked_parameters.threshold is empty. Read frozen pipeline/config; never guess.")
        for field in ["checkpoint", "threshold_file", "inference_entrypoint"]:
            item = model.get(field)
            if item:
                file_to_roles.setdefault(item, set()).add(f"{key}.{field}")
        for item in model.get("additional_files", []):
            if not isinstance(item, str) or not item:
                problems.append(f"Invalid file in {key}.additional_files")
            else:
                file_to_roles.setdefault(item, set()).add(f"{key}.additional_files")
    for item in cfg.get("shared_files", []):
        file_to_roles.setdefault(item, set()).add("shared_files")

    records = []
    for item, roles in sorted(file_to_roles.items()):
        try:
            path = resolve_inside(root, item)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        if not path.exists():
            problems.append(f"File/dir not found: {item}")
            continue
        files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
        if not files:
            problems.append(f"No files under: {item}")
            continue
        for file in files:
            records.append((rel(root, file), sorted(roles), file))
    # Collapse files included by overlapping dirs/explicit paths.
    collapsed = {}
    for name, roles, path in records:
        if name not in collapsed:
            collapsed[name] = (set(), path)
        collapsed[name][0].update(roles)
    return [(name, sorted(roles), path) for name, (roles, path) in sorted(collapsed.items())], problems


def inventory(root: Path, out_path: Path):
    found = discover(root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(out_path, json_dump(found))
    print(f"INVENTORY written: {out_path}")
    for section, entries in found.items():
        print(f"\n{section} ({len(entries)})")
        for p in entries[:80]:
            print(f"  {p}")
        if len(entries) > 80:
            print(f"  ... {len(entries) - 80} more in inventory JSON")
    return 0


def freeze(root: Path, cfg_path: Path, manifest_path: Path, env_path: Path):
    if manifest_path.exists():
        print(f"STOP: Release manifest exists; never overwrite a frozen release: {manifest_path}")
        return 2
    if not cfg_path.exists():
        print(f"STOP: Config missing. Run init and fill exact paths: {cfg_path}")
        return 2
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    files, problems = gather(root, cfg)
    hash_expected = cfg.get("expected_sha256", {})
    for item, expected in hash_expected.items():
        p = resolve_inside(root, item)
        if not p.is_file():
            problems.append(f"Pinned file absent: {item}")
        elif digest(p).lower() != expected.lower():
            problems.append(f"PINNED HASH MISMATCH: {item} (existing module lock may have changed)")
    if problems:
        print("NOT LOCKED: fix all missing items before freezing:")
        for item in problems:
            print(f"  - {item}")
        return 2
    env = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": get_version("torch"),
        "transformers": get_version("transformers"),
        "tokenizers": get_version("tokenizers"),
        "numpy": get_version("numpy")
    }
    try:
        env["pip_freeze"] = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True, stderr=subprocess.DEVNULL
        ).splitlines()
    except (subprocess.CalledProcessError, FileNotFoundError):
        env["pip_freeze"] = None
    env_path.parent.mkdir(parents=True, exist_ok=True)
    # Store environment first; release manifest pins its content.
    atomic_write(env_path, json_dump(env))
    file_records = [
        {"path": path, "size": file.stat().st_size, "sha256": digest(file), "roles": roles}
        for path, roles, file in files
    ]
    manifest = {
        "release_id": cfg["release_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root_relative": True,
        "config": cfg,
        "files": file_records,
        "environment_file": rel(root, env_path),
        "environment_sha256": digest(env_path),
        "status": "HASH_LOCKED_ONLY_REGRESSION_REQUIRED",
        "important": "Hash verification does not establish end-to-end inference regression or real-world validity."
    }
    atomic_write(manifest_path, json_dump(manifest))
    print(f"HASH LOCK CREATED: {len(file_records)} files | {sum(r['size'] for r in file_records):,} bytes")
    print(f"MANIFEST: {manifest_path}")
    print("STATUS: HASH_LOCKED_ONLY_REGRESSION_REQUIRED")
    print("NEXT: run the existing Module 2 regression test and smoke-test all three wrapped endpoints.")
    return 0


def verify(root: Path, manifest_path: Path):
    if not manifest_path.exists():
        print(f"NOT LOCKED: manifest absent: {manifest_path}")
        return 2
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []
    for record in m["files"]:
        p = resolve_inside(root, record["path"])
        if not p.is_file():
            errors.append(f"MISSING: {record['path']}")
        elif p.stat().st_size != record["size"] or digest(p) != record["sha256"]:
            errors.append(f"MODIFIED: {record['path']}")
    env = resolve_inside(root, m["environment_file"])
    if not env.is_file() or digest(env) != m["environment_sha256"]:
        errors.append("MODIFIED/MISSING: saved environment record")
    for path_str, expected in m.get("config", {}).get("expected_sha256", {}).items():
        p = resolve_inside(root, path_str)
        if not p.is_file() or digest(p) != expected:
            errors.append(f"PINNED HASH MISMATCH: {path_str}")
    if errors:
        print("VERIFY FAIL:")
        for error in errors:
            print(" -", error)
        return 1
    print(f"VERIFY PASS: {len(m['files'])} files unchanged (SHA-256)")
    print("NOTE: Environment versions recorded, not compared to current; run regression separately.")
    return 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["inventory", "init", "freeze", "verify"])
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = p.parse_args()
    root = args.root.resolve()
    release = root / "system" / "release"
    cfg = release / "model_lock_config.json"
    manifest = release / "model_lock_manifest_v1.json"
    env = release / "environment_v1.json"
    if args.action == "inventory":
        return inventory(root, release / "model_inventory_v1.json")
    if args.action == "init":
        if cfg.exists():
            print(f"Config exists; not overwriting: {cfg}")
        else:
            atomic_write(cfg, json_dump(template()))
            print(f"Config template created: {cfg}")
        print("Fill missing checkpoint paths, tokenizer snapshot/revisions, and consistency threshold before freeze.")
        return 0
    if args.action == "freeze":
        return freeze(root, cfg, manifest, env)
    return verify(root, manifest)


if __name__ == "__main__":
    raise SystemExit(main())
