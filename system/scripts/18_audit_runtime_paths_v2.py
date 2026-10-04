from __future__ import annotations

import os
import sys
from pathlib import Path

# Avoid Windows cp1252 crashes when printing Vietnamese.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]

FILES = [
    ROOT / "system/backend/app/adapters/spelling.py",
    ROOT / "system/backend/app/adapters/_source.py",
    ROOT / "system/backend/app/adapters/clickbait.py",
    ROOT / "system/backend/app/adapters/headline_consistency.py",
    ROOT / "system/backend/app/services/model_registry.py",
    ROOT / "system/backend/app/api/main.py",
    ROOT / "module1/scripts/16_evaluate_phobert_test.py",
    ROOT / "module2/inference/headline_integrity_pipeline.py",
    ROOT / "Dockerfile",
    ROOT / ".dockerignore",
    ROOT / "docker-compose.yml",
]

KEYS = (
    "module1",
    "module2",
    "module2_headline_integrity",
    "scripts",
    "models",
    "results",
    "best_model",
    "final_model_lock",
    "headline_integrity_pipeline",
    "16_evaluate_phobert_test",
    "PROJECT_ROOT",
    "MODULE_ROOT",
    "Path(",
    "parents[",
    "import_original",
    "SAMSUNG_DEVICE",
)

REQUIRED = [
    ROOT / "module1/scripts/16_evaluate_phobert_test.py",
    ROOT / "module1/models/module1_phobert_mil_full/best_model.pt",
    ROOT / "module1/models/module1_phobert_mil_full/tokenizer",
    ROOT / "module1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json",
    ROOT / "module2/inference/headline_integrity_pipeline.py",
    ROOT / "module2/models/clickbait/mbert_title_only_v1/best_model.pt",
    ROOT / "module2/models/consistency/baseline_title_description_mbert/best_model.pt",
    ROOT / "module2/models/consistency/full_article_max_mil_mbert/best_model.pt",
    ROOT / "module2/results/clickbait/final_title_only_v1/final_model_lock.json",
    ROOT / "module2/results/consistency/final_hybrid_v1/final_model_lock.json",
]

print("=" * 100)
print("PROJECTSAMSUNG V1 - RUNTIME PATH AUDIT V2")
print("=" * 100)
print("ROOT:", ROOT)

print("\nA. REQUIRED RUNTIME ARTIFACTS")
for x in REQUIRED:
    print(("[OK]      " if x.exists() else "[MISSING] ") + str(x.relative_to(ROOT)))

print("\nB. IMPORTANT SOURCE/PATH LINES")
for f in FILES:
    print("\n" + "-" * 100)
    print(f.relative_to(ROOT))
    print("-" * 100)
    if not f.exists():
        print("[MISSING FILE]")
        continue
    text = f.read_text(encoding="utf-8", errors="replace")
    for no, line in enumerate(text.splitlines(), 1):
        if any(k.lower() in line.lower() for k in KEYS):
            print(f"{no:04d}: {line}")

print("\nC. OLD PATH NAME SCAN")
old_terms = [
    "module2_headline_integrity",
    "scripts/16_evaluate_phobert_test.py",
    r"scripts\16_evaluate_phobert_test.py",
    "/models/module1_phobert_mil_full",
    r"\models\module1_phobert_mil_full",
]
hits = 0
for base, dirs, files in os.walk(ROOT):
    # Keep scan small and deployment-focused.
    dirs[:] = [d for d in dirs if d not in {
        ".git", ".venv", "venv", "node_modules", "__pycache__",
        "data", "docs", "tools", "reports",
    }]
    for name in files:
        f = Path(base) / name
        if f.suffix.lower() not in {".py", ".txt", ".md", ".json", ".yml", ".yaml"} and name not in {
            "Dockerfile", ".dockerignore"
        }:
            continue
        try:
            if f.stat().st_size > 2 * 1024 * 1024:
                continue
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for no, line in enumerate(text.splitlines(), 1):
            if any(term.lower() in line.lower() for term in old_terms):
                print(f"{f.relative_to(ROOT)}:{no}: {line[:350]}")
                hits += 1
print("OLD PATH HITS:", hits)

print("\nD. DOCKERIGNORE FULL CONTENT")
di = ROOT / ".dockerignore"
print(di.read_text(encoding="utf-8", errors="replace") if di.exists() else "[MISSING]")

print("\nE. RESULT")
missing = [x for x in REQUIRED if not x.exists()]
if missing:
    print("RESULT: BLOCKED - required runtime artifact(s) missing.")
else:
    print("RESULT: ARTIFACTS_PRESENT")
    print("Next: update backend path references and .dockerignore to the new module1/module2 layout, then rebuild Docker.")
