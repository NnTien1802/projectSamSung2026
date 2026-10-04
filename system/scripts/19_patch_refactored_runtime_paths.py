from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[2]
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
BACKUP = ROOT / "system" / "reports" / "deployment_patch_backups" / STAMP
BACKUP.mkdir(parents=True, exist_ok=True)

def backup(path: Path):
    if not path.exists():
        return
    dst = BACKUP / path.relative_to(ROOT)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, dst)

def replace_exact(path: Path, old: str, new: str, required: bool = True):
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        print(f"[SKIP] missing: {path.relative_to(ROOT)}")
        return False

    text = path.read_text(encoding="utf-8")
    if old not in text:
        if required:
            raise RuntimeError(
                f"Expected text not found in {path.relative_to(ROOT)}:\n{old}"
            )
        print(f"[SKIP] pattern absent: {path.relative_to(ROOT)}")
        return False

    backup(path)
    path.write_text(text.replace(old, new), encoding="utf-8")
    print(f"[PATCH] {path.relative_to(ROOT)}")
    return True

print("=" * 100)
print("ProjectSamSung V1 - patch refactored runtime paths")
print("=" * 100)
print("ROOT  :", ROOT)
print("BACKUP:", BACKUP)

# 1) Runtime backend paths
replace_exact(
    ROOT / "system/backend/app/adapters/spelling.py",
    'source = ROOT / "scripts" / "16_evaluate_phobert_test.py"',
    'source = ROOT / "module1" / "scripts" / "16_evaluate_phobert_test.py"',
)

replace_exact(
    ROOT / "system/backend/app/adapters/spelling.py",
    'model_dir = ROOT / "models" / "module1_phobert_mil_full"',
    'model_dir = ROOT / "module1" / "models" / "module1_phobert_mil_full"',
)

replace_exact(
    ROOT / "system/backend/app/adapters/spelling.py",
    'threshold_path = ROOT / "results" / "module1_phobert_mil_full" / "threshold_tuning" / "best_threshold.json"',
    'threshold_path = ROOT / "module1" / "results" / "module1_phobert_mil_full" / "threshold_tuning" / "best_threshold.json"',
)

replace_exact(
    ROOT / "system/backend/app/services/model_registry.py",
    'src = ROOT / "module2_headline_integrity" / "inference" / "headline_integrity_pipeline.py"',
    'src = ROOT / "module2" / "inference" / "headline_integrity_pipeline.py"',
)

# 2) Host-side smoke/profile helpers used after deployment.
for rel in [
    "system/scripts/05_smoke_test_adapters.py",
    "system/scripts/07_profile_gpu_memory.py",
    "system/scripts/09_smoke_test_http_api.py",
]:
    path = ROOT / rel
    if path.exists():
        text = path.read_text(encoding="utf-8")
        if "module2_headline_integrity" in text:
            backup(path)
            path.write_text(
                text.replace("module2_headline_integrity", "module2"),
                encoding="utf-8",
            )
            print(f"[PATCH] {rel}")

# 3) Refactored Module 2 diagnostic/regression imports.
# Do not touch headline_integrity_pipeline.py itself.
inf_dir = ROOT / "module2" / "inference"
for path in sorted(inf_dir.glob("*.py")):
    if path.name == "headline_integrity_pipeline.py":
        continue
    text = path.read_text(encoding="utf-8")
    old = "module2_headline_integrity.inference.headline_integrity_pipeline"
    new = "module2.inference.headline_integrity_pipeline"
    if old in text:
        backup(path)
        path.write_text(text.replace(old, new), encoding="utf-8")
        print(f"[PATCH] {path.relative_to(ROOT)}")

# 4) Deployment-oriented .dockerignore for new module1/module2 layout.
dockerignore_text = """# ------------------------------------------------------------------
# Local/dev environments
# ------------------------------------------------------------------
.venv
venv
__pycache__
**/__pycache__
*.py[cod]
.pytest_cache
.mypy_cache
.ipynb_checkpoints
.git
.gitignore

# ------------------------------------------------------------------
# Frontend is deployed separately
# ------------------------------------------------------------------
frontend/node_modules
frontend/dist

# ------------------------------------------------------------------
# Datasets are not required for inference
# ------------------------------------------------------------------
data
module1/data
module2/data

# ------------------------------------------------------------------
# Documentation / reports / local tools are not runtime dependencies
# ------------------------------------------------------------------
docs
tools
*.docx
*.pdf
system/reports

# ------------------------------------------------------------------
# Old branches, if any remain
# ------------------------------------------------------------------
module2_headline_integrity
module3_plagiarism_detection

# ------------------------------------------------------------------
# MODULE 1
# Keep:
#   module1/scripts/16_evaluate_phobert_test.py
#   module1/models/module1_phobert_mil_full/**
#   module1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json
# ------------------------------------------------------------------
module1/scripts/*
!module1/scripts/16_evaluate_phobert_test.py

module1/models/*
!module1/models/module1_phobert_mil_full
!module1/models/module1_phobert_mil_full/**

module1/results/*
!module1/results/module1_phobert_mil_full
module1/results/module1_phobert_mil_full/*
!module1/results/module1_phobert_mil_full/threshold_tuning
module1/results/module1_phobert_mil_full/threshold_tuning/*
!module1/results/module1_phobert_mil_full/threshold_tuning/best_threshold.json

# ------------------------------------------------------------------
# MODULE 2 / logical M2+M3
# Keep runtime inference code, three final checkpoints, and two locks.
# ------------------------------------------------------------------
module2/scripts/*

module2/models/*
!module2/models/clickbait
!module2/models/consistency

module2/models/clickbait/*
!module2/models/clickbait/mbert_title_only_v1
module2/models/clickbait/mbert_title_only_v1/*
!module2/models/clickbait/mbert_title_only_v1/best_model.pt

module2/models/consistency/*
!module2/models/consistency/baseline_title_description_mbert
!module2/models/consistency/full_article_max_mil_mbert

module2/models/consistency/baseline_title_description_mbert/*
!module2/models/consistency/baseline_title_description_mbert/best_model.pt

module2/models/consistency/full_article_max_mil_mbert/*
!module2/models/consistency/full_article_max_mil_mbert/best_model.pt

module2/results/*
!module2/results/clickbait
!module2/results/consistency

module2/results/clickbait/*
!module2/results/clickbait/final_title_only_v1
module2/results/clickbait/final_title_only_v1/*
!module2/results/clickbait/final_title_only_v1/final_model_lock.json

module2/results/consistency/*
!module2/results/consistency/final_hybrid_v1
module2/results/consistency/final_hybrid_v1/*
!module2/results/consistency/final_hybrid_v1/final_model_lock.json

# ------------------------------------------------------------------
# Misc
# ------------------------------------------------------------------
*.log
npm-debug.log*
"""

dockerignore = ROOT / ".dockerignore"
backup(dockerignore)
dockerignore.write_text(dockerignore_text, encoding="utf-8")
print("[PATCH] .dockerignore")

# 5) Validate required artifacts after patch
required = [
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

print("\nVALIDATION")
missing = []
for item in required:
    ok = item.exists()
    print(f"{'[OK]' if ok else '[MISSING]'} {item.relative_to(ROOT)}")
    if not ok:
        missing.append(item)

if missing:
    print("\nRESULT: BLOCKED")
    raise SystemExit(2)

print("\nRESULT: PATCH PASS")
print("No checkpoint, threshold, or headline_integrity_pipeline.py content was modified.")
print("Historical system/release manifests were intentionally NOT rewritten.")
