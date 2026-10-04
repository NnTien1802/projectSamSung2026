from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Iterable

# ============================================================
# ProjectSamSung V1 - deployment tree audit
#
# Purpose:
# - Print the current directory tree after refactoring.
# - Find deployment-critical runtime files.
# - Detect references to old Module 1 locations.
# - Detect absolute Windows paths / old project paths.
# - Inspect Docker files and possible .dockerignore exclusions.
#
# Pure Python stdlib. Safe/read-only.
# ============================================================

DEFAULT_MAX_DEPTH = 5
DEFAULT_MAX_CHILDREN = 80

# Directories that are usually not useful in a deployment tree report.
SKIP_DIR_NAMES = {
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ipynb_checkpoints",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "build",
}

# Large/research-only directories that can make tree output noisy.
SOFT_SKIP_DIR_NAMES = {
    "data",
}

# File patterns important for current deployment.
IMPORTANT_BASENAMES = {
    "Dockerfile",
    ".dockerignore",
    "requirements_docker.txt",
    "docker-compose.yml",
    "README_V1.md",
}

# Search strings that reveal stale references after refactor.
STALE_PATTERNS = [
    r"scripts[\\/]+16_evaluate_phobert_test\.py",
    r"C:[\\/]+1_projectSamSung",
    r"C:[\\/]+projectSamSung_v1",
    r"module1_phobert_mil_full",
]

TEXT_EXTENSIONS = {
    ".py", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".txt", ".md", ".env", ".dockerignore"
}


def human_size(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(n)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{n} B"


def rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def is_skipped_dir(path: Path, include_data: bool) -> bool:
    if path.name in SKIP_DIR_NAMES:
        return True
    if not include_data and path.name in SOFT_SKIP_DIR_NAMES:
        return True
    return False


def iter_tree(
    root: Path,
    max_depth: int,
    max_children: int,
    include_data: bool,
) -> None:
    print("=" * 100)
    print("A. CURRENT PROJECT TREE")
    print("=" * 100)
    print(f"PROJECT ROOT: {root}")
    print(
        f"TREE: depth <= {max_depth}, "
        f"<= {max_children} children/dir, "
        f"include_data={include_data}"
    )
    print()

    def walk(path: Path, depth: int) -> None:
        if depth > max_depth:
            return

        try:
            children = sorted(
                path.iterdir(),
                key=lambda p: (not p.is_dir(), p.name.lower())
            )
        except PermissionError:
            print("  " * depth + "[permission denied]")
            return

        visible = []
        for child in children:
            if child.is_dir() and is_skipped_dir(child, include_data):
                continue
            visible.append(child)

        truncated = len(visible) > max_children
        visible = visible[:max_children]

        for child in visible:
            prefix = "  " * depth
            if child.is_dir():
                print(f"{prefix}{child.name}/")
                walk(child, depth + 1)
            else:
                try:
                    size = human_size(child.stat().st_size)
                except OSError:
                    size = "?"
                print(f"{prefix}{child.name}  [{size}]")

        if truncated:
            print(
                "  " * depth
                + f"... ({len(children) - max_children} more children)"
            )

    walk(root, 0)
    print()


def find_named(root: Path, name: str, include_data: bool) -> list[Path]:
    found: list[Path] = []
    for base, dirs, files in os.walk(root):
        base_path = Path(base)

        dirs[:] = [
            d for d in dirs
            if not is_skipped_dir(base_path / d, include_data)
        ]

        if name in files:
            found.append(base_path / name)

        # Also support looking for directories by name.
        for d in dirs:
            if d == name:
                found.append(base_path / d)

    return found


def print_critical_files(root: Path, include_data: bool) -> None:
    print("=" * 100)
    print("B. DEPLOYMENT-CRITICAL FILE DISCOVERY")
    print("=" * 100)

    targets = [
        "Dockerfile",
        ".dockerignore",
        "requirements_docker.txt",
        "docker-compose.yml",
        "main.py",
        "model_registry.py",
        "spelling.py",
        "headline_integrity_pipeline.py",
        "best_model.pt",
        "final_model_lock.json",
    ]

    for target in targets:
        matches = find_named(root, target, include_data)
        print(f"\n[{target}] {len(matches)} match(es)")
        for p in matches[:30]:
            if p.is_file():
                try:
                    print(
                        f"  - {rel(p, root)} "
                        f"({human_size(p.stat().st_size)})"
                    )
                except OSError:
                    print(f"  - {rel(p, root)}")
            else:
                print(f"  - {rel(p, root)}/")

        if len(matches) > 30:
            print(f"  ... {len(matches) - 30} more")


def candidate_module1_dirs(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        n = child.name.lower()
        if "module1" in n or n in {"module_1", "m1"}:
            candidates.append(child)

    # Also check one level deeper, e.g. system/module1.
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if child.name in SKIP_DIR_NAMES:
            continue
        try:
            for grandchild in child.iterdir():
                if not grandchild.is_dir():
                    continue
                n = grandchild.name.lower()
                if "module1" in n or n in {"module_1", "m1"}:
                    candidates.append(grandchild)
        except PermissionError:
            pass

    # stable de-dup
    seen = set()
    out = []
    for p in candidates:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(p)
    return out


def print_module1_summary(root: Path) -> None:
    print()
    print("=" * 100)
    print("C. MODULE 1 REFACTOR SUMMARY")
    print("=" * 100)

    candidates = candidate_module1_dirs(root)

    if not candidates:
        print("No obvious Module 1 directory was detected by name.")
        print("This is not automatically an error; inspect section A/B.")
        return

    for module_dir in candidates:
        print(f"\nCandidate: {rel(module_dir, root)}/")
        files = []
        for p in module_dir.rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts:
                files.append(p)

        total_size = 0
        for p in files:
            try:
                total_size += p.stat().st_size
            except OSError:
                pass

        print(f"  Files: {len(files)}")
        print(f"  Total size: {human_size(total_size)}")

        py_files = [p for p in files if p.suffix.lower() == ".py"]
        model_files = [
            p for p in files
            if p.suffix.lower() in {".pt", ".bin", ".safetensors"}
        ]
        tokenizer_files = [
            p for p in files
            if p.name in {
                "vocab.txt", "bpe.codes", "tokenizer.json",
                "tokenizer_config.json", "special_tokens_map.json",
                "added_tokens.json"
            }
        ]

        print(f"  Python files: {len(py_files)}")
        print(f"  Model files: {len(model_files)}")
        print(f"  Tokenizer files: {len(tokenizer_files)}")

        for p in model_files[:10]:
            try:
                print(
                    f"    MODEL: {rel(p, root)} "
                    f"({human_size(p.stat().st_size)})"
                )
            except OSError:
                print(f"    MODEL: {rel(p, root)}")


def iter_text_files(root: Path, include_data: bool) -> Iterable[Path]:
    for base, dirs, files in os.walk(root):
        base_path = Path(base)
        dirs[:] = [
            d for d in dirs
            if not is_skipped_dir(base_path / d, include_data)
        ]

        for file_name in files:
            p = base_path / file_name
            suffix = p.suffix.lower()

            # Dockerfile has no suffix; .dockerignore has suffix-ish name.
            if (
                suffix in TEXT_EXTENSIONS
                or p.name in {"Dockerfile", ".dockerignore"}
            ):
                try:
                    if p.stat().st_size <= 5 * 1024 * 1024:
                        yield p
                except OSError:
                    continue


def scan_stale_references(root: Path, include_data: bool) -> None:
    print()
    print("=" * 100)
    print("D. STALE / ABSOLUTE PATH REFERENCES")
    print("=" * 100)

    compiled = [(pat, re.compile(pat, re.IGNORECASE)) for pat in STALE_PATTERNS]

    hits = []
    for path in iter_text_files(root, include_data):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        for line_no, line in enumerate(text.splitlines(), start=1):
            for raw, pattern in compiled:
                if pattern.search(line):
                    hits.append(
                        (path, line_no, raw, line.strip())
                    )

    if not hits:
        print("No stale-reference patterns found.")
        return

    for path, line_no, raw, line in hits[:200]:
        print(f"{rel(path, root)}:{line_no}")
        print(f"  PATTERN: {raw}")
        print(f"  LINE   : {line[:300]}")

    if len(hits) > 200:
        print(f"... {len(hits) - 200} more hit(s)")

    print(f"\nTotal stale/reference hits: {len(hits)}")


def scan_imports_for_module1(root: Path, include_data: bool) -> None:
    print()
    print("=" * 100)
    print("E. MODULE 1 IMPORT / FILE-PATH REFERENCES")
    print("=" * 100)

    patterns = [
        re.compile(r"module1", re.IGNORECASE),
        re.compile(r"SpellingAdapter"),
        re.compile(r"import_original"),
        re.compile(r"evaluate_phobert", re.IGNORECASE),
    ]

    hits = []
    for path in iter_text_files(root, include_data):
        if path.suffix.lower() != ".py":
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        for line_no, line in enumerate(text.splitlines(), start=1):
            if any(p.search(line) for p in patterns):
                hits.append((path, line_no, line.strip()))

    for path, line_no, line in hits[:250]:
        print(
            f"{rel(path, root)}:{line_no}: "
            f"{line[:320]}"
        )

    if len(hits) > 250:
        print(f"... {len(hits) - 250} more hit(s)")

    print(f"\nTotal Module 1 related source hits: {len(hits)}")


def inspect_dockerignore(root: Path) -> None:
    print()
    print("=" * 100)
    print("F. DOCKERIGNORE REVIEW")
    print("=" * 100)

    path = root / ".dockerignore"
    if not path.exists():
        print(".dockerignore NOT FOUND")
        return

    text = path.read_text(encoding="utf-8", errors="ignore")
    print(text)

    print("-" * 100)
    module1_lines = [
        line for line in text.splitlines()
        if "module1" in line.lower()
        or "scripts" in line.lower()
        or line.strip() == "models"
        or line.strip().startswith("models/")
    ]

    print("Lines that may affect Module 1 / runtime source:")
    if module1_lines:
        for line in module1_lines:
            print(f"  {line}")
    else:
        print("  (none detected)")


def inspect_backend_paths(root: Path) -> None:
    print()
    print("=" * 100)
    print("G. BACKEND RUNTIME FILES")
    print("=" * 100)

    likely = [
        root / "system" / "backend" / "app" / "api" / "main.py",
        root / "system" / "backend" / "app" / "services" / "model_registry.py",
        root / "system" / "backend" / "app" / "adapters" / "spelling.py",
        root / "system" / "backend" / "app" / "adapters" / "_source.py",
    ]

    for path in likely:
        print(
            f"{'[OK]' if path.exists() else '[MISSING]'} "
            f"{rel(path, root)}"
        )

        if path.exists() and path.suffix == ".py":
            print("  Relevant lines:")
            try:
                lines = path.read_text(
                    encoding="utf-8",
                    errors="ignore"
                ).splitlines()
                for i, line in enumerate(lines, start=1):
                    lowered = line.lower()
                    if (
                        "module1" in lowered
                        or "spelling" in lowered
                        or "path(" in lowered
                        or "import_original" in lowered
                        or "samsung_device" in lowered
                    ):
                        print(f"    {i:04d}: {line[:300]}")
            except OSError as exc:
                print(f"    read error: {exc}")


def summary(root: Path) -> None:
    print()
    print("=" * 100)
    print("H. NEXT-STEP CHECKLIST")
    print("=" * 100)

    checks = {
        "Dockerfile": (root / "Dockerfile").exists(),
        ".dockerignore": (root / ".dockerignore").exists(),
        "requirements_docker.txt": (root / "requirements_docker.txt").exists(),
        "FastAPI main.py": (
            root
            / "system"
            / "backend"
            / "app"
            / "api"
            / "main.py"
        ).exists(),
        "Model registry": (
            root
            / "system"
            / "backend"
            / "app"
            / "services"
            / "model_registry.py"
        ).exists(),
        "Spelling adapter": (
            root
            / "system"
            / "backend"
            / "app"
            / "adapters"
            / "spelling.py"
        ).exists(),
    }

    for name, ok in checks.items():
        print(f"{'[PASS]' if ok else '[FAIL]'} {name}")

    print()
    print("After running this script, send these sections back:")
    print("  A. CURRENT PROJECT TREE")
    print("  B. DEPLOYMENT-CRITICAL FILE DISCOVERY")
    print("  C. MODULE 1 REFACTOR SUMMARY")
    print("  D. STALE / ABSOLUTE PATH REFERENCES")
    print("  E. MODULE 1 IMPORT / FILE-PATH REFERENCES")
    print("  F. DOCKERIGNORE REVIEW")
    print("  G. BACKEND RUNTIME FILES")
    print("  H. NEXT-STEP CHECKLIST")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit current ProjectSamSung V1 tree before Docker deployment."
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Project root. Default: current directory.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=DEFAULT_MAX_DEPTH,
    )
    parser.add_argument(
        "--max-children",
        type=int,
        default=DEFAULT_MAX_CHILDREN,
    )
    parser.add_argument(
        "--include-data",
        action="store_true",
        help="Include data/ in tree/search. Usually unnecessary for deployment audit.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()

    if not root.exists() or not root.is_dir():
        print(f"ERROR: invalid project root: {root}", file=sys.stderr)
        return 2

    print("\nProjectSamSung V1 - Deployment Tree Audit")
    print(f"Python: {sys.version}")
    print()

    iter_tree(
        root,
        max_depth=args.max_depth,
        max_children=args.max_children,
        include_data=args.include_data,
    )
    print_critical_files(root, args.include_data)
    print_module1_summary(root)
    scan_stale_references(root, args.include_data)
    scan_imports_for_module1(root, args.include_data)
    inspect_dockerignore(root)
    inspect_backend_paths(root)
    summary(root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
