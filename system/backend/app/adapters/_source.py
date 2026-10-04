"""Read-only imports from original research scripts; never mutate their code."""
from pathlib import Path
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[4]


def import_original(path: Path, module_name: str):
    if not path.is_file():
        raise FileNotFoundError(f"Original research source missing: {path}")
    cached = sys.modules.get(module_name)
    if cached is not None:
        if Path(cached.__file__).resolve() != path.resolve():
            raise RuntimeError(f"Module name collision: {module_name}")
        return cached
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import original script: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return mod
