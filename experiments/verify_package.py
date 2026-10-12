"""Verify immutable bundle files; generated plans/results are intentionally outside this list."""
import hashlib
import json
from pathlib import Path


def verify(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    for name, expected in manifest["sha256"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing/invalid package file: {name}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Package file changed: {name}; use a fresh extraction for this experiment")
    print(f"Package integrity OK: {len(manifest['sha256'])} files", flush=True)
    return manifest


if __name__ == "__main__":
    verify(Path(__file__).resolve().parents[1])
