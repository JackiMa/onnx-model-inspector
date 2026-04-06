#!/usr/bin/env python3
"""Create a source zip of the project for redistribution or republishing."""

from __future__ import annotations

from pathlib import Path
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = PROJECT_ROOT / "dist"

EXCLUDE_PREFIXES = {
    ".git",
    "dist",
    "__pycache__",
}
EXCLUDE_SUFFIXES = {
    ".vsix",
    ".pyc",
}


def should_include(path: Path) -> bool:
    relative = path.relative_to(PROJECT_ROOT)
    if any(part in EXCLUDE_PREFIXES for part in relative.parts):
        return False
    if path.suffix in EXCLUDE_SUFFIXES:
        return False
    return path.is_file()


def main() -> None:
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    package_json = (PROJECT_ROOT / "package.json").read_text(encoding="utf-8")
    import json
    meta = json.loads(package_json)
    archive_path = DIST_DIR / f"{meta['name']}-{meta['version']}-source.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(PROJECT_ROOT.rglob("*")):
            if should_include(path):
                archive.write(path, path.relative_to(PROJECT_ROOT).as_posix())
    print(f"Created source archive: {archive_path}")


if __name__ == "__main__":
    main()
