#!/usr/bin/env python3
"""Offline VSIX packager for the ONNX Model Inspector extension.

This script creates a valid VSIX archive without relying on `vsce`.
It is intentionally conservative and only packages the runtime files and
user-facing documentation that should ship with the extension.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIST_DIR = PROJECT_ROOT / "dist"

RUNTIME_FILES = [
    "package.json",
    "extension.js",
    "model-parser.js",
    "README.md",
    "CHANGELOG.md",
    "LICENSE.txt",
    "THIRD_PARTY_NOTICES.md",
    "scripts/inspect_pt.py",
    "scripts/inspect_safetensors.py",
    "scripts/inspect_torchscript.py",
    "scripts/detect_format.py",
]

RUNTIME_DIRS = [
    "docs",
    "media",
    "images",
]

CONTENT_TYPES = {
    ".json": "application/json",
    ".vsixmanifest": "text/xml",
    ".js": "application/javascript",
    ".mjs": "application/javascript",
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".png": "image/png",
    ".css": "text/css",
    ".html": "text/html",
    ".svg": "image/svg+xml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package the extension as a .vsix archive.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DIST_DIR,
        help="Directory where the VSIX file will be written. Defaults to ./dist",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the package inputs and print the planned VSIX path without writing the archive.",
    )
    return parser.parse_args()


def read_package_json() -> dict:
    package_path = PROJECT_ROOT / "package.json"
    with package_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def gather_files() -> list[Path]:
    files: list[Path] = []
    for relative in RUNTIME_FILES:
        path = PROJECT_ROOT / relative
        if path.exists():
            files.append(path)
    for relative_dir in RUNTIME_DIRS:
        base = PROJECT_ROOT / relative_dir
        if not base.exists():
            continue
        files.extend(
            sorted(
                path
                for path in base.rglob("*")
                if path.is_file() and "superpowers" not in path.relative_to(PROJECT_ROOT).parts
            )
        )
    return files


def validate(package: dict, files: Iterable[Path]) -> None:
    required = ["name", "displayName", "version", "publisher", "description", "engines"]
    for key in required:
        if key not in package:
            raise SystemExit(f"package.json is missing required field: {key}")
    if "vscode" not in package["engines"]:
        raise SystemExit("package.json.engines.vscode is required")
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in files if not path.exists()]
    if missing:
        raise SystemExit(f"Missing package files: {missing}")


def rel_extension_path(path: Path) -> str:
    relative = path.relative_to(PROJECT_ROOT).as_posix()
    return f"extension/{relative}"


def build_content_types(files: Iterable[Path]) -> str:
    suffixes = sorted({path.suffix.lower() for path in files if path.suffix} | {".vsixmanifest"})
    defaults = [
        f'<Default Extension="{escape(suffix)}" ContentType="{escape(CONTENT_TYPES.get(suffix, "application/octet-stream"))}"/>'
        for suffix in suffixes
    ]
    joined = "".join(defaults)
    return (
        "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        f"{joined}</Types>"
    )


def build_assets(package: dict) -> list[tuple[str, str]]:
    assets: list[tuple[str, str]] = [
        ("Microsoft.VisualStudio.Code.Manifest", "extension/package.json"),
        ("Microsoft.VisualStudio.Services.Content.Details", "extension/README.md"),
        ("Microsoft.VisualStudio.Services.Content.Changelog", "extension/CHANGELOG.md"),
        ("Microsoft.VisualStudio.Services.Content.License", "extension/LICENSE.txt"),
    ]
    icon_path = package.get("icon")
    if isinstance(icon_path, str) and icon_path:
        assets.append(("Microsoft.VisualStudio.Services.Icons.Default", f"extension/{icon_path}"))
    return assets


def build_manifest(package: dict) -> str:
    identity = {
        "Id": package["name"],
        "Version": package["version"],
        "Publisher": package["publisher"],
        "Language": "en-US",
    }
    properties = [
        ("Microsoft.VisualStudio.Code.Engine", package["engines"]["vscode"]),
        ("Microsoft.VisualStudio.Code.ExtensionDependencies", ",".join(package.get("extensionDependencies", []))),
        ("Microsoft.VisualStudio.Code.ExtensionPack", ",".join(package.get("extensionPack", []))),
        ("Microsoft.VisualStudio.Code.ExtensionKind", ",".join(package.get("extensionKind", []))),
        ("Microsoft.VisualStudio.Code.LocalizedLanguages", ""),
    ]
    if package.get("preview"):
        properties.append(("Microsoft.VisualStudio.Code.PreRelease", "true"))

    repository = package.get("repository")
    if isinstance(repository, dict) and repository.get("url"):
        properties.append(("Microsoft.VisualStudio.Services.Links.Source", repository["url"]))
        properties.append(("Microsoft.VisualStudio.Services.Links.GitHub", repository["url"]))
    if isinstance(package.get("homepage"), str):
        properties.append(("Microsoft.VisualStudio.Services.Links.Learn", package["homepage"]))
    bugs = package.get("bugs")
    if isinstance(bugs, dict) and bugs.get("url"):
        properties.append(("Microsoft.VisualStudio.Services.Links.Support", bugs["url"]))

    property_xml = "\n".join(
        f'    <Property Id="{escape(key)}" Value="{escape(value)}" />'
        for key, value in properties
    )
    asset_xml = "\n".join(
        f'    <Asset Type="{escape(asset_type)}" Path="{escape(path)}" Addressable="true" />'
        for asset_type, path in build_assets(package)
    )

    tags = ",".join(package.get("keywords", []))
    categories = ",".join(package.get("categories", []))
    icon_path = package.get("icon")
    icon_xml = f"\n  <Icon>extension/{escape(icon_path)}</Icon>" if icon_path else ""

    return f'''<?xml version="1.0" encoding="utf-8"?>
<PackageManifest Version="2.0.0" xmlns="http://schemas.microsoft.com/developer/vsx-schema/2011" xmlns:d="http://schemas.microsoft.com/developer/vsx-schema-design/2011">
  <Metadata>
    <Identity Language="{identity['Language']}" Id="{escape(identity['Id'])}" Version="{escape(identity['Version'])}" Publisher="{escape(identity['Publisher'])}" />
    <DisplayName>{escape(package['displayName'])}</DisplayName>
    <Description xml:space="preserve">{escape(package['description'])}</Description>
    <Tags>{escape(tags)}</Tags>
    <Categories>{escape(categories)}</Categories>
    <GalleryFlags>Public</GalleryFlags>
    <Properties>
{property_xml}
    </Properties>
    <License>extension/LICENSE.txt</License>{icon_xml}
  </Metadata>
  <Installation>
    <InstallationTarget Id="Microsoft.VisualStudio.Code" />
  </Installation>
  <Dependencies />
  <Assets>
{asset_xml}
  </Assets>
</PackageManifest>
'''


def build_vsix(package: dict, files: list[Path], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / f"{package['name']}-{package['version']}.vsix"
    content_types_xml = build_content_types(files)
    manifest_xml = build_manifest(package)

    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types_xml)
        archive.writestr("extension.vsixmanifest", manifest_xml)
        for path in files:
            archive.write(path, rel_extension_path(path))
    return archive_path


def main() -> None:
    args = parse_args()
    package = read_package_json()
    files = gather_files()
    validate(package, files)
    planned_path = args.output_dir / f"{package['name']}-{package['version']}.vsix"

    if args.dry_run:
        print(f"Dry run successful. Planned output: {planned_path}")
        return

    archive_path = build_vsix(package, files, args.output_dir)
    print(f"Created VSIX: {archive_path}")


if __name__ == "__main__":
    main()
