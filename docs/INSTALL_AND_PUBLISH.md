# Installation and publishing guide

## 1. Local installation from VSIX

### VS Code

1. Open the Extensions view.
2. Open the Extensions menu (`...`).
3. Choose **Install from VSIX...**.
4. Select the packaged `.vsix` file.
5. Open a `.onnx` file.

### Cursor

1. Open the Extensions view.
2. Choose **Install from VSIX...** or drag the `.vsix` file into the pane.
3. Open a `.onnx` file.

## 2. Rebuilding the VSIX offline

This project includes a packaging script that does not rely on `vsce`.

From the project root:

```bash
python3 ./scripts/package_vsix.py
```

By default it writes the built package into `dist/`.

### Dry run

```bash
python3 ./scripts/package_vsix.py --dry-run
```

That validates the manifest inputs and prints the planned output path without writing a VSIX.

## 3. Rebuilding the source archive

If you want to hand a clean source bundle to someone else or attach one to a release:

```bash
python3 ./scripts/package_source_zip.py
```

## 4. Fields to customize before publishing

At minimum, update `package.json`:

- `publisher`
- `name` if you want a different extension identifier
- `displayName` if you want a different marketplace label

You may also want to add or update:

- `repository`
- `homepage`
- `bugs`
- `keywords`
- `version`
- `description`

## 5. Publishing to VS Code Marketplace

The standard path is to publish through the official VS Code extension tooling after creating your publisher identity and access token.

Typical flow:

1. Update `package.json`
2. Build or repackage the extension
3. Authenticate your publisher tooling
4. Publish with the official CLI

If you already use `vsce`, that remains the standard route for Marketplace publication.

## 6. Publishing to Open VSX

Open VSX is useful for Cursor-adjacent and non-Microsoft extension ecosystems.

Typical flow:

1. Create your namespace / publisher on Open VSX
2. Generate an access token
3. Publish with the Open VSX CLI

## 7. Recommended release workflow

### Fastest path

- Edit `package.json`
- Run `python3 ./scripts/package_vsix.py`
- Install locally and smoke-test
- Publish using your preferred official CLI

### Safer path

- Bump `version`
- Rebuild the VSIX
- Install locally in a clean profile or fresh machine
- Open at least one small ONNX model and one larger real model
- Verify:
  - Overview renders
  - detected export/file timestamps appear when present
  - Graph opens in a vertical compact layout
  - embedded weight rows appear inside operators
  - merged Constant rows appear inside shape / helper operators instead of cluttering the canvas
  - simple activations and shape ops render in their compact card sizes
  - selection highlighting keeps non-selected nodes readable and adds a strong glow to the active node
  - operator insight cards render in the detail pane, including activation plots when applicable
  - edge labels look readable on your real models
  - graph search can find weight / bias names and lands on the owning operator
  - Ctrl+wheel zoom works
  - background drag-to-pan works
  - explicit **Reveal in graph** centering works
  - Metadata pretty JSON and raw value both appear
  - Expand all / Collapse all works for JSON metadata
  - I/O & Weights tab renders
- Publish

## 8. What is already publish-ready

This project already includes:

- extension manifest (`package.json`)
- extension icon
- readme
- changelog
- license
- third-party notices
- offline VSIX packager
- offline source archive packager

So the only missing marketplace-specific pieces are generally your publisher identity and any branding / repository fields you want to set.

## 9. If you prefer the standard CLIs

You can also use the standard CLIs instead of the included Python packagers.

Typical tools:

- `@vscode/vsce` for VS Code Marketplace
- `ovsx` for Open VSX

The included Python scripts are there mainly so this project can be repackaged even in environments where installing the JavaScript packaging CLIs is inconvenient.
