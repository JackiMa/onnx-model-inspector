# ONNX Model Inspector

> A self-contained ONNX viewer for VS Code and Cursor. Zero dependencies — install and go.

Open `.onnx` files in a custom read-only editor with four coordinated views:

| View | What it shows |
|------|---------------|
| **Overview** | Model identity, opsets, counts, file/export timestamps, quick metadata preview |
| **Graph** | Vertical Netron-inspired operator flow with embedded parameters, search, zoom, pan |
| **Metadata** | Model-level and graph-level `metadata_props` with pretty JSON + raw values |
| **I/O & Weights** | Inputs, outputs, value info, and initializer summaries |

No Python, no Netron desktop, no background service, no external runtime.

> **Cursor users:** This extension is available on [Open VSX](https://open-vsx.org/extension/gema/onnx-model-inspector). You can also install directly from a `.vsix` file.

## Why this extension exists

Cursor uses the Open VSX registry and does not always surface every VS Code Marketplace extension. This project bundles the ONNX decoding, graph layout, and metadata inspection logic directly inside the extension so you can install one `.vsix` and start using it in any compatible editor.

## What is included

- Custom read-only editor that automatically opens `*.onnx`
- Bundled ONNX protobuf decoding and summarization logic
- Vertical layered graph layout with routed edges, smaller simple-op cards, and a Netron-inspired operator-card style
- Embedded weight / initializer rows inside operators instead of a sparse cloud of standalone parameter nodes
- Graph search by node name, op type, tensor terminal, embedded parameter name, or folded Constant-helper provenance
- Stable selection behavior: graph clicks inspect only and preserve the current camera, while left-side result clicks and **Reveal in graph** explicitly recenter the viewport
- Background drag-to-pan, **Ctrl + wheel** zoom, zoom slider, reset, and fit controls
- Pretty JSON metadata tree with field-level syntax highlighting, expand-all / collapse-all controls, and raw-value comparison
- File and export metadata surfacing, including detected generated / exported timestamps when present in ONNX metadata
- Operator insight cards in the right-side detail pane, including activation function plots, clearer plain-language explanations, shape before/after summaries, and remembered collapsible sections
- Weight / initializer summaries including shape, element count, estimated byte size, and storage mode
- Offline packaging scripts for both `.vsix` and a source archive
- Publish notes for VS Code Marketplace and Open VSX

## Installation

### From Marketplace (VS Code)

1. Open the Extensions view and search for **ONNX Model Inspector**.
2. Click **Install**.

### From Open VSX (Cursor)

1. Open the Extensions view and search for **ONNX Model Inspector**.
2. Click **Install**.

### From VSIX (manual)

1. Download the `.vsix` from [GitHub Releases](https://github.com/JackiMa/onnx-model-inspector/releases).
2. In VS Code / Cursor: Extensions view → `...` → **Install from VSIX...** → select the file.
3. Open any `.onnx` file.

## Usage

### Open a model

- Double-click any `.onnx` file in the Explorer.
- Or run **ONNX Inspector: Open File** from the Command Palette.

### Work with the Graph tab

- The graph now reads **top to bottom**, closer to Netron’s default mental model.
- Weight and bias tensors that are initializer-backed show up as **embedded rows inside operator cards**.
- Search by node name, operator type, tensor terminal, or embedded parameter name such as `bias`, `weight`, or a full tensor name.
- Small helper `Constant` nodes such as `axes` inputs are automatically **folded into the consuming operator** when they are single-use and safe to inline.
- The right-hand detail pane explains common operators directly. For activations such as **ELU**, it can draw the operator's function curve so the transformation is easier to understand at a glance.
- Click a search result to smoothly reveal it from the current viewport.
- Click a node in the graph canvas to inspect it **without** forcing a viewport jump.
- Use **Ctrl + mouse wheel** to zoom.
- Drag the graph background to pan.
- Use the right-side zoom overlay for **+ / − / 100% / Fit** and the zoom slider.
- Use the **Reveal in graph** button in the detail pane when you explicitly want to re-center.
- Inspect direct **Upstream** and **Downstream** neighbors from the detail pane.
- Expand or collapse detail sections on the right; the inspector remembers your preferred section layout. Toolbar guidance for **Reveal in graph**, **Expand all**, and **Collapse all** now appears as hover tooltips instead of taking vertical space.

### Work with the Metadata tab

- Search metadata keys and values.
- If a value parses as JSON, the extension shows:
  1. an interactive pretty JSON tree
  2. the original raw stored string
- Use **Expand all** and **Collapse all** for large JSON payloads.
- Copy either the raw value or the pretty JSON rendering.

### Review export and file timestamps

The Overview tab now surfaces:

- detected ONNX export / generated timestamps from metadata fields when present
- file created time
- file modified time
- last reload time
- file size

This is useful when you are verifying which artifact actually made it into a deployment bundle.

### Refresh after regenerating a model

Use the **Reload** button in the editor header to re-read the ONNX file from disk.

## Commands

- `ONNX Inspector: Open File`

## How it works

At a high level:

1. The extension registers a **custom readonly editor** for `*.onnx`.
2. When a model is opened, the extension host reads the raw ONNX bytes and basic file stats.
3. A bundled parser decodes the ONNX protobuf payload locally.
4. The extension derives a serializable inspection model containing:
   - summary fields
   - opsets
   - metadata
   - detected temporal metadata
   - inputs / outputs / value info
   - initializers / weight summaries
   - graph nodes and edges
   - embedded operator parameters for display
   - vertical graph layout coordinates and routed edge points
5. The webview renders the inspection model with no external network calls.

More detail is available in [`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md).

## Project layout

- `extension.js` — extension entrypoint and custom editor provider
- `media/main.js` — webview UI logic
- `media/styles.css` — webview styling
- `media/lib/` — bundled ONNX parser modules and graph layout runtime
- `scripts/package_vsix.py` — offline VSIX packager
- `scripts/package_source_zip.py` — offline source archive packager
- `docs/` — installation, architecture, and publishing notes

## Packaging a new VSIX

```bash
python3 ./scripts/package_vsix.py
```

Output goes to `dist/` by default.

## Packaging a source archive

```bash
python3 ./scripts/package_source_zip.py
```

## Publishing

See [`docs/INSTALL_AND_PUBLISH.md`](docs/INSTALL_AND_PUBLISH.md) for VS Code Marketplace, Open VSX, and GitHub Release workflows.

## Limitations

- The editor is intentionally **read-only**.
- The graph renderer is strongly inspired by Netron, but it is still **not a full Netron clone**.
- External tensor files are surfaced as references in initializer metadata, but this extension does not open or merge external weight files.
- Weight contents are summarized rather than dumped in full.
- Temporal metadata detection works from stored metadata fields and file system stats; if the model simply does not store an export time, the inspector cannot infer the exact original generation moment.

## License

This project is released under the MIT License. See [`LICENSE.txt`](LICENSE.txt).

## Third-party notices

This extension includes ONNX-related parser modules and a compact graph layout runtime derived from Netron source files. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
