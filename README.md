# ONNX Model Inspector

> Inspect ONNX, PyTorch, and safetensors model files directly in VS Code and Cursor — no external services, no cloud uploads.

Open `.onnx`, `.pt`, `.pth`, and `.safetensors` files in a custom read-only editor with four coordinated views:

| View | ONNX | PT / TorchScript | safetensors |
|------|------|-----------------|-------------|
| **Overview** | Model identity, opsets, counts, export timestamps | Checkpoint root type, tensor counts, scalar metadata | Header metadata, tensor count, total parameter size |
| **Graph** | Vertical operator flow with embedded parameters | Module hierarchy tree (state dict) or operator DAG (TorchScript) | Tensor list |
| **Metadata** | `metadata_props` with pretty JSON + raw values | Checkpoint sections and scalar metadata | Header `__metadata__` dict |
| **I/O & Weights** | Inputs, outputs, initializer summaries | Tensor summaries and checkpoint sections | Tensor dtype, shape, and offset summaries |

For ONNX, there is no Python runtime dependency. For PT / PTH / safetensors inspection, the extension shells out to a bundled Python helper.

> **Cursor users:** This extension is available on [Open VSX](https://open-vsx.org/extension/gema/onnx-model-inspector). You can also install directly from a `.vsix` file.

## Why this extension exists

Cursor uses the Open VSX registry and does not always surface every VS Code Marketplace extension. This project bundles the ONNX decoding, graph layout, and metadata inspection logic directly inside the extension so you can install one `.vsix` and start using it in any compatible editor.

## What is included

- Custom read-only editor that automatically opens `*.onnx`, `*.pt`, `*.pth`, and `*.safetensors`
- Bundled ONNX protobuf decoding and summarization — no Python needed for ONNX
- Bundled Python helpers for PT, TorchScript, and safetensors inspection (requires Python; PyTorch only needed for PT files)
- Format auto-detection: classifies PT archives as plain-state-dict, TorchScript, lightning-checkpoint, exported-program, or full-model-pickle before loading anything
- TorchScript **offline** computation-graph extraction: parses the generated IR source with the standard `ast` module (no `torch` runtime required), renders a real operator DAG with layered layout
- Module hierarchy graph for plain state-dict checkpoints with layer type inference (Conv2d, Linear, MultiheadAttention, AttentionBlock, BatchNorm, LayerNorm, GroupNorm, DepthwiseConv2d...)
- safetensors header inspection with pure Python standard library (no third-party packages needed)
- Vertical layered graph layout with routed edges, smaller simple-op cards, and a Netron-inspired operator-card style
- Embedded weight / initializer rows inside operators instead of a sparse cloud of standalone parameter nodes
- Graph search by node name, op type, tensor terminal, embedded parameter name, or folded Constant-helper provenance
- Stable selection behavior: graph clicks inspect only and preserve the current camera; left-side result clicks and **Reveal in graph** explicitly recenter the viewport
- Background drag-to-pan, **Ctrl + wheel** zoom, zoom slider, reset, and fit controls
- Pretty JSON metadata tree with field-level syntax highlighting, expand-all / collapse-all controls, and raw-value comparison
- File and export metadata surfacing, including detected generated / exported timestamps when present in ONNX metadata
- Operator insight cards in the right-side detail pane, including activation function plots and plain-language explanations
- Offline packaging scripts for both `.vsix` and a source archive

## Safety

- PyTorch `.pt` files are opened with `weights_only=True` by default. Unsafe pickle deserialization (which can execute arbitrary code) is disabled unless you explicitly:
  1. Trust the workspace in VS Code
  2. Enable `onnxInspector.allowFullPickleLoad` in settings
  3. Confirm the one-time per-file warning dialog
- Python subprocesses run under a 120-second hard timeout and a 16 MB stdout cap
- Files larger than 500 MB skip the webview-payload read; the Python inspector reads directly from disk

## Installation

### From Open VSX (Cursor / VS Code)

1. Open the Extensions view and search for **ONNX Model Inspector**.
2. Click **Install**.

### From VSIX (manual)

1. Download the `.vsix` from [GitHub Releases](https://github.com/JackiMa/onnx-model-inspector/releases).
2. In VS Code / Cursor: Extensions view → `...` → **Install from VSIX...** → select the file.
3. Open any `.onnx`, `.pt`, `.pth`, or `.safetensors` file.

## Requirements

| Format | Python | PyTorch |
|--------|--------|---------|
| ONNX | Not required | Not required |
| safetensors | Required | Not required |
| PT / TorchScript (graph only) | Required | Not required |
| PT (weights / metadata) | Required | Required |

Point the extension at a specific interpreter with `onnxInspector.pythonPath`. If empty, it falls back to `ONNX_INSPECTOR_PYTHON`, `CONDA_PYTHON_EXE`, then `python3` / `py -3`.

## Usage

### Open a model

- Double-click any `.onnx`, `.pt`, `.pth`, or `.safetensors` file in the Explorer.
- Or run **ONNX Inspector: Open File** from the Command Palette.

### Work with the Graph tab

- The graph reads **top to bottom**, closer to Netron's default mental model.
- For TorchScript archives, the graph shows a real **operator DAG** extracted offline from the IR source — no `torch` runtime needed.
- For plain state-dict checkpoints, the graph shows a **module hierarchy tree** with inferred layer types.
- For ONNX, weight and bias tensors that are initializer-backed show up as **embedded rows inside operator cards**.
- Search by node name, operator type, tensor terminal, or embedded parameter name.
- Small helper `Constant` nodes are automatically **folded into the consuming operator** when single-use.
- Click a search result to smoothly reveal it. Click a node in the canvas to inspect it without forcing a viewport jump.
- Use **Ctrl + mouse wheel** to zoom. Drag the background to pan.
- Use the right-side zoom overlay for **+ / − / 100% / Fit** and the zoom slider.
- Use the **Reveal in graph** button in the detail pane when you want to re-center.
- Inspect direct **Upstream** and **Downstream** neighbors from the detail pane.

### Work with the Metadata tab

- Search metadata keys and values.
- If a value parses as JSON, the extension shows an interactive pretty JSON tree alongside the original raw string.
- Use **Expand all** and **Collapse all** for large JSON payloads.
- Copy either the raw value or the pretty JSON rendering.

### Review export and file timestamps

The Overview tab surfaces:

- Detected ONNX export / generated timestamps from metadata when present
- File created and modified times
- Last reload time and file size

### Refresh after regenerating a model

Use the **Reload** button in the editor header to re-read the file from disk.

## Commands

| Command | Description |
|---------|-------------|
| `ONNX Inspector: Open File` | Pick a model file to open in the inspector |
| `ONNX Inspector: Reset Load Consent` | Clear saved per-file consent for unsafe pickle loads |

## Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `onnxInspector.pythonPath` | `""` | Path to the Python interpreter. Leave empty to auto-discover. |
| `onnxInspector.allowFullPickleLoad` | `false` | Allow unsafe pickle deserialization for PT checkpoints that fail `weights_only=True`. Requires workspace trust and per-file confirmation. |

## How it works

1. The extension registers a **custom readonly editor** for `*.onnx`, `*.pt`, `*.pth`, and `*.safetensors`.
2. When a model is opened, the extension host reads basic file stats and routes to the right parser.
3. For ONNX files, a bundled parser decodes the protobuf payload locally in the webview — no subprocess.
4. For PT files, a bundled `detect_format.py` classifies the archive, then the appropriate inspector (`inspect_pt.py` or `inspect_torchscript.py`) runs as a subprocess.
5. For safetensors files, `inspect_safetensors.py` reads the 8-byte header length and parses the JSON header — no third-party packages.
6. The extension derives a serializable inspection model containing summary fields, metadata, tensor summaries, and graph data.
7. The webview renders the inspection model with no external network calls.

More detail is available in [`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md).

## Project layout

- `extension.js` — extension entrypoint and custom editor provider
- `model-parser.js` — format detection, subprocess orchestration, graph-view builders
- `media/main.js` — webview UI logic
- `media/styles.css` — webview styling
- `media/lib/` — bundled ONNX parser modules and graph layout runtime
- `scripts/detect_format.py` — classifies PT archives without deserialization
- `scripts/inspect_pt.py` — PT checkpoint inspector (requires PyTorch)
- `scripts/inspect_torchscript.py` — TorchScript offline graph extractor (stdlib only)
- `scripts/inspect_safetensors.py` — safetensors header reader (stdlib only)
- `scripts/package_vsix.py` — offline VSIX packager
- `docs/` — installation, architecture, and publishing notes

## Packaging a new VSIX

```bash
python3 ./scripts/package_vsix.py
```

Output goes to `dist/` by default.

## Publishing

See [`docs/INSTALL_AND_PUBLISH.md`](docs/INSTALL_AND_PUBLISH.md) for Open VSX and GitHub Release workflows.

## Limitations

- The editor is intentionally **read-only**.
- The graph renderer is strongly inspired by Netron but is not a full Netron clone.
- External tensor files are surfaced as references in initializer metadata; the extension does not open or merge external weight files.
- Weight contents are summarized rather than dumped in full.
- PT files with `weights_only=True` failures require explicit consent, workspace trust, and the `allowFullPickleLoad` setting to proceed.
- TorchScript graph extraction is AST-based: complex control flow and some dynamic dispatch patterns may not be fully recovered (a "Partial TorchScript graph" banner appears when this happens).

## License

This project is released under the MIT License. See [`LICENSE.txt`](LICENSE.txt).

## Third-party notices

This extension includes ONNX-related parser modules and a compact graph layout runtime derived from Netron source files. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
