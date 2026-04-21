# How the extension works

## Design goal

The main requirement for this build was: **install once, then open model artifacts immediately**.

That led to three design choices:

1. **No Python runtime dependency for ONNX inspection**
2. **No external Netron service or desktop app dependency**
3. **No network fetches from inside the extension**

PT / PTH support is a deliberate exception to the first point: checkpoint inspection shells out to a local Python + PyTorch runtime when available, because those formats are PyTorch-native rather than protobuf-native.

## Runtime architecture

### 1. Custom readonly editor

The extension registers a custom readonly editor for `*.onnx`, `*.pt`, and `*.pth` files.

That means opening a model launches a purpose-built webview instead of a generic binary editor.

### 2. Extension-host parsing

The extension host does the parsing work.

Flow:

1. VS Code / Cursor asks the provider to open the model file.
2. The extension reads raw bytes through the workspace file system API.
3. The extension also reads basic file stats such as `ctime`, `mtime`, and file size.
4. If the file is ONNX, a bundled ONNX protobuf decoder parses the `ModelProto` payload.
5. If the file is PT / PTH, the extension invokes `scripts/inspect_pt.py`, which uses local PyTorch to load the checkpoint on CPU and summarize its structure.
6. The extension converts the decoded structure into a simplified, serializable inspection model.
7. The webview receives that inspection model and renders it.

This keeps the webview simple and avoids shipping raw file bytes into the browser UI.

### 3. Format-dependent output

- **ONNX** produces the full graph-oriented inspection model.
- **PT / PTH** produces a summary-oriented inspection model: checkpoint sections, scalar metadata, and tensor entries.

The PT path intentionally does not attempt to reconstruct a computation graph in V1.

### 4. Safety and limitations of PT inspection

The PT helper loads checkpoints via local PyTorch and walks the resulting object tree to extract:

- mapping / sequence sections
- scalar metadata values
- tensor names, dtypes, shapes, element counts, and estimated byte sizes

This path is meant for trusted local artifacts. PT / PTH formats are not as structurally self-describing as ONNX, so the extension focuses on summary inspection rather than general object introspection or graph reconstruction.

If Python or PyTorch is unavailable, PT inspection fails with a structured error shown in the editor.

The Python executable for PT inspection is resolved in this order:
1. `onnxInspector.pythonPath` setting
2. `ONNX_INSPECTOR_PYTHON`
3. `CONDA_PYTHON_EXE`
4. `python3`

If `onnxInspector.pythonPath` contains `${workspaceFolder}`, it is expanded relative to the current workspace.

## Bundled parser and layout runtime

The bundled runtime lives under `media/lib/` and includes:

- `onnx-proto.mjs`
- `protobuf.mjs`
- `text.mjs`
- `onnx-parser.mjs`
- `dagre.mjs`

The first three provide low-level ONNX protobuf decoding. `onnx-parser.mjs` is the extension-specific summarizer. `dagre.mjs` provides the compact layered graph layout engine bundled into the extension so the graph remains usable without external dependencies.

## What the parser extracts

### Model summary

- graph name
- IR version
- producer name / version
- domain
- model version
- opset imports
- top-level doc string
- file size and object counts

### Metadata

- model-level `metadata_props`
- graph-level `metadata_props`
- JSON detection for values that parse as JSON
- interactive pretty-JSON tree generation for JSON metadata values
- preservation of the original raw metadata string for side-by-side inspection
- detection of common generated / exported / created timestamps from both flat metadata and nested JSON metadata

### Inputs / outputs / value info

- name
- type summary
- tensor shape summary where available
- metadata counts

### Initializers / weights

- tensor name
- data type
- shape
- element count
- estimated byte size when possible
- storage summary (`raw_data`, `float_data`, etc.)
- external tensor reference entries if present

### Graph structure

The extension builds a display graph by:

1. treating non-initializer graph inputs as source terminals
2. treating operator nodes as transformation nodes
3. embedding initializer-backed parameter tensors **inside the owning operator card** instead of drawing them as separate graph nodes
4. folding small, single-use helper `Constant` nodes into the consuming operator when they are safe to inline
5. treating graph outputs as terminal consumers
6. connecting tensors from producer to consumer in a vertical top-to-bottom flow

After that, the parser runs a layered layout pass and stores:

- node boxes with final width / height / position
- routed edge points for cleaner edge rendering
- overall graph bounds for fit-to-viewport behavior

That is why the graph is substantially less sparse than a naive left-column / right-column placement and also less cluttered than rendering every weight tensor as a standalone node.

## Why the graph is lightweight instead of a full Netron clone

The goal here is not to replicate every Netron feature. The goal is to make **ONNX structure + metadata inspection** convenient inside Cursor / VS Code without external setup.

So the graph renderer focuses on:

- compact readable operator flow
- a vertical layout that is easier to read for sequential models
- embedded parameter rows for initializer-backed inputs
- inline Constant folding for tiny helper tensors such as `axes` and small shape helpers
- routed tensor connectivity
- lightweight edge labels when type / shape summaries are available
- stable selection behavior where graph clicks inspect only and sidebar results explicitly center the target
- explicit reveal-to-selection control
- direct upstream / downstream inspection
- searchability

and intentionally skips heavier features such as deep tensor data browsing, every advanced grouping mode, and external tensor loading.

## Why the editor is read-only

This extension is meant for inspection, packaging review, and metadata contract checks. It does not modify ONNX files.

That keeps the implementation smaller and more reliable because it does not need save, undo/redo, backup, or conflict-resolution logic.

## UI structure

### Overview

Best for a quick sanity pass:

- identity and versions
- counts
- doc string
- metadata preview
- file created / modified / reload times
- detected ONNX export timestamps when present

### Graph

Best for flow inspection:

- vertical layered layout
- embedded operator parameters
- routed edges with compact labels
- node search
- operator insight cards in the detail pane, including activation plots, plain-language operator summaries, and shape before/after cards for common shape ops
- click-to-inspect without forced viewport jumps
- left-side graph results that explicitly center the chosen node
- explicit reveal button when you *do* want re-centering
- drag-to-pan
- Ctrl + wheel zoom
- right-side zoom slider / fit / reset controls
- direct upstream and downstream navigation
- remembered collapsible detail sections on the right-side inspector pane

### Metadata

Best for deployment contracts:

- search key/value pairs
- pretty JSON first, raw value second
- expand and collapse JSON by field
- expand-all / collapse-all controls
- copy raw value
- copy pretty JSON if the value parses cleanly

### I/O & Weights

Best for interface review:

- input tensor types
- output tensor types
- value info
- weight tensor summaries

## Reload behavior

The extension does not auto-watch and auto-reparse every external file change. Instead, it exposes a **Reload** button in the editor header.

That choice keeps the behavior predictable and avoids maintaining file watchers across every workspace / remote scenario.

## What to change if you want to extend it

### Add stricter metadata validation

A good next step is to add a command that validates required metadata keys or schema fragments against your internal contract.

### Add alternate graph modes

A useful extension point is a toggle for hiding edge labels, dimming terminals, or switching between the embedded-parameter view and a more tensor-centric debugging view on very large graphs.

### Add richer graph rendering

You could layer in grouping, subgraph folding, tensor provenance overlays, or a minimap, but that would increase bundle size and maintenance cost.

### Add node-level metadata surfacing elsewhere

Node-level metadata already appears in the selection details pane. If you need node metadata in a standalone searchable table, add another tab or a filter mode inside the metadata pane.
