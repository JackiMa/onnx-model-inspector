# Changelog

## 0.6.0

- Added read-only `.pt` and `.pth` checkpoint inspection with tensor, section, and scalar metadata summaries.
- Added configurable Python executable resolution for PT / PTH inspection through `onnxInspector.pythonPath`, `ONNX_INSPECTOR_PYTHON`, `CONDA_PYTHON_EXE`, then `python3`.
- Updated the inspector UI and docs to distinguish full ONNX graph inspection from PT / PTH summary-only inspection.
- Fixed VSIX packaging so the PT parser bridge and bundled Python helper are included while internal implementation plans stay out of release artifacts.

## 0.5.2

- Fixed the remaining graph-selection navigation bug that could reset the view to the top after selecting certain blocks. The inspector now preserves both the outer tab scroll position and the inner graph viewport when you click a node directly in the graph.
- Kept explicit recentering limited to the left-side result list and **Reveal in graph**, so graph clicks are now consistently inspect-only.
- Moved the right-side navigation guidance off the page and into hover tooltips on **Reveal in graph**, **Expand all**, and **Collapse all** to free up detail-pane space.
- Simplified the plain-language explanations for shape-only operators such as **Squeeze** and **Unsqueeze**.
- Added a UI runtime test that simulates DOM replacement and verifies that graph-click selection preserves the current viewport while sidebar-driven selection still recenters the target node.

## 0.5.1

- Fixed a regression that could leave the Graph tab blank when opening it.
- Restored the missing type-flow summary renderer used by the right-hand explanation panel.
- Restored lexical graph-node sorting used by the upstream / downstream relationship panels.
- Added a runtime UI smoke test that opens the Graph tab and renders operator explanations during test runs.

## 0.5.0

- Fixed graph navigation semantics so clicking a node inside the graph only changes the selection, while clicks from the left-side result list or **Reveal in graph** explicitly center the target node.
- Reworked graph-focus scheduling to avoid stale focus callbacks, which previously caused viewport jumps that could feel inconsistent after rapid selection changes.
- Added remembered collapsible sections to the right-side selection pane, along with **Expand all** / **Collapse all** controls for faster inspection workflows.
- Rewrote operator explanations in simpler plain language and added clearer effect badges and bullet lists so shape ops such as **Squeeze / Unsqueeze** are easier to understand.
- Added shape before/after summary cards for common shape-only operators and kept activation-function plots for nonlinear layers such as **Elu**.
- Added an explicit navigation hint in the selection pane and the graph-search sidebar so the “inspect vs. center” interaction model is easier to discover.
- Updated README and architecture notes to document the new navigation behavior and remembered detail-pane layout.

## 0.4.0

- Folded small, single-use `Constant` helper nodes into their consuming operators so axes / shape helper tensors no longer clutter the graph, and model-specific cases like `/Constant_1` stop behaving like awkward standalone nodes.
- Added richer operator profiling so simple activation layers such as **Elu** render as smaller micro nodes and shape operators such as **Squeeze / Unsqueeze** render as more compact mini cards.
- Reworked graph highlighting so non-selected nodes stay readable and the active selection is emphasized with a clearer glow / border treatment instead of globally dimming the whole graph.
- Added operator insight cards in the detail pane, including activation-function plots for common activations and clearer natural-language explanations for shape, linear, recurrent, normalization, and constant operators.
- Added merged-Constant provenance to graph search so searching for helper constant names still reveals the owning operator even after folding.
- Improved tensor preview decoding for small initializer / constant values and surfaced those previews in graph-detail summaries.
- Added parser and UI checks covering merged Constant helpers, micro / mini operator rendering categories, and operator-insight UI presence.

## 0.3.0

- Reworked the graph into a **vertical top-to-bottom layout** so model flow reads more like Netron and uses the canvas space much more efficiently.
- Stopped rendering most initializer tensors as standalone graph nodes; instead, initializer / weight inputs are now shown as **embedded parameter rows inside operator cards**.
- Re-styled operator nodes into a more Netron-like card shape with a colored operator header and inline parameter rows.
- Added compact **edge labels** for available tensor type / shape summaries.
- Added search support for **embedded parameters**, so searching for a weight or bias now reveals the owning operator.
- Added ONNX **temporal metadata detection** for common exported / generated / created timestamp fields, including nested JSON metadata values.
- Added **file metadata** surfacing in the Overview tab, including file created time, modified time, reload time, file size, and detected export timestamps when available.
- Added parser tests covering vertical graph layout, embedded parameter rendering, and timestamp metadata detection.

## 0.2.0

- Replaced the old sparse graph layout with a compact layered layout powered by a bundled Dagre runtime derived from Netron.
- Added routed graph edges and a more Netron-inspired graph visual style.
- Added background drag-to-pan in the graph canvas.
- Added first-open automatic fit-to-viewport behavior for graph rendering.
- Added selection-neighborhood highlighting so the selected node and its direct neighbors stand out more clearly.
- Added explicit **Reveal in graph** behavior in the details pane so recentering is user-controlled.
- Added direct upstream / downstream neighbor navigation from the graph detail pane.
- Expanded graph zoom range and updated zoom controls accordingly.
- Added metadata **Expand all** / **Collapse all** controls for pretty JSON trees.
- Added parser and packaging checks for the bundled graph layout runtime and compact layout expectations.

## 0.1.1

- Added graph zoom with **Ctrl + mouse wheel**.
- Added an in-canvas zoom control with slider, zoom in/out, reset, and fit actions.
- Fixed graph viewport resets when selecting nodes from the graph canvas or the left-side graph result list.
- Updated graph focusing so explicit node jumps animate from the current viewport instead of from the top-left corner.
- Reworked metadata JSON rendering to show an interactive pretty JSON tree before the raw stored value.
- Added field-level syntax highlighting and foldable expand/collapse sections for JSON metadata values.

## 0.1.0

- Initial release.
- Added a read-only custom editor for `.onnx` files.
- Added Overview, Graph, Metadata, and I/O & Weights tabs.
- Bundled ONNX protobuf parsing directly into the extension.
- Added an offline Python-based VSIX packaging script.
- Added publishing and architecture documentation.
