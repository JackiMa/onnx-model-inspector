# Third-party notices

This extension bundles parser-oriented and graph-layout source modules derived from the following project:

## Netron

- Project: `lutzroeder/netron`
- Repository: <https://github.com/lutzroeder/netron>
- License: MIT

Included derivative files in this extension are based on Netron source modules for:

- ONNX protobuf decoding
- supporting text / protobuf readers
- compact Dagre-based graph layout

Those files are located under `media/lib/` and were adapted for this extension packaging layout.

## ONNX format

This extension reads ONNX model files and surfaces metadata and graph structure from the ONNX container format. ONNX itself is a separate specification and ecosystem project.
