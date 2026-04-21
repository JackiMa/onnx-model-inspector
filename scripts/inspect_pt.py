#!/usr/bin/env python3
import json
import sys


def main() -> int:
    if len(sys.argv) != 2:
        print(json.dumps({"error": "usage: inspect_pt.py <file>"}))
        return 1

    target = sys.argv[1]
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        print(json.dumps({"error": f"PyTorch is not available: {exc}"}))
        return 2

    try:
        checkpoint = torch.load(target, map_location="cpu")
    except Exception as exc:
        print(json.dumps({"error": f"Unable to load checkpoint: {exc}"}))
        return 3

    payload = {
        "producerVersion": getattr(torch, "__version__", ""),
        "rootType": type(checkpoint).__name__,
        "sections": [],
        "metadata": [],
        "tensors": [],
    }

    def walk(value, prefix="root"):
        if isinstance(value, dict):
            payload["sections"].append({
                "path": prefix,
                "kind": "mapping",
                "summary": f"{len(value)} entries",
                "entryCount": len(value),
            })
            for key, child in value.items():
                walk(child, f"{prefix}.{key}")
            return

        if isinstance(value, (list, tuple)):
            payload["sections"].append({
                "path": prefix,
                "kind": "sequence",
                "summary": f"{len(value)} items",
                "entryCount": len(value),
            })
            for index, child in enumerate(value):
                walk(child, f"{prefix}[{index}]")
            return

        if hasattr(torch, "Tensor") and isinstance(value, torch.Tensor):
            payload["tensors"].append({
                "path": prefix,
                "dtype": str(value.dtype).replace("torch.", ""),
                "shape": list(value.shape),
                "numel": int(value.numel()),
                "bytes": int(value.numel() * value.element_size()),
                "storage": "tensor",
            })
            return

        if isinstance(value, (str, int, float, bool)) or value is None:
            payload["metadata"].append({"path": prefix, "value": value})
            return

        payload["sections"].append({
            "path": prefix,
            "kind": type(value).__name__,
            "summary": type(value).__name__,
            "entryCount": None,
        })

    walk(checkpoint)
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
