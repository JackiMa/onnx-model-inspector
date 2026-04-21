#!/usr/bin/env python3
"""Parse a .safetensors file header.

Pure standard library. Reads only the 8-byte header length + JSON header.
Tensor data is never materialized.

Output shape matches inspect_pt.py so the Node side can reuse
normalizePtInspection.
"""
import argparse
import json
import os
import signal
import struct
import sys
import threading


MAX_HEADER_BYTES = 100 * 1024 * 1024  # codex R2 #3: explicit cap


DTYPE_TO_ELEMSIZE = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
    "F8_E5M2": 1,
    "F8_E4M3": 1,
}


DTYPE_TO_NUMPY = {
    "BOOL": "bool",
    "U8": "uint8",
    "I8": "int8",
    "U16": "uint16",
    "I16": "int16",
    "F16": "float16",
    "BF16": "bfloat16",
    "U32": "uint32",
    "I32": "int32",
    "F32": "float32",
    "U64": "uint64",
    "I64": "int64",
    "F64": "float64",
    "F8_E5M2": "float8_e5m2",
    "F8_E4M3": "float8_e4m3fn",
}


def _install_timeout(seconds: int) -> None:
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, lambda *_: _abort_timeout(seconds))
        signal.alarm(seconds)
        return
    timer = threading.Timer(seconds, lambda: _abort_timeout(seconds))
    timer.daemon = True
    timer.start()


def _abort_timeout(seconds: int) -> None:
    sys.stderr.write(f"inspect_safetensors.py timed out after {seconds}s\n")
    sys.stderr.flush()
    os._exit(124)


def read_header(path: str) -> dict:
    size = os.path.getsize(path)
    if size < 10:
        raise ValueError("file too small to be safetensors")
    with open(path, "rb") as fh:
        header_len_bytes = fh.read(8)
        if len(header_len_bytes) != 8:
            raise ValueError("truncated header length")
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        if header_len < 2 or header_len > MAX_HEADER_BYTES:
            raise ValueError(f"implausible header length: {header_len}")
        if header_len + 8 > size:
            raise ValueError("header length exceeds file size")
        raw = fh.read(header_len)
    try:
        header = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise ValueError(f"header is not valid UTF-8 JSON: {exc}")
    if not isinstance(header, dict):
        raise ValueError("header top-level is not an object")
    return header


def _product(shape):
    acc = 1
    for dim in shape:
        acc *= int(dim)
    return acc


def to_payload(header: dict) -> dict:
    metadata_map = header.get("__metadata__") if isinstance(header.get("__metadata__"), dict) else {}

    tensors = []
    section_children = []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(info, dict):
            continue
        dtype_raw = info.get("dtype", "")
        shape = info.get("shape", []) or []
        offsets = info.get("data_offsets", [0, 0]) or [0, 0]
        try:
            numel = _product(shape)
        except Exception:
            numel = 0
        try:
            byte_len = int(offsets[1]) - int(offsets[0])
        except Exception:
            byte_len = 0
        dtype_str = DTYPE_TO_NUMPY.get(str(dtype_raw).upper(), str(dtype_raw).lower())
        tensors.append({
            "path": f"root.{name}",
            "dtype": dtype_str,
            "shape": list(shape),
            "numel": numel,
            "bytes": byte_len,
            "storage": "safetensors",
        })
        section_children.append(name)

    sections = [
        {"path": "root", "kind": "mapping", "summary": f"{len(tensors)} tensors", "entryCount": len(tensors)}
    ]

    metadata = [
        {"path": f"__metadata__.{key}", "value": value}
        for key, value in metadata_map.items()
    ]

    return {
        "producerVersion": "",
        "rootType": "safetensors",
        "loadMode": "header_only",
        "detection": {"subFormat": "safetensors", "reasons": ["header parsed"], "warnings": []},
        "sections": sections,
        "metadata": metadata,
        "tensors": tensors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()
    _install_timeout(max(5, args.timeout))
    try:
        header = read_header(args.path)
    except Exception as exc:
        print(json.dumps({"error": f"Unable to read safetensors header: {exc}"}))
        return 3
    print(json.dumps(to_payload(header)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
