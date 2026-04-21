#!/usr/bin/env python3
"""Detect a model file's format without triggering any unsafe deserialization.

Only Python standard library is used. The output is JSON with:
    {"subFormat": "<label>", "reasons": [...], "warnings": [...]}

Supported labels:
    safetensors
    safetensors-index
    torchscript
    exported-program
    full-model-pickle
    lightning-checkpoint
    optimizer-checkpoint
    plain-state-dict
    legacy-pickle
    onnx
    unknown
"""
import io
import json
import os
import pickletools
import struct
import sys
import zipfile


MAX_HEADER_BYTES = 100 * 1024 * 1024  # safetensors header cap
ONNX_MAGIC_PREFIXES = (b"\x08", b"\x12")  # protobuf field tags commonly starting ONNX model.proto


def _read_head(path, n=16):
    with open(path, "rb") as fh:
        return fh.read(n)


def _read_zip_names(path):
    try:
        with zipfile.ZipFile(path) as zf:
            return zf.namelist()
    except zipfile.BadZipFile:
        return None


def _has_code_directory(name):
    """True if the zip member path indicates a TorchScript `code/` subtree.

    PyTorch places TorchScript IR under either `code/` (legacy) or
    `<archive>/code/` (current) where <archive> is the top-level folder
    named after the file. We accept both and also an `archive/code/` variant
    seen in some exports.
    """
    if name.startswith("code/") or name.startswith("archive/code/"):
        return True
    idx = name.find("/code/")
    # Require the prefix to be a single segment (archive name), no other slashes.
    return idx > 0 and "/" not in name[:idx]


def _find_pickle_member(zip_names):
    """Returns the pickle member name in a PyTorch zip.

    Both legacy (root `data.pkl`) and PyTorch 2.x (`<archive>/data.pkl`) layouts
    exist in the wild. Prefer root, then archive.
    """
    if "data.pkl" in zip_names:
        return "data.pkl"
    for name in zip_names:
        if name.endswith("/data.pkl"):
            return name
    return None


def _shallow_scan_pickle(zip_path, member_name):
    """Read a pickle member from a zip and extract heuristic signals.

    Returns a dict with:
        globals: set of "module.Name" strings referenced by GLOBAL / STACK_GLOBAL
        short_strings: list of short BINUNICODE / SHORT_BINSTRING values
    Never deserializes an object.
    """
    globals_set = set()
    short_strings = []
    try:
        with zipfile.ZipFile(zip_path) as zf:
            with zf.open(member_name) as member:
                data = member.read(4 * 1024 * 1024)  # cap at 4MB of pickle bytecode
        stack = []
        for op, arg, _pos in pickletools.genops(io.BytesIO(data)):
            name = op.name
            if name == "GLOBAL":
                # arg is 'module\nName'
                try:
                    module, cls = str(arg).split("\n", 1)
                    globals_set.add(f"{module}.{cls}")
                except ValueError:
                    pass
            elif name == "STACK_GLOBAL":
                if len(stack) >= 2:
                    cls = stack[-1]
                    module = stack[-2]
                    globals_set.add(f"{module}.{cls}")
            elif name in ("SHORT_BINUNICODE", "BINUNICODE", "SHORT_BINSTRING", "BINSTRING", "BINUNICODE8"):
                s = str(arg)
                stack.append(s)
                if len(s) <= 64:
                    short_strings.append(s)
                if len(stack) > 4:
                    stack = stack[-4:]
            elif name in ("MEMOIZE", "PUT", "BINPUT", "LONG_BINPUT"):
                pass
    except Exception:
        pass  # scan is heuristic only
    return {"globals": globals_set, "short_strings": short_strings}


def _looks_like_safetensors(path):
    """Returns (is_safetensors, reasons)."""
    reasons = []
    size = os.path.getsize(path)
    if size < 10:
        return False, ["file too small"]
    with open(path, "rb") as fh:
        header_len_bytes = fh.read(8)
        if len(header_len_bytes) < 8:
            return False, ["no header length"]
        header_len = struct.unpack("<Q", header_len_bytes)[0]
        if header_len < 2 or header_len > MAX_HEADER_BYTES:
            return False, [f"implausible header length: {header_len}"]
        if header_len + 8 > size:
            return False, ["header exceeds file"]
        header_bytes = fh.read(header_len)
    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return False, [f"header not valid JSON: {exc}"]
    if not isinstance(header, dict):
        return False, ["header is not an object"]

    # Schema check: every non-__metadata__ key must map to {dtype, shape, data_offsets}
    valid_tensor_count = 0
    for key, value in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(value, dict):
            return False, [f"entry {key!r} is not an object"]
        for required in ("dtype", "shape", "data_offsets"):
            if required not in value:
                return False, [f"entry {key!r} missing {required}"]
        if not isinstance(value["shape"], list):
            return False, [f"entry {key!r} shape not a list"]
        if not isinstance(value["data_offsets"], list) or len(value["data_offsets"]) != 2:
            return False, [f"entry {key!r} data_offsets malformed"]
        valid_tensor_count += 1

    if valid_tensor_count == 0 and "__metadata__" not in header:
        return False, ["no tensor entries"]
    reasons.append(f"header {header_len} bytes, {valid_tensor_count} tensors")
    return True, reasons


def _looks_like_safetensors_index(path):
    try:
        with open(path, "rb") as fh:
            data = fh.read(MAX_HEADER_BYTES)
        obj = json.loads(data.decode("utf-8"))
    except Exception:
        return False
    return isinstance(obj, dict) and isinstance(obj.get("weight_map"), dict)


def _classify_pickle_archive(zip_names, pickle_member, zip_path):
    scan = _shallow_scan_pickle(zip_path, pickle_member)
    globals_found = scan["globals"]
    short_strings = scan["short_strings"]

    has_module_refs = any(g.startswith("torch.nn.modules") for g in globals_found)
    has_optimizer = any(g.startswith("torch.optim") for g in globals_found)

    sentinel_strings = set(short_strings)

    if "pytorch_lightning_version" in sentinel_strings or any(g.startswith("pytorch_lightning") for g in globals_found):
        return "lightning-checkpoint", [f"lightning marker found"]
    if "optimizer_state_dict" in sentinel_strings or "optimizer" in sentinel_strings and has_optimizer:
        return "optimizer-checkpoint", [f"optimizer state detected"]
    if has_module_refs:
        sample = next((g for g in globals_found if g.startswith("torch.nn.modules")), "")
        return "full-model-pickle", [f"module ref {sample}"]
    return "plain-state-dict", ["no module/lightning/optimizer markers detected"]


def _looks_like_onnx(head):
    # ONNX is a protobuf stream. First byte is tag byte for field 1 (ir_version = varint) which is 0x08.
    # Protobuf field 7 (producer_name, string) = 0x3a. Using a whitelist is better than a blacklist.
    return len(head) >= 2 and head[:1] in (b"\x08", b"\x12", b"\x1a", b"\x22", b"\x2a") and b"PK\x03\x04" not in head[:4]


def detect(path):
    reasons = []
    warnings = []

    try:
        size = os.path.getsize(path)
    except OSError as exc:
        return {"subFormat": "unknown", "reasons": [f"stat failed: {exc}"], "warnings": []}

    if size == 0:
        return {"subFormat": "unknown", "reasons": ["empty file"], "warnings": []}

    # Fast-path by extension for safetensors index JSON.
    if path.endswith(".safetensors.index.json"):
        if _looks_like_safetensors_index(path):
            return {"subFormat": "safetensors-index", "reasons": ["safetensors index JSON"], "warnings": []}

    head = _read_head(path, 32)

    # Safetensors header must come BEFORE zip check (header length is first 8 bytes).
    is_safe, reason_list = _looks_like_safetensors(path)
    if is_safe:
        return {"subFormat": "safetensors", "reasons": reason_list, "warnings": []}

    if head.startswith(b"PK\x03\x04"):
        names = _read_zip_names(path) or []
        if any(_has_code_directory(n) for n in names):
            reasons.append("zip contains code/ (TorchScript IR)")
            return {"subFormat": "torchscript", "reasons": reasons, "warnings": []}
        if any(n.endswith("serialized_model.json") or "pt2_archive_format" in n for n in names):
            reasons.append("PT2 archive markers")
            return {"subFormat": "exported-program", "reasons": reasons, "warnings": []}
        pickle_member = _find_pickle_member(names)
        if pickle_member:
            label, pickle_reasons = _classify_pickle_archive(names, pickle_member, path)
            return {
                "subFormat": label,
                "reasons": [f"zip+pickle member {pickle_member}"] + pickle_reasons,
                "warnings": []
            }
        return {"subFormat": "unknown", "reasons": ["zip without recognizable PyTorch layout"], "warnings": []}

    # Legacy pickle (pre-PyTorch 1.3): starts with pickle protocol marker.
    if head[:2] in (b"\x80\x02", b"\x80\x03", b"\x80\x04", b"\x80\x05"):
        warnings.append("legacy pickle format; proceed with caution")
        return {"subFormat": "legacy-pickle", "reasons": ["pickle proto marker"], "warnings": warnings}

    if _looks_like_onnx(head):
        return {"subFormat": "onnx", "reasons": ["protobuf tag matches ONNX"], "warnings": []}

    return {"subFormat": "unknown", "reasons": ["no known magic matched"], "warnings": warnings}


def main():
    if len(sys.argv) != 2:
        print(json.dumps({"error": "usage: detect_format.py <file>"}))
        return 1
    print(json.dumps(detect(sys.argv[1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
