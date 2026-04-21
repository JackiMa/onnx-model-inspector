#!/usr/bin/env python3
import argparse
import json
import os
import signal
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import detect_format  # noqa: E402
except Exception:  # pragma: no cover
    detect_format = None


class _SafeLoadError(Exception):
    """Raised when safe (weights_only) load fails and caller requested no fallback."""

    def __init__(self, detail: str):
        super().__init__(detail)
        self.detail = detail


def _install_timeout(seconds: int) -> None:
    """Best-effort timeout guard for the Python subprocess itself.

    Uses SIGALRM on POSIX; threading.Timer fallback on Windows. The Node side
    already has its own timeout, this is the belt-and-suspenders layer.
    """
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, lambda *_: _abort_timeout(seconds))
        signal.alarm(seconds)
        return

    timer = threading.Timer(seconds, lambda: _abort_timeout(seconds))
    timer.daemon = True
    timer.start()


def _abort_timeout(seconds: int) -> None:
    sys.stderr.write(f"inspect_pt.py timed out after {seconds}s\n")
    sys.stderr.flush()
    import os
    os._exit(124)


def safe_load(path: str, allow_full: bool, lazy: bool = False):
    """Load a checkpoint defensively.

    Default path uses weights_only=True which PyTorch documents as safe
    against arbitrary code execution from pickle reducers. If that fails
    and the caller has explicit user consent (--allow-full-load) we fall
    back to the legacy unsafe path. Otherwise we return a sentinel the
    Node side turns into a confirmation dialog.

    When ``lazy`` is set, attempt mmap-enabled loading so tensor storages
    stay file-mapped rather than being materialized in RAM. Falls back
    silently (feature probe, not version parsing) when torch doesn't
    support ``mmap`` for this archive.
    """
    import torch

    try:
        if lazy:
            checkpoint = lazy_safe_load(path)
            return checkpoint, "weights_only_lazy"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        return checkpoint, "weights_only"
    except Exception as exc_safe:
        if not allow_full:
            raise _SafeLoadError(str(exc_safe))
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        return checkpoint, "full_load"


def lazy_safe_load(path: str):
    """Try mmap-enabled safe load. Falls back when torch version or archive
    layout does not support it. Feature probing, not version parsing."""
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError as exc:
        if "mmap" in str(exc).lower():
            return torch.load(path, map_location="cpu", weights_only=True)
        raise
    except RuntimeError as exc:
        if "mmap" in str(exc).lower():
            return torch.load(path, map_location="cpu", weights_only=True)
        raise


def walk(value, prefix, payload, torch_mod):
    if isinstance(value, dict):
        payload["sections"].append({
            "path": prefix,
            "kind": "mapping",
            "summary": f"{len(value)} entries",
            "entryCount": len(value),
        })
        for key, child in value.items():
            walk(child, f"{prefix}.{key}", payload, torch_mod)
        return

    if isinstance(value, (list, tuple)):
        payload["sections"].append({
            "path": prefix,
            "kind": "sequence",
            "summary": f"{len(value)} items",
            "entryCount": len(value),
        })
        for index, child in enumerate(value):
            walk(child, f"{prefix}[{index}]", payload, torch_mod)
        return

    if torch_mod is not None and hasattr(torch_mod, "Tensor") and isinstance(value, torch_mod.Tensor):
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--allow-full-load", action="store_true",
                        help="Permit torch.load with weights_only=False after explicit user consent.")
    parser.add_argument("--lazy", action="store_true",
                        help="Attempt mmap-backed load (torch.load(mmap=True)) to avoid materializing tensor data.")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Hard timeout in seconds (double-protection over Node's timeout).")
    args = parser.parse_args()

    _install_timeout(max(5, args.timeout))

    detection = None
    if detect_format is not None:
        try:
            detection = detect_format.detect(args.path)
        except Exception:
            detection = None

    try:
        import torch
    except Exception as exc:  # pragma: no cover
        print(json.dumps({"error": f"PyTorch is not available: {exc}", "detection": detection}))
        return 2

    try:
        checkpoint, mode = safe_load(args.path, allow_full=args.allow_full_load, lazy=args.lazy)
    except _SafeLoadError as exc:
        print(json.dumps({
            "requiresFullLoad": True,
            "safeLoadError": exc.detail,
            "producerVersion": getattr(torch, "__version__", ""),
            "detection": detection,
        }))
        return 0
    except Exception as exc:
        print(json.dumps({"error": f"Unable to load checkpoint: {exc}", "detection": detection}))
        return 3

    payload = {
        "producerVersion": getattr(torch, "__version__", ""),
        "rootType": type(checkpoint).__name__,
        "loadMode": mode,
        "detection": detection,
        "sections": [],
        "metadata": [],
        "tensors": [],
    }
    walk(checkpoint, "root", payload, torch)
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
