#!/usr/bin/env python3
"""Offline TorchScript archive inspector.

Phase A implementation per docs/PT_SUPPORT_PLAN_V2.md section 7. Parses the
``code/**/*.py`` files inside a TorchScript zip archive using only the Python
standard library. Extracts an operator graph (ops + edges + module hierarchy)
without importing torch. Output is a JSON payload compatible with
normalizePtInspection on the Node side, with an extra ``torchscript`` block.

Failure contract: on any unrecoverable error OR zero recovered ops, prints
``{"error": "...", "detection": {...}}`` and exits with code 3 so the caller
falls back to the module-hierarchy view.
"""
import argparse
import ast
import json
import os
import signal
import sys
import threading
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import detect_format  # noqa: E402
except Exception:
    detect_format = None


TORCH_ROOTS = ("torch", "ops", "aten", "prim")
# TorchScript archives commonly use a single archive-root directory, so a
# code member can appear as ``code/...``, ``archive/code/...`` or
# ``<anything>/code/...``. We check for any of these by splitting on
# ``/code/`` below; these constants are kept for documentation.
CODE_PREFIXES = ("code/", "archive/code/")


def _install_timeout(seconds: int) -> None:
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, lambda *_: _abort_timeout(seconds))
        signal.alarm(seconds)
        return
    timer = threading.Timer(seconds, lambda: _abort_timeout(seconds))
    timer.daemon = True
    timer.start()


def _abort_timeout(seconds: int) -> None:
    sys.stderr.write(f"inspect_torchscript.py timed out after {seconds}s\n")
    sys.stderr.flush()
    os._exit(124)


def _strip_to_code_relative(name):
    """Return the ``code/...`` relative path for a zip member, or None.

    Accepts ``code/foo.py``, ``archive/code/foo.py``, or
    ``<archive_root>/code/foo.py``.
    """
    if not name.endswith(".py"):
        return None
    for prefix in CODE_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    # Handle generic archive-root prefix: ``<root>/code/...``.
    idx = name.find("/code/")
    if idx != -1:
        return name[idx + len("/code/"):]
    return None


def _read_code_modules(zip_path):
    """Return list of (rel_path, parsed_ast) for every code/*.py member.

    ``rel_path`` is the path relative to the ``code/`` directory (e.g.
    ``__torch__/torch/nn/modules/container.py``).
    """
    out = []
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            rel = _strip_to_code_relative(name)
            if rel is None:
                continue
            try:
                raw = zf.read(name)
            except KeyError:
                continue
            try:
                tree = ast.parse(raw, filename=name)
            except SyntaxError:
                continue
            out.append((rel, tree))
    return out


def _dotted_name(node):
    """Best-effort reconstruction of dotted attribute access as a string.
    Returns ``None`` for expressions we can't reduce to a simple dotted path.
    """
    parts = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return None


def _call_target(func):
    """Classify the callee of an ast.Call.

    Returns (kind, label, primary_name) where kind is one of:
        'torch'      - torch.*, torch.nn.functional.*, torch.ops.*.*, ops.aten.*
        'submodule'  - self.<attr>(...)
        'method'     - <var>.<method>(...)
        'other'      - unrecognized
    label is a human-readable callee string; primary_name is the receiver
    variable for 'method' calls (so we can treat it as an input).
    """
    if isinstance(func, ast.Attribute):
        # self.<attr>(...)
        if isinstance(func.value, ast.Name) and func.value.id == "self":
            return "submodule", f"self.{func.attr}", None

        dotted = _dotted_name(func)
        if dotted is not None:
            root = dotted.split(".", 1)[0]
            if root in TORCH_ROOTS:
                return "torch", dotted, None

        # <var>.<method>(...)
        if isinstance(func.value, ast.Name):
            return "method", f"{func.value.id}.{func.attr}", func.value.id

        # Fallback: still treat it as a method-like call on an expression.
        if dotted is not None:
            return "other", dotted, None

    if isinstance(func, ast.Name):
        # Bare name call; likely a helper. Not a torch op.
        return "other", func.id, None

    return "other", "<dynamic>", None


def _op_type_from_label(kind, label):
    if kind == "torch":
        return label.rsplit(".", 1)[-1]
    if kind == "submodule":
        return label  # 'self.encoder' — resolved later to a class if possible
    if kind == "method":
        return label.rsplit(".", 1)[-1]
    return label


def _targets_to_names(target):
    """Flatten an assignment target into a list of variable names.
    Returns list[str|None] where None marks a non-Name slot (e.g. starred,
    subscript).
    """
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, (ast.Tuple, ast.List)):
        out = []
        for elt in target.elts:
            if isinstance(elt, ast.Name):
                out.append(elt.id)
            elif isinstance(elt, ast.Starred) and isinstance(elt.value, ast.Name):
                out.append(elt.value.id)
            else:
                out.append(None)
        return out
    return [None]


def _collect_arg_names(call):
    """Return variable names referenced in positional + keyword args."""
    names = []
    for arg in call.args:
        if isinstance(arg, ast.Name):
            names.append(arg.id)
        elif isinstance(arg, ast.Starred) and isinstance(arg.value, ast.Name):
            names.append(arg.value.id)
    for kw in call.keywords:
        if kw.value is not None and isinstance(kw.value, ast.Name):
            names.append(kw.value.id)
    return names


def _extract_submodule_alias(value):
    """If ``value`` aliases a submodule of ``self``, return ``"self.<attr>"``.

    Handles ``self.foo`` and ``getattr(self, "name")`` forms. Returns None
    otherwise.
    """
    if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name) \
            and value.value.id == "self":
        return f"self.{value.attr}"
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) \
            and value.func.id == "getattr" and len(value.args) >= 2:
        target = value.args[0]
        attr = value.args[1]
        if isinstance(target, ast.Name) and target.id == "self" \
                and isinstance(attr, ast.Constant) and isinstance(attr.value, str):
            return f"self.{attr.value}"
    return None


class _ForwardVisitor(ast.NodeVisitor):
    """Walks a forward() body collecting ops, edges, and counters.

    A single instance is reused per forward method. It records nodes in
    top-level statements only; bodies of For/While/If/With/Try are counted
    as 'unparsed' ops to keep us honest per spec.
    """

    def __init__(self, module_path, class_name, op_id_counter):
        self.module_path = module_path
        self.class_name = class_name
        self.counter = op_id_counter
        self.ops = []
        self.edges = []
        self.warnings = []
        self.var_to_op = {}  # var_name -> op_id producing it
        self.submodule_alias = {}  # local var -> "self.<attr>" (from getattr / self.x)
        self.returned = set()
        self.op_count_total = 0
        self.op_count_extracted = 0
        self.op_count_unparsed = 0
        self.partial = False

    def _next_id(self):
        idx = self.counter[0]
        self.counter[0] += 1
        return f"op:{idx}"

    def _record_edges(self, op_id, inputs):
        for var_name in inputs:
            producer = self.var_to_op.get(var_name)
            if producer is not None and producer != op_id:
                self.edges.append({
                    "sourceId": producer,
                    "targetId": op_id,
                    "tensorName": var_name,
                })

    def _make_op(self, call, outputs, in_place=False):
        kind, label, receiver = _call_target(call.func)
        # Resolve ``(_0).forward(x)`` patterns: if receiver is a local var
        # previously aliased to ``self.<attr>``, treat as a submodule call.
        if kind == "method" and receiver is not None \
                and receiver in self.submodule_alias:
            # e.g. ``_0.forward`` → callee becomes ``self.0``
            alias = self.submodule_alias[receiver]
            method_name = label.rsplit(".", 1)[-1]
            if method_name == "forward":
                kind = "submodule"
                label = alias
                receiver = None
            else:
                label = f"{alias}.{method_name}"
        op_type = _op_type_from_label(kind, label)
        inputs = list(_collect_arg_names(call))
        if kind == "method" and receiver is not None:
            # Receiver variable is the primary input.
            inputs.insert(0, receiver)

        op_id = self._next_id()
        op = {
            "id": op_id,
            "module": self.module_path,
            "opType": op_type,
            "callee": label,
            "kind": kind,
            "inputs": inputs,
            "outputs": outputs,
            "sourceLine": getattr(call, "lineno", None),
            "inPlace": in_place or op_type.endswith("_"),
        }
        self.ops.append(op)
        self._record_edges(op_id, inputs)
        for out in outputs:
            if out is not None:
                self.var_to_op[out] = op_id
        if op["inPlace"]:
            for inp in inputs:
                # in-place ops also "produce" their receiver
                if inp is not None:
                    self.var_to_op[inp] = op_id
        self.op_count_extracted += 1
        return op  # noqa: returned for callers that want to introspect

    # --- Statement handlers ---------------------------------------------

    def visit_Assign(self, node):
        # Alias: ``_0 = getattr(self, "0")`` or ``_0 = self.foo``.
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            alias = _extract_submodule_alias(node.value)
            if alias is not None:
                self.submodule_alias[name] = alias
                # Track inherited aliases (``_a = _0``).
                return
            if isinstance(node.value, ast.Name) \
                    and node.value.id in self.submodule_alias:
                self.submodule_alias[name] = self.submodule_alias[node.value.id]
                return

        # a = call(...), or (a, b) = call(...)
        if len(node.targets) == 1 and isinstance(node.value, ast.Call):
            call = node.value
            outputs = _targets_to_names(node.targets[0])
            self.op_count_total += 1
            self._make_op(call, outputs)
            return
        # Nested calls within more complex assignments still count.
        self._scan_nested_calls(node)

    def visit_AnnAssign(self, node):
        if isinstance(node.value, ast.Call):
            outputs = _targets_to_names(node.target)
            self.op_count_total += 1
            self._make_op(node.value, outputs)
            return
        self._scan_nested_calls(node)

    def visit_AugAssign(self, node):
        # x += y  -> synthetic in-place 'add_' op if target is a Name.
        if isinstance(node.target, ast.Name):
            op_id = self._next_id()
            target_name = node.target.id
            op_name = _AUGMENTED_OP_NAMES.get(type(node.op), "iadd") + "_"
            inputs = [target_name]
            if isinstance(node.value, ast.Name):
                inputs.append(node.value.id)
            op = {
                "id": op_id,
                "module": self.module_path,
                "opType": op_name,
                "callee": f"augassign.{op_name}",
                "kind": "augassign",
                "inputs": inputs,
                "outputs": [target_name],
                "sourceLine": node.lineno,
                "inPlace": True,
            }
            self.ops.append(op)
            self._record_edges(op_id, inputs)
            self.var_to_op[target_name] = op_id
            self.op_count_total += 1
            self.op_count_extracted += 1
            return
        self._scan_nested_calls(node)

    def visit_Expr(self, node):
        # Bare expression statement, possibly an in-place method call.
        if isinstance(node.value, ast.Call):
            call = node.value
            self.op_count_total += 1
            kind, label, receiver = _call_target(call.func)
            op_type = _op_type_from_label(kind, label)
            outputs = []
            if op_type.endswith("_") and kind == "method" and receiver is not None:
                outputs = [receiver]
            self._make_op(call, outputs, in_place=op_type.endswith("_"))

    def visit_Return(self, node):
        if isinstance(node.value, ast.Name):
            self.returned.add(node.value.id)
            return
        if isinstance(node.value, (ast.Tuple, ast.List)):
            for elt in node.value.elts:
                if isinstance(elt, ast.Name):
                    self.returned.add(elt.id)
            return
        # ``return <call>(...)`` — treat the call as a terminal op whose
        # synthetic output is named ``<module_path>.return``.
        if isinstance(node.value, ast.Call):
            self.op_count_total += 1
            synthetic = f"__return_{getattr(node, 'lineno', '0')}__"
            op = self._make_op(node.value, [synthetic])
            if op is not None:
                self.returned.add(synthetic)
            return
        # Other compound expressions: count nested calls as unparsed.
        self._scan_nested_calls(node)

    # --- Control flow / dynamic dispatch: mark partial ------------------

    def _count_calls_in(self, node, reason):
        call_count = sum(1 for n in ast.walk(node) if isinstance(n, ast.Call))
        if call_count:
            self.op_count_total += call_count
            self.op_count_unparsed += call_count
            self.partial = True
            self.warnings.append(
                f"{reason} at line {getattr(node, 'lineno', '?')}: "
                f"{call_count} call(s) not extracted"
            )

    def visit_For(self, node):
        self._count_calls_in(node, "for-loop")

    def visit_While(self, node):
        self._count_calls_in(node, "while-loop")

    def visit_If(self, node):
        self._count_calls_in(node, "if-branch")

    def visit_With(self, node):
        self._count_calls_in(node, "with-block")

    def visit_AsyncFor(self, node):
        self._count_calls_in(node, "async-for")

    def visit_Try(self, node):
        self._count_calls_in(node, "try-block")

    def visit_FunctionDef(self, node):
        # Nested def inside forward
        self._count_calls_in(node, "nested-def")

    def visit_AsyncFunctionDef(self, node):
        self._count_calls_in(node, "nested-async-def")

    def visit_Lambda(self, node):
        self._count_calls_in(node, "lambda")

    # --- Helpers --------------------------------------------------------

    def _scan_nested_calls(self, node):
        """When an assignment's RHS is not a bare Call (e.g. BinOp, Subscript,
        IfExp, list literals with calls inside), we can't build op nodes but
        must still count the calls for partial-rate accounting."""
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                self.op_count_total += 1
                self.op_count_unparsed += 1
                if not self.partial:
                    self.partial = True
                    self.warnings.append(
                        f"nested call in compound expression at line "
                        f"{getattr(node, 'lineno', '?')} not extracted"
                    )

    def scan_for_dynamic_dispatch(self, body):
        for n in ast.walk(ast.Module(body=body, type_ignores=[])):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) \
                    and n.func.id == "getattr":
                # ``getattr(self, "const")`` is resolvable statically; only
                # flag when arguments are dynamic.
                if _extract_submodule_alias(n) is not None:
                    continue
                self.partial = True
                self.warnings.append(
                    f"dynamic getattr() dispatch at line {n.lineno}"
                )


_AUGMENTED_OP_NAMES = {
    ast.Add: "add",
    ast.Sub: "sub",
    ast.Mult: "mul",
    ast.Div: "div",
    ast.FloorDiv: "floordiv",
    ast.Mod: "mod",
    ast.Pow: "pow",
    ast.MatMult: "matmul",
    ast.BitAnd: "bitand",
    ast.BitOr: "bitor",
    ast.BitXor: "bitxor",
    ast.LShift: "lshift",
    ast.RShift: "rshift",
}


def _class_forward(class_node):
    """Return the primary forward FunctionDef of a class, or None."""
    for item in class_node.body:
        if isinstance(item, ast.FunctionDef) and item.name == "forward":
            return item
    return None


def _class_init_children(class_node):
    """Extract submodule children from a ClassDef.

    Two common TorchScript patterns are recognized:

    1. ``self.<attr> = <ClassRef>(...)`` inside ``__init__``.
    2. ``__annotations__["<attr>"] = <ClassRef>`` at class body level, as
       emitted by ``torch.jit.script`` for ``nn.Sequential`` and friends.

    Returns dict of attr_name -> dotted class reference (str) or None when
    the RHS can't be resolved.
    """
    children = {}

    # Pattern 2: class-body ``__annotations__["name"] = Foo`` assignments.
    for stmt in class_node.body:
        if not isinstance(stmt, ast.Assign):
            continue
        if len(stmt.targets) != 1:
            continue
        target = stmt.targets[0]
        if isinstance(target, ast.Subscript) \
                and isinstance(target.value, ast.Name) \
                and target.value.id == "__annotations__":
            slice_node = target.slice
            if isinstance(slice_node, ast.Constant) \
                    and isinstance(slice_node.value, str):
                key = slice_node.value
                dotted = _dotted_name(stmt.value)
                children[key] = dotted

    # Pattern 3: class-level annotated attributes like
    # ``fc1 : __torch__.torch.nn.modules.linear.Linear``.
    _builtin_attrs = {
        "training", "__parameters__", "__buffers__",
        "_is_full_backward_hook", "_is_full_backward_hook_",
    }
    for stmt in class_node.body:
        if not isinstance(stmt, ast.AnnAssign):
            continue
        if not isinstance(stmt.target, ast.Name):
            continue
        attr = stmt.target.id
        if attr in _builtin_attrs or attr.startswith("__"):
            continue
        dotted = _dotted_name(stmt.annotation)
        # Only record when the annotation points to a torchscript class (a
        # dotted path, typically starting with ``__torch__``). Plain types
        # like ``Tensor`` or ``int`` are tensors/scalars, not submodules.
        if dotted is None:
            continue
        if "." not in dotted:
            continue
        if attr not in children:
            children[attr] = dotted

    # Pattern 1: ``__init__`` body.
    for item in class_node.body:
        if not (isinstance(item, ast.FunctionDef) and item.name == "__init__"):
            continue
        for stmt in ast.walk(item):
            if not isinstance(stmt, ast.Assign):
                continue
            if len(stmt.targets) != 1:
                continue
            target = stmt.targets[0]
            if not (isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"):
                continue
            attr = target.attr
            value = stmt.value
            if isinstance(value, ast.Call):
                dotted = _dotted_name(value.func)
                children[attr] = dotted
            else:
                children[attr] = None
    return children


def _index_classes(modules):
    """Build a mapping of short and dotted class names to their ClassDef.

    Multiple classes with the same short name may exist (torchscript name
    mangling); we keep the last one as a best effort.
    """
    by_short = {}
    by_dotted = {}
    for rel, tree in modules:
        # ``rel`` is already ``code``-relative (e.g. ``__torch__/MyNet.py``).
        mod_dotted = rel[:-3].replace("/", ".") if rel.endswith(".py") else rel
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                by_short[node.name] = (mod_dotted, node)
                by_dotted[f"{mod_dotted}.{node.name}"] = (mod_dotted, node)
    return by_short, by_dotted


def _resolve_child_class(class_ref, by_short, by_dotted):
    if class_ref is None:
        return None
    if class_ref in by_dotted:
        return class_ref
    tail = class_ref.rsplit(".", 1)[-1]
    if tail in by_short:
        mod, node = by_short[tail]
        return f"{mod}.{node.name}"
    return None


def _pick_root_class(by_short, by_dotted, zip_path):
    """Best-effort root class selection.

    Priority:
      1. Class referenced from ``data.pkl`` / ``archive/data.pkl`` via a
         pickletools shallow scan (GLOBAL/STACK_GLOBAL) - falls back silently.
      2. A class with no other class referencing it via __init__.
      3. The first class we found.
    """
    # 2: find classes that are not children of another class
    referenced = set()
    for mod, node in by_dotted.values():
        for _, class_ref in _class_init_children(node).items():
            resolved = _resolve_child_class(class_ref, by_short, by_dotted)
            if resolved:
                referenced.add(resolved)
    roots = [d for d in by_dotted.keys() if d not in referenced]
    # Also prefer classes that have a forward() method
    roots_with_forward = [
        d for d in roots if _class_forward(by_dotted[d][1]) is not None
    ]
    if roots_with_forward:
        return roots_with_forward[0]
    if roots:
        return roots[0]
    if by_dotted:
        return next(iter(by_dotted.keys()))
    return None


def _walk_module_tree(root_dotted, by_short, by_dotted):
    """Produce a flat module list starting from root_dotted.

    Each entry: {path, className, children: [attr,...]}. ``path`` is the
    submodule path (empty string for the root). Traversal is BFS with cycle
    guard.
    """
    if root_dotted is None or root_dotted not in by_dotted:
        return []
    modules = []
    seen_paths = set()
    # queue entries: (path, dotted_class_name)
    queue = [("", root_dotted)]
    while queue:
        path, dotted = queue.pop(0)
        if path in seen_paths:
            continue
        seen_paths.add(path)
        _, class_node = by_dotted[dotted]
        children_map = _class_init_children(class_node)
        child_attrs = sorted(children_map.keys())
        modules.append({
            "path": path,
            "className": dotted,
            "children": child_attrs,
        })
        for attr in child_attrs:
            child_ref = children_map[attr]
            resolved = _resolve_child_class(child_ref, by_short, by_dotted)
            if resolved is not None:
                new_path = attr if not path else f"{path}.{attr}"
                queue.append((new_path, resolved))
    return modules


def _count_storage_entries(zip_path):
    """Count ``data/<int>`` members (tensor storages) in the archive."""
    count = 0
    storage_keys = []
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                # PyTorch archives: '<archive>/data/0', '<archive>/data/1'...
                parts = name.rsplit("/", 2)
                if len(parts) >= 2 and parts[-2] == "data":
                    key = parts[-1]
                    if key.isdigit():
                        count += 1
                        storage_keys.append(key)
    except zipfile.BadZipFile:
        pass
    return count, storage_keys


def _visit_forward(class_node, module_path, class_dotted, op_id_counter):
    forward = _class_forward(class_node)
    if forward is None:
        return None
    visitor = _ForwardVisitor(module_path, class_dotted, op_id_counter)
    visitor.scan_for_dynamic_dispatch(forward.body)
    for stmt in forward.body:
        visitor.visit(stmt)
    return visitor


def build_payload(zip_path, partial_threshold=0.8):
    detection = None
    if detect_format is not None:
        try:
            detection = detect_format.detect(zip_path)
        except Exception:
            detection = None

    modules_parsed = _read_code_modules(zip_path)
    if not modules_parsed:
        raise RuntimeError("no code/**/*.py members found in archive")

    by_short, by_dotted = _index_classes(modules_parsed)
    if not by_dotted:
        raise RuntimeError("no class definitions found in code/")

    root_dotted = _pick_root_class(by_short, by_dotted, zip_path)
    module_entries = _walk_module_tree(root_dotted, by_short, by_dotted)

    # Walk every class with a forward() — even non-root ones, since they're
    # reachable submodules. Index their ops by module path.
    op_id_counter = [0]
    all_ops = []
    all_edges = []
    warnings = []
    op_total = 0
    op_extracted = 0
    op_unparsed = 0
    partial_any = False

    # For each entry in the module tree, find its class and visit forward.
    path_to_class = {e["path"]: e["className"] for e in module_entries}
    for entry in module_entries:
        dotted = entry["className"]
        if dotted not in by_dotted:
            continue
        _, class_node = by_dotted[dotted]
        visitor = _visit_forward(class_node, entry["path"], dotted, op_id_counter)
        if visitor is None:
            continue
        all_ops.extend(visitor.ops)
        all_edges.extend(visitor.edges)
        warnings.extend(visitor.warnings)
        op_total += visitor.op_count_total
        op_extracted += visitor.op_count_extracted
        op_unparsed += visitor.op_count_unparsed
        partial_any = partial_any or visitor.partial

    if op_extracted == 0:
        raise RuntimeError("parsed 0 operators from forward() methods")

    # Recovery rate check
    if op_total > 0:
        rate = op_extracted / op_total
        if rate < partial_threshold:
            partial_any = True
            warnings.append(
                f"recovery rate {rate:.2%} below threshold "
                f"{partial_threshold:.0%}: {op_extracted}/{op_total} ops"
            )

    # Tensor storage listing (offline — dtype/shape unknown)
    _, storage_keys = _count_storage_entries(zip_path)
    tensors = [
        {
            "path": f"data/{key}",
            "dtype": "unknown",
            "shape": [],
            "numel": 0,
            "bytes": 0,
            "storage": "torchscript-constant",
        }
        for key in sorted(storage_keys, key=lambda k: int(k))
    ]

    root_class_name = root_dotted.rsplit(".", 1)[-1] if root_dotted else "ScriptModule"

    reasons = [f"parsed {len(modules_parsed)} code/*.py file(s)"]
    det_subformat = "torchscript"
    det_warnings = []
    if detection is not None:
        reasons.extend(detection.get("reasons", []))
        det_warnings.extend(detection.get("warnings", []))
        det_subformat = detection.get("subFormat", "torchscript") or "torchscript"

    payload = {
        "producerVersion": "",
        "rootType": "ScriptModule",
        "loadMode": "torchscript_ast",
        "detection": {
            "subFormat": det_subformat,
            "reasons": reasons,
            "warnings": det_warnings,
        },
        "sections": [{
            "path": "root",
            "kind": "mapping",
            "summary": f"{len(module_entries)} submodule(s), {op_extracted} op(s)",
            "entryCount": len(module_entries),
        }],
        "metadata": [],
        "tensors": tensors,
        "torchscript": {
            "rootClass": root_class_name,
            "modules": module_entries,
            "ops": all_ops,
            "edges": all_edges,
            "partial": partial_any,
            "opCountTotal": op_total,
            "opCountExtracted": op_extracted,
            "opCountUnparsed": op_unparsed,
            "warnings": warnings,
        },
    }
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--partial-threshold", type=float, default=0.8)
    args = parser.parse_args()

    _install_timeout(max(5, args.timeout))

    detection_for_error = None
    if detect_format is not None:
        try:
            detection_for_error = detect_format.detect(args.path)
        except Exception:
            detection_for_error = None

    try:
        payload = build_payload(args.path, partial_threshold=args.partial_threshold)
    except Exception as exc:
        print(json.dumps({
            "error": f"torchscript AST inspection failed: {exc}",
            "detection": detection_for_error,
        }))
        return 3

    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
