# PT / TorchScript / safetensors 支持 · V3.1 实施计划

> **V3 是 V2 经 codex 第一轮深度审查后的重写。V3.1 在 codex 第二轮审查后应用了 5 处必改。**
>
> V2 → V3 的核心修复：
> - 架构守则在 `media/main.js` 三个位置已经泄露（L508 / L1599 / L989）→ 可验证契约测试
> - `torch.load` 在 `untrustedWorkspaces.supported: true` 声明下构成 RCE → safe_load + 用户确认
> - Phase G 在渲染层折叠无法降低 dagre 成本（真瓶颈在解析阶段）→ 剥离为独立项目
> - Phase H 的自定义 Unpickler 安全复杂度高 → 走官方 `torch.load(weights_only=True, mmap=True)`
> - 时间估算 11.5h 在加入正确的安全策略与契约测试后，实际约 25-35h
>
> V3 → V3.1（codex R2 必改）：
> - § 2.1 untrusted workspace 下即使用户同意也不 full load；consent 不循环弹窗
> - § 2.4 Windows `py -3` 必须拆成 command + args（`execFile` 契约）
> - § 4.1 格式检测加 `archive/data.pkl` / `safetensors-index` / safetensors schema 验证
> - § 1.2 ONNX 零增量契约用 AST import 解析，不用文件 hash
> - § 9.7 大 payload 真实降级路径（分页 / lazy node detail / 硬拒绝），不只告警

---

## 0. 目标与约束

### 目标
6 个改进（删除原 Phase G，剥离为独立项目）：
- **Phase -1**（新）：安全与架构前置（torch.load RCE 修复、三处泄露点修复、子进程 hardening）
- **B + D**（并行）：格式分发器 + safetensors 支持，互为多格式架构验证
- **C** 合并进 B（只是 subFormat UI + ONNX 导出向导）
- **E** 层类型推断补全
- **H** 懒加载（走官方 `torch.load(weights_only=True, mmap=True)`）
- **A** TorchScript 真实图（独立立项，1-2 天）

### 非目标
- **G 可折叠树**：从本计划剥离，另立 `docs/ONNX_LARGE_GRAPH_COLLAPSE_PLAN.md`（必须在 `layoutGraph` 之前做，否则不降低 dagre 成本——不与 PT 绑定）
- `.pt2` / ExecuTorch `.pte` 图解析
- 运行时 TorchScript 图执行

### 硬约束
1. **ONNX 性能 3 通道基线**，每通道回归 ≤ **15 %**（不是绝对 ms）：
   - extension-host 冷启动（require + `detectModelFormat` + `parseModelFile`）
   - webview 首帧（`modelData` 事件 → 首个 DOM 渲染完成）
   - 交互响应（`setTab` / `selectGraphNode` → DOM 更新完成）
2. **torch.load 默认 `weights_only=True`**；需要 full load 时强制用户显式确认。
3. **架构契约测试**：每个 phase 开始前的 regression gate——强制检测 codex 指出的 3 处泄露不能复现。
4. **torch-less CI 必须通过**：纯函数单测 / 格式检测 / safetensors 测试始终跑；`sample.pt` 集成测试条件跳过。

---

## 1. 架构守则（彻底重写）

### 1.1 V2 守则被证伪

codex 给出 3 个具体反例，证明"靠编程纪律"不可行：

| ID | 位置 | 问题 | V3 修复 |
|----|------|------|---------|
| L1 | [media/main.js:508](../media/main.js#L508) `detectParsedFormat` | 未知格式默认回落 `'onnx'` | 显式白名单，未知返回 `'unknown'`，render 端 switch 默认显示错误 |
| L2 | [media/main.js:1599](../media/main.js#L1599) `aria-label="ONNX graph visualization"` | 硬编码 ONNX 字样，PT 视图也用 | 按 `parsed.format` 动态生成 label |
| L3 | [media/main.js:989](../media/main.js#L989) `insightMarkup` | PT operator 走 ONNX 的 opset-based insight 渲染 | insight 构造器按 format 分派，PT 用模块层级解释 |

### 1.2 V3 新守则（机器可验证）

不再是 `.md` 里的宣言，而是写入 `test/run-tests.cjs` 的新 `runArchitectureContractTests()`：

1. **格式白名单**：`detectParsedFormat({format: 'foo'}) === 'unknown'`；未知 → render 显示错误卡片，**不**走 ONNX / PT 分支。
2. **反向注入测试**：构造一个 PT parsed 对象，调用 ONNX 专属的 `renderGraphSvg` / `renderOperatorInsight` → 断言产出标明 `unsupported-format` 错误而非静默渲染。
3. **ONNX 模块零增量**（codex R2 #4 修复）：不用文件 hash（会阻塞正常编辑）。改为**解析静态 import 列表**——用简单 regex 抽取 `^import .* from ['"](.+)['"]` 语句，与基线 import 白名单比对。白名单变更需 PR 显式更新。
4. **ONNX 渲染路径无 PT symbol 引用**：扫描 `media/main.js` 里所有 `renderXxxForOnnx`（改名后的）函数体，不能出现 `'pt'` 字面量或 `pt-*` node id 前缀。
5. **共享状态无跨格式泄漏**（codex R2 §3 第 3 条）：`state.graphSearch` / `state.selectedGraphNodeId` 等共享 state 字段在 format 切换时（打开不同文件）必须正确隔离。测试：先打开 ONNX 设 graphSearch="conv"，切到 PT 文件，断言 PT 渲染不携带 ONNX 的搜索词污染。

---

## 2. Phase -1 · 安全与架构前置（4-6 h · 必做）

### 2.1 torch.load RCE 修复

**现状**：[scripts/inspect_pt.py:19](../scripts/inspect_pt.py#L19) 无条件 `torch.load(target, map_location="cpu")`；[package.json:45-48](../package.json#L45-L48) 声明 `untrustedWorkspaces.supported: true`。恶意 `.pt` 文件可在打开时执行任意 Python 代码。

**修复流程**：

```python
# scripts/inspect_pt.py
def safe_load(path, allow_full=False):
    try:
        return torch.load(path, map_location="cpu", weights_only=True), "weights_only"
    except Exception as exc:
        if allow_full:
            return torch.load(path, map_location="cpu", weights_only=False), "full_load"
        return None, ("needs_consent", str(exc))
```

- 默认走 `weights_only=True`。
- 失败则 payload 标签 `requires_full_load: true` + 错误信息。
- `extension.js` 收到标签 → `vscode.window.showWarningMessage(...)` 弹窗："This checkpoint failed safe-load. It likely contains Python class references that require unsafe pickle deserialization. Continue only if you trust the source."`["Allow once", "Cancel"]`。
- **untrusted workspace 守门（codex R2 #1）**：只有当新增配置项 `onnxInspector.allowFullPickleLoad === true` 且**不**在 untrusted workspace 时，`--allow-full-load` 才会被传递给 Python。否则即使用户点 Allow，也只返回 "full-load disabled by policy"。
  ```js
  // extension.js
  const settings = vscode.workspace.getConfiguration('onnxInspector');
  const isTrusted = vscode.workspace.isTrusted;
  const allowFull = isTrusted && settings.get('allowFullPickleLoad', false);
  ```
- **不循环弹窗（codex R2 §3 第 2 条）**：每个 document URI 的 consent 状态存内存 `Map<uri, 'pending' | 'allowed' | 'denied'>`；user 拒绝后该 URI 不再弹。手动 revoke 需 Command Palette 的 "ONNX Inspector: Reset load consent"。
- 全流程记录到 VSCode output channel。

### 2.2 子进程 hardening

- `model-parser.js:40-42` `execFile` 加 `timeout: 120_000`（`maxBuffer: 8MB` 保留）。
- Python 子进程内加 `signal.alarm(120)` 做双保险（仅 POSIX；Windows 用 `threading.Timer`）。
- 文件大小 preflight：[extension.js:49](../extension.js#L49) 全量 `fs.readFileSync` 改为——`file.size > 500 * 1024 * 1024` 时不读 bytes 到 webview payload，只给 Python 子进程路径。
- **stderr 展示给用户**：当前 extension 吞掉 stderr，改为写入 output channel。

### 2.3 三处泄露点修复（L1/L2/L3）

见 1.1 表。具体 diff：

- **L1** [media/main.js:508](../media/main.js#L508)：
  ```js
  function detectParsedFormat(parsed) {
      const explicit = parsed?.format;
      if (explicit === 'onnx' || explicit === 'pt' || explicit === 'safetensors') return explicit;
      return 'unknown';  // 不再默认 'onnx'
  }
  ```
  所有 render 函数入口改为 `switch (format) { ... default: renderUnsupportedFormatCard(parsed); }`。

- **L2** [media/main.js:1599](../media/main.js#L1599)：
  ```js
  const ariaLabel = {
      onnx: 'ONNX graph visualization',
      pt: 'PyTorch module hierarchy',
      safetensors: 'Safetensors module hierarchy'
  }[format] || 'Model graph';
  ```

- **L3** [media/main.js:989](../media/main.js#L989)：`insightMarkup` 提取到 `renderOperatorInsight(format, node)`，按 format 分派。PT 用 "Inferred layer type: Conv2d (high confidence) — weight shape suggests 64 filters × 3 input channels × 3×3 kernel."（基于已推断的 opType + 形状），而不是 ONNX 的 opset 字典查询。

### 2.4 Windows Python 发现

`resolvePythonExecutable` ([model-parser.js:570](../model-parser.js#L570)) 默认 `'python3'`——Windows 上无效。

**codex R2 #2 修复**：`execFile` 只接受单个 command，**不能**传 `'py -3'` 作为一个字符串。必须把 command 和 args 分开：

```js
// 返回 [{command, prefixArgs}]，调用方 execFile(command, [...prefixArgs, ...realArgs])
function defaultPythonCandidates() {
    if (process.platform === 'win32') {
        return [
            {command: process.env.ONNX_INSPECTOR_PYTHON, prefixArgs: []},
            {command: process.env.CONDA_PYTHON_EXE, prefixArgs: []},
            {command: 'py', prefixArgs: ['-3']},
            {command: 'python', prefixArgs: []}
        ].filter((c) => c.command);
    }
    return [
        {command: process.env.ONNX_INSPECTOR_PYTHON, prefixArgs: []},
        {command: process.env.CONDA_PYTHON_EXE, prefixArgs: []},
        {command: 'python3', prefixArgs: []}
    ].filter((c) => c.command);
}
```
按顺序尝试 `<command> <...prefixArgs> --version`，取第一个退出码 0 的。

### 2.5 架构契约测试

新增 `test/run-tests.cjs` 的 `runArchitectureContractTests()` 段，放在所有 Phase 前跑。详见第 10 节。

### DoD
- 安全 smoke：`test/fixtures/malicious.pt`（合法 pickle 调用 `os.system("touch /tmp/pwned")`）加载后，**不**创建 `/tmp/pwned`；用户看到 dialog。
- 三条契约测试全部通过。
- Windows 手动验证（或在文档中标注需要）。

---

## 3. Phase 0 · 多通道性能基线（2 h）

### 3.1 维度

```
baseline.onnx_small  = { host_parse_ms, first_paint_ms, interaction_ms }   # test/sample.onnx (3 nodes)
baseline.onnx_large  = { ... }                                              # 条件性 fixture
baseline.pt_sample   = { ... }                                              # test/sample.pt
```

- 每个维度跑 10 次，去掉最高最低各 1 次，取中位数。
- 不提交机器相关 `test/.perf-baseline.json`——改用 **same-job before/after** 模式：

```
npm run perf:baseline     # 在 main 分支上记录
npm run perf:compare      # 在当前改动上对比
```

基线数据缓存到 `.perf-cache/<branch>-<sha>.json`（加到 `.gitignore`）。

### 3.2 大 ONNX fixture

- 加 `scripts/generate_large_onnx.py`：生成 ~300 节点 ONNX（纯 ONNX Python API，不依赖 torch）。
- fixture 落到 `test/fixtures/large.onnx`，**不进 VSIX 打包**（加到 `.vscodeignore`）。
- 测试条件性运行：`if (fs.existsSync(fixture)) { ... }`；否则打印 warning 跳过。

### 3.3 预算
- smoke（3 节点）**不作为 gate**——噪声太大，只打印结果。
- 大 ONNX 每通道回归 > 15 % 则 `assert.fail`。
- 交互响应 gate：任何 < 50 ms 的 baseline 绝对值不作为 regression 源（避免抖动假阳）。
- **CI 抖动对抗**（codex R2 §14 第 6 条）：每次对比跑 **3 次**，只要 2 次超阈值才判定回归（去掉单次抖动假阳）。

---

## 4. Phase B + D · 格式分发与 safetensors（并行 · 3-5 h + 2 h）

### 4.1 Phase B：格式分发矩阵

**不**依赖 `torch.load` 做检测（codex R1 #2）。新建 `scripts/detect_format.py`（纯标准库）：

| 检测条件 | subFormat 标签 |
|---------|---------------|
| 前 8 bytes u64 LE header_len ∈ [2, 100MB] **且** header 是合法 JSON **且** 顶层 keys 全部匹配 `{tensor-name → {dtype, shape, data_offsets}}` schema（单独一个 `__metadata__` 例外） | `safetensors` |
| 文件名 `*.safetensors.index.json` 或内容是 `{"metadata":...,"weight_map":...}` | `safetensors-index`（告知；指向分片文件） |
| `PK\x03\x04` + zip 含 `code/` 目录（含 `archive/code/` 前缀变体） | `torchscript` |
| `PK\x03\x04` + zip 含 `*.pt2` / `serialized_model.json` / `pt2_archive_format` | `exported-program` |
| `PK\x03\x04` + zip 含 `data.pkl` 或 `archive/data.pkl` + `pickletools` 扫出 `torch.nn.modules` GLOBAL 引用 | `full-model-pickle` |
| `PK\x03\x04` + `data.pkl` + 扫出 `pytorch_lightning_version` key | `lightning-checkpoint` |
| `PK\x03\x04` + `data.pkl` + 扫出 `optimizer` top-level key | `optimizer-checkpoint` |
| `PK\x03\x04` + `data.pkl` 存在但无特殊标记 | `plain-state-dict` |
| 文件名 `pytorch_model.bin` + `PK\x03\x04` | `plain-state-dict`（HF 惯例） |
| 前 2 bytes `\x80\x02..\x80\x05`（pickle proto，非 zip） | `legacy-pickle` |
| 其他 | `unknown` |

**codex R2 #2 修复的关键细节**：
- `archive/` 前缀：PyTorch ≥ 2.0 的 zip 结构把 pickle 放在 `archive/data.pkl`（而不是根 `data.pkl`）。检测必须同时查两种路径。
- safetensors schema 验证：header length 只过 size 关还不够——必须做 JSON schema check，避免任何前 8 bytes 凑巧解析成合法 u64 的文件被误判（codex R2 §3 第 4 条）。
- `safetensors-index`：告知用户这是 HF 分片索引，指向实际 `.safetensors` 分片文件，本扩展暂不聚合分片（属于 F-后续）。

**pickle 浅扫实现**：`pickletools.genops(stream)` 迭代 opcode。识别 `c` (GLOBAL)。`STACK_GLOBAL` (`\x93`) 需要简化的栈模拟——跟踪最近两个 `SHORT_BINUNICODE` / `SHORT_BINSTRING` 字符串作为 module/class 名。**但仅作为启发式标签**（codex R2 §14 第 8 条），不作为信任边界——真正的 RCE 防护仍是 `weights_only=True`。

### 4.2 Phase D：safetensors

新建 `scripts/inspect_safetensors.py`（纯标准库）：

```python
import json, struct, sys
with open(sys.argv[1], "rb") as fh:
    header_len = struct.unpack("<Q", fh.read(8))[0]
    if header_len > 100 * 1024 * 1024:  # 100 MB cap (codex #16)
        raise ValueError(f"safetensors header too large: {header_len}")
    header = json.loads(fh.read(header_len))
# 转 tensors + metadata，输出 JSON
```

### 4.3 归一化重构

**修复 codex 指出的 [model-parser.js:86-88](../model-parser.js#L86-L88) `format: 'pt'` 硬编码**：

- 当前 `normalizePtInspection(inspected, fileSizeBytes)` → 重构为 `normalizeTensorInspection(inspected, fileSizeBytes, { format, subFormat })`。
- `format: 'pt' | 'safetensors'` 由调用方传入。
- `summary.graphName` 动态：`inspected.rootType || (format === 'safetensors' ? 'safetensors tensors' : 'PyTorch checkpoint')`。
- `producerName` 同理动态。

### 4.4 合并 Phase C（ONNX 导出向导）

在 `renderGraphTab` 里：

```js
if (format === 'pt' && ['plain-state-dict', 'full-model-pickle',
                        'lightning-checkpoint', 'optimizer-checkpoint'].includes(subFormat)) {
    prepend(renderOnnxExportWizard(parsed));
}
```

`renderOnnxExportWizard` 显示代码片段 + 复制按钮。剪贴板实现：

- 优先 `navigator.clipboard.writeText`（现代 webview 有 user activation 时可用）。
- fallback `postMessage({cmd: 'copyToClipboard', text: ...})` → extension.js 调 `vscode.env.clipboard.writeText`（绕过 CSP）。
- **codex R2 §3 第 3 条**：命令名必须和 extension.js 里已有的消息 dispatch 保持一致；测试里加一个断言验证 `cmd` 字段值与 `extension.js` 的 handler 匹配，避免手滑把 `copyToClipboard` 写成 `copyText`。

### 4.5 扩展契约点（codex #15 "D 不止 1h"）

D 涉及：
- `package.json` customEditors / menus / activationEvents — 4 处改动
- `extension.js` file open dialog filters
- `test/run-tests.cjs` 新 safetensors 集成测试 + 扩展契约测试
- `normalizePtInspection` 重构为多格式

### 4.6 为什么并行
D 触及四个扩展契约点，B 只改 parser dispatch。并行做**暴露"加一种新格式需要改多少处"的真实成本**，是多格式架构验证的核心。

---

## 5. Phase E · 层推断补全（1.5 h）

扩展 `inferPtLayerType` 规则，每条带 `confidence: 'high' | 'medium' | 'low'`：

| 触发条件 | opType | confidence |
|---------|--------|-----------|
| `in_proj_weight` + `out_proj.weight` 同父 | `MultiheadAttention` | high |
| `q_proj.weight` + `k_proj.weight` + `v_proj.weight` 同父 | 聚合为 `AttentionBlock`（子 linear 保留） | high |
| `weight.shape[1] == 1` + rank 4 | `DepthwiseConv2d` | medium（groups 无法从 state_dict 恢复） |
| rank 1 weight + bias，无 running_*，模块名含 `ln`/`layer_norm` | `LayerNorm` | high |
| rank 1 weight + bias，无 running_*，模块名含 `group` | `GroupNorm` | medium |
| rank 1 weight + bias，无 running_*，其他 | `LayerNorm` | medium + warning "Could also be GroupNorm/InstanceNorm" |

每条规则附带测试注释，说明"如何 falsify"（即什么反例会 break 这条推断）。

UI：operator 卡片 subtitle 后在 medium/low confidence 时加 `ⓘ` 图标，hover 显示"inferred from: weight [C], bias [C]; could also be GroupNorm"。

---

## 6. Phase H · 懒加载 via 官方 API（2 h）

**采纳 codex 方案**（第 5 条反馈），放弃 V2 的自定义 Unpickler。

### 6.1 实施

**codex R2 #4 修复**：不按 torch 版本号 string 解析（`"2.1.0+cu121"` 会炸）。改为**特性探测**：

```python
# scripts/inspect_pt.py
def lazy_load(path):
    try:
        # 特性探测：直接试 mmap 参数是否被接受
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError as exc:
        if "mmap" in str(exc):
            return torch.load(path, map_location="cpu", weights_only=True)  # 旧 torch
        raise
    except RuntimeError as exc:
        # mmap 对非 zip checkpoint / legacy pickle 不支持
        if "mmap" in str(exc).lower():
            return torch.load(path, map_location="cpu", weights_only=True)
        raise
```

- 文件 > 500 MB 时启用 lazy。
- `mmap=True` 失败时自动回退，不破坏用户体验。
- safetensors 天然 lazy（只读 header）。

### 6.2 测试
- 用 `resource.getrusage(RUSAGE_CHILDREN)` 测 Python 子进程峰值 RSS。
- 500 MB fixture 的 RSS 峰值应 < 100 MB（mmap 有效时）。
- 无 mmap 支持时打印 warning 但不失败。

---

## 7. Phase A · TorchScript 真实图（独立立项 · 1-2 天）

### 7.1 Scope 校正（codex #3）

V2 承诺 3-4 小时，低估了静态 AST 匹配的边角：
- tuple 解包：`x, y = torch.split(...)` → 多输出绑定
- 返回 tuple/list/dict
- namespaces：`prim::*`、`ops.aten.*`、`torch.*`、`torch.nn.functional.*`
- in-place ops：`x.add_(y)` → 输入也是输出
- 动态模块：`getattr(self, name)(x)` → 必须查 `__init__` 解析
- 控制流：`for i in range(...): y = self.layers[i](y)` → 循环展开或标记"dynamic"
- 生成的辅助方法：`forward__0`、`__torch__.MyNet.forward_1`

### 7.2 验收（防止"authoritative-looking 错图"）

1. **假阴性率**：准备 8-10 个开源 TorchScript 模型（ResNet18 / MobileNetV2 / BERT-tiny / LSTM / Seq2Seq / Transformer encoder / UNet / YOLOv5 等），通过 `torch.jit.load(...).graph.str()` 获得 ground-truth op 数。我们静态 AST 提取出的 op 数 ≥ 80% 即通过。
2. **部分解析必须显式标注**：任何解析出的 graph 在 UI 上显示 `Partial parse: 87 ops recovered, ~23 may be missing (see: loops / conditionals / dynamic dispatch detected)`。**绝不静默展示**。
3. **硬降级**：解析失败 / ops < 30% → fallback 到模块层级树，UI 显示黄色 banner "TorchScript AST could not be fully parsed. Showing module hierarchy instead."

### 7.3 排在最后

- 前面几个 phase 先稳定多格式架构。
- A 本身复杂度高；做错误导比没有更糟。
- 独立立项允许 A 的工期放宽而不 block 其他 phase。

---

## 8. Phase G（已剥离）

原因：codex #4——ONNX 大图 layout 成本发生在 [media/lib/onnx-parser.mjs:53-62](../media/lib/onnx-parser.mjs#L53-L62) 的 `buildGraphView` + `:1174-1221` 的 `dagreCompactLayout`，都在 webview 渲染**之前**。渲染侧折叠 DOM 不降低 dagre 成本。

**剥离后的独立项目**要点（不在本计划）：
- 在 `buildGraphView` 里按前缀聚合节点（ONNX 常见 `/block0/conv1/Conv`）。
- 聚合阈值 + 交互展开需要独立 UX 设计。
- 独立性能基线（大 ONNX fixture）。
- 独立于本计划的 PR。

---

## 9. 安全与可移植性（贯穿全计划）

### 9.1 torch.load 安全
见 Phase -1 (2.1)。

### 9.2 文件大小
见 Phase -1 (2.2)。

### 9.3 子进程 hang
见 Phase -1 (2.2)。

### 9.4 Windows Python 发现
见 Phase -1 (2.4)。

### 9.5 webview 剪贴板 / CSP
见 Phase C (4.4)。

### 9.6 `retainContextWhenHidden` 内存
[extension.js:40](../extension.js#L40) 声明 `retainContextWhenHidden: true` → 每个打开的 webview 常驻内存。大模型下是泄漏风险。

**修复**：加配置项 `onnxInspector.retainContextWhenHidden`（默认 true 保持现状），文档说明"打开多个大模型时可关闭"。不默认改 true→false（会影响 ONNX 用户体验）。

### 9.7 webview payload 大小（codex R2 #5 修复）

**"只告警不降级"不够**（codex R2 §3 第 5 条：webview 会 freeze）。真实降级路径：

- parsed.initializers / parsed.metadata / parsed.graphMetadata 超过阈值（> 5000 条）→ 分页：只发前 500 + 尾 500，UI 显示 "Showing 1000 of N items; use filter to find specific entries."
- parsed.graphView.displayNodes 超过 2000 → 发送 summary 模式（只含节点 id + 位置，节点细节按需通过 `postMessage({type: 'requestNodeDetail', id})` 惰性请求）。
- payload serialized size > 50 MB → 硬拒绝，UI 显示 "Model too large to render; extension will upgrade streaming support in a future release." + 引导用户用命令行工具（或考虑 G 项目的聚合折叠）。
- 阈值加到 package.json 配置项（用户可调）。

---

## 10. 测试策略（分层）

### 10.1 分层表

| 层 | 依赖 | 何时跑 | 示例 |
|----|------|-------|------|
| 架构契约 | 无 | **每次** `npm test` | `runArchitectureContractTests` |
| 纯函数单测 | 无 | **每次** | `normalizeTensorInspection`、`inferPtLayerType`、`splitTensorPath` |
| 格式检测 | fixture bytes | **每次** | `detect_format.py` 对各 fixture 返回预期 subFormat |
| safetensors 集成 | 无 | **每次** | 纯 stdlib |
| PT 集成 | torch | **条件** | `sample.pt`；无 torch 打印 skip |
| 性能基线 | fixture | **条件** | 大 ONNX 存在时跑 |
| 安全 smoke | 无（fixture 是恶意 pickle） | **每次** | `malicious.pt` 加载后断言 `/tmp/pwned` 不存在 |

### 10.2 Fixture 管理

- `scripts/generate_fixtures.py`：全部 fixture 生成代码集中管理。
- 有 torch：CI / 开发者本地可重新生成。
- 无 torch：用 git 跟踪的静态 bytes 作为预生成 fixture。
- fixtures 目录：`test/fixtures/`（加 `.vscodeignore`，不进 VSIX）。

### 10.3 架构契约测试样例

```js
function runArchitectureContractTests() {
    // L1: unknown → 'unknown'，不是 'onnx'
    assert.equal(detectParsedFormat({format: 'foo'}), 'unknown');

    // L2: aria-label 动态
    assert.match(renderGraphSvg({format: 'pt', ...mockPt}), /aria-label="PyTorch module hierarchy"/);

    // L3: PT insight 不走 ONNX 路径
    const ptInsight = renderOperatorInsight('pt', {details: {opType: 'Conv2d'}});
    assert.doesNotMatch(ptInsight, /opset|ONNX/i);
    assert.match(ptInsight, /inferred|module/i);

    // 反向注入：PT parsed 走 ONNX 专属 render → 错误卡片
    const wrong = renderOnnxGraphSvg({format: 'pt'});
    assert.match(wrong, /unsupported-format/);
}
```

---

## 11. 风险矩阵（修订）

| # | 风险 | 概率 | 影响 | 缓解 |
|---|------|------|------|------|
| 1 | torch.load RCE 在 untrusted workspace | 中 | **极高** | Phase -1 首要修复 |
| 2 | ONNX 渲染路径被 PT 代码意外污染 | 高 | 高 | 架构契约测试 + 显式白名单 |
| 3 | TorchScript AST 产出误导性"完整图" | 高 | 高 | Phase A 假阴性率 + 显式 partial 标注 + 硬降级 |
| 4 | `weights_only=True` 对 pre-1.5 checkpoint 失败 | 中 | 中 | 用户确认 fallback |
| 5 | Windows python 发现失败 | 中 | 中 | Phase -1 平台 candidates |
| 6 | 大文件 OOM（extension.js full read） | 中 | 中 | 500 MB preflight + 路径传递 |
| 7 | 子进程 hang | 低 | 中 | `execFile` timeout + Python `signal.alarm` |
| 8 | webview 剪贴板被 CSP 禁 | 中 | 低 | postMessage → host clipboard |
| 9 | safetensors header 过大 | 低 | 低 | 100 MB cap |
| 10 | 无 torch 的 CI 失败 | 高 | 中 | 条件 skip + 静态 fixture |
| 11 | Phase A 时间严重超支 | 高 | 中 | 独立立项，不 block 其他 phase |
| 12 | `retainContextWhenHidden` 多模型 OOM | 低 | 中 | 配置项 + 文档 |
| 13 | pickle 浅扫误判格式 | 中 | 低 | 多重验证（zip 目录 + 浅扫 + 白名单） |

---

## 12. 工时估算（修订）

| Phase | V2 估计 | V3 估计 | 说明 |
|-------|--------|--------|------|
| **-1** 安全+架构前置 | — | 4-6 h | codex 发现的核心缺口 |
| **0** 基线 | 0.5 h | 2 h | 3 通道 + 大 ONNX fixture |
| **B+D** 分发+safetensors | 2 h + 1 h | 3-5 h + 2 h | 并行，含扩展契约点 |
| **C** 向导 | 0.5 h | 合并进 B | 0 |
| **E** 层推断 | 1 h | 1.5 h | 含 confidence |
| **H** 懒加载 | 1.5 h | 2 h | 官方 API |
| **A** TorchScript | 3.5 h | **1-2 天**（独立） | 最后做 |
| **G** 折叠 | 2 h | **剥离** | 另立项 |
| **合计（不含 A / G）** | 6 h | **15-19 h** | 真实工时 |
| **含 A** | 11.5 h | **25-35 h** | |

---

## 13. 执行顺序

```
Phase -1 (安全+架构) ──┐
                       ├─→ Phase 0 基线 ──→ [Phase B+D 并行] ──→ Phase E ──→ Phase H ──→ Phase A
                       │
                       └─→ 每 phase 入口跑架构契约测试（无新泄露）
```

- 严格串行：-1 → 0 → B+D → E → H → A。
- 每个 phase 合入前必须通过架构契约测试 + 性能基线。
- Phase A 独立立项，完成时机不 block 前面的发布。
- Phase G 另立计划，与本文档无依赖。

---

## 14. 交给 codex 的第二轮审查重点

1. Phase -1 的 `weights_only=True` fallback 流程是否还有漏洞（比如 dialog 可绕过）？
2. 格式检测矩阵（4.1）是否覆盖真实世界 95% checkpoint？漏了 `torch.package` / `onnx_torch_interop` / `safetensors.Index` 吗？
3. 架构契约测试（10.3）是否真防住未来侵入？还是只检查"显式错误"漏了"静默降级"？
4. Phase H 的 `weights_only=True + mmap=True` 组合在 torch 2.1/2.2/2.3/2.4/2.5/2.6 上是否都可靠？有版本雷区吗？
5. Phase A 的假阴性率 ≥ 80% 阈值合理吗？选哪些开源 TorchScript 模型做 ground truth？
6. 15% 相对回归预算 + same-job before/after 模式是否足够可靠？CI 机器抖动怎么办？
7. V3 总工时 15-19 h（不含 A）是否仍然乐观？哪个 phase 最可能再次超支？
8. pickle 浅扫 (`pickletools.genops`) 判断 `full-model-pickle` 可靠吗？有没有恶意构造绕过的方法？
9. 架构守则里"ONNX 模块零增量 import 白名单"是否实施得太严（阻塞正常演进）？还是刚好？
10. 剥离 Phase G 是对的决定吗？还是应该合并进本计划？

---

## 附录 · 文件改动清单（V3）

```
scripts/
  detect_format.py            [新] Phase B，纯标准库格式检测
  inspect_pt.py               [改] safe_load + timeout + signal.alarm
  inspect_safetensors.py      [新] Phase D
  generate_fixtures.py        [新] 所有 fixture 生成
  generate_large_onnx.py      [新] Phase 0 大 ONNX fixture

model-parser.js               [改] 格式分发 + normalizeTensorInspection + Windows python + timeout
extension.js                  [改] file.size preflight + 用户确认 dialog + clipboard handler + stderr channel
media/main.js                 [改] detectParsedFormat 白名单 + aria-label 动态 + insight 分派 + 向导卡片
media/styles.css              [改] 向导卡片 + 部分解析 banner + confidence 图标

test/run-tests.cjs            [改] 架构契约测试 + 性能基线 + 分层 + 条件 skip
test/fixtures/
  malicious.pt                [新] 安全 smoke
  plain_state_dict.pt         [新]
  torchscript_linear.pt       [新]
  attention_sample.pt         [新]
  tiny.safetensors            [新]
  large.onnx                  [新 · 本地生成，不进 VSIX]

package.json                  [改] .safetensors + timeout 配置项 + retainContextWhenHidden 配置项
.vscodeignore                 [改] 排除 test/fixtures/large.onnx 和 .perf-cache/

docs/PT_SUPPORT_PLAN_V2.md    [本文档，V3 内容]
docs/ONNX_LARGE_GRAPH_COLLAPSE_PLAN.md  [新，Phase G 独立立项]
```
