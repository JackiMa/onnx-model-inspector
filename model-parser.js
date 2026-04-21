const fs = require('node:fs');
const path = require('node:path');
const { execFile } = require('node:child_process');
const { promisify } = require('node:util');
const { pathToFileURL } = require('node:url');

const execFileAsync = promisify(execFile);

const SUBPROCESS_TIMEOUT_MS = 120 * 1000;
const SUBPROCESS_MAX_BUFFER = 16 * 1024 * 1024;

const FORMAT_WHITELIST = new Set(['onnx', 'pt', 'safetensors']);

function detectModelFormat(filePath) {
    const ext = path.extname(filePath || '').toLowerCase();
    if (ext === '.onnx') {
        return 'onnx';
    }
    if (ext === '.pt' || ext === '.pth') {
        return 'pt';
    }
    if (ext === '.safetensors') {
        return 'safetensors';
    }
    return 'unknown';
}

function isSupportedFormat(format) {
    return FORMAT_WHITELIST.has(format);
}

async function parseModelFile(options) {
    const { filePath, bytes, extensionPath } = options;
    const format = detectModelFormat(filePath);
    if (format === 'onnx') {
        const parserPath = path.join(extensionPath, 'media', 'lib', 'onnx-parser.mjs');
        const parser = await import(pathToFileURL(parserPath).href);
        const parsed = parser.parseOnnxBytes(bytes);
        if (parsed && !parsed.format) {
            parsed.format = 'onnx';
        }
        return { ok: true, parsed, format };
    }
    if (format === 'pt') {
        const pythonCommand = await resolvePythonCommand(options);
        const subFormat = await detectPtSubFormat(filePath, extensionPath, pythonCommand).catch(() => null);
        const sizeBytes = bytes.length || fileSizeFromDisk(filePath);

        if (subFormat === 'torchscript') {
            const tsInspected = await inspectTorchScriptFile(filePath, extensionPath, pythonCommand).catch((err) => ({
                error: err?.message || String(err)
            }));
            if (tsInspected && !tsInspected.error && tsInspected.torchscript && tsInspected.torchscript.opCountExtracted > 0) {
                return {
                    ok: true,
                    format,
                    parsed: normalizePtInspection(tsInspected, sizeBytes, { format: 'pt', subFormat: 'torchscript' })
                };
            }
            // fall through to inspect_pt.py, with a warning embedded
        }

        const inspected = await inspectPtFile(filePath, extensionPath, pythonCommand, {
            allowFullLoad: Boolean(options.allowFullLoad)
        });
        if (inspected?.requiresFullLoad) {
            return {
                ok: false,
                format,
                requiresFullLoad: true,
                safeLoadError: inspected.safeLoadError || 'safe load failed',
                producerVersion: inspected.producerVersion || ''
            };
        }
        return { ok: true, parsed: normalizePtInspection(inspected, sizeBytes, { format: 'pt' }), format };
    }
    if (format === 'safetensors') {
        const pythonCommand = await resolvePythonCommand(options);
        const inspected = await inspectSafetensorsFile(filePath, extensionPath, pythonCommand);
        const sizeBytes = bytes.length || fileSizeFromDisk(filePath);
        return { ok: true, parsed: normalizePtInspection(inspected, sizeBytes, { format: 'safetensors' }), format };
    }
    throw new Error(`Unsupported model format for ${filePath}`);
}

function fileSizeFromDisk(filePath) {
    try {
        return fs.statSync(filePath).size;
    } catch {
        return 0;
    }
}

async function detectPtSubFormat(filePath, extensionPath, pythonCommand) {
    const scriptPath = path.join(extensionPath, 'scripts', 'detect_format.py');
    const { command, prefixArgs } = normalizeCommandSpec(pythonCommand);
    const { stdout } = await execFileAsync(command, [...prefixArgs, scriptPath, filePath], {
        cwd: extensionPath,
        maxBuffer: 1 * 1024 * 1024,
        timeout: 30 * 1000,
        killSignal: 'SIGKILL'
    });
    const result = JSON.parse(stdout);
    return result?.subFormat || null;
}

async function inspectTorchScriptFile(filePath, extensionPath, pythonCommand) {
    const scriptPath = path.join(extensionPath, 'scripts', 'inspect_torchscript.py');
    const { command, prefixArgs } = normalizeCommandSpec(pythonCommand);
    try {
        const { stdout } = await execFileAsync(command, [...prefixArgs, scriptPath, filePath, '--timeout', '110'], {
            cwd: extensionPath,
            maxBuffer: SUBPROCESS_MAX_BUFFER,
            timeout: SUBPROCESS_TIMEOUT_MS,
            killSignal: 'SIGKILL'
        });
        const parsed = JSON.parse(stdout);
        if (parsed.error) {
            throw new Error(parsed.error);
        }
        return parsed;
    } catch (error) {
        const stderr = error?.stderr ? String(error.stderr).trim() : '';
        const detail = stderr || error?.message || String(error);
        const err = new Error(`torchscript inspection failed: ${detail.slice(0, 400)}`);
        err.stderr = stderr;
        throw err;
    }
}

async function inspectSafetensorsFile(filePath, extensionPath, pythonCommand) {
    const scriptPath = path.join(extensionPath, 'scripts', 'inspect_safetensors.py');
    const { command, prefixArgs } = normalizeCommandSpec(pythonCommand);
    try {
        const { stdout } = await execFileAsync(command, [...prefixArgs, scriptPath, filePath, '--timeout', '25'], {
            cwd: extensionPath,
            maxBuffer: SUBPROCESS_MAX_BUFFER,
            timeout: SUBPROCESS_TIMEOUT_MS,
            killSignal: 'SIGKILL'
        });
        const parsed = JSON.parse(stdout);
        if (parsed.error) {
            throw new Error(parsed.error);
        }
        return parsed;
    } catch (error) {
        const stderr = error?.stderr ? String(error.stderr).trim() : '';
        const stdout = error?.stdout ? String(error.stdout).trim() : '';
        const detail = stderr || stdout || error?.message || String(error);
        const err = new Error(`safetensors inspection failed: ${detail.slice(0, 400)}`);
        err.stderr = stderr;
        err.stdout = stdout;
        throw err;
    }
}

const LAZY_LOAD_THRESHOLD_BYTES = 500 * 1024 * 1024;

async function inspectPtFile(filePath, extensionPath, pythonCommand, { allowFullLoad = false } = {}) {
    const scriptPath = path.join(extensionPath, 'scripts', 'inspect_pt.py');
    const scriptArgs = [scriptPath, filePath, '--timeout', '110'];
    if (allowFullLoad) {
        scriptArgs.push('--allow-full-load');
    }
    if (fileSizeFromDisk(filePath) > LAZY_LOAD_THRESHOLD_BYTES) {
        scriptArgs.push('--lazy');
    }
    const { command, prefixArgs } = normalizeCommandSpec(pythonCommand);
    try {
        const { stdout } = await execFileAsync(command, [...prefixArgs, ...scriptArgs], {
            cwd: extensionPath,
            maxBuffer: SUBPROCESS_MAX_BUFFER,
            timeout: SUBPROCESS_TIMEOUT_MS,
            killSignal: 'SIGKILL'
        });
        return JSON.parse(stdout);
    } catch (error) {
        const stderr = error?.stderr ? String(error.stderr).trim() : '';
        const stdout = error?.stdout ? String(error.stdout).trim() : '';
        const detail = stderr || stdout || error?.message || String(error);
        const summary = stderr ? `${stderr.split('\n').slice(0, 5).join(' | ')}` : detail.slice(0, 400);
        const err = new Error(`PT inspection failed: ${summary}`);
        err.stderr = stderr;
        err.stdout = stdout;
        throw err;
    }
}

function normalizeCommandSpec(spec) {
    if (!spec) {
        return { command: 'python3', prefixArgs: [] };
    }
    if (typeof spec === 'string') {
        return { command: spec, prefixArgs: [] };
    }
    return {
        command: spec.command,
        prefixArgs: Array.isArray(spec.prefixArgs) ? spec.prefixArgs : []
    };
}

function normalizePtInspection(inspected, fileSizeBytes, context = {}) {
    const format = context.format || 'pt';
    const subFormat = context.subFormat
        || inspected?.detection?.subFormat
        || (format === 'safetensors' ? 'safetensors' : 'plain-state-dict');
    const tensors = (inspected?.tensors || []).map((item) => ({
        name: item.path,
        dataTypeName: item.dtype || 'unknown',
        shape: Array.isArray(item.shape) ? item.shape : [],
        shapeSummary: formatShape(item.shape),
        elementCount: Number.isFinite(item.numel) ? item.numel : 0,
        estimatedBytesValue: Number.isFinite(item.bytes) ? item.bytes : 0,
        estimatedBytesKnown: Number.isFinite(item.bytes),
        estimatedBytes: Number.isFinite(item.bytes) ? formatBytes(item.bytes) : 'Unknown',
        storage: { summary: item.storage || 'tensor' },
        externalData: [],
        metadata: []
    }));

    const metadata = (inspected?.metadata || []).map((item) => ({
        key: item.path,
        value: stringifyScalar(item.value),
        valuePreview: stringifyScalar(item.value),
        isJson: false,
        jsonValue: null
    }));

    const sections = (inspected?.sections || []).map((section, index) => ({
        id: `section:${index}:${section.path}`,
        path: section.path,
        kind: section.kind || 'unknown',
        summary: section.summary || '',
        entryCount: Number.isFinite(section.entryCount) ? section.entryCount : null
    }));

    const rootType = inspected?.rootType || (format === 'safetensors' ? 'safetensors' : 'checkpoint');
    const producerName = format === 'safetensors' ? 'safetensors' : 'PyTorch';

    const torchscript = inspected?.torchscript || null;
    const graphView = (torchscript && Array.isArray(torchscript.ops) && torchscript.ops.length > 0)
        ? buildTorchScriptGraphView(torchscript, rootType)
        : buildPtGraphView(tensors, rootType);

    return {
        format,
        subFormat,
        torchscript: torchscript ? {
            rootClass: torchscript.rootClass || '',
            partial: Boolean(torchscript.partial),
            opCountTotal: torchscript.opCountTotal || 0,
            opCountExtracted: torchscript.opCountExtracted || 0,
            opCountUnparsed: torchscript.opCountUnparsed || 0,
            warnings: Array.isArray(torchscript.warnings) ? torchscript.warnings : []
        } : null,
        summary: {
            graphName: torchscript?.rootClass || rootType,
            irVersion: '',
            irVersionLabel: '',
            producerName,
            producerVersion: inspected?.producerVersion || '',
            subFormat,
            domain: '',
            modelVersion: '',
            docString: '',
            opsets: [],
            fileSizeBytes,
            fileSizeLabel: formatBytes(fileSizeBytes),
            temporalMetadata: { primary: null, candidates: [] },
            stats: {
                nodeCount: graphView.stats.operatorNodeCount,
                inputCount: 0,
                outputCount: 0,
                valueInfoCount: 0,
                initializerCount: tensors.length,
                metadataCount: metadata.length,
                graphMetadataCount: sections.length,
                externalTensorCount: 0,
                estimatedParameterBytes: tensors.reduce((sum, item) => sum + (item.estimatedBytesValue || 0), 0),
                estimatedParameterBytesLabel: formatBytes(tensors.reduce((sum, item) => sum + (item.estimatedBytesValue || 0), 0))
            }
        },
        metadata,
        graphMetadata: sections.map((section) => ({
            key: section.path,
            value: `${section.kind}${section.summary ? ` · ${section.summary}` : ''}`,
            valuePreview: `${section.kind}${section.summary ? ` · ${section.summary}` : ''}`,
            isJson: false,
            jsonValue: null
        })),
        checkpointSections: sections,
        inputs: [],
        outputs: [],
        valueInfos: [],
        initializers: tensors,
        nodes: [],
        graphView
    };
}

const PT_PARAM_SUFFIXES = new Set([
    'weight',
    'bias',
    'running_mean',
    'running_var',
    'num_batches_tracked',
    'weight_v',
    'weight_g'
]);

const PT_RECURRENT_SUFFIXES = /^(weight|bias)_(ih|hh)_l\d+(_reverse)?$/;
// Generic param-name pattern: things like "in_proj_weight", "gate_bias", "weight_g".
// Covers common PyTorch idioms without enumerating every module's parameter name.
const PT_PARAM_GENERIC = /^(weight|bias)([_.].+)?$|[_.](weight|bias)$/;

function splitTensorPath(fullPath) {
    const prefixes = ['root.state_dict.', 'root.model.', 'root.net.', 'root.module.', 'root.'];
    let remainder = fullPath;
    for (const prefix of prefixes) {
        if (remainder.startsWith(prefix)) {
            remainder = remainder.slice(prefix.length);
            break;
        }
    }

    const segments = remainder.split('.');
    if (segments.length < 2) {
        return { modulePath: '', paramName: remainder, displayPath: remainder };
    }
    const last = segments[segments.length - 1];
    if (PT_PARAM_SUFFIXES.has(last)
        || PT_RECURRENT_SUFFIXES.test(last)
        || PT_PARAM_GENERIC.test(last)) {
        return {
            modulePath: segments.slice(0, -1).join('.'),
            paramName: last,
            displayPath: remainder
        };
    }
    return { modulePath: remainder, paramName: '', displayPath: remainder };
}

function inferPtLayerType(params, moduleHints = {}) {
    const paramNames = new Set(params.map((item) => item.paramName));
    const byName = new Map(params.map((item) => [item.paramName, item]));
    const hasRecurrent = [...paramNames].some((name) => PT_RECURRENT_SUFFIXES.test(name));
    const lowerPath = `${moduleHints.modulePath || ''}`.toLowerCase();
    const siblingKeys = moduleHints.siblingModuleNames instanceof Set
        ? moduleHints.siblingModuleNames
        : new Set();

    if (hasRecurrent) {
        const ih = [...byName.entries()].find(([name]) => name.startsWith('weight_ih_l'));
        const hiddenFactor = ih ? inferRecurrentHiddenFactor(ih[1]) : 1;
        if (hiddenFactor === 4) return { opType: 'LSTM', family: 'activation', confidence: 'high' };
        if (hiddenFactor === 3) return { opType: 'GRU', family: 'activation', confidence: 'high' };
        return { opType: 'RNN', family: 'activation', confidence: 'medium' };
    }

    if (paramNames.has('in_proj_weight')) {
        const hasOutProj = siblingKeys.has('out_proj') || paramNames.has('out_proj.weight');
        return {
            opType: 'MultiheadAttention',
            family: 'activation',
            confidence: hasOutProj ? 'high' : 'medium',
            warnings: hasOutProj ? [] : ['Missing sibling out_proj module; naming non-standard.']
        };
    }

    const weight = byName.get('weight');
    const bias = byName.get('bias');
    const hasRunning = paramNames.has('running_mean') || paramNames.has('running_var');

    if (weight) {
        const shape = weight.tensor.shape || [];
        const rank = shape.length;
        if (hasRunning) {
            if (rank === 1) return { opType: 'BatchNorm', family: 'normalization', confidence: 'high' };
            return { opType: 'Normalization', family: 'normalization', confidence: 'medium' };
        }
        if (rank === 4) {
            if (shape[1] === 1) {
                return {
                    opType: 'DepthwiseConv2d',
                    family: 'convolution',
                    confidence: 'medium',
                    warnings: ['Depthwise signature (weight.shape[1] == 1); exact groups cannot be recovered from a state_dict alone.']
                };
            }
            return { opType: 'Conv2d', family: 'convolution', confidence: 'high' };
        }
        if (rank === 5) return { opType: 'Conv3d', family: 'convolution', confidence: 'high' };
        if (rank === 3) return { opType: 'Conv1d', family: 'convolution', confidence: 'high' };
        if (rank === 2) {
            if (bias) {
                const biasShape = bias.tensor.shape || [];
                if (biasShape.length === 1 && biasShape[0] === shape[0]) {
                    return { opType: 'Linear', family: 'linear', confidence: 'high' };
                }
            }
            return {
                opType: 'Embedding',
                family: 'linear',
                confidence: 'medium',
                warnings: ['Inferred as Embedding because there is no matching bias; could also be a Linear without bias.']
            };
        }
        if (rank === 1) {
            if (bias) {
                if (lowerPath.includes('group') || lowerPath.includes('gn')) {
                    return {
                        opType: 'GroupNorm',
                        family: 'normalization',
                        confidence: 'medium',
                        warnings: ['Name hint suggests GroupNorm; the state_dict cannot distinguish it from LayerNorm alone.']
                    };
                }
                if (lowerPath.includes('instance') || lowerPath.includes('in_norm')) {
                    return {
                        opType: 'InstanceNorm',
                        family: 'normalization',
                        confidence: 'medium'
                    };
                }
                return {
                    opType: 'LayerNorm',
                    family: 'normalization',
                    confidence: 'medium',
                    warnings: ['Could also be GroupNorm or InstanceNorm; state_dict alone is ambiguous.']
                };
            }
            return { opType: 'Parameter', family: 'other', confidence: 'low' };
        }
        if (rank === 0) return { opType: 'Parameter', family: 'other', confidence: 'low' };
    }

    if (params.length === 1) return { opType: 'Parameter', family: 'other', confidence: 'low' };
    return { opType: 'Module', family: 'other', confidence: 'low' };
}

function inferRecurrentHiddenFactor(param) {
    const shape = param.tensor.shape || [];
    if (shape.length === 2 && Number.isFinite(shape[0]) && Number.isFinite(shape[1]) && shape[1] > 0) {
        const ratio = shape[0] / shape[1];
        if (Math.abs(ratio - Math.round(ratio)) < 0.0001) {
            return Math.round(ratio);
        }
    }
    return 1;
}

function buildTorchScriptGraphView(torchscript, rootType) {
    const ops = Array.isArray(torchscript.ops) ? torchscript.ops : [];
    const tsEdges = Array.isArray(torchscript.edges) ? torchscript.edges : [];

    const producedBy = new Map();
    for (const op of ops) {
        for (const out of op.outputs || []) {
            if (!producedBy.has(out)) producedBy.set(out, op.id);
        }
    }

    const referencedInputs = new Set();
    for (const op of ops) {
        for (const inputName of op.inputs || []) {
            if (!producedBy.has(inputName)) {
                referencedInputs.add(inputName);
            }
        }
    }

    const rootId = 'ts-root';
    const rootNode = {
        id: rootId,
        kind: 'input',
        name: torchscript.rootClass || rootType || 'input',
        title: torchscript.rootClass || rootType || 'input',
        subtitle: `${ops.length} op${ops.length === 1 ? '' : 's'}`,
        details: {
            typeSummary: torchscript.rootClass || rootType || 'input',
            opCountExtracted: torchscript.opCountExtracted,
            opCountTotal: torchscript.opCountTotal,
            partial: Boolean(torchscript.partial),
            warnings: torchscript.warnings || []
        },
        searchTokens: [torchscript.rootClass || 'input', ...Array.from(referencedInputs)]
    };

    const displayNodes = [rootNode];
    const edges = [];
    const edgeKeys = new Set();

    const addEdge = (sourceId, targetId, tensorName, extras = {}) => {
        const key = `${sourceId}→${targetId}:${tensorName}`;
        if (edgeKeys.has(key)) return;
        edgeKeys.add(key);
        edges.push({
            id: `edge:${edges.length}`,
            sourceId,
            targetId,
            tensorName: tensorName || '',
            typeSummary: '',
            shapeLabel: '',
            isModelInput: sourceId === rootId,
            isOutputEdge: false,
            ...extras
        });
    };

    const familyForOp = (opType) => {
        const normalized = `${opType || ''}`.toLowerCase();
        if (['relu', 'elu', 'leakyrelu', 'sigmoid', 'tanh', 'gelu', 'softplus'].includes(normalized)) return 'activation';
        if (['reshape', 'view', 'permute', 'transpose', 'squeeze', 'unsqueeze', 'flatten', 'cat', 'concat', 'split', 'slice'].includes(normalized)) return 'shape';
        if (normalized.startsWith('batch_norm') || normalized.startsWith('layer_norm') || normalized.startsWith('group_norm')) return 'normalization';
        if (['linear', 'matmul', 'conv', 'conv1d', 'conv2d', 'conv3d'].includes(normalized)) return 'linear';
        if (['lstm', 'gru', 'rnn'].includes(normalized)) return 'recurrent';
        if (['add', 'sub', 'mul', 'div', 'pow'].includes(normalized)) return 'utility';
        return 'other';
    };

    for (const op of ops) {
        const parameterRows = (op.inputs || []).map((inputName) => ({
            kind: 'input',
            label: inputName,
            shape: '',
            dtype: '',
            name: inputName,
            preview: '',
            secondaryValue: ''
        }));
        const family = familyForOp(op.opType);
        const displayName = op.opType || op.callee || op.id;
        displayNodes.push({
            id: op.id,
            kind: 'operator',
            name: op.id,
            title: displayName,
            subtitle: op.callee || op.kind || '',
            renderStyle: {
                family,
                variant: parameterRows.length === 0 ? 'compact' : 'standard',
                headerHeight: 34,
                rowHeight: 20,
                bodyVisible: true
            },
            details: {
                opType: op.opType || 'op',
                displayName,
                family,
                callee: op.callee || '',
                inputs: op.inputs || [],
                outputs: op.outputs || [],
                inPlace: Boolean(op.inPlace),
                module: op.module || '',
                sourceLine: op.sourceLine || 0,
                embeddedParameters: [],
                embeddedConstants: [],
                dataInputs: (op.inputs || []).map((name, index) => ({ index, name, typeSummary: '' })),
                outputInfos: (op.outputs || []).map((name, index) => ({ index, name, typeSummary: '' })),
                primaryInput: (op.inputs || [])[0] ? { name: op.inputs[0], typeSummary: '' } : null
            },
            parameterRows,
            searchTokens: [
                displayName, op.opType || '', op.callee || '', op.module || '',
                ...(op.inputs || []), ...(op.outputs || [])
            ].filter(Boolean)
        });
    }

    // Connect synthetic root to every op whose inputs are graph-external.
    for (const op of ops) {
        for (const inputName of op.inputs || []) {
            if (!producedBy.has(inputName)) {
                addEdge(rootId, op.id, inputName);
            }
        }
    }

    // Connect use-def edges.
    for (const edge of tsEdges) {
        addEdge(edge.sourceId, edge.targetId, edge.tensorName || '');
    }

    const layout = layoutDag(displayNodes, edges, rootId);

    return {
        name: torchscript.rootClass || rootType || '',
        displayNodes,
        edges,
        layout,
        terminalNodes: [rootNode],
        stats: {
            displayNodeCount: displayNodes.length,
            edgeCount: edges.length,
            operatorNodeCount: ops.length,
            collapsedConstantCount: 0
        }
    };
}

function layoutDag(displayNodes, edges, rootId) {
    const sizeById = new Map();
    for (const node of displayNodes) {
        sizeById.set(node.id, measurePtNode(node));
    }
    const incoming = new Map();
    const outgoing = new Map();
    for (const node of displayNodes) {
        incoming.set(node.id, new Set());
        outgoing.set(node.id, new Set());
    }
    for (const edge of edges) {
        if (incoming.has(edge.targetId)) incoming.get(edge.targetId).add(edge.sourceId);
        if (outgoing.has(edge.sourceId)) outgoing.get(edge.sourceId).add(edge.targetId);
    }

    // Layer assignment: BFS from root.
    const layer = new Map();
    layer.set(rootId, 0);
    const queue = [rootId];
    while (queue.length) {
        const id = queue.shift();
        const d = layer.get(id) || 0;
        for (const next of outgoing.get(id) || []) {
            const nextLayer = Math.max(layer.get(next) || 0, d + 1);
            if (!layer.has(next) || nextLayer > layer.get(next)) {
                layer.set(next, nextLayer);
                queue.push(next);
            }
        }
    }
    for (const node of displayNodes) {
        if (!layer.has(node.id)) layer.set(node.id, 0);
    }

    const layers = new Map();
    for (const node of displayNodes) {
        const l = layer.get(node.id) || 0;
        if (!layers.has(l)) layers.set(l, []);
        layers.get(l).push(node.id);
    }

    const horizontalGap = 32;
    const verticalGap = 52;

    const rowHeightByLayer = new Map();
    let maxRowWidth = 0;
    for (const [l, ids] of layers.entries()) {
        let width = 0;
        let height = 0;
        for (const id of ids) {
            const size = sizeById.get(id) || { width: 120, height: 40 };
            width += size.width;
            height = Math.max(height, size.height);
        }
        width += Math.max(0, ids.length - 1) * horizontalGap;
        rowHeightByLayer.set(l, height);
        maxRowWidth = Math.max(maxRowWidth, width);
    }

    const positions = {};
    const sortedLayers = [...layers.keys()].sort((a, b) => a - b);
    let runningY = 24;
    for (const l of sortedLayers) {
        const ids = layers.get(l);
        let rowWidth = 0;
        for (const id of ids) {
            rowWidth += sizeById.get(id).width;
        }
        rowWidth += Math.max(0, ids.length - 1) * horizontalGap;
        let x = 24 + Math.max(0, (maxRowWidth - rowWidth) / 2);
        for (const id of ids) {
            const size = sizeById.get(id);
            positions[id] = { x, y: runningY, width: size.width, height: size.height };
            x += size.width + horizontalGap;
        }
        runningY += (rowHeightByLayer.get(l) || 40) + verticalGap;
    }

    return {
        algorithm: 'ts-layered-tb',
        width: Math.max(900, maxRowWidth + 48),
        height: Math.max(420, runningY + 24),
        positions,
        routedEdges: {}
    };
}

function buildPtGraphView(tensors, rootType) {
    if (!tensors.length) {
        return {
            name: rootType || '',
            displayNodes: [],
            edges: [],
            layout: { algorithm: 'none', width: 900, height: 420, positions: {}, routedEdges: {} },
            terminalNodes: [],
            stats: { displayNodeCount: 0, edgeCount: 0, operatorNodeCount: 0, collapsedConstantCount: 0 }
        };
    }

    const moduleOrder = [];
    const moduleMap = new Map();
    for (const tensor of tensors) {
        const { modulePath, paramName } = splitTensorPath(tensor.name);
        if (!moduleMap.has(modulePath)) {
            moduleMap.set(modulePath, { path: modulePath, params: [] });
            moduleOrder.push(modulePath);
        }
        moduleMap.get(modulePath).params.push({ paramName: paramName || '(value)', tensor });
    }

    const containerPaths = new Set();
    for (const modulePath of moduleOrder) {
        if (!modulePath) continue;
        const segments = modulePath.split('.');
        for (let i = 1; i < segments.length; i += 1) {
            containerPaths.add(segments.slice(0, i).join('.'));
        }
    }
    for (const leaf of moduleOrder) {
        containerPaths.delete(leaf);
    }

    // Build parent -> Set<child name> index for sibling-aware inference (Phase E).
    const childrenByParent = new Map();
    for (const leaf of moduleOrder) {
        if (!leaf) continue;
        const segments = leaf.split('.');
        for (let i = 0; i < segments.length; i += 1) {
            const parentPath = segments.slice(0, i).join('.');
            const childName = segments[i];
            if (!childrenByParent.has(parentPath)) {
                childrenByParent.set(parentPath, new Set());
            }
            childrenByParent.get(parentPath).add(childName);
        }
    }
    const ATTENTION_PROJECTIONS = new Set(['q_proj', 'k_proj', 'v_proj']);
    const attentionBlockParents = new Set();
    for (const [parent, children] of childrenByParent.entries()) {
        const hits = [...ATTENTION_PROJECTIONS].filter((name) => children.has(name));
        if (hits.length >= 2) {
            attentionBlockParents.add(parent);
        }
    }

    const rootId = 'pt-root';
    const displayNodes = [];
    const edges = [];
    const edgeKeys = new Set();

    const addEdge = (sourceId, targetId) => {
        const key = `${sourceId}→${targetId}`;
        if (edgeKeys.has(key)) return;
        edgeKeys.add(key);
        edges.push({
            id: `edge:${edges.length}`,
            sourceId,
            targetId,
            tensorName: '',
            typeSummary: '',
            shapeLabel: '',
            isModelInput: sourceId === rootId,
            isOutputEdge: false
        });
    };

    const rootNode = {
        id: rootId,
        kind: 'input',
        name: rootType || 'checkpoint',
        title: rootType || 'checkpoint',
        subtitle: `${tensors.length} tensor${tensors.length === 1 ? '' : 's'}`,
        details: { typeSummary: rootType || 'checkpoint' },
        searchTokens: [rootType || 'checkpoint']
    };
    displayNodes.push(rootNode);

    const containerPathList = [...containerPaths].sort((a, b) => {
        const depthDiff = a.split('.').length - b.split('.').length;
        return depthDiff !== 0 ? depthDiff : a.localeCompare(b);
    });
    const idForContainer = (containerPath) => `pt-container:${containerPath || '(root)'}`;

    for (const containerPath of containerPathList) {
        const segments = containerPath.split('.');
        const nodeName = segments[segments.length - 1];
        const isAttention = attentionBlockParents.has(containerPath);
        const subtitle = isAttention ? 'AttentionBlock' : 'Module';
        const opType = isAttention ? 'AttentionBlock' : 'Module';
        const family = isAttention ? 'activation' : 'other';
        displayNodes.push({
            id: idForContainer(containerPath),
            kind: 'operator',
            name: nodeName,
            title: nodeName,
            subtitle,
            renderStyle: { family, variant: 'mini', headerHeight: 30, rowHeight: 16, bodyVisible: false },
            details: {
                opType,
                displayName: containerPath,
                family,
                modulePath: containerPath,
                confidence: isAttention ? 'high' : 'low',
                inferenceWarnings: [],
                embeddedParameters: [],
                embeddedConstants: [],
                dataInputs: [],
                outputInfos: [],
                primaryInput: null
            },
            parameterRows: [],
            searchTokens: [containerPath, nodeName, opType]
        });
    }

    for (const modulePath of moduleOrder) {
        const moduleEntry = moduleMap.get(modulePath);
        const parentPath = modulePath.includes('.') ? modulePath.slice(0, modulePath.lastIndexOf('.')) : '';
        const siblingModuleNames = childrenByParent.get(parentPath) || new Set();
        const hints = { modulePath, siblingModuleNames };
        const inference = inferPtLayerType(moduleEntry.params, hints);
        const { opType, family, confidence = 'medium', warnings = [] } = inference;
        const parameterRows = moduleEntry.params.map((param) => ({
            kind: 'parameter',
            label: param.paramName,
            shape: param.tensor.shapeSummary || formatShape(param.tensor.shape),
            dtype: param.tensor.dataTypeName || '',
            name: param.tensor.name,
            preview: '',
            secondaryValue: param.tensor.shapeSummary || param.tensor.dataTypeName || '—'
        }));
        const embeddedParameters = moduleEntry.params.map((param, index) => ({
            label: param.paramName,
            name: param.tensor.name,
            dataTypeName: param.tensor.dataTypeName || '',
            shapeLabel: param.tensor.shapeSummary || formatShape(param.tensor.shape),
            valuePreview: '',
            inputIndex: index
        }));
        const segments = modulePath ? modulePath.split('.') : ['(root)'];
        const leafName = segments[segments.length - 1];
        const title = leafName || '(root)';
        const nodeId = `pt-module:${modulePath || '(root)'}`;
        const variant = parameterRows.length === 0 ? 'micro' : (parameterRows.length <= 2 ? 'compact' : 'standard');
        displayNodes.push({
            id: nodeId,
            kind: 'operator',
            name: title,
            title,
            subtitle: opType,
            renderStyle: { family, variant, headerHeight: 34, rowHeight: 22, bodyVisible: true },
            details: {
                opType,
                displayName: modulePath || '(root)',
                family,
                modulePath,
                confidence,
                inferenceWarnings: warnings,
                embeddedParameters,
                embeddedConstants: [],
                dataInputs: [],
                outputInfos: [],
                primaryInput: null,
                parameters: moduleEntry.params.map((param) => ({
                    name: param.paramName,
                    tensorName: param.tensor.name,
                    dataTypeName: param.tensor.dataTypeName,
                    shapeLabel: param.tensor.shapeSummary,
                    elementCount: param.tensor.elementCount,
                    estimatedBytes: param.tensor.estimatedBytes
                }))
            },
            parameterRows,
            searchTokens: [
                modulePath,
                title,
                opType,
                ...moduleEntry.params.flatMap((param) => [param.paramName, param.tensor.name, param.tensor.dataTypeName, param.tensor.shapeSummary])
            ].filter(Boolean)
        });
    }

    const nodeIdForPath = (modulePath, { leaf }) => {
        if (leaf && moduleMap.has(modulePath)) return `pt-module:${modulePath || '(root)'}`;
        if (containerPaths.has(modulePath)) return idForContainer(modulePath);
        return null;
    };

    const resolveParentId = (modulePath) => {
        if (!modulePath) return rootId;
        const segments = modulePath.split('.');
        for (let i = segments.length - 1; i >= 1; i -= 1) {
            const candidate = segments.slice(0, i).join('.');
            const id = nodeIdForPath(candidate, { leaf: moduleMap.has(candidate) });
            if (id) return id;
        }
        return rootId;
    };

    for (const containerPath of containerPathList) {
        addEdge(resolveParentId(containerPath), idForContainer(containerPath));
    }
    for (const modulePath of moduleOrder) {
        addEdge(resolveParentId(modulePath), `pt-module:${modulePath || '(root)'}`);
    }

    const layout = layoutPtTree(displayNodes, edges, rootId);
    return {
        name: rootType || '',
        displayNodes,
        edges,
        layout,
        terminalNodes: [rootNode],
        stats: {
            displayNodeCount: displayNodes.length,
            edgeCount: edges.length,
            operatorNodeCount: moduleOrder.length,
            collapsedConstantCount: 0
        }
    };
}

function measurePtNode(node) {
    if (node.kind === 'input') {
        const title = `${node.title || ''}`;
        const subtitle = `${node.subtitle || ''}`;
        const width = Math.min(260, Math.max(120, 24 + Math.max(title.length * 6.4, subtitle.length * 5.2)));
        return { width, height: 40 };
    }
    const style = node.renderStyle || { variant: 'standard', rowHeight: 22 };
    const title = `${node.details?.opType || node.title || ''}`;
    const subtitle = `${node.subtitle || ''}`;
    const parameterRows = Array.isArray(node.parameterRows) ? node.parameterRows : [];
    const titleWidth = title.length * 6.8;
    const subtitleWidth = subtitle.length * 5.1;
    const parameterWidth = parameterRows.reduce((max, row) => {
        const valueText = row.shape || row.preview || row.secondaryValue || row.dtype || '';
        return Math.max(max, (row.label?.length || 0) * 6.4 + `${valueText}`.length * 5.3 + 44);
    }, 0);

    if (style.variant === 'micro') {
        return { width: Math.min(130, Math.max(78, titleWidth + 24)), height: 34 };
    }
    if (style.variant === 'compact') {
        return {
            width: Math.min(220, Math.max(108, titleWidth + 32, subtitleWidth + 28, parameterWidth)),
            height: 34 + parameterRows.length * 18 + (parameterRows.length ? 12 : 0)
        };
    }
    return {
        width: Math.min(288, Math.max(148, titleWidth + 36, subtitleWidth + 32, parameterWidth)),
        height: 40 + parameterRows.length * 20 + (parameterRows.length ? 12 : 0)
    };
}

function layoutPtTree(displayNodes, edges, rootId) {
    const childrenByParent = new Map();
    for (const edge of edges) {
        if (!childrenByParent.has(edge.sourceId)) childrenByParent.set(edge.sourceId, []);
        childrenByParent.get(edge.sourceId).push(edge.targetId);
    }
    const nodeById = new Map(displayNodes.map((node) => [node.id, node]));
    const sizeById = new Map(displayNodes.map((node) => [node.id, measurePtNode(node)]));

    const horizontalGap = 24;
    const verticalGap = 42;
    const rowHeightByDepth = new Map();

    const subtreeWidth = new Map();
    const visiting = new Set();

    const computeWidth = (id) => {
        if (subtreeWidth.has(id)) return subtreeWidth.get(id);
        if (visiting.has(id)) {
            subtreeWidth.set(id, sizeById.get(id)?.width || 120);
            return subtreeWidth.get(id);
        }
        visiting.add(id);
        const selfWidth = sizeById.get(id)?.width || 120;
        const kids = childrenByParent.get(id) || [];
        let childrenWidth = 0;
        for (const kid of kids) {
            childrenWidth += computeWidth(kid);
        }
        if (kids.length > 1) childrenWidth += (kids.length - 1) * horizontalGap;
        const width = Math.max(selfWidth, childrenWidth);
        visiting.delete(id);
        subtreeWidth.set(id, width);
        return width;
    };
    computeWidth(rootId);

    const positions = {};
    const assigned = new Set();

    const assign = (id, leftEdge, depth) => {
        if (assigned.has(id)) return;
        assigned.add(id);
        const size = sizeById.get(id) || { width: 120, height: 40 };
        const width = subtreeWidth.get(id) || size.width;
        const cx = leftEdge + width / 2;
        const x = cx - size.width / 2;
        const currentMaxHeight = rowHeightByDepth.get(depth) || 0;
        if (size.height > currentMaxHeight) rowHeightByDepth.set(depth, size.height);
        positions[id] = { x, y: 0, width: size.width, height: size.height, _depth: depth };

        const kids = childrenByParent.get(id) || [];
        let childLeft = cx - width / 2;
        const totalChildWidth = kids.reduce((sum, kid, index) => sum + (subtreeWidth.get(kid) || 0) + (index > 0 ? horizontalGap : 0), 0);
        childLeft = cx - totalChildWidth / 2;
        for (const kid of kids) {
            const kidWidth = subtreeWidth.get(kid) || (sizeById.get(kid)?.width || 120);
            assign(kid, childLeft, depth + 1);
            childLeft += kidWidth + horizontalGap;
        }
    };

    assign(rootId, 24, 0);

    for (const node of displayNodes) {
        if (!assigned.has(node.id)) {
            assign(node.id, 24, 0);
        }
    }

    const sortedDepths = [...rowHeightByDepth.keys()].sort((a, b) => a - b);
    const yByDepth = new Map();
    let runningY = 24;
    for (const depth of sortedDepths) {
        yByDepth.set(depth, runningY);
        runningY += rowHeightByDepth.get(depth) + verticalGap;
    }

    let minX = Number.POSITIVE_INFINITY;
    let minY = Number.POSITIVE_INFINITY;
    let maxX = Number.NEGATIVE_INFINITY;
    let maxY = Number.NEGATIVE_INFINITY;
    for (const [id, position] of Object.entries(positions)) {
        position.y = yByDepth.get(position._depth) || 0;
        delete position._depth;
        minX = Math.min(minX, position.x);
        minY = Math.min(minY, position.y);
        maxX = Math.max(maxX, position.x + position.width);
        maxY = Math.max(maxY, position.y + position.height);
    }

    const offsetX = Number.isFinite(minX) ? 24 - minX : 0;
    const offsetY = Number.isFinite(minY) ? 24 - minY : 0;
    for (const position of Object.values(positions)) {
        position.x += offsetX;
        position.y += offsetY;
    }

    const width = Number.isFinite(maxX) ? maxX - minX + 48 : 900;
    const height = Number.isFinite(maxY) ? maxY - minY + 48 : 420;

    return {
        algorithm: 'pt-tree-tb',
        width,
        height,
        positions,
        routedEdges: {}
    };
}

function resolvePythonExecutable(options = {}) {
    const explicit = options.pythonExecutable;
    if (explicit) {
        return explicit;
    }

    const configured = options.pythonPath;
    if (configured) {
        return normalizeConfiguredPythonPath(configured, options.extensionPath);
    }

    return process.env.ONNX_INSPECTOR_PYTHON
        || process.env.CONDA_PYTHON_EXE
        || 'python3';
}

function defaultPythonCandidates() {
    const base = [
        process.env.ONNX_INSPECTOR_PYTHON ? { command: process.env.ONNX_INSPECTOR_PYTHON, prefixArgs: [] } : null,
        process.env.CONDA_PYTHON_EXE ? { command: process.env.CONDA_PYTHON_EXE, prefixArgs: [] } : null
    ].filter(Boolean);

    if (process.platform === 'win32') {
        return [
            ...base,
            { command: 'py', prefixArgs: ['-3'] },
            { command: 'python', prefixArgs: [] }
        ];
    }
    return [
        ...base,
        { command: 'python3', prefixArgs: [] }
    ];
}

async function probePythonCandidate(candidate) {
    try {
        await execFileAsync(candidate.command, [...candidate.prefixArgs, '--version'], {
            timeout: 5000,
            maxBuffer: 1 * 1024 * 1024
        });
        return true;
    } catch {
        return false;
    }
}

async function resolvePythonCommand(options = {}) {
    const explicit = options.pythonExecutable;
    if (explicit) {
        return normalizeCommandSpec(explicit);
    }

    const configured = options.pythonPath;
    if (configured) {
        return normalizeCommandSpec(normalizeConfiguredPythonPath(configured, options.extensionPath));
    }

    for (const candidate of defaultPythonCandidates()) {
        if (await probePythonCandidate(candidate)) {
            return candidate;
        }
    }
    return normalizeCommandSpec(process.platform === 'win32' ? 'python' : 'python3');
}

function normalizeConfiguredPythonPath(pythonPath, extensionPath) {
    if (!pythonPath) {
        return pythonPath;
    }
    if (pythonPath.includes('${workspaceFolder}')) {
        const workspaceRoot = extensionPath || process.cwd();
        return path.normalize(pythonPath.replaceAll('${workspaceFolder}', workspaceRoot));
    }
    return pythonPath;
}

function formatShape(shape) {
    if (!Array.isArray(shape) || shape.length === 0) {
        return 'scalar';
    }
    return `[${shape.map((item) => String(item)).join(' × ')}]`;
}

function stringifyScalar(value) {
    if (value === null || value === undefined) {
        return '';
    }
    if (typeof value === 'string') {
        return value;
    }
    if (typeof value === 'number' || typeof value === 'boolean') {
        return String(value);
    }
    return JSON.stringify(value);
}

function formatBytes(bytes) {
    if (!Number.isFinite(bytes)) {
        return 'Unknown';
    }
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let value = bytes;
    let index = 0;
    while (value >= 1024 && index < units.length - 1) {
        value /= 1024;
        index += 1;
    }
    return `${index === 0 ? value.toFixed(0) : value.toFixed(value >= 10 ? 1 : 2)} ${units[index]}`;
}

module.exports = {
    detectModelFormat,
    isSupportedFormat,
    parseModelFile,
    normalizePtInspection,
    resolvePythonExecutable,
    resolvePythonCommand,
    defaultPythonCandidates,
    normalizeConfiguredPythonPath,
    FORMAT_WHITELIST: Array.from(FORMAT_WHITELIST),
    SUBPROCESS_TIMEOUT_MS
};
