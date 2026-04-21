const fs = require('node:fs');
const path = require('node:path');
const { execFile } = require('node:child_process');
const { promisify } = require('node:util');
const { pathToFileURL } = require('node:url');

const execFileAsync = promisify(execFile);

function detectModelFormat(filePath) {
    const ext = path.extname(filePath || '').toLowerCase();
    if (ext === '.onnx') {
        return 'onnx';
    }
    if (ext === '.pt' || ext === '.pth') {
        return 'pt';
    }
    return 'unknown';
}

async function parseModelFile(options) {
    const { filePath, bytes, extensionPath } = options;
    const pythonExecutable = resolvePythonExecutable(options);
    const nextOptions = { ...options, pythonExecutable };
    const format = detectModelFormat(filePath);
    if (format === 'onnx') {
        const parserPath = path.join(extensionPath, 'media', 'lib', 'onnx-parser.mjs');
        const parser = await import(pathToFileURL(parserPath).href);
        return { ok: true, parsed: parser.parseOnnxBytes(bytes), format };
    }
    if (format === 'pt') {
        const inspected = await inspectPtFile(filePath, extensionPath, nextOptions.pythonExecutable);
        return { ok: true, parsed: normalizePtInspection(inspected, bytes.length), format };
    }
    throw new Error(`Unsupported model format for ${filePath}`);
}

async function inspectPtFile(filePath, extensionPath, pythonExecutable) {
    const scriptPath = path.join(extensionPath, 'scripts', 'inspect_pt.py');
    try {
        const { stdout } = await execFileAsync(pythonExecutable, [scriptPath, filePath], {
            cwd: extensionPath,
            maxBuffer: 8 * 1024 * 1024
        });
        return JSON.parse(stdout);
    } catch (error) {
        const stderr = error?.stderr ? String(error.stderr).trim() : '';
        const stdout = error?.stdout ? String(error.stdout).trim() : '';
        const detail = stderr || stdout || error?.message || String(error);
        throw new Error(`PT inspection failed: ${detail}`);
    }
}

function normalizePtInspection(inspected, fileSizeBytes) {
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

    return {
        format: 'pt',
        summary: {
            graphName: inspected?.rootType || 'PyTorch checkpoint',
            irVersion: '',
            irVersionLabel: '',
            producerName: 'PyTorch',
            producerVersion: inspected?.producerVersion || '',
            domain: '',
            modelVersion: '',
            docString: '',
            opsets: [],
            fileSizeBytes,
            fileSizeLabel: formatBytes(fileSizeBytes),
            temporalMetadata: { primary: null, candidates: [] },
            stats: {
                nodeCount: 0,
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
        graphView: {
            name: '',
            displayNodes: [],
            edges: [],
            layout: { algorithm: 'none', width: 0, height: 0, positions: {} },
            terminalNodes: [],
            stats: {
                displayNodeCount: 0,
                edgeCount: 0,
                operatorNodeCount: 0,
                collapsedConstantCount: 0
            }
        }
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
    parseModelFile,
    normalizePtInspection,
    resolvePythonExecutable,
    normalizeConfiguredPythonPath
};
