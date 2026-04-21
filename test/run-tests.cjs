const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { execFileSync } = require('node:child_process');
const { pathToFileURL } = require('node:url');
const vm = require('node:vm');

const root = path.resolve(__dirname, '..');
const distDir = path.join(root, 'dist');

(async () => {
  await runParserChecks();
  runUiChecks();
  runPackagingChecks();
  console.log('All tests passed.');
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

async function runParserChecks() {
  const parser = await import(pathToFileURL(path.join(root, 'media', 'lib', 'onnx-parser.mjs')).href + `?t=${Date.now()}`);
  const modelParser = require(path.join(root, 'model-parser.js'));
  const sample = fs.readFileSync(path.join(root, 'test', 'sample.onnx'));
  const parsed = parser.parseOnnxBytes(new Uint8Array(sample));

  assert.equal(modelParser.detectModelFormat('/tmp/model.onnx'), 'onnx');
  assert.equal(modelParser.detectModelFormat('/tmp/checkpoint.pt'), 'pt');
  assert.equal(modelParser.detectModelFormat('/tmp/checkpoint.pth'), 'pt');
  assert.equal(modelParser.detectModelFormat('/tmp/weights.bin'), 'unknown');
  assert.equal(modelParser.resolvePythonExecutable({ pythonExecutable: '/tmp/custom-python' }), '/tmp/custom-python');
  assert.equal(modelParser.resolvePythonExecutable({ pythonPath: '/tmp/config-python' }), '/tmp/config-python');
  assert.equal(modelParser.resolvePythonExecutable({}), process.env.ONNX_INSPECTOR_PYTHON || process.env.CONDA_PYTHON_EXE || 'python3');

  const pythonPathRoot = '/tmp/python-path-root';
  const configuredPython = modelParser.resolvePythonExecutable({ pythonPath: '${workspaceFolder}/env/bin/python', extensionPath: pythonPathRoot });
  assert.equal(configuredPython, path.join(pythonPathRoot, 'env', 'bin', 'python'));

  const normalizedPt = modelParser.normalizePtInspection({
    producerVersion: '2.6.0',
    rootType: 'OrderedDict',
    sections: [
      { path: 'state_dict', kind: 'mapping', summary: '2 entries', entryCount: 2 },
      { path: 'epoch', kind: 'scalar', summary: '12' }
    ],
    metadata: [
      { path: 'epoch', value: 12 },
      { path: 'best_metric', value: 0.987 }
    ],
    tensors: [
      { path: 'state_dict.layer.weight', dtype: 'float32', shape: [2, 2], numel: 4, bytes: 16, storage: 'tensor' },
      { path: 'state_dict.layer.bias', dtype: 'float32', shape: [2], numel: 2, bytes: 8, storage: 'tensor' }
    ]
  }, sample.length);

  assert.equal(normalizedPt.format, 'pt');
  assert.equal(normalizedPt.summary.producerName, 'PyTorch');
  assert.equal(normalizedPt.summary.stats.initializerCount, 2);
  assert.equal(normalizedPt.summary.stats.metadataCount, 2);
  assert.equal(normalizedPt.checkpointSections.length, 2);
  assert.equal(normalizedPt.initializers[0].name, 'state_dict.layer.weight');
  assert.equal(normalizedPt.initializers[0].shapeSummary, '[2 × 2]');
  assert.equal(normalizedPt.graphView.stats.displayNodeCount, 0);

  const realPtPath = path.join(root, 'test', 'sample.pt');
  assert.ok(fs.existsSync(realPtPath), 'Expected real PT fixture at test/sample.pt');
  const realPtBytes = fs.readFileSync(realPtPath);
  const realPtParsed = await modelParser.parseModelFile({
    filePath: realPtPath,
    bytes: new Uint8Array(realPtBytes),
    extensionPath: root
  });
  assert.equal(realPtParsed.ok, true);
  assert.equal(realPtParsed.format, 'pt');
  assert.equal(realPtParsed.parsed.summary.producerName, 'PyTorch');
  assert.ok(realPtParsed.parsed.checkpointSections.some((item) => item.path === 'root.state_dict'));
  assert.ok(realPtParsed.parsed.initializers.some((item) => item.name === 'root.state_dict.layer.weight'));
  assert.ok(realPtParsed.parsed.metadata.some((item) => item.key === 'root.epoch' && item.value === '3'));

  assert.equal(parsed.summary.graphName, 'SimpleGraph');
  assert.equal(parsed.summary.stats.nodeCount, 3);
  assert.equal(parsed.summary.stats.inputCount, 1);
  assert.equal(parsed.summary.stats.outputCount, 1);
  assert.deepEqual(parsed.metadata.map((entry) => entry.key), ['task', 'owner', 'contract', 'notes']);
  assert.equal(parsed.inputs[0].typeSummary, 'tensor<FLOAT [1 × 3]>');
  assert.equal(parsed.initializers[0].dataTypeName, 'FLOAT');
  assert.equal(parsed.graphView.stats.displayNodeCount, 5, 'Initializers should be embedded in operator cards rather than rendered as standalone graph nodes');
  assert.ok(parsed.graphView.edges.length >= 3);
  assert.equal(parsed.graphView.layout.algorithm, 'dagre-tb');
  assert.ok(parsed.graphView.layout.height > parsed.graphView.layout.width, 'Expected a vertical graph layout');
  const positions = parsed.graphView.layout.positions;
  assert.ok(Math.abs(positions['node:1'].x - positions['node:0'].x) < 24, 'Operators should stay vertically aligned');
  assert.ok(positions['node:1'].y > positions['node:0'].y, 'Later operators should appear below earlier operators');
  assert.equal(parsed.graphView.displayNodes.find((node) => node.id === 'node:0').parameterRows[0].label, 'B');
  assert.equal(parsed.graphView.displayNodes.find((node) => node.id === 'node:1').parameterRows[0].label, 'B');
  assert.equal(parsed.graphView.displayNodes.find((node) => node.id === 'node:2').renderStyle.variant, 'micro');
  assert.equal(parsed.graphView.stats.collapsedConstantCount, 0);
  assert.equal(parsed.summary.temporalMetadata.primary, null);

  const uploadedPath = '/mnt/data/2026-03-20_17-04-44_109000.onnx';
  if (fs.existsSync(uploadedPath)) {
    const uploadedParsed = parser.parseOnnxBytes(new Uint8Array(fs.readFileSync(uploadedPath)));
    assert.ok(uploadedParsed.graphView.stats.collapsedConstantCount >= 3, 'Expected small Constant helper nodes to be merged into operator cards');
    const eluNode = uploadedParsed.graphView.displayNodes.find((node) => node.details?.opType === 'Elu');
    assert.equal(eluNode?.renderStyle?.variant, 'micro');
    const squeezeNode = uploadedParsed.graphView.displayNodes.find((node) => node.details?.displayName === '/Squeeze');
    assert.ok((squeezeNode?.searchTokens || []).some((token) => `${token}`.includes('/Constant_1')), 'Merged constant provenance should remain searchable');
    const gruNode = uploadedParsed.graphView.displayNodes.find((node) => node.details?.opType === 'GRU');
    assert.deepEqual((gruNode?.parameterRows || []).slice(0, 3).map((row) => row.label), ['W', 'R', 'B']);
    const unsqueezeNode = uploadedParsed.graphView.displayNodes.find((node) => node.details?.displayName === '/Unsqueeze');
    assert.equal(unsqueezeNode?.renderStyle?.variant, 'mini');
  }

  const temporal = parser.detectTemporalMetadata(
    [{ key: 'exported_at', value: '2025-04-05T10:11:12Z', valuePreview: '', isJson: false, jsonValue: null }],
    [{ key: 'contract', value: '{"generated":{"timestamp":1712311112000}}', valuePreview: '', isJson: true, jsonValue: { generated: { timestamp: 1712311112000 } } }]
  );
  assert.equal(temporal.primary.path, 'model.exported_at');
  assert.equal(temporal.candidates.length, 2);
  assert.equal(parser.inferParameterLabel('BatchNormalization', 4, 'running_var'), 'Variance');
}

async function runUiChecks() {
  const mainJs = fs.readFileSync(path.join(root, 'media', 'main.js'), 'utf8');
  const stylesCss = fs.readFileSync(path.join(root, 'media', 'styles.css'), 'utf8');
  assert.match(mainJs, /function renderOperatorInsight\(/);
  assert.match(mainJs, /function renderFunctionPlotSvg\(/);
  assert.match(mainJs, /function scheduleGraphFocusIfRequested\(/);
  assert.match(mainJs, /function renderTypeFlowSummaryCard\(/);
  assert.match(mainJs, /function lexicalSort\(/);
  assert.match(mainJs, /function capturePageViewportFromDom\(/);
  assert.match(mainJs, /function restorePageViewportAfterRender\(/);
  assert.match(mainJs, /title="Center the selected node in the graph viewport"/);
  assert.match(mainJs, /Plain-language explanation/);
  assert.match(mainJs, /detail-sections-expand-all/);
  assert.match(mainJs, /renderRememberedSection\(/);
  assert.match(stylesCss, /activation-plot/);
  assert.match(stylesCss, /graph-node\.selected/);
  assert.match(stylesCss, /insight-card/);
  assert.match(stylesCss, /detail-accordion/);

  await runUiRuntimeSmoke(mainJs);
}

async function runUiRuntimeSmoke(mainJs) {
  const parser = await import(pathToFileURL(path.join(root, 'media', 'lib', 'onnx-parser.mjs')).href + `?ui=${Date.now()}`);
  const sample = fs.readFileSync(path.join(root, 'test', 'sample.onnx'));
  const parsed = parser.parseOnnxBytes(new Uint8Array(sample));

  class FakeElement {
    constructor(id = '', options = {}) {
      this.id = id;
      this.dataset = options.dataset || {};
      this._innerHTML = '';
      this.textContent = '';
      this.scrollLeft = 0;
      this.scrollTop = 0;
      this.style = {};
      this.clientWidth = options.clientWidth || 1280;
      this.clientHeight = options.clientHeight || 840;
      this.classList = {
        add() {},
        remove() {},
        contains() { return false; }
      };
      this.open = false;
      this.value = '';
      this.onInnerHTML = options.onInnerHTML || null;
    }
    get innerHTML() {
      return this._innerHTML;
    }
    set innerHTML(value) {
      this._innerHTML = String(value);
      if (typeof this.onInnerHTML === 'function') {
        this.onInnerHTML(this._innerHTML);
      }
    }
    closest() { return null; }
    scrollTo(options = {}) {
      if (typeof options.left === 'number') {
        this.scrollLeft = options.left;
      }
      if (typeof options.top === 'number') {
        this.scrollTop = options.top;
      }
    }
    setPointerCapture() {}
    getBoundingClientRect() {
      return { left: 0, top: 0, width: this.clientWidth, height: this.clientHeight };
    }
  }

  class FakeInputElement extends FakeElement {}
  class FakeDetailsElement extends FakeElement {}
  class FakeSvgElement extends FakeElement {}

  const documentListeners = new Map();
  const windowListeners = new Map();
  const createdElements = new Map();
  let persistedState = undefined;
  const vscodeMessages = [];

  const addListener = (store, type, handler) => {
    if (!store.has(type)) {
      store.set(type, []);
    }
    store.get(type).push(handler);
  };

  const dispatch = (store, type, event) => {
    for (const handler of store.get(type) || []) {
      handler(event);
    }
  };

  const pageContent = new FakeElement('page-content', { clientWidth: 1400, clientHeight: 980 });
  const graphScroll = new FakeElement('graph-scroll-container', { clientWidth: 1100, clientHeight: 760 });
  const graphCanvas = new FakeElement('graph-canvas', { dataset: { baseWidth: '1200', baseHeight: '2200' } });
  graphCanvas.dataset = { baseWidth: '1200', baseHeight: '2200' };
  const graphSvg = new FakeSvgElement('graph-svg');
  const graphZoomLabel = new FakeElement('graph-zoom-label');
  const graphZoomSlider = new FakeInputElement('graph-zoom-slider');
  graphZoomSlider.dataset = { role: 'graph-zoom' };
  graphZoomSlider.value = '100';

  const appElement = new FakeElement('app', {
    onInnerHTML() {
      pageContent.scrollLeft = 0;
      pageContent.scrollTop = 0;
      graphScroll.scrollLeft = 0;
      graphScroll.scrollTop = 0;
    }
  });

  createdElements.set('app', appElement);
  createdElements.set('page-content', pageContent);
  createdElements.set('graph-scroll-container', graphScroll);
  createdElements.set('graph-canvas', graphCanvas);
  createdElements.set('graph-svg', graphSvg);
  createdElements.set('graph-zoom-label', graphZoomLabel);

  const documentStub = {
    getElementById(id) {
      return createdElements.get(id) || null;
    },
    addEventListener(type, handler) {
      addListener(documentListeners, type, handler);
    },
    querySelector(selector) {
      if (selector === '[data-role="graph-zoom"]') {
        return graphZoomSlider;
      }
      return null;
    },
    querySelectorAll() {
      return [];
    }
  };

  const windowStub = {
    addEventListener(type, handler) {
      addListener(windowListeners, type, handler);
    },
    removeEventListener() {},
    setTimeout,
    clearTimeout,
    requestAnimationFrame(callback) {
      return setTimeout(() => callback(Date.now()), 0);
    },
    cancelAnimationFrame(handle) {
      clearTimeout(handle);
    },
    get document() {
      return documentStub;
    }
  };

  const payload = {
    uri: 'file:///sample.onnx',
    fileName: 'sample.onnx',
    displayPath: 'test/sample.onnx',
    updatedAt: new Date('2026-04-06T00:00:00.000Z').toISOString(),
    fileStat: {
      mtime: new Date('2026-04-05T12:34:56.000Z').toISOString(),
      ctime: new Date('2026-04-05T12:34:56.000Z').toISOString(),
      size: sample.length
    },
    parseResult: {
      ok: true,
      parsed
    },
    byteLength: sample.length
  };

  const context = vm.createContext({
    console,
    setTimeout,
    clearTimeout,
    queueMicrotask,
    requestAnimationFrame: windowStub.requestAnimationFrame,
    cancelAnimationFrame: windowStub.cancelAnimationFrame,
    document: documentStub,
    window: windowStub,
    Element: FakeElement,
    HTMLElement: FakeElement,
    HTMLInputElement: FakeInputElement,
    HTMLDetailsElement: FakeDetailsElement,
    SVGElement: FakeSvgElement,
    acquireVsCodeApi() {
      return {
        postMessage(message) {
          vscodeMessages.push(message);
        },
        getState() {
          return persistedState;
        },
        setState(nextState) {
          persistedState = nextState;
        }
      };
    }
  });

  const instrumented = `${mainJs}\nwindow.__uiTest = { state, setTab, render, selectGraphNode, getAppHtml: () => app.innerHTML };`;
  vm.runInContext(instrumented, context, { filename: 'media/main.js' });

  assert.ok(vscodeMessages.some((message) => message?.command === 'requestModelData'), 'UI should request model data on startup');
  dispatch(windowListeners, 'message', { data: { command: 'modelData', payload } });
  await new Promise((resolve) => setTimeout(resolve, 10));

  const ui = context.window.__uiTest;
  assert.ok(ui, 'Expected UI test exports to be available');

  ui.setTab('graph');
  await new Promise((resolve) => setTimeout(resolve, 10));

  const graphHtml = ui.getAppHtml();
  assert.match(graphHtml, /Graph view/);
  assert.match(graphHtml, /graph-svg/);
  assert.match(graphHtml, /Plain-language explanation/);
  assert.match(graphHtml, /title="Center the selected node in the graph viewport"/);
  assert.doesNotMatch(graphHtml, /Click a block inside the graph to inspect it without moving the camera/);

  pageContent.scrollTop = 280;
  pageContent.scrollLeft = 12;
  graphScroll.scrollLeft = 140;
  graphScroll.scrollTop = 520;
  ui.selectGraphNode('node:1', { focusInView: false, interactionSource: 'graph-or-detail' });
  await new Promise((resolve) => setTimeout(resolve, 10));
  assert.equal(ui.state.selectedGraphNodeId, 'node:1');
  assert.equal(pageContent.scrollTop, 280, 'Selecting a node from the graph should preserve the outer panel scroll position');
  assert.equal(pageContent.scrollLeft, 12, 'Selecting a node from the graph should preserve the outer panel horizontal scroll position');
  assert.equal(graphScroll.scrollLeft, 140, 'Selecting a node from the graph should preserve the graph viewport X position');
  assert.equal(graphScroll.scrollTop, 520, 'Selecting a node from the graph should preserve the graph viewport Y position');

  graphScroll.scrollLeft = 0;
  graphScroll.scrollTop = 0;
  ui.selectGraphNode('node:2', { focusInView: true, interactionSource: 'sidebar' });
  await new Promise((resolve) => setTimeout(resolve, 10));
  assert.equal(ui.state.selectedGraphNodeId, 'node:2');
  assert.ok(graphScroll.scrollTop > 0 || graphScroll.scrollLeft > 0, 'Sidebar-driven selection should center the target node in view');
  assert.equal(pageContent.scrollTop, 280, 'Centering a graph node should still preserve the outer panel scroll position');
  assert.match(ui.getAppHtml(), /Graph search/);

  const ptPayload = {
    uri: 'file:///sample.pt',
    fileName: 'sample.pt',
    displayPath: 'test/sample.pt',
    updatedAt: new Date('2026-04-06T00:00:00.000Z').toISOString(),
    fileStat: {
      mtime: new Date('2026-04-05T12:34:56.000Z').toISOString(),
      ctime: new Date('2026-04-05T12:34:56.000Z').toISOString(),
      size: 2048
    },
    parseResult: {
      ok: true,
      format: 'pt',
      parsed: {
        format: 'pt',
        summary: {
          graphName: 'OrderedDict',
          irVersion: '',
          irVersionLabel: '',
          producerName: 'PyTorch',
          producerVersion: '2.6.0',
          domain: '',
          modelVersion: '',
          docString: '',
          opsets: [],
          fileSizeBytes: 2048,
          fileSizeLabel: '2.00 KB',
          temporalMetadata: { primary: null, candidates: [] },
          stats: {
            nodeCount: 0,
            inputCount: 0,
            outputCount: 0,
            valueInfoCount: 0,
            initializerCount: 2,
            metadataCount: 2,
            graphMetadataCount: 2,
            externalTensorCount: 0,
            estimatedParameterBytes: 24,
            estimatedParameterBytesLabel: '24 B'
          }
        },
        metadata: [
          { key: 'root.epoch', value: '12', valuePreview: '12', isJson: false, jsonValue: null },
          { key: 'root.best_metric', value: '0.98', valuePreview: '0.98', isJson: false, jsonValue: null }
        ],
        graphMetadata: [
          { key: 'root.state_dict', value: 'mapping · 2 entries', valuePreview: 'mapping · 2 entries', isJson: false, jsonValue: null },
          { key: 'root.optimizer', value: 'mapping · 1 entries', valuePreview: 'mapping · 1 entries', isJson: false, jsonValue: null }
        ],
        checkpointSections: [
          { id: 'section:0:root.state_dict', path: 'root.state_dict', kind: 'mapping', summary: '2 entries', entryCount: 2 },
          { id: 'section:1:root.optimizer', path: 'root.optimizer', kind: 'mapping', summary: '1 entries', entryCount: 1 }
        ],
        inputs: [],
        outputs: [],
        valueInfos: [],
        initializers: [
          { name: 'root.state_dict.layer.weight', dataTypeName: 'float32', shapeSummary: '[2 × 2]', elementCount: 4, estimatedBytes: '16 B', storage: { summary: 'tensor' }, metadata: [] },
          { name: 'root.state_dict.layer.bias', dataTypeName: 'float32', shapeSummary: '[2]', elementCount: 2, estimatedBytes: '8 B', storage: { summary: 'tensor' }, metadata: [] }
        ],
        nodes: [],
        graphView: { displayNodes: [], edges: [], layout: { positions: {} }, stats: { displayNodeCount: 0, edgeCount: 0, operatorNodeCount: 0, collapsedConstantCount: 0 } }
      }
    },
    byteLength: 2048
  };

  dispatch(windowListeners, 'message', { data: { command: 'modelData', payload: ptPayload } });
  await new Promise((resolve) => setTimeout(resolve, 10));
  ui.setTab('overview');
  await new Promise((resolve) => setTimeout(resolve, 10));
  assert.match(ui.getAppHtml(), /PyTorch Checkpoint Inspector/);
  assert.match(ui.getAppHtml(), /Checkpoint overview/);
  assert.match(ui.getAppHtml(), /Tensor preview/);

  ui.setTab('graph');
  await new Promise((resolve) => setTimeout(resolve, 10));
  assert.match(ui.getAppHtml(), /Graph view unavailable for PT checkpoints/);

  ui.setTab('metadata');
  await new Promise((resolve) => setTimeout(resolve, 10));
  assert.match(ui.getAppHtml(), /Checkpoint detail/);
  assert.match(ui.getAppHtml(), /root\.state_dict/);

  ui.setTab('io');
  await new Promise((resolve) => setTimeout(resolve, 10));
  assert.match(ui.getAppHtml(), /Tensor and checkpoint contents/);
  assert.match(ui.getAppHtml(), /Checkpoint sections/);
  assert.match(ui.getAppHtml(), /root\.state_dict\.layer\.weight/);
}

function runPackagingChecks() {
  execFileSync('python3', [path.join(root, 'scripts', 'package_vsix.py'), '--dry-run'], { stdio: 'inherit' });
  execFileSync('python3', [path.join(root, 'scripts', 'package_vsix.py')], { stdio: 'inherit' });
  execFileSync('python3', [path.join(root, 'scripts', 'package_source_zip.py')], { stdio: 'inherit' });

  const packageJson = JSON.parse(fs.readFileSync(path.join(root, 'package.json'), 'utf8'));
  const archivePath = path.join(distDir, `${packageJson.name}-${packageJson.version}.vsix`);
  const sourcePath = path.join(distDir, `${packageJson.name}-${packageJson.version}-source.zip`);
  assert.ok(fs.existsSync(archivePath), 'VSIX file was not created');
  assert.ok(fs.existsSync(sourcePath), 'Source zip was not created');

  const pythonCheck = [
    'import sys, zipfile',
    `archive = r"${archivePath}"`,
    'expected = {"[Content_Types].xml", "extension.vsixmanifest", "extension/package.json", "extension/extension.js", "extension/README.md", "extension/CHANGELOG.md", "extension/LICENSE.txt", "extension/images/icon.png", "extension/media/main.js", "extension/media/styles.css", "extension/media/lib/onnx-parser.mjs", "extension/media/lib/dagre.mjs", "extension/docs/HOW_IT_WORKS.md", "extension/docs/INSTALL_AND_PUBLISH.md"}',
    'with zipfile.ZipFile(archive) as zf:',
    '    names = set(zf.namelist())',
    'missing = sorted(expected - names)',
    'assert not missing, f"Missing files in VSIX: {missing}"',
    'print("VSIX structure looks correct.")'
  ].join('\n');
  execFileSync('python3', ['-c', pythonCheck], { stdio: 'inherit' });
}
