const vscode = typeof acquireVsCodeApi === 'function'
    ? acquireVsCodeApi()
    : {
        postMessage() {},
        getState() { return undefined; },
        setState() {}
    };

const persisted = vscode.getState() || {};
const state = {
    activeTab: persisted.activeTab || 'overview',
    graphSearch: persisted.graphSearch || '',
    metadataSearch: persisted.metadataSearch || '',
    ioSearch: persisted.ioSearch || '',
    selectedGraphNodeId: persisted.selectedGraphNodeId || '',
    selectedMetadataId: persisted.selectedMetadataId || '',
    graphZoom: clampGraphZoom(Number(persisted.graphZoom) || 1),
    graphScrollLeft: Number(persisted.graphScrollLeft) || 0,
    graphScrollTop: Number(persisted.graphScrollTop) || 0,
    graphViewportInitialized: Boolean(persisted.graphViewportInitialized),
    sectionOpen: normalizeSectionOpenState(persisted.sectionOpen),
    tabScroll: normalizeTabScrollState(persisted.tabScroll),
    modelData: null
};

let scheduledStatePersist = 0;
let graphPanSession = null;
let graphInitialFitPending = !state.graphViewportInitialized && state.graphScrollLeft === 0 && state.graphScrollTop === 0 && Math.abs(state.graphZoom - 1) < 0.0001;
let graphFocusRequest = null;
let graphFocusRequestVersion = 0;
let graphFocusAnimationFrame = 0;
let lastRenderedTab = state.activeTab;
let pageScrollRestoreAnimationFrame = 0;
let graphViewportRestoreDepth = 0;
const DEFAULT_DETAIL_SECTIONS = [
    'detail:explanation',
    'detail:upstream',
    'detail:downstream',
    'detail:data-inputs',
    'detail:outputs',
    'detail:embedded-parameters',
    'detail:merged-constants',
    'detail:attributes',
    'detail:metadata',
    'detail:tensor-summary'
];

const app = document.getElementById('app');

window.addEventListener('message', (event) => {
    const message = event.data;
    if (!message || message.command !== 'modelData') {
        return;
    }
    state.modelData = message.payload;
    graphInitialFitPending = !state.graphViewportInitialized && state.graphScrollLeft === 0 && state.graphScrollTop === 0 && Math.abs(state.graphZoom - 1) < 0.0001;
    initializeSelections();
    render();
});

document.addEventListener('click', (event) => {
    const actionElement = event.target.closest('[data-action]');
    if (!actionElement) {
        return;
    }
    const action = actionElement.dataset.action;
    switch (action) {
        case 'set-tab':
            setTab(actionElement.dataset.tab);
            break;
        case 'request-reload':
            vscode.postMessage({ command: 'requestReload' });
            break;
        case 'select-graph-node':
            selectGraphNode(actionElement.dataset.nodeId, { focusInView: false, interactionSource: 'graph-or-detail' });
            break;
        case 'select-graph-node-focus':
            selectGraphNode(actionElement.dataset.nodeId, { focusInView: true, interactionSource: 'sidebar' });
            break;
        case 'open-graph-node':
            state.activeTab = 'graph';
            selectGraphNode(actionElement.dataset.nodeId, { focusInView: true, interactionSource: 'overview' });
            break;
        case 'select-metadata':
            state.selectedMetadataId = actionElement.dataset.metadataId || '';
            persistUiState();
            render();
            break;
        case 'copy-metadata-value':
            copySelectedMetadataValue(false);
            break;
        case 'copy-metadata-json':
            copySelectedMetadataValue(true);
            break;
        case 'metadata-expand-all':
            setJsonExpansion(true);
            break;
        case 'metadata-collapse-all':
            setJsonExpansion(false);
            break;
        case 'focus-selected-graph-node':
            selectGraphNode(state.selectedGraphNodeId, { focusInView: true, interactionSource: 'detail' });
            break;
        case 'copy-text': {
            const text = actionElement.dataset.text || '';
            if (text) {
                vscode.postMessage({ command: 'copyText', payload: { text } });
            }
            break;
        }
        case 'graph-zoom-in':
            setGraphZoom(state.graphZoom * 1.15);
            break;
        case 'graph-zoom-out':
            setGraphZoom(state.graphZoom / 1.15);
            break;
        case 'graph-zoom-reset':
            setGraphZoom(1);
            break;
        case 'graph-zoom-fit':
            fitGraphToViewport();
            break;
        case 'detail-sections-expand-all':
            setSectionScopeExpanded('detail', true);
            break;
        case 'detail-sections-collapse-all':
            setSectionScopeExpanded('detail', false);
            break;
        default:
            break;
    }
});

document.addEventListener('input', (event) => {
    const target = event.target;
    if (!(target instanceof HTMLInputElement)) {
        return;
    }
    switch (target.dataset.role) {
        case 'graph-search':
            state.graphSearch = target.value;
            syncGraphSelectionToSearch();
            persistUiState();
            render();
            break;
        case 'metadata-search':
            state.metadataSearch = target.value;
            syncMetadataSelectionToSearch();
            persistUiState();
            render();
            break;
        case 'io-search':
            state.ioSearch = target.value;
            persistUiState();
            render();
            break;
        case 'graph-zoom': {
            const nextZoom = Number(target.value) / 100;
            if (Number.isFinite(nextZoom)) {
                setGraphZoom(nextZoom);
            }
            break;
        }
        default:
            break;
    }
});

document.addEventListener('toggle', (event) => {
    const target = event.target;
    if (!(target instanceof HTMLDetailsElement) || target.dataset.role !== 'remembered-section') {
        return;
    }
    const sectionKey = target.dataset.sectionKey || '';
    if (!sectionKey) {
        return;
    }
    state.sectionOpen[sectionKey] = target.open;
    schedulePersistUiState();
}, true);

document.addEventListener('scroll', (event) => {
    const target = event.target;
    if (!(target instanceof HTMLElement) || target.id !== 'graph-scroll-container') {
        return;
    }
    if (graphViewportRestoreDepth > 0) {
        return;
    }
    state.graphScrollLeft = target.scrollLeft;
    state.graphScrollTop = target.scrollTop;
    state.graphViewportInitialized = true;
    schedulePersistUiState();
}, true);

document.addEventListener('wheel', (event) => {
    if (!event.ctrlKey) {
        return;
    }
    const target = event.target;
    if (!(target instanceof Element)) {
        return;
    }
    const container = target.closest('#graph-scroll-container');
    if (!(container instanceof HTMLElement)) {
        return;
    }
    event.preventDefault();
    const factor = event.deltaY < 0 ? 1.1 : 1 / 1.1;
    setGraphZoom(state.graphZoom * factor, {
        anchorClientX: event.clientX,
        anchorClientY: event.clientY
    });
}, { passive: false });

const completeGraphPan = () => {
    if (!graphPanSession) {
        return;
    }
    graphPanSession.container.classList.remove('is-panning');
    graphPanSession = null;
    state.graphViewportInitialized = true;
    schedulePersistUiState();
};

document.addEventListener('pointerdown', (event) => {
    if (event.button !== 0) {
        return;
    }
    const target = event.target;
    if (!(target instanceof Element)) {
        return;
    }
    const container = target.closest('#graph-scroll-container');
    if (!(container instanceof HTMLElement)) {
        return;
    }
    if (target.closest('.graph-node') || target.closest('.graph-zoom-overlay') || target.closest('button, input, textarea, select')) {
        return;
    }
    graphPanSession = {
        pointerId: event.pointerId,
        startClientX: event.clientX,
        startClientY: event.clientY,
        startScrollLeft: container.scrollLeft,
        startScrollTop: container.scrollTop,
        container
    };
    container.classList.add('is-panning');
    container.setPointerCapture?.(event.pointerId);
    event.preventDefault();
});

document.addEventListener('pointermove', (event) => {
    if (!graphPanSession || event.pointerId !== graphPanSession.pointerId) {
        return;
    }
    const deltaX = event.clientX - graphPanSession.startClientX;
    const deltaY = event.clientY - graphPanSession.startClientY;
    graphPanSession.container.scrollLeft = graphPanSession.startScrollLeft - deltaX;
    graphPanSession.container.scrollTop = graphPanSession.startScrollTop - deltaY;
    state.graphScrollLeft = graphPanSession.container.scrollLeft;
    state.graphScrollTop = graphPanSession.container.scrollTop;
});

document.addEventListener('pointerup', (event) => {
    if (!graphPanSession || event.pointerId !== graphPanSession.pointerId) {
        return;
    }
    completeGraphPan();
});

document.addEventListener('pointercancel', completeGraphPan);
window.addEventListener('blur', completeGraphPan);

requestInitialData();
render();

function requestInitialData() {
    vscode.postMessage({ command: 'requestModelData' });
}

function initializeSelections() {
    const parsed = getParsed();
    if (!parsed) {
        return;
    }

    const graphNodes = parsed.graphView?.displayNodes || [];
    if (!state.selectedGraphNodeId || !graphNodes.some((node) => node.id === state.selectedGraphNodeId)) {
        const preferred = graphNodes.find((node) => node.kind === 'operator') || graphNodes[0];
        state.selectedGraphNodeId = preferred?.id || '';
    }

    const metadataEntries = getCombinedMetadataEntries(parsed);
    if (!state.selectedMetadataId || !metadataEntries.some((entry) => entry.id === state.selectedMetadataId)) {
        state.selectedMetadataId = metadataEntries[0]?.id || '';
    }
    persistUiState();
}

function persistUiState() {
    vscode.setState({
        activeTab: state.activeTab,
        graphSearch: state.graphSearch,
        metadataSearch: state.metadataSearch,
        ioSearch: state.ioSearch,
        selectedGraphNodeId: state.selectedGraphNodeId,
        selectedMetadataId: state.selectedMetadataId,
        graphZoom: state.graphZoom,
        graphScrollLeft: state.graphScrollLeft,
        graphScrollTop: state.graphScrollTop,
        graphViewportInitialized: state.graphViewportInitialized,
        sectionOpen: state.sectionOpen,
        tabScroll: state.tabScroll
    });
}

function schedulePersistUiState() {
    if (scheduledStatePersist) {
        return;
    }
    scheduledStatePersist = window.setTimeout(() => {
        scheduledStatePersist = 0;
        persistUiState();
    }, 80);
}

function setTab(tab) {
    if (!tab || tab === state.activeTab) {
        return;
    }
    state.activeTab = tab;
    persistUiState();
    render();
}

function normalizeSectionOpenState(sectionOpen) {
    if (!sectionOpen || typeof sectionOpen !== 'object') {
        return {};
    }
    return Object.fromEntries(
        Object.entries(sectionOpen)
            .filter(([key]) => typeof key === 'string' && key)
            .map(([key, value]) => [key, Boolean(value)])
    );
}

function normalizeTabScrollState(tabScroll) {
    if (!tabScroll || typeof tabScroll !== 'object') {
        return {};
    }
    return Object.fromEntries(
        Object.entries(tabScroll)
            .filter(([key]) => typeof key === 'string' && key)
            .map(([key, value]) => {
                const left = Math.max(0, Number(value?.left) || 0);
                const top = Math.max(0, Number(value?.top) || 0);
                return [key, { left, top }];
            })
    );
}

function capturePageViewportFromDom(tabKey = lastRenderedTab || state.activeTab) {
    if (!tabKey) {
        return;
    }
    const container = document.getElementById('page-content');
    if (!(container instanceof HTMLElement)) {
        return;
    }
    state.tabScroll[tabKey] = {
        left: container.scrollLeft,
        top: container.scrollTop
    };
}

function restorePageViewportAfterRender(tabKey = state.activeTab) {
    if (pageScrollRestoreAnimationFrame) {
        cancelAnimationFrame(pageScrollRestoreAnimationFrame);
        pageScrollRestoreAnimationFrame = 0;
    }

    const apply = () => {
        const container = document.getElementById('page-content');
        if (!(container instanceof HTMLElement)) {
            return;
        }
        const snapshot = state.tabScroll[tabKey] || { left: 0, top: 0 };
        container.scrollLeft = snapshot.left || 0;
        container.scrollTop = snapshot.top || 0;
    };

    apply();
    pageScrollRestoreAnimationFrame = requestAnimationFrame(() => {
        pageScrollRestoreAnimationFrame = 0;
        apply();
    });
}

function beginGraphViewportRestore() {
    graphViewportRestoreDepth += 1;
}

function endGraphViewportRestore() {
    graphViewportRestoreDepth = Math.max(0, graphViewportRestoreDepth - 1);
}

function withGraphViewportRestoreLock(callback) {
    beginGraphViewportRestore();
    try {
        return callback();
    } finally {
        endGraphViewportRestore();
    }
}

function selectGraphNode(nodeId, options = {}) {
    const nextId = nodeId || '';
    const focusInView = Boolean(options.focusInView);
    cancelPendingGraphFocus({ preserveCurrentScroll: focusInView });
    state.selectedGraphNodeId = nextId;
    if (focusInView && nextId) {
        graphFocusRequest = {
            nodeId: nextId,
            version: ++graphFocusRequestVersion
        };
    }
    persistUiState();
    render();
}

function cancelPendingGraphFocus(options = {}) {
    graphFocusRequestVersion += 1;
    graphFocusRequest = null;
    if (graphFocusAnimationFrame) {
        cancelAnimationFrame(graphFocusAnimationFrame);
        graphFocusAnimationFrame = 0;
    }
    if (!options.preserveCurrentScroll) {
        stopGraphAutoScroll();
    }
}

function stopGraphAutoScroll() {
    const container = document.getElementById('graph-scroll-container');
    if (!(container instanceof HTMLElement)) {
        return;
    }
    const currentLeft = container.scrollLeft;
    const currentTop = container.scrollTop;
    withGraphViewportRestoreLock(() => {
        container.scrollTo({ left: currentLeft, top: currentTop, behavior: 'auto' });
    });
    state.graphScrollLeft = currentLeft;
    state.graphScrollTop = currentTop;
}

function copySelectedMetadataValue(asJson) {
    const parsed = getParsed();
    if (!parsed) {
        return;
    }
    const entry = getCombinedMetadataEntries(parsed).find((item) => item.id === state.selectedMetadataId);
    if (!entry) {
        return;
    }
    const text = asJson && entry.isJson ? entry.formattedJson : entry.value;
    if (text) {
        vscode.postMessage({ command: 'copyText', payload: { text } });
    }
}

function getParsed() {
    return state.modelData?.parseResult?.ok ? state.modelData.parseResult.parsed : null;
}

function getCombinedMetadataEntries(parsed) {
    return [
        ...(parsed.metadata || []).map((entry, index) => ({
            ...entry,
            id: `model:${index}:${entry.key}`,
            scope: 'model'
        })),
        ...(parsed.graphMetadata || []).map((entry, index) => ({
            ...entry,
            id: `graph:${index}:${entry.key}`,
            scope: 'graph'
        }))
    ];
}

function syncGraphSelectionToSearch() {
    const parsed = getParsed();
    if (!parsed) {
        return;
    }
    const query = state.graphSearch.trim().toLowerCase();
    const candidates = query
        ? (parsed.graphView?.displayNodes || []).filter((node) => searchableNodeText(node).includes(query))
        : (parsed.graphView?.displayNodes || []).filter((node) => node.kind === 'operator');
    if (candidates.length) {
        cancelPendingGraphFocus();
        state.selectedGraphNodeId = candidates[0].id;
    }
}

function syncMetadataSelectionToSearch() {
    const parsed = getParsed();
    if (!parsed) {
        return;
    }
    const query = state.metadataSearch.trim().toLowerCase();
    const entries = getCombinedMetadataEntries(parsed);
    const matches = query
        ? entries.filter((entry) => (`${entry.scope} ${entry.key} ${entry.value}`).toLowerCase().includes(query))
        : entries;
    if (matches.length) {
        state.selectedMetadataId = matches[0].id;
    }
}

function render() {
    capturePageViewportFromDom(lastRenderedTab);
    if (lastRenderedTab === 'graph') {
        captureGraphViewportFromDom();
    }

    const lockGraphViewport = state.activeTab === 'graph';
    if (lockGraphViewport) {
        beginGraphViewportRestore();
    }

    try {
        if (!state.modelData) {
            app.innerHTML = renderLoading('Waiting for model data…');
            return;
        }

        if (!state.modelData.parseResult?.ok) {
            app.innerHTML = renderError(state.modelData);
            return;
        }

        const parsed = getParsed();
        app.innerHTML = `
            <div class="page-shell">
                ${renderHeader(parsed)}
                ${renderTabs()}
                <div class="page-content" id="page-content">
                    ${renderActiveTab(parsed)}
                </div>
            </div>
        `;
        if (state.activeTab === 'graph') {
            afterGraphRender(parsed);
        }
    } finally {
        restorePageViewportAfterRender(state.activeTab);
        lastRenderedTab = state.activeTab;
        if (lockGraphViewport) {
            endGraphViewportRestore();
        }
    }
}

function renderLoading(message) {
    return `
        <div class="loading-state">
            <div class="spinner"></div>
            <div>
                <div class="loading-title">Loading ONNX model…</div>
                <div class="loading-subtitle">${escapeHtml(message)}</div>
            </div>
        </div>
    `;
}

function renderError(modelData) {
    const error = modelData.parseResult?.error || {};
    return `
        <div class="page-shell">
            <div class="header-card">
                <div>
                    <div class="eyebrow">ONNX Model Inspector</div>
                    <h1 class="title-row">${escapeHtml(modelData.fileName || 'Unknown file')}</h1>
                    <div class="subtle-path">${escapeHtml(modelData.displayPath || modelData.uri || '')}</div>
                </div>
                <button class="primary-button" data-action="request-reload">Reload</button>
            </div>
            <div class="error-card">
                <h2>Could not parse this ONNX file</h2>
                <p>${escapeHtml(error.message || 'Unknown parsing error')}</p>
                <div class="kv-grid single-column">
                    <div class="kv-card">
                        <div class="kv-label">Error type</div>
                        <div class="kv-value monospace">${escapeHtml(error.name || 'Error')}</div>
                    </div>
                    <div class="kv-card">
                        <div class="kv-label">Stack trace</div>
                        <pre class="code-block">${escapeHtml(error.stack || 'No stack trace available.')}</pre>
                    </div>
                </div>
            </div>
        </div>
    `;
}

function renderHeader(parsed) {
    const summary = parsed.summary;
    const primaryTemporal = summary.temporalMetadata?.primary || null;
    const fileStat = state.modelData.fileStat || {};
    return `
        <div class="header-card">
            <div class="header-main">
                <div class="eyebrow">ONNX Model Inspector</div>
                <h1 class="title-row">${escapeHtml(state.modelData.fileName)}</h1>
                <div class="subtle-path">${escapeHtml(state.modelData.displayPath || state.modelData.uri)}</div>
                <div class="badge-row">
                    ${renderBadge(summary.graphName ? `Graph: ${summary.graphName}` : 'Unnamed graph')}
                    ${renderBadge(`IR ${escapeHtml(summary.irVersion || '—')}${summary.irVersionLabel ? ` · ${escapeHtml(summary.irVersionLabel)}` : ''}`)}
                    ${renderBadge(summary.fileSizeLabel || formatBytes(state.modelData.byteLength || 0))}
                    ${renderBadge(`${summary.stats.nodeCount} node${summary.stats.nodeCount === 1 ? '' : 's'}`)}
                    ${renderBadge(`${summary.stats.metadataCount + summary.stats.graphMetadataCount} metadata item${summary.stats.metadataCount + summary.stats.graphMetadataCount === 1 ? '' : 's'}`)}
                    ${primaryTemporal ? renderBadge(`Exported ${formatTimestamp(primaryTemporal.isoValue)}`) : ''}
                </div>
            </div>
            <div class="header-actions">
                ${fileStat.mtime ? `<div class="subtle-meta">File modified ${escapeHtml(formatTimestamp(fileStat.mtime))}</div>` : ''}
                <div class="subtle-meta">Reloaded ${escapeHtml(formatTimestamp(state.modelData.updatedAt))}</div>
                <button class="primary-button" data-action="request-reload">Reload</button>
            </div>
        </div>
    `;
}

function renderTabs() {
    const tabs = [
        ['overview', 'Overview'],
        ['graph', 'Graph'],
        ['metadata', 'Metadata'],
        ['io', 'I/O & Weights']
    ];
    return `
        <div class="tab-bar">
            ${tabs.map(([key, label]) => `
                <button
                    class="tab-button ${state.activeTab === key ? 'active' : ''}"
                    data-action="set-tab"
                    data-tab="${key}">
                    ${escapeHtml(label)}
                </button>
            `).join('')}
        </div>
    `;
}

function renderActiveTab(parsed) {
    switch (state.activeTab) {
        case 'graph':
            return renderGraphTab(parsed);
        case 'metadata':
            return renderMetadataTab(parsed);
        case 'io':
            return renderIoTab(parsed);
        case 'overview':
        default:
            return renderOverviewTab(parsed);
    }
}

function renderOverviewTab(parsed) {
    const summary = parsed.summary;
    const producer = [summary.producerName, summary.producerVersion].filter(Boolean).join(' ');
    const fileStat = state.modelData.fileStat || {};
    const temporal = summary.temporalMetadata || { primary: null, candidates: [] };
    return `
        <div class="overview-grid">
            <div class="panel-card">
                <h2>Summary</h2>
                <div class="stats-grid">
                    ${renderStatCard('Nodes', summary.stats.nodeCount)}
                    ${renderStatCard('Inputs', summary.stats.inputCount)}
                    ${renderStatCard('Outputs', summary.stats.outputCount)}
                    ${renderStatCard('Initializers', summary.stats.initializerCount)}
                    ${renderStatCard('Value Info', summary.stats.valueInfoCount)}
                    ${renderStatCard('Merged Constants', parsed.graphView?.stats?.collapsedConstantCount || 0)}
                    ${renderStatCard('Estimated Parameters', summary.stats.estimatedParameterBytesLabel)}
                </div>
            </div>
            <div class="panel-card">
                <h2>Model information</h2>
                <div class="kv-grid">
                    ${renderKvCard('Graph name', summary.graphName || '—')}
                    ${renderKvCard('Producer', producer || '—')}
                    ${renderKvCard('Domain', summary.domain || '—')}
                    ${renderKvCard('Model version', summary.modelVersion || '—')}
                    ${renderKvCard('IR version', `${summary.irVersion || '—'}${summary.irVersionLabel ? ` · ${summary.irVersionLabel}` : ''}`)}
                    ${renderKvCard('Opsets', summary.opsets.length ? summary.opsets.map((item) => `${item.domain}:${item.version}`).join(', ') : '—')}
                </div>
                ${summary.docString ? `<div class="doc-card"><div class="section-label">Model doc string</div><div class="doc-string">${escapeHtml(summary.docString)}</div></div>` : ''}
            </div>
            <div class="panel-card">
                <h2>File & export metadata</h2>
                <div class="kv-grid">
                    ${renderKvCard('Detected export time', temporal.primary ? formatTimestamp(temporal.primary.isoValue) : '—')}
                    ${renderKvCard('Timestamp candidates', temporal.candidates?.length || 0)}
                    ${renderKvCard('File modified', fileStat.mtime ? formatTimestamp(fileStat.mtime) : '—')}
                    ${renderKvCard('File created', fileStat.ctime ? formatTimestamp(fileStat.ctime) : '—')}
                    ${renderKvCard('Last reloaded', state.modelData.updatedAt ? formatTimestamp(state.modelData.updatedAt) : '—')}
                    ${renderKvCard('File size', summary.fileSizeLabel || formatBytes(fileStat.size || 0))}
                </div>
                ${temporal.candidates?.length ? `
                    <div class="detail-section">
                        <div class="section-label">Detected time fields</div>
                        <div class="detail-table">
                            ${temporal.candidates.slice(0, 8).map((candidate) => `
                                <div class="detail-row">
                                    <div class="detail-key">${escapeHtml(candidate.path)}</div>
                                    <div class="detail-value">${escapeHtml(formatTimestamp(candidate.isoValue))}</div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                ` : `<div class="subtle-meta">No obvious generated/exported timestamp was detected inside the ONNX metadata fields.</div>`}
            </div>
            <div class="panel-card">
                <h2>Metadata overview</h2>
                ${renderMetadataPreview(parsed)}
            </div>
            <div class="panel-card overview-wide-card">
                <h2>Top graph nodes</h2>
                <div class="chip-list">
                    ${(parsed.nodes || []).slice(0, 18).map((node) => `
                        <button class="chip-button" data-action="open-graph-node" data-node-id="${escapeAttribute(node.id)}" title="Open graph view and inspect ${escapeHtml(node.displayName)}">${escapeHtml(node.displayName)}</button>
                    `).join('')}
                    ${(parsed.nodes || []).length > 18 ? `<span class="subtle-meta">+${parsed.nodes.length - 18} more</span>` : ''}
                </div>
            </div>
        </div>
    `;
}

function renderMetadataPreview(parsed) {
    const entries = getCombinedMetadataEntries(parsed);
    if (!entries.length) {
        return `<div class="empty-state compact">No model-level or graph-level metadata was found.</div>`;
    }
    return `
        <div class="metadata-preview-list">
            ${entries.slice(0, 8).map((entry) => `
                <div class="metadata-preview-item">
                    <div>
                        <div class="section-label">${escapeHtml(entry.scope)}</div>
                        <div class="metadata-key">${escapeHtml(entry.key)}</div>
                    </div>
                    <div class="metadata-value-preview">${escapeHtml(entry.valuePreview)}</div>
                </div>
            `).join('')}
            ${entries.length > 8 ? `<div class="subtle-meta">Open the Metadata tab to inspect all ${entries.length} items.</div>` : ''}
        </div>
    `;
}

function renderGraphTab(parsed) {
    const graphView = parsed.graphView;
    const nodes = graphView.displayNodes || [];
    const query = state.graphSearch.trim().toLowerCase();
    const resultNodes = query
        ? nodes.filter((node) => searchableNodeText(node).includes(query))
        : nodes.filter((node) => node.kind === 'operator').slice(0, 250);

    const selectedNode = nodes.find((node) => node.id === state.selectedGraphNodeId)
        || nodes.find((node) => node.kind === 'operator')
        || nodes[0]
        || null;

    return `
        <div class="graph-layout">
            <div class="sidebar-card">
                <h2>Graph search</h2>
                <input
                    class="search-input"
                    type="text"
                    placeholder="Search nodes, operators, weights, or tensors"
                    value="${escapeAttribute(state.graphSearch)}"
                    data-role="graph-search" />
                <div class="subtle-meta">${resultNodes.length} result${resultNodes.length === 1 ? '' : 's'} · click a result to center it · graph clicks only change selection</div>
                <div class="result-list">
                    ${resultNodes.length ? resultNodes.map((node) => `
                        <button class="result-item ${node.id === selectedNode?.id ? 'selected' : ''}" data-action="select-graph-node-focus" data-node-id="${escapeAttribute(node.id)}">
                            <div class="result-title">${escapeHtml(node.title)}</div>
                            <div class="result-subtitle">${escapeHtml(node.kind === 'operator' ? `${node.details?.opType || node.subtitle}` : (node.subtitle || node.kind))}</div>
                        </button>
                    `).join('') : `<div class="empty-state compact">No graph nodes match your search.</div>`}
                </div>
            </div>
            <div class="panel-card graph-panel">
                <div class="panel-toolbar">
                    <div>
                        <h2>Graph view</h2>
                        <div class="subtle-meta">${graphView.stats.displayNodeCount} visual node${graphView.stats.displayNodeCount === 1 ? '' : 's'} · ${graphView.stats.edgeCount} edge${graphView.stats.edgeCount === 1 ? '' : 's'}${graphView.stats.collapsedConstantCount ? ` · ${graphView.stats.collapsedConstantCount} small Constant input${graphView.stats.collapsedConstantCount === 1 ? '' : 's'} merged into operators` : ''} · vertical layout · Ctrl + wheel to zoom · drag background to pan</div>
                    </div>
                    <div class="badge-row">
                        ${renderBadge('Vertical layered layout')}
                        ${renderBadge('Netron-like operator cards')}
                        ${renderBadge('Parameters embedded in operators')}
                        ${graphView.stats.collapsedConstantCount ? renderBadge(`${graphView.stats.collapsedConstantCount} Constant node${graphView.stats.collapsedConstantCount === 1 ? '' : 's'} merged`) : ''}
                        ${renderBadge('Inputs = blue')}
                        ${renderBadge('Outputs = purple')}
                    </div>
                </div>
                <div class="graph-stage">
                    <div id="graph-scroll-container" class="graph-scroll-container">
                        ${renderGraphSvg(parsed, selectedNode?.id || '', query)}
                    </div>
                    <div class="graph-zoom-overlay" aria-label="Graph zoom controls">
                        <button class="secondary-button graph-zoom-button" data-action="graph-zoom-in" title="Zoom in">+</button>
                        <div class="graph-zoom-slider-shell">
                            <input
                                class="graph-zoom-slider"
                                type="range"
                                min="20"
                                max="500"
                                step="5"
                                value="${Math.round(state.graphZoom * 100)}"
                                data-role="graph-zoom"
                                aria-label="Graph zoom percentage" />
                        </div>
                        <button class="secondary-button graph-zoom-button" data-action="graph-zoom-out" title="Zoom out">−</button>
                        <button class="secondary-button graph-zoom-pill" data-action="graph-zoom-reset" title="Reset zoom to 100%">100%</button>
                        <button class="secondary-button graph-zoom-pill" data-action="graph-zoom-fit" title="Fit graph into the viewport">Fit</button>
                        <div id="graph-zoom-label" class="graph-zoom-label">${Math.round(state.graphZoom * 100)}%</div>
                    </div>
                </div>
            </div>
            <div class="sidebar-card">
                <h2>Selection details</h2>
                ${selectedNode ? renderGraphSelectionDetails(selectedNode, parsed) : `<div class="empty-state compact">Select a graph node to inspect its details.</div>`}
            </div>
        </div>
    `;
}

function renderGraphSelectionDetails(node, parsed) {
    const details = node.details || {};
    const neighborhood = getGraphNeighborhood(parsed.graphView, node.id);
    const headerBadges = [renderBadge(node.kind)];
    if (node.kind === 'operator' && details.opType) {
        headerBadges.push(renderBadge(details.opType));
    }
    if (node.kind === 'operator' && details.renderStyle?.family) {
        headerBadges.push(renderBadge(details.renderStyle.family));
    }
    if (node.kind === 'operator' && details.domain) {
        headerBadges.push(renderBadge(details.domain));
    }
    if (node.kind !== 'operator' && details.typeSummary) {
        headerBadges.push(renderBadge(details.typeSummary));
    }
    if (node.kind !== 'operator' && details.dataTypeName) {
        headerBadges.push(renderBadge(details.dataTypeName));
    }

    const headerToolbar = `
        <div class="detail-toolbar">
            <button class="secondary-button" data-action="focus-selected-graph-node" title="Center the selected node in the graph viewport" aria-label="Center the selected node in the graph viewport">Reveal in graph</button>
            <button class="secondary-button subtle-button" data-action="detail-sections-expand-all" title="Expand every section in this details panel" aria-label="Expand every section in this details panel">Expand all</button>
            <button class="secondary-button subtle-button" data-action="detail-sections-collapse-all" title="Collapse every section in this details panel" aria-label="Collapse every section in this details panel">Collapse all</button>
        </div>
    `;

    const upstreamSection = renderRememberedSection(
        'detail:upstream',
        'Upstream',
        renderRelatedNodeButtons(neighborhood.incoming, 'No direct upstream nodes.'),
        { defaultOpen: true, summaryMeta: `${neighborhood.incoming.length}` }
    );
    const downstreamSection = renderRememberedSection(
        'detail:downstream',
        'Downstream',
        renderRelatedNodeButtons(neighborhood.outgoing, 'No direct downstream nodes.'),
        { defaultOpen: true, summaryMeta: `${neighborhood.outgoing.length}` }
    );

    if (node.kind === 'operator') {
        const embeddedParameters = details.embeddedParameters || [];
        const embeddedConstants = details.embeddedConstants || [];
        const dataInputs = details.dataInputs || [];
        const outputs = details.outputInfos || (details.outputs || []).map((name, index) => ({ index, name, typeSummary: '' }));
        const insightMarkup = renderOperatorInsight(details);
        return `
            <div class="detail-group">
                <div class="detail-header-row">
                    <div>
                        <div class="detail-title">${escapeHtml(details.displayName || node.title)}</div>
                        <div class="badge-row">${headerBadges.join('')}${renderBadge(`${neighborhood.incoming.length} ↑`)}${renderBadge(`${neighborhood.outgoing.length} ↓`)}</div>
                    </div>
                    ${headerToolbar}
                </div>
                ${details.docString ? `<div class="doc-string compact">${escapeHtml(details.docString)}</div>` : ''}
                ${details.displayName && details.opType && details.displayName !== details.opType ? `
                    <div class="detail-section-inline">
                        <div class="section-label">Node name</div>
                        <div class="detail-value monospace">${escapeHtml(details.displayName)}</div>
                    </div>
                ` : ''}
                ${renderRememberedSection('detail:explanation', 'Plain-language explanation', insightMarkup || `<div class="subtle-meta">No extra explanation is available for this operator yet.</div>`, { defaultOpen: true })}
                ${upstreamSection}
                ${downstreamSection}
                ${renderTensorInfoTableSection('detail:data-inputs', 'Data inputs', dataInputs, 'This operator has no runtime tensor inputs.', { defaultOpen: true })}
                ${renderTensorInfoTableSection('detail:outputs', 'Outputs', outputs, 'No outputs were recorded for this operator.', { defaultOpen: true })}
                ${renderKeyValueTableSection('detail:embedded-parameters', 'Embedded parameters', embeddedParameters.map((item) => ({
                    key: item.label,
                    value: [item.shapeLabel ? `⟨${compactShapeLabel(item.shapeLabel)}⟩` : '', item.dataTypeName, item.valuePreview].filter(Boolean).join(' · ')
                })), 'No embedded initializer parameters on this operator.', { defaultOpen: true })}
                ${renderKeyValueTableSection('detail:merged-constants', 'Merged Constant inputs', embeddedConstants.map((item) => ({
                    key: `${item.label}${item.sourceNodeTitle ? ` (${item.sourceNodeTitle})` : ''}`,
                    value: [item.shapeLabel ? `⟨${compactShapeLabel(item.shapeLabel)}⟩` : '', item.dataTypeName, item.valuePreview || item.valueSummary].filter(Boolean).join(' · ')
                })), 'No small Constant inputs were merged into this operator.', { defaultOpen: false })}
                ${renderKeyValueTableSection('detail:attributes', 'Attributes', (details.attributes || []).map((attribute) => ({
                    key: attribute.name || '(unnamed)',
                    value: [attribute.type, attribute.valueSummary].filter(Boolean).join(' · ')
                })), 'This operator has no explicit attributes.', { defaultOpen: false })}
                ${renderKeyValueTableSection('detail:metadata', 'Metadata', (details.metadata || []).map((entry) => ({
                    key: entry.key,
                    value: entry.valuePreview
                })), 'No operator-level metadata was found.', { defaultOpen: false })}
            </div>
        `;
    }

    const info = details;
    return `
        <div class="detail-group">
            <div class="detail-header-row">
                <div>
                    <div class="detail-title">${escapeHtml(node.title)}</div>
                    <div class="badge-row">${headerBadges.join('')}${renderBadge(`${neighborhood.incoming.length} ↑`)}${renderBadge(`${neighborhood.outgoing.length} ↓`)}</div>
                </div>
                ${headerToolbar}
            </div>
            ${info.docString ? `<div class="doc-string compact">${escapeHtml(info.docString)}</div>` : ''}
            ${upstreamSection}
            ${downstreamSection}
            ${renderRememberedSection('detail:tensor-summary', 'Tensor summary', `
                <div class="detail-table">
                    <div class="detail-row">
                        <div class="detail-key">Name</div>
                        <div class="detail-value monospace">${escapeHtml(info.name || node.title)}</div>
                    </div>
                    <div class="detail-row">
                        <div class="detail-key">Type</div>
                        <div class="detail-value">${escapeHtml(info.typeSummary || info.dataTypeName || '—')}</div>
                    </div>
                    ${info.denotation ? `
                        <div class="detail-row">
                            <div class="detail-key">Denotation</div>
                            <div class="detail-value">${escapeHtml(info.denotation)}</div>
                        </div>
                    ` : ''}
                </div>
            `, { defaultOpen: true })}
            ${renderKeyValueTableSection('detail:metadata', 'Metadata', (info.metadata || []).map((entry) => ({ key: entry.key, value: entry.valuePreview })), 'No metadata was found.', { defaultOpen: false })}
        </div>
    `;
}

function renderTensorInfoTableSection(sectionKey, label, items, emptyLabel, options = {}) {
    const body = (!items || !items.length)
        ? `<div class="subtle-meta">${escapeHtml(emptyLabel)}</div>`
        : `
            <div class="detail-table detail-table-wide">
                ${items.map((item) => `
                    <div class="detail-row detail-row-wide">
                        <div class="detail-key monospace">${escapeHtml(item.name || '(optional)')}</div>
                        <div class="detail-value">${escapeHtml(item.typeSummary || '—')}</div>
                    </div>
                `).join('')}
            </div>
        `;
    return renderRememberedSection(sectionKey, label, body, {
        defaultOpen: options.defaultOpen !== false,
        summaryMeta: items?.length ? `${items.length}` : '0'
    });
}

function renderKeyValueTableSection(sectionKey, label, rows, emptyLabel, options = {}) {
    const body = (!rows || !rows.length)
        ? `<div class="subtle-meta">${escapeHtml(emptyLabel)}</div>`
        : `
            <div class="detail-table">
                ${rows.map((row) => `
                    <div class="detail-row">
                        <div class="detail-key">${escapeHtml(row.key || '(unnamed)')}</div>
                        <div class="detail-value">${escapeHtml(row.value || '—')}</div>
                    </div>
                `).join('')}
            </div>
        `;
    return renderRememberedSection(sectionKey, label, body, {
        defaultOpen: options.defaultOpen !== false,
        summaryMeta: rows?.length ? `${rows.length}` : '0'
    });
}

function renderRememberedSection(sectionKey, title, body, options = {}) {
    const open = isRememberedSectionOpen(sectionKey, options.defaultOpen !== false);
    const summaryMeta = options.summaryMeta ? `<span class="detail-accordion-meta">${escapeHtml(options.summaryMeta)}</span>` : '';
    return `
        <details class="remembered-section detail-accordion" data-role="remembered-section" data-section-scope="detail" data-section-key="${escapeAttribute(sectionKey)}" ${open ? 'open' : ''}>
            <summary class="detail-accordion-summary">
                <span>${escapeHtml(title)}</span>
                ${summaryMeta}
            </summary>
            <div class="detail-accordion-body">${body}</div>
        </details>
    `;
}

function isRememberedSectionOpen(sectionKey, fallbackOpen = true) {
    return Object.prototype.hasOwnProperty.call(state.sectionOpen, sectionKey)
        ? Boolean(state.sectionOpen[sectionKey])
        : fallbackOpen;
}

function setSectionScopeExpanded(scope, expanded) {
    if (!scope) {
        return;
    }
    const defaults = scope === 'detail' ? DEFAULT_DETAIL_SECTIONS : [];
    defaults.forEach((key) => {
        state.sectionOpen[key] = expanded;
    });
    const nodes = document.querySelectorAll(`details[data-role="remembered-section"][data-section-scope="${scope}"]`);
    nodes.forEach((node) => {
        if (!(node instanceof HTMLDetailsElement)) {
            return;
        }
        node.open = expanded;
        const key = node.dataset.sectionKey || '';
        if (key) {
            state.sectionOpen[key] = expanded;
        }
    });
    persistUiState();
}

function renderOperatorInsight(details) {
    const opType = `${details.opType || ''}`.toLowerCase();
    const inputSummary = details.primaryInput?.typeSummary || '';
    const outputSummary = (details.outputInfos || [])[0]?.typeSummary || '';
    const inputType = shortTensorTypeLabel(inputSummary);
    const outputType = shortTensorTypeLabel(outputSummary);

    const activationSpec = getActivationPlotSpec(details);
    if (activationSpec) {
        return `
            <div class="insight-card">
                <div class="insight-title">${escapeHtml(activationSpec.title)}</div>
                <div class="insight-description">${escapeHtml(activationSpec.description)}</div>
                ${renderInsightFlagList(['Changes values element-wise', 'Keeps tensor shape'])}
                ${renderInsightBulletList([
                    'Each element is transformed independently.',
                    'The tensor shape usually stays the same before and after this step.',
                    opType === 'elu' ? 'Negative values are smoothly compressed instead of being cut off sharply.' : 'This adds non-linearity so the network can model more than a single linear projection.'
                ])}
                ${renderFunctionPlotSvg(activationSpec)}
                <div class="insight-meta-row monospace">${escapeHtml(activationSpec.formula)}</div>
                ${(inputType || outputType) ? renderTypeFlowSummaryCard(inputType, details.opType || 'Op', outputType) : ''}
            </div>
        `;
    }

    if (opType === 'unsqueeze' || opType === 'squeeze' || opType === 'reshape' || opType === 'transpose') {
        const summary = opType === 'unsqueeze'
            ? 'Adds a new dimension whose size is 1. It changes only the shape, not the numeric values.'
            : opType === 'squeeze'
                ? 'Removes dimensions whose size is 1. It changes only the shape, not the numeric values.'
                : opType === 'reshape'
                    ? 'Keeps the same values but reorganizes them under a new shape.'
                    : 'Reorders the axes of the tensor. The values stay the same, but they are viewed in a different axis order.';
        const bullets = opType === 'unsqueeze'
            ? [
                'It adds a size-1 axis so the next operator sees the shape it expects.',
                'Example: 1×169 becomes 1×1×169. The numbers stay the same.'
            ]
            : opType === 'squeeze'
                ? [
                    'It removes an axis whose size is 1, so the tensor becomes simpler to read and use.',
                    'Example: 1×1×512 becomes 1×512. The numbers stay the same.'
                ]
                : opType === 'reshape'
                    ? [
                        'The element count must stay the same before and after reshape.',
                        'This is usually used to prepare data for a linear layer or to recover a desired view.'
                    ]
                    : [
                        'Transpose changes axis order, such as swapping sequence-first and batch-first layouts.',
                        'The underlying numeric values are not recomputed here.'
                    ];
        return `
            <div class="insight-card">
                <div class="insight-title">${escapeHtml(details.opType || 'Shape transform')}</div>
                <div class="insight-description">${escapeHtml(summary)}</div>
                ${renderInsightFlagList(['Changes tensor shape', 'Keeps numeric values'])}
                ${renderInsightBulletList(bullets)}
                ${renderShapeEffectCard(inputSummary, outputSummary)}
                ${(details.embeddedConstants || []).length ? `
                    <div class="insight-meta-row monospace">${escapeHtml((details.embeddedConstants || []).map((item) => `${item.label}=${item.valuePreview || item.valueSummary || item.shapeLabel || '—'}`).join(' · '))}</div>
                ` : ''}
                ${(inputType || outputType) ? renderTypeFlowSummaryCard(inputType, details.opType || 'Op', outputType) : ''}
            </div>
        `;
    }

    if (opType === 'gemm' || opType === 'matmul') {
        const equation = opType === 'gemm' ? 'y = A·B + C' : 'y = A·B';
        return `
            <div class="insight-card">
                <div class="insight-title">Linear projection</div>
                <div class="insight-description">This mixes the incoming features with learned weights to produce a new feature vector.</div>
                ${renderInsightFlagList(['Changes values', 'Learns from weights'])}
                ${renderInsightBulletList([
                    'Think of this as a fully connected layer or matrix multiply.',
                    'The output size is set by the learned weight matrix.',
                    opType === 'gemm' ? 'A bias term is usually added after the matrix multiply.' : 'This step is a pure matrix multiply without an extra bias input.'
                ])}
                <div class="insight-meta-row monospace">${escapeHtml(equation)}</div>
                ${(inputType || outputType) ? renderTypeFlowSummaryCard(inputType, details.opType || 'Op', outputType) : ''}
            </div>
        `;
    }

    if (opType === 'gru' || opType === 'lstm' || opType === 'rnn') {
        return `
            <div class="insight-card">
                <div class="insight-title">Recurrent memory update</div>
                <div class="insight-description">This block combines the current input with the previous hidden state, then produces an updated hidden state for the next step.</div>
                ${renderInsightFlagList(['Uses state / memory', 'Changes values'])}
                ${renderInsightBulletList([
                    'Use this when the model needs memory across time steps.',
                    'The hidden-state output is what gets fed back into the next recurrent step.',
                    'Sequence output and hidden-state output can both appear depending on the model export.'
                ])}
                ${(inputType || outputType) ? renderTypeFlowSummaryCard(inputType, details.opType || 'Op', outputType) : ''}
            </div>
        `;
    }

    if (opType === 'batchnormalization' || opType === 'instancenormalization' || opType === 'layernormalization') {
        return `
            <div class="insight-card">
                <div class="insight-title">Normalization step</div>
                <div class="insight-description">This recenters and rescales the tensor so later layers see a more stable value range.</div>
                ${renderInsightFlagList(['Changes values', 'Keeps tensor shape'])}
                ${renderInsightBulletList([
                    'Learned scale and bias tune the final normalized output.',
                    'Stored mean and variance describe the typical value range seen during export or training.'
                ])}
                ${(inputType || outputType) ? renderTypeFlowSummaryCard(inputType, details.opType || 'Op', outputType) : ''}
            </div>
        `;
    }

    if (opType === 'constant') {
        const constantValue = details.attributes?.find((attribute) => `${attribute.name || ''}`.toLowerCase() === 'value')?.valueSummary || 'constant value';
        return `
            <div class="insight-card">
                <div class="insight-title">Constant tensor</div>
                <div class="insight-description">This creates a fixed value stored directly inside the ONNX graph.</div>
                ${renderInsightFlagList(['Creates fixed data', 'No runtime computation'])}
                ${renderInsightBulletList([
                    'Small helper constants are often merged into the operator that uses them.',
                    'Typical uses are axes, reshape targets, clipping bounds, or lookup constants.'
                ])}
                <div class="insight-meta-row monospace">${escapeHtml(constantValue)}</div>
            </div>
        `;
    }

    return `
        <div class="insight-card">
            <div class="insight-title">${escapeHtml(details.opType || 'Operator')}</div>
            <div class="insight-description">This operator transforms one or more input tensors into one or more output tensors.</div>
            ${renderInsightFlagList(['Inspect inputs and outputs below'])}
            ${(inputType || outputType) ? renderTypeFlowSummaryCard(inputType, details.opType || 'Op', outputType) : ''}
        </div>
    `;
}

function renderInsightFlagList(items) {
    if (!items || !items.length) {
        return '';
    }
    return `
        <div class="insight-flag-list">
            ${items.map((item) => `<span class="insight-flag">${escapeHtml(item)}</span>`).join('')}
        </div>
    `;
}

function renderInsightBulletList(items) {
    const filtered = (items || []).filter(Boolean);
    if (!filtered.length) {
        return '';
    }
    return `
        <ul class="insight-list">
            ${filtered.map((item) => `<li>${escapeHtml(item)}</li>`).join('')}
        </ul>
    `;
}

function renderShapeEffectCard(inputSummary, outputSummary) {
    const inputParts = extractTensorTypeParts(inputSummary);
    const outputParts = extractTensorTypeParts(outputSummary);
    if (!inputParts && !outputParts) {
        return '';
    }
    const inputValue = inputParts?.shape || inputParts?.dtype || 'unknown';
    const outputValue = outputParts?.shape || outputParts?.dtype || 'unknown';
    const dtypeLine = inputParts?.dtype || outputParts?.dtype
        ? `<div class="shape-effect-dtype monospace">dtype: ${escapeHtml(inputParts?.dtype || outputParts?.dtype || 'unknown')}</div>`
        : '';
    return `
        <div class="shape-effect-card">
            <div class="shape-effect-stage">
                <div class="type-flow-label">Before</div>
                <div class="shape-effect-value monospace">${escapeHtml(inputValue)}</div>
            </div>
            <div class="shape-effect-arrow">→</div>
            <div class="shape-effect-stage">
                <div class="type-flow-label">After</div>
                <div class="shape-effect-value monospace">${escapeHtml(outputValue)}</div>
            </div>
            ${dtypeLine}
        </div>
    `;
}

function extractTensorTypeParts(typeSummary) {
    const text = `${typeSummary || ''}`;
    const match = text.match(/^tensor<([^\s>]+)(?:\s+\[([^\]]+)\])?>$/i);
    if (!match) {
        return text ? { dtype: '', shape: text } : null;
    }
    return {
        dtype: match[1] || '',
        shape: match[2] ? compactShapeLabel(match[2]) : ''
    };
}

function getActivationPlotSpec(details) {
    const opType = `${details.opType || ''}`.toLowerCase();
    const alpha = getNumericAttribute(details, 'alpha', opType === 'leakyrelu' ? 0.01 : 1);
    const beta = getNumericAttribute(details, 'beta', 0.5);
    switch (opType) {
        case 'relu':
            return {
                title: 'ReLU activation',
                description: 'Passes positive values through unchanged and clamps negative values to zero.',
                formula: 'f(x) = max(0, x)',
                sample: (x) => Math.max(0, x),
                xMin: -4,
                xMax: 4
            };
        case 'elu':
            return {
                title: 'ELU activation',
                description: 'Keeps positive values linear while mapping negatives onto a smooth exponential curve.',
                formula: `f(x) = x (x>0), else α(e^x - 1); α=${formatCompactNumber(alpha)}`,
                sample: (x) => (x >= 0 ? x : alpha * (Math.exp(x) - 1)),
                xMin: -4,
                xMax: 4
            };
        case 'leakyrelu':
            return {
                title: 'LeakyReLU activation',
                description: 'Leaves a small negative slope instead of fully clamping the negative side.',
                formula: `f(x) = x (x>0), else αx; α=${formatCompactNumber(alpha)}`,
                sample: (x) => (x >= 0 ? x : alpha * x),
                xMin: -4,
                xMax: 4
            };
        case 'sigmoid':
            return {
                title: 'Sigmoid activation',
                description: 'Compresses the signal into the range (0, 1), often used for probabilities or gates.',
                formula: 'f(x) = 1 / (1 + e^-x)',
                sample: (x) => 1 / (1 + Math.exp(-x)),
                xMin: -6,
                xMax: 6
            };
        case 'tanh':
            return {
                title: 'Tanh activation',
                description: 'Compresses the signal into the range (-1, 1) with zero-centered output.',
                formula: 'f(x) = tanh(x)',
                sample: (x) => Math.tanh(x),
                xMin: -4,
                xMax: 4
            };
        case 'softplus':
            return {
                title: 'Softplus activation',
                description: 'A smooth approximation to ReLU that never has a hard corner.',
                formula: 'f(x) = log(1 + e^x)',
                sample: (x) => Math.log1p(Math.exp(x)),
                xMin: -6,
                xMax: 6
            };
        case 'softsign':
            return {
                title: 'Softsign activation',
                description: 'Squashes values into (-1, 1) using a rational function.',
                formula: 'f(x) = x / (1 + |x|)',
                sample: (x) => x / (1 + Math.abs(x)),
                xMin: -6,
                xMax: 6
            };
        case 'hardsigmoid':
            return {
                title: 'HardSigmoid activation',
                description: 'Uses a clipped linear approximation of sigmoid for cheap gating.',
                formula: `f(x) = clip(αx + β, 0, 1); α=${formatCompactNumber(alpha)}, β=${formatCompactNumber(beta)}`,
                sample: (x) => Math.max(0, Math.min(1, alpha * x + beta)),
                xMin: -6,
                xMax: 6
            };
        case 'selu': {
            const seluAlpha = getNumericAttribute(details, 'alpha', 1.6732632423543772);
            const gamma = getNumericAttribute(details, 'gamma', 1.0507009873554805);
            return {
                title: 'SELU activation',
                description: 'Scaled ELU used to keep activations in a stable range during self-normalizing training.',
                formula: `f(x) = γx (x>0), else γα(e^x - 1); γ=${formatCompactNumber(gamma)}, α=${formatCompactNumber(seluAlpha)}`,
                sample: (x) => (x >= 0 ? gamma * x : gamma * seluAlpha * (Math.exp(x) - 1)),
                xMin: -4,
                xMax: 4
            };
        }
        default:
            return null;
    }
}

function renderFunctionPlotSvg(spec) {
    const width = 232;
    const height = 140;
    const padding = { left: 18, right: 10, top: 12, bottom: 18 };
    const xMin = spec.xMin ?? -4;
    const xMax = spec.xMax ?? 4;
    const samples = [];
    for (let index = 0; index <= 96; index += 1) {
        const x = xMin + (index / 96) * (xMax - xMin);
        const y = spec.sample(x);
        if (Number.isFinite(y)) {
            samples.push({ x, y });
        }
    }
    const yValues = samples.map((item) => item.y).concat([0]);
    const yMin = Math.min(...yValues);
    const yMax = Math.max(...yValues);
    const yPad = Math.max(0.3, (yMax - yMin) * 0.12);
    const plotYMin = yMin - yPad;
    const plotYMax = yMax + yPad;
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;
    const toX = (value) => padding.left + ((value - xMin) / Math.max(0.0001, xMax - xMin)) * plotWidth;
    const toY = (value) => padding.top + (1 - ((value - plotYMin) / Math.max(0.0001, plotYMax - plotYMin))) * plotHeight;
    const xAxisY = toY(0);
    const yAxisX = toX(0);
    const path = samples.map((point, index) => `${index === 0 ? 'M' : 'L'} ${toX(point.x).toFixed(2)} ${toY(point.y).toFixed(2)}`).join(' ');
    return `
        <svg class="activation-plot" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(spec.title)} function plot">
            <rect class="activation-plot-bg" x="0.5" y="0.5" width="${width - 1}" height="${height - 1}" rx="12" ry="12"></rect>
            <line class="activation-plot-axis" x1="${padding.left}" y1="${xAxisY}" x2="${width - padding.right}" y2="${xAxisY}"></line>
            <line class="activation-plot-axis" x1="${yAxisX}" y1="${padding.top}" x2="${yAxisX}" y2="${height - padding.bottom}"></line>
            <path class="activation-plot-curve" d="${path}"></path>
            <text class="activation-plot-label" x="${padding.left}" y="${height - 4}">${escapeHtml(formatCompactNumber(xMin))}</text>
            <text class="activation-plot-label" x="${width - padding.right}" y="${height - 4}" text-anchor="end">${escapeHtml(formatCompactNumber(xMax))}</text>
            <text class="activation-plot-label" x="${width - 8}" y="${Math.max(12, xAxisY - 4)}" text-anchor="end">x</text>
            <text class="activation-plot-label" x="${Math.min(width - 10, yAxisX + 10)}" y="${padding.top + 8}">y</text>
        </svg>
    `;
}

function getNumericAttribute(details, name, fallbackValue) {
    const entry = (details.attributes || []).find((attribute) => `${attribute.name || ''}`.toLowerCase() === `${name || ''}`.toLowerCase());
    if (!entry) {
        return fallbackValue;
    }
    const direct = Number(entry.valueDetail);
    if (Number.isFinite(direct)) {
        return direct;
    }
    const fromSummary = Number.parseFloat(entry.valueSummary || '');
    return Number.isFinite(fromSummary) ? fromSummary : fallbackValue;
}

function shortTensorTypeLabel(typeSummary) {
    const text = `${typeSummary || ''}`;
    const tensorMatch = text.match(/^tensor<([^\s>]+)(?:\s+\[([^\]]+)\])?>$/i);
    if (tensorMatch) {
        const [, dtype, shape] = tensorMatch;
        const shapeLabel = shape ? compactShapeLabel(shape) : '';
        return shapeLabel ? `${dtype} · ${shapeLabel}` : dtype;
    }
    return text;
}

function renderTypeFlowSummaryCard(inputLabel, operatorLabel, outputLabel) {
    const before = inputLabel || 'input tensor';
    const op = operatorLabel || 'Op';
    const after = outputLabel || 'output tensor';
    return `
        <div class="type-flow-card">
            <div class="type-flow-stage">
                <div class="type-flow-label">Before</div>
                <div class="type-flow-value monospace">${escapeHtml(before)}</div>
            </div>
            <div class="type-flow-arrow monospace">${escapeHtml(op)} →</div>
            <div class="type-flow-stage">
                <div class="type-flow-label">After</div>
                <div class="type-flow-value monospace">${escapeHtml(after)}</div>
            </div>
        </div>
    `;
}

function compactShapeLabel(shapeLabel) {
    return `${shapeLabel || ''}`.replace(/^\[|\]$/g, '').replace(/\s*×\s*/g, '×');
}

function formatCompactNumber(value) {
    if (!Number.isFinite(value)) {
        return String(value);
    }
    if (Number.isInteger(value)) {
        return String(value);
    }
    return `${value}`.length <= 6 ? `${value}` : value.toFixed(4).replace(/\.0+$/, '').replace(/(\.\d*?)0+$/, '$1');
}

function renderRelatedNodeButtons(nodes, emptyLabel) {
    if (!nodes.length) {
        return `<div class="subtle-meta">${escapeHtml(emptyLabel)}</div>`;
    }
    return `
        <div class="chip-list related-node-list">
            ${nodes.map((node) => `
                <button class="chip-button related-node-button" data-action="select-graph-node" data-node-id="${escapeAttribute(node.id)}" title="Inspect ${escapeHtml(node.title)}">
                    <span class="related-node-title">${escapeHtml(truncate(node.title, 26))}</span>
                    <span class="related-node-subtitle">${escapeHtml(truncate(node.subtitle || node.kind, 24))}</span>
                </button>
            `).join('')}
        </div>
    `;
}

function renderGraphSvg(parsed, selectedNodeId, query) {
    const graphView = parsed.graphView;
    const { positions, width, height, routedEdges = {} } = graphView.layout;
    const matchingIds = new Set(
        query
            ? graphView.displayNodes.filter((node) => searchableNodeText(node).includes(query)).map((node) => node.id)
            : graphView.displayNodes.map((node) => node.id)
    );
    const neighborhood = selectedNodeId ? getGraphNeighborhood(graphView, selectedNodeId) : null;

    const edgeMarkup = graphView.edges.map((edge) => {
        const source = positions[edge.sourceId];
        const target = positions[edge.targetId];
        if (!source || !target) {
            return '';
        }
        const points = resolveEdgePoints(edge, positions, routedEdges);
        const path = roundedPolylinePath(points, 10);
        const queryDimmed = query && !(matchingIds.has(edge.sourceId) || matchingIds.has(edge.targetId));
        const classes = [
            'graph-edge-group',
            edge.isModelInput ? 'input-edge' : '',
            edge.isOutputEdge ? 'output-edge' : '',
            queryDimmed ? 'query-dimmed' : '',
            neighborhood?.edgeIds.has(edge.id) ? 'related' : '',
            edge.sourceId === selectedNodeId || edge.targetId === selectedNodeId ? 'selected' : ''
        ].filter(Boolean).join(' ');
        const labelMarkup = renderGraphEdgeLabel(edge, points);
        return `
            <g class="${classes}">
                <path class="graph-edge" d="${path}"></path>
                ${labelMarkup}
                <title>${escapeHtml(edge.tensorName)}${edge.typeSummary ? ` · ${escapeHtml(edge.typeSummary)}` : ''}</title>
            </g>
        `;
    }).join('');

    const nodeMarkup = graphView.displayNodes.map((node) => {
        const position = positions[node.id];
        if (!position) {
            return '';
        }
        const selected = node.id === selectedNodeId;
        const queryDimmed = query && !matchingIds.has(node.id);
        const classes = [
            'graph-node',
            `kind-${node.kind}`,
            selected ? 'selected' : '',
            neighborhood && neighborhood.nodeIds.has(node.id) && !selected ? 'related' : '',
            queryDimmed ? 'query-dimmed' : ''
        ].filter(Boolean).join(' ');
        return renderGraphNodeMarkup(node, position, classes);
    }).join('');

    return `
        <div class="graph-canvas" id="graph-canvas" data-base-width="${width}" data-base-height="${height}">
            <svg class="graph-svg" id="graph-svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" role="img" aria-label="ONNX graph visualization">
                <defs>
                    <marker id="arrowhead" markerWidth="10" markerHeight="8" refX="8" refY="4" orient="auto" markerUnits="strokeWidth">
                        <path d="M 0 0 L 10 4 L 0 8 z" class="graph-arrow"></path>
                    </marker>
                </defs>
                ${edgeMarkup}
                ${nodeMarkup}
            </svg>
        </div>
    `;
}

function renderGraphNodeMarkup(node, position, classes) {
    if (node.kind === 'operator') {
        const style = node.renderStyle || node.details?.renderStyle || { family: 'other', variant: 'standard', headerHeight: 34, rowHeight: 22, bodyVisible: true };
        const opLabel = truncate(node.details?.opType || node.subtitle || node.title || 'Operator', style.variant === 'micro' ? 14 : 26);
        const rows = Array.isArray(node.parameterRows) ? node.parameterRows : [];
        const familyClass = `family-${style.family || 'other'}`;
        const variantClass = `variant-${style.variant || 'standard'}`;

        if (style.variant === 'micro') {
            return `
                <g class="${classes} ${familyClass} ${variantClass}" data-action="select-graph-node" data-node-id="${escapeAttribute(node.id)}" transform="translate(${position.x}, ${position.y})">
                    <rect class="graph-node-rect graph-node-operator graph-node-pill" width="${position.width}" height="${position.height}" rx="${style.cornerRadius || 14}" ry="${style.cornerRadius || 14}"></rect>
                    <text class="graph-node-title graph-operator-title graph-operator-title-centered" x="${position.width / 2}" y="${position.height / 2 + 1}" text-anchor="middle">${escapeHtml(opLabel)}</text>
                    <title>${escapeHtml(node.title)}${node.details?.opType ? ` · ${escapeHtml(node.details.opType)}` : ''}</title>
                </g>
            `;
        }

        const headerHeight = style.headerHeight || 34;
        const rowHeight = style.rowHeight || 22;
        const rowMarkup = rows.map((row, index) => {
            const y = headerHeight + 15 + index * rowHeight;
            const dividerY = headerHeight + index * rowHeight + 5;
            return `
                <line class="graph-param-divider" x1="0" y1="${dividerY}" x2="${position.width}" y2="${dividerY}"></line>
                <text class="graph-param-label graph-param-kind-${escapeAttribute(row.kind || 'parameter')}" x="12" y="${y}">${escapeHtml(truncate(row.label || 'Param', 16))}</text>
                <text class="graph-param-value" x="${position.width - 12}" y="${y}" text-anchor="end">${escapeHtml(truncate(formatGraphRowValue(row), 24))}</text>
            `;
        }).join('');
        return `
            <g class="${classes} ${familyClass} ${variantClass}" data-action="select-graph-node" data-node-id="${escapeAttribute(node.id)}" transform="translate(${position.x}, ${position.y})">
                <rect class="graph-node-rect graph-node-operator" width="${position.width}" height="${position.height}" rx="${style.cornerRadius || 12}" ry="${style.cornerRadius || 12}"></rect>
                <rect class="graph-node-header" width="${position.width}" height="${headerHeight}" rx="${style.cornerRadius || 12}" ry="${style.cornerRadius || 12}"></rect>
                <line class="graph-node-divider" x1="0" y1="${headerHeight}" x2="${position.width}" y2="${headerHeight}"></line>
                <text class="graph-node-title graph-operator-title" x="12" y="${Math.max(17, headerHeight / 2 + 2)}">${escapeHtml(opLabel)}</text>
                ${rowMarkup}
                <title>${escapeHtml(node.title)}${node.details?.opType ? ` · ${escapeHtml(node.details.opType)}` : ''}</title>
            </g>
        `;
    }
    const title = truncate(node.title || '(unnamed)', 22);
    const subtitle = truncate(node.subtitle || node.kind, 26);
    return `
        <g class="${classes}" data-action="select-graph-node" data-node-id="${escapeAttribute(node.id)}" transform="translate(${position.x}, ${position.y})">
            <rect class="graph-node-rect graph-node-terminal" width="${position.width}" height="${position.height}" rx="10" ry="10"></rect>
            <text class="graph-node-title graph-terminal-title" x="12" y="16">${escapeHtml(title)}</text>
            <text class="graph-node-subtitle graph-terminal-subtitle" x="12" y="28">${escapeHtml(subtitle)}</text>
            <title>${escapeHtml(node.title)}${node.subtitle ? ` · ${escapeHtml(node.subtitle)}` : ''}</title>
        </g>
    `;
}

function formatGraphRowValue(row) {
    if (row?.shape) {
        return `⟨${compactShapeLabel(row.shape)}⟩`;
    }
    if (row?.secondaryValue) {
        return `${row.secondaryValue}`;
    }
    if (row?.preview) {
        return `${row.preview}`;
    }
    if (row?.dtype) {
        return `${row.dtype}`;
    }
    return '—';
}

function renderGraphEdgeLabel(edge, points) {
    const rawLabel = edge.shapeLabel || '';
    if (!rawLabel) {
        return '';
    }
    const label = truncate(rawLabel, 30);
    const point = edgeLabelPoint(points);
    if (!point) {
        return '';
    }
    const width = Math.max(44, Math.min(220, label.length * 6.2 + 14));
    return `
        <g class="graph-edge-label-group" transform="translate(${point.x - width / 2}, ${point.y - 10})">
            <rect class="graph-edge-label-bg" width="${width}" height="18" rx="9" ry="9"></rect>
            <text class="graph-edge-label" x="${width / 2}" y="12" text-anchor="middle">${escapeHtml(label)}</text>
        </g>
    `;
}

function edgeLabelPoint(points) {
    if (!Array.isArray(points) || points.length < 2) {
        return null;
    }
    const midIndex = Math.floor((points.length - 1) / 2);
    const left = points[midIndex];
    const right = points[midIndex + 1] || left;
    return {
        x: (left.x + right.x) / 2,
        y: (left.y + right.y) / 2
    };
}

function renderMetadataTab(parsed) {
    const allEntries = getCombinedMetadataEntries(parsed);
    const query = state.metadataSearch.trim().toLowerCase();
    const entries = query
        ? allEntries.filter((entry) => (`${entry.scope} ${entry.key} ${entry.value}`).toLowerCase().includes(query))
        : allEntries;
    const selectedEntry = entries.find((entry) => entry.id === state.selectedMetadataId)
        || allEntries.find((entry) => entry.id === state.selectedMetadataId)
        || entries[0]
        || null;

    return `
        <div class="split-layout">
            <div class="sidebar-card">
                <h2>Metadata entries</h2>
                <input
                    class="search-input"
                    type="text"
                    placeholder="Search metadata keys or values"
                    value="${escapeAttribute(state.metadataSearch)}"
                    data-role="metadata-search" />
                <div class="subtle-meta">${entries.length} visible of ${allEntries.length}</div>
                <div class="result-list metadata-results">
                    ${entries.length ? entries.map((entry) => `
                        <button class="result-item ${entry.id === selectedEntry?.id ? 'selected' : ''}" data-action="select-metadata" data-metadata-id="${escapeAttribute(entry.id)}">
                            <div class="result-title">${escapeHtml(entry.key)}</div>
                            <div class="result-subtitle">${escapeHtml(entry.scope)}${entry.isJson ? ' · JSON' : ''}</div>
                        </button>
                    `).join('') : `<div class="empty-state compact">No metadata matches your search.</div>`}
                </div>
            </div>
            <div class="panel-card metadata-panel">
                <div class="panel-toolbar">
                    <div>
                        <h2>Metadata detail</h2>
                        <div class="subtle-meta">Model-level and graph-level metadata from the ONNX container.</div>
                    </div>
                    ${selectedEntry ? `
                        <div class="toolbar-actions">
                            <button class="secondary-button" data-action="copy-metadata-value">Copy value</button>
                            ${selectedEntry.isJson ? `<button class="secondary-button" data-action="copy-metadata-json">Copy JSON</button>` : ''}
                            ${selectedEntry.isJson ? `<button class="secondary-button" data-action="metadata-expand-all">Expand all</button>` : ''}
                            ${selectedEntry.isJson ? `<button class="secondary-button" data-action="metadata-collapse-all">Collapse all</button>` : ''}
                        </div>
                    ` : ''}
                </div>
                ${selectedEntry ? `
                    <div class="metadata-detail-grid">
                        ${renderKvCard('Scope', selectedEntry.scope)}
                        ${renderKvCard('Key', selectedEntry.key)}
                    </div>
                    ${selectedEntry.isJson ? `
                        <div class="detail-section">
                            <div class="section-label">Pretty JSON</div>
                            <div class="json-tree-card">
                                ${renderJsonTree(selectedEntry.jsonValue)}
                            </div>
                        </div>
                    ` : ''}
                    <div class="detail-section">
                        <div class="section-label">Raw value</div>
                        <pre class="code-block">${escapeHtml(selectedEntry.value || '')}</pre>
                    </div>
                ` : `<div class="empty-state">This model does not expose metadata at the model or graph level.</div>`}
            </div>
        </div>
    `;
}

function getGraphNeighborhood(graphView, selectedNodeId) {
    const nodeById = new Map((graphView.displayNodes || []).map((node) => [node.id, node]));
    const incoming = [];
    const outgoing = [];
    const nodeIds = new Set(selectedNodeId ? [selectedNodeId] : []);
    const edgeIds = new Set();

    for (const edge of graphView.edges || []) {
        if (edge.targetId === selectedNodeId) {
            const source = nodeById.get(edge.sourceId);
            if (source) {
                incoming.push(source);
                nodeIds.add(source.id);
                edgeIds.add(edge.id);
            }
        }
        if (edge.sourceId === selectedNodeId) {
            const target = nodeById.get(edge.targetId);
            if (target) {
                outgoing.push(target);
                nodeIds.add(target.id);
                edgeIds.add(edge.id);
            }
        }
    }

    incoming.sort(graphNodeSort);
    outgoing.sort(graphNodeSort);
    return { incoming, outgoing, nodeIds, edgeIds };
}

function graphNodeSort(left, right) {
    const rank = (value) => {
        switch (value?.kind) {
            case 'operator': return 0;
            case 'input': return 1;
            case 'initializer': return 2;
            case 'output': return 3;
            default: return 4;
        }
    };
    return rank(left) - rank(right) || lexicalSort(left?.title, right?.title);
}

function lexicalSort(left, right) {
    return `${left || ''}`.localeCompare(`${right || ''}`, undefined, {
        numeric: true,
        sensitivity: 'base'
    });
}

function resolveEdgePoints(edge, positions, routedEdges) {
    const routed = routedEdges?.[edge.id];
    if (Array.isArray(routed) && routed.length >= 2) {
        return routed;
    }
    const source = positions[edge.sourceId];
    const target = positions[edge.targetId];
    if (!source || !target) {
        return [];
    }
    const startX = source.x + source.width / 2;
    const startY = source.y + source.height;
    const endX = target.x + target.width / 2;
    const endY = target.y;
    const midY = startY + Math.max(26, (endY - startY) / 2);
    return [
        { x: startX, y: startY },
        { x: startX, y: midY },
        { x: endX, y: midY },
        { x: endX, y: endY }
    ];
}

function roundedPolylinePath(points, radius = 8) {
    if (!Array.isArray(points) || points.length === 0) {
        return '';
    }
    if (points.length === 1) {
        return `M ${points[0].x} ${points[0].y}`;
    }
    if (points.length === 2) {
        return `M ${points[0].x} ${points[0].y} L ${points[1].x} ${points[1].y}`;
    }
    let path = `M ${points[0].x} ${points[0].y}`;
    for (let index = 1; index < points.length - 1; index += 1) {
        const previous = points[index - 1];
        const current = points[index];
        const next = points[index + 1];
        const previousDistance = Math.hypot(current.x - previous.x, current.y - previous.y);
        const nextDistance = Math.hypot(next.x - current.x, next.y - current.y);
        if (previousDistance < 0.001 || nextDistance < 0.001) {
            continue;
        }
        const effectiveRadius = Math.min(radius, previousDistance / 2, nextDistance / 2);
        const start = {
            x: current.x - ((current.x - previous.x) / previousDistance) * effectiveRadius,
            y: current.y - ((current.y - previous.y) / previousDistance) * effectiveRadius
        };
        const end = {
            x: current.x + ((next.x - current.x) / nextDistance) * effectiveRadius,
            y: current.y + ((next.y - current.y) / nextDistance) * effectiveRadius
        };
        path += ` L ${start.x} ${start.y} Q ${current.x} ${current.y} ${end.x} ${end.y}`;
    }
    const last = points[points.length - 1];
    path += ` L ${last.x} ${last.y}`;
    return path;
}

function renderIoTab(parsed) {
    const query = state.ioSearch.trim().toLowerCase();
    const filterItems = (items, getter) => query ? items.filter((item) => getter(item).includes(query)) : items;
    const inputs = filterItems(parsed.inputs || [], (item) => `${item.name} ${item.typeSummary}`.toLowerCase());
    const outputs = filterItems(parsed.outputs || [], (item) => `${item.name} ${item.typeSummary}`.toLowerCase());
    const initializers = filterItems(parsed.initializers || [], (item) => `${item.name} ${item.dataTypeName} ${item.shapeSummary}`.toLowerCase());
    const valueInfos = filterItems(parsed.valueInfos || [], (item) => `${item.name} ${item.typeSummary}`.toLowerCase());

    return `
        <div class="panel-card">
            <div class="panel-toolbar">
                <div>
                    <h2>Inputs, outputs, and weights</h2>
                    <div class="subtle-meta">Filter by name, shape, or data type.</div>
                </div>
                <input
                    class="search-input compact"
                    type="text"
                    placeholder="Filter I/O and weights"
                    value="${escapeAttribute(state.ioSearch)}"
                    data-role="io-search" />
            </div>
            <div class="io-sections">
                ${renderValueInfoSection('Inputs', inputs, ['Name', 'Type', 'Metadata'], (item) => [item.name, item.typeSummary, `${item.metadata.length}`])}
                ${renderValueInfoSection('Outputs', outputs, ['Name', 'Type', 'Metadata'], (item) => [item.name, item.typeSummary, `${item.metadata.length}`])}
                ${renderValueInfoSection('Initializers / Weights', initializers, ['Name', 'Type', 'Shape', 'Elements', 'Estimated size', 'Storage'], (item) => [item.name, item.dataTypeName, item.shapeSummary || '—', item.elementCount, item.estimatedBytes, item.storage.summary])}
                ${renderValueInfoSection('Value Info', valueInfos, ['Name', 'Type', 'Metadata'], (item) => [item.name, item.typeSummary, `${item.metadata.length}`])}
            </div>
        </div>
    `;
}

function renderValueInfoSection(title, items, columns, rowRenderer) {
    return `
        <details class="io-section" open>
            <summary>${escapeHtml(title)} <span class="summary-count">${items.length}</span></summary>
            ${items.length ? `
                <div class="table-scroll">
                    <table class="data-table">
                        <thead>
                            <tr>${columns.map((column) => `<th>${escapeHtml(column)}</th>`).join('')}</tr>
                        </thead>
                        <tbody>
                            ${items.map((item) => `
                                <tr>
                                    ${rowRenderer(item).map((value) => `<td>${escapeHtml(value)}</td>`).join('')}
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
            ` : `<div class="empty-state compact">No entries in this section.</div>`}
        </details>
    `;
}

function renderStatCard(label, value) {
    return `
        <div class="stat-card">
            <div class="stat-label">${escapeHtml(label)}</div>
            <div class="stat-value">${escapeHtml(String(value))}</div>
        </div>
    `;
}

function renderKvCard(label, value) {
    return `
        <div class="kv-card">
            <div class="kv-label">${escapeHtml(label)}</div>
            <div class="kv-value">${escapeHtml(String(value))}</div>
        </div>
    `;
}

function renderBadge(text) {
    return `<span class="badge">${escapeHtml(text)}</span>`;
}

function renderPillList(values, code) {
    if (!values || !values.length) {
        return `<div class="subtle-meta">None</div>`;
    }
    return `
        <div class="pill-list">
            ${values.map((value) => `<span class="pill ${code ? 'monospace' : ''}">${escapeHtml(value || '(optional)')}</span>`).join('')}
        </div>
    `;
}

function captureGraphViewportFromDom() {
    const container = document.getElementById('graph-scroll-container');
    if (!(container instanceof HTMLElement)) {
        return;
    }
    state.graphScrollLeft = container.scrollLeft;
    state.graphScrollTop = container.scrollTop;
}

function afterGraphRender(parsed) {
    const container = document.getElementById('graph-scroll-container');
    if (!(container instanceof HTMLElement)) {
        return;
    }
    applyGraphZoomToDom();

    if (graphInitialFitPending) {
        graphInitialFitPending = false;
        requestAnimationFrame(() => {
            fitGraphToViewport({ markInitialized: true });
            scheduleGraphFocusIfRequested(parsed);
        });
        return;
    }

    withGraphViewportRestoreLock(() => {
        container.scrollLeft = state.graphScrollLeft;
        container.scrollTop = state.graphScrollTop;
    });
    scheduleGraphFocusIfRequested(parsed);
    updateGraphZoomUi();
}

function scheduleGraphFocusIfRequested(parsed) {
    if (!graphFocusRequest?.nodeId) {
        return;
    }
    if (graphFocusAnimationFrame) {
        cancelAnimationFrame(graphFocusAnimationFrame);
        graphFocusAnimationFrame = 0;
    }
    const request = { ...graphFocusRequest };
    graphFocusAnimationFrame = requestAnimationFrame(() => {
        graphFocusAnimationFrame = 0;
        if (!graphFocusRequest || graphFocusRequest.version !== request.version || graphFocusRequest.nodeId !== request.nodeId) {
            return;
        }
        centerGraphOnNode(parsed, request.nodeId, 'smooth');
        graphFocusRequest = null;
        schedulePersistUiState();
    });
}

function clampGraphZoom(value) {
    if (!Number.isFinite(value)) {
        return 1;
    }
    return Math.min(5, Math.max(0.2, value));
}

function applyGraphZoomToDom() {
    const canvas = document.getElementById('graph-canvas');
    const svg = document.getElementById('graph-svg');
    if (!(canvas instanceof HTMLElement) || !(svg instanceof SVGElement)) {
        return;
    }
    const baseWidth = Number(canvas.dataset.baseWidth) || 0;
    const baseHeight = Number(canvas.dataset.baseHeight) || 0;
    if (!(baseWidth > 0) || !(baseHeight > 0)) {
        return;
    }
    const scaledWidth = Math.max(1, Math.round(baseWidth * state.graphZoom));
    const scaledHeight = Math.max(1, Math.round(baseHeight * state.graphZoom));
    canvas.style.width = `${scaledWidth}px`;
    canvas.style.height = `${scaledHeight}px`;
    svg.style.width = `${scaledWidth}px`;
    svg.style.height = `${scaledHeight}px`;
}

function updateGraphZoomUi() {
    const slider = document.querySelector('[data-role="graph-zoom"]');
    if (slider instanceof HTMLInputElement) {
        slider.value = String(Math.round(state.graphZoom * 100));
    }
    const label = document.getElementById('graph-zoom-label');
    if (label) {
        label.textContent = `${Math.round(state.graphZoom * 100)}%`;
    }
}

function setGraphZoom(nextZoom, options = {}) {
    const container = document.getElementById('graph-scroll-container');
    if (!(container instanceof HTMLElement)) {
        state.graphZoom = clampGraphZoom(nextZoom);
        state.graphViewportInitialized = true;
        persistUiState();
        return;
    }

    const rect = container.getBoundingClientRect();
    const previousZoom = state.graphZoom;
    const targetZoom = clampGraphZoom(nextZoom);
    if (Math.abs(targetZoom - previousZoom) < 0.0001) {
        updateGraphZoomUi();
        return;
    }

    const fallbackOffsetX = container.clientWidth / 2;
    const fallbackOffsetY = container.clientHeight / 2;
    const offsetX = Number.isFinite(options.anchorClientX)
        ? options.anchorClientX - rect.left
        : fallbackOffsetX;
    const offsetY = Number.isFinite(options.anchorClientY)
        ? options.anchorClientY - rect.top
        : fallbackOffsetY;
    const contentX = (container.scrollLeft + offsetX) / previousZoom;
    const contentY = (container.scrollTop + offsetY) / previousZoom;

    state.graphZoom = targetZoom;
    state.graphViewportInitialized = true;
    graphInitialFitPending = false;
    applyGraphZoomToDom();

    const nextLeft = Math.max(0, contentX * targetZoom - offsetX);
    const nextTop = Math.max(0, contentY * targetZoom - offsetY);
    withGraphViewportRestoreLock(() => {
        container.scrollLeft = nextLeft;
        container.scrollTop = nextTop;
    });
    state.graphScrollLeft = container.scrollLeft;
    state.graphScrollTop = container.scrollTop;
    updateGraphZoomUi();
    schedulePersistUiState();
}

function fitGraphToViewport(options = {}) {
    const parsed = getParsed();
    const container = document.getElementById('graph-scroll-container');
    if (!parsed || !(container instanceof HTMLElement)) {
        return;
    }
    const layout = parsed.graphView?.layout;
    if (!layout?.width || !layout?.height) {
        return;
    }
    const padding = 72;
    const zoomX = (container.clientWidth - padding) / layout.width;
    const zoomY = (container.clientHeight - padding) / layout.height;
    const nextZoom = clampGraphZoom(Math.min(zoomX, zoomY));
    state.graphZoom = nextZoom;
    state.graphViewportInitialized = options.markInitialized !== false;
    graphInitialFitPending = false;
    applyGraphZoomToDom();
    withGraphViewportRestoreLock(() => {
        container.scrollLeft = 0;
        container.scrollTop = 0;
    });
    state.graphScrollLeft = 0;
    state.graphScrollTop = 0;
    updateGraphZoomUi();
    schedulePersistUiState();
}

function centerGraphOnNode(parsed, nodeId, behavior = 'smooth') {
    const container = document.getElementById('graph-scroll-container');
    if (!(container instanceof HTMLElement)) {
        return;
    }
    const position = parsed.graphView?.layout?.positions?.[nodeId];
    if (!position) {
        return;
    }
    const targetLeft = Math.max(0, (position.x + position.width / 2) * state.graphZoom - container.clientWidth / 2);
    const targetTop = Math.max(0, (position.y + position.height / 2) * state.graphZoom - container.clientHeight / 2);
    withGraphViewportRestoreLock(() => {
        container.scrollTo({ left: targetLeft, top: targetTop, behavior });
    });
    state.graphScrollLeft = targetLeft;
    state.graphScrollTop = targetTop;
    state.graphViewportInitialized = true;
    graphInitialFitPending = false;
}

function setJsonExpansion(expanded) {
    const items = document.querySelectorAll('.json-tree-card details');
    items.forEach((item) => {
        item.open = expanded;
    });
}

function renderJsonTree(value, depth = 0, label = '') {
    if (value === null || typeof value !== 'object') {
        return renderJsonLeaf(value, label);
    }
    const isArray = Array.isArray(value);
    const entries = isArray ? value.map((item, index) => [String(index), item]) : Object.entries(value);
    const summaryLabel = label
        ? `<span class="json-key">${escapeHtml(label)}</span><span class="json-separator">: </span>`
        : '';
    const bracketOpen = isArray ? '[' : '{';
    const bracketClose = isArray ? ']' : '}';
    const countLabel = `${entries.length} ${isArray ? (entries.length === 1 ? 'item' : 'items') : (entries.length === 1 ? 'field' : 'fields')}`;
    return `
        <details class="json-node ${isArray ? 'json-array' : 'json-object'}" ${depth < 2 ? 'open' : ''}>
            <summary class="json-summary">
                ${summaryLabel}
                <span class="json-bracket">${bracketOpen}</span>
                <span class="json-count">${countLabel}</span>
                <span class="json-bracket">${bracketClose}</span>
            </summary>
            <div class="json-children">
                ${entries.length
                    ? entries.map(([key, item]) => renderJsonTree(item, depth + 1, isArray ? `[${key}]` : key)).join('')
                    : `<div class="json-empty">${isArray ? 'Empty array' : 'Empty object'}</div>`}
            </div>
        </details>
    `;
}

function renderJsonLeaf(value, label = '') {
    const type = value === null ? 'null' : typeof value;
    let display = '';
    let className = 'json-null';
    if (type === 'string') {
        className = 'json-string';
        display = JSON.stringify(value);
    } else if (type === 'number') {
        className = 'json-number';
        display = String(value);
    } else if (type === 'boolean') {
        className = 'json-boolean';
        display = String(value);
    } else if (type === 'null') {
        className = 'json-null';
        display = 'null';
    } else {
        className = 'json-string';
        display = JSON.stringify(value);
    }
    return `
        <div class="json-leaf">
            ${label ? `<span class="json-key">${escapeHtml(label)}</span><span class="json-separator">: </span>` : ''}
            <span class="${className}">${escapeHtml(display)}</span>
        </div>
    `;
}

function searchableNodeText(node) {
    const details = node.details || {};
    const embedded = (details.embeddedParameters || []).flatMap((item) => [item.label, item.name, item.shapeLabel, item.dataTypeName, item.valuePreview]);
    const embeddedConstants = (details.embeddedConstants || []).flatMap((item) => [item.label, item.tensorName, item.sourceNodeTitle, item.valuePreview, item.valueSummary, item.shapeLabel]);
    const dataInputs = (details.dataInputs || []).flatMap((item) => [item.name, item.typeSummary]);
    const outputs = (details.outputInfos || []).flatMap((item) => [item.name, item.typeSummary]);
    const searchTokens = Array.isArray(node.searchTokens) ? node.searchTokens : [];
    return `${node.title || ''} ${node.subtitle || ''} ${node.kind || ''} ${details.opType || ''} ${searchTokens.join(' ')} ${embedded.join(' ')} ${embeddedConstants.join(' ')} ${dataInputs.join(' ')} ${outputs.join(' ')}`.toLowerCase();
}

function formatTimestamp(value) {
    try {
        const date = new Date(value);
        return new Intl.DateTimeFormat(undefined, {
            year: 'numeric',
            month: 'short',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit'
        }).format(date);
    } catch {
        return value || '';
    }
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

function truncate(value, limit) {
    if (!value || value.length <= limit) {
        return value || '';
    }
    return `${value.slice(0, limit - 1)}…`;
}

function escapeHtml(value) {
    return String(value ?? '')
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
}

function escapeAttribute(value) {
    return escapeHtml(value);
}
