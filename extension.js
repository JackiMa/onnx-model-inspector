const vscode = require('vscode');
const path = require('node:path');
const modelParser = require('./model-parser');

const VIEW_TYPE = 'onnxInspector.viewer';
const LARGE_FILE_THRESHOLD_BYTES = 500 * 1024 * 1024;

let sharedOutputChannel = null;

function getOutputChannel() {
    if (!sharedOutputChannel) {
        sharedOutputChannel = vscode.window.createOutputChannel('ONNX Model Inspector');
    }
    return sharedOutputChannel;
}

function logDiagnostic(message) {
    const timestamp = new Date().toISOString();
    getOutputChannel().appendLine(`[${timestamp}] ${message}`);
}

class OnnxInspectorDocument {
    constructor(uri, bytes, parseResult) {
        this.uri = uri;
        this.bytes = bytes;
        this.parseResult = parseResult;
        this.updatedAt = Date.now();
    }

    update(bytes, parseResult) {
        this.bytes = bytes;
        this.parseResult = parseResult;
        this.updatedAt = Date.now();
    }

    dispose() {
        this.bytes = new Uint8Array();
        this.parseResult = null;
    }
}

class OnnxInspectorProvider {
    constructor(context) {
        this.context = context;
        this.panelsByResource = new Map();
        this.consentByUri = new Map();
    }

    static register(context) {
        const provider = new OnnxInspectorProvider(context);
        const registration = vscode.window.registerCustomEditorProvider(
            VIEW_TYPE,
            provider,
            {
                webviewOptions: {
                    retainContextWhenHidden: true
                },
                supportsMultipleEditorsPerDocument: true
            }
        );
        return { provider, registration };
    }

    async openCustomDocument(uri, openContext, token) {
        const bytes = await this.readFileBytes(uri, openContext);
        const parseResult = await this.parseModel(uri, bytes);
        return new OnnxInspectorDocument(uri, bytes, parseResult);
    }

    async readFileBytes(uri, openContext) {
        if (openContext?.untitledDocumentData) {
            return openContext.untitledDocumentData;
        }
        // ONNX is parsed in-process from bytes, so we must always read it
        // regardless of size. PT and safetensors are handled by a Python
        // subprocess that reads from disk directly; for those we can skip the
        // full-read when the file is large to keep the webview payload small.
        const format = modelParser.detectModelFormat(uri.fsPath);
        const subprocessBacked = format === 'pt' || format === 'safetensors';
        if (subprocessBacked) {
            try {
                const stat = await vscode.workspace.fs.stat(uri);
                if (stat.size > LARGE_FILE_THRESHOLD_BYTES) {
                    logDiagnostic(`Skipping webview-payload read for large ${format} file (${stat.size} bytes): ${uri.fsPath}`);
                    return new Uint8Array();
                }
            } catch {
                // ignore stat failure, let readFile fail meaningfully
            }
        }
        return await vscode.workspace.fs.readFile(uri);
    }

    async resolveCustomEditor(document, webviewPanel, token) {
        webviewPanel.webview.options = {
            enableScripts: true,
            localResourceRoots: [
                vscode.Uri.joinPath(this.context.extensionUri, 'media'),
                vscode.Uri.joinPath(this.context.extensionUri, 'images')
            ]
        };

        webviewPanel.webview.html = this.getHtmlForWebview(webviewPanel.webview);
        this.addPanel(document, webviewPanel);

        const receiveDisposable = webviewPanel.webview.onDidReceiveMessage(async (message) => {
            try {
                switch (message?.command) {
                    case 'requestModelData':
                        await this.postDocumentState(webviewPanel, document);
                        break;
                    case 'requestReload':
                        await this.reloadDocument(document);
                        this.broadcastDocumentState(document);
                        break;
                    case 'copyText':
                    case 'copyToClipboard': {
                        const text = typeof message?.payload?.text === 'string' ? message.payload.text : '';
                        if (text) {
                            await vscode.env.clipboard.writeText(text);
                            vscode.window.setStatusBarMessage('ONNX Inspector: copied to clipboard.', 2000);
                        }
                        break;
                    }
                    case 'showErrorMessage': {
                        const text = typeof message?.payload?.text === 'string' ? message.payload.text : '';
                        if (text) {
                            vscode.window.showErrorMessage(text);
                        }
                        break;
                    }
                    case 'requestFullLoadConsent': {
                        await this.handleFullLoadConsentRequest(document);
                        this.broadcastDocumentState(document);
                        break;
                    }
                    default:
                        break;
                }
            } catch (error) {
                logDiagnostic(`Message handler error: ${error?.stack || error}`);
                vscode.window.showErrorMessage(`ONNX Inspector: ${error?.message || String(error)}`);
            }
        });

        webviewPanel.onDidDispose(() => {
            receiveDisposable.dispose();
            this.removePanel(document, webviewPanel);
        });

        void this.postDocumentState(webviewPanel, document);
    }

    async reloadDocument(document) {
        const bytes = await this.readFileBytes(document.uri);
        const parseResult = await this.parseModel(document.uri, bytes);
        document.update(bytes, parseResult);
    }

    async handleFullLoadConsentRequest(document) {
        const uriKey = document.uri.toString();
        const currentState = this.consentByUri.get(uriKey);
        if (currentState === 'denied') {
            logDiagnostic(`Full-load consent previously denied for ${document.uri.fsPath}; not reprompting. Use command "ONNX Inspector: Reset Load Consent" to retry.`);
            return;
        }
        if (!vscode.workspace.isTrusted) {
            vscode.window.showErrorMessage('ONNX Inspector: this workspace is untrusted. Full (unsafe) pickle deserialization is disabled by policy. Trust the workspace to enable.');
            logDiagnostic(`Full-load blocked in untrusted workspace: ${document.uri.fsPath}`);
            return;
        }
        const settings = vscode.workspace.getConfiguration('onnxInspector');
        if (!settings.get('allowFullPickleLoad', false)) {
            vscode.window.showErrorMessage('ONNX Inspector: full pickle load is disabled. Enable "onnxInspector.allowFullPickleLoad" in settings if you trust this checkpoint source.');
            logDiagnostic(`Full-load blocked by setting: ${document.uri.fsPath}`);
            return;
        }
        const choice = await vscode.window.showWarningMessage(
            'This checkpoint failed safe (weights_only) load. Loading it requires unsafe pickle deserialization, which can execute arbitrary Python code. Only continue if you trust the source.',
            { modal: true },
            'Allow once',
            'Cancel'
        );
        if (choice !== 'Allow once') {
            this.consentByUri.set(uriKey, 'denied');
            logDiagnostic(`User denied full-load consent for ${document.uri.fsPath}`);
            return;
        }
        this.consentByUri.set(uriKey, 'allowed');
        logDiagnostic(`User granted one-shot full-load consent for ${document.uri.fsPath}`);
        const bytes = await this.readFileBytes(document.uri);
        const parseResult = await this.parseModel(document.uri, bytes, { allowFullLoad: true });
        document.update(bytes, parseResult);
    }

    resetConsent(uri) {
        if (uri) {
            this.consentByUri.delete(uri.toString());
        } else {
            this.consentByUri.clear();
        }
    }

    async parseModel(uri, bytes, { allowFullLoad = false } = {}) {
        try {
            const configuration = vscode.workspace.getConfiguration('onnxInspector');
            const pythonPath = configuration.get('pythonPath', '');
            const effectiveAllow = allowFullLoad && vscode.workspace.isTrusted && configuration.get('allowFullPickleLoad', false);
            const result = await modelParser.parseModelFile({
                filePath: uri.fsPath,
                bytes,
                extensionPath: this.context.extensionPath,
                pythonPath,
                allowFullLoad: effectiveAllow
            });
            if (result?.requiresFullLoad) {
                logDiagnostic(`Safe load failed; checkpoint requires full load: ${uri.fsPath} (${result.safeLoadError || 'no detail'})`);
            }
            return result;
        } catch (error) {
            logDiagnostic(`Parse error for ${uri.fsPath}: ${error?.stack || error}`);
            return {
                ok: false,
                error: {
                    name: error?.name || 'Error',
                    message: error?.message || String(error),
                    stack: error?.stack || ''
                }
            };
        }
    }

    addPanel(document, panel) {
        const key = document.uri.toString();
        if (!this.panelsByResource.has(key)) {
            this.panelsByResource.set(key, new Set());
        }
        this.panelsByResource.get(key).add(panel);
    }

    removePanel(document, panel) {
        const key = document.uri.toString();
        const panels = this.panelsByResource.get(key);
        if (!panels) {
            return;
        }
        panels.delete(panel);
        if (panels.size === 0) {
            this.panelsByResource.delete(key);
        }
    }

    broadcastDocumentState(document) {
        const key = document.uri.toString();
        const panels = this.panelsByResource.get(key);
        if (!panels) {
            return;
        }
        for (const panel of panels) {
            void this.postDocumentState(panel, document);
        }
    }

    async postDocumentState(panel, document) {
        let fileStat = null;
        try {
            const stat = await vscode.workspace.fs.stat(document.uri);
            fileStat = {
                ctime: Number.isFinite(stat.ctime) ? new Date(stat.ctime).toISOString() : '',
                mtime: Number.isFinite(stat.mtime) ? new Date(stat.mtime).toISOString() : '',
                size: Number.isFinite(stat.size) ? stat.size : document.bytes.length
            };
        } catch {
            fileStat = {
                ctime: '',
                mtime: '',
                size: document.bytes.length
            };
        }
        const payload = {
            uri: document.uri.toString(),
            fileName: path.posix.basename(document.uri.path),
            displayPath: vscode.workspace.asRelativePath(document.uri, false),
            updatedAt: new Date(document.updatedAt).toISOString(),
            parseResult: document.parseResult,
            byteLength: document.bytes.length,
            fileStat
        };
        panel.webview.postMessage({ command: 'modelData', payload });
    }

    getHtmlForWebview(webview) {
        const scriptUri = webview.asWebviewUri(vscode.Uri.joinPath(this.context.extensionUri, 'media', 'main.js'));
        const styleUri = webview.asWebviewUri(vscode.Uri.joinPath(this.context.extensionUri, 'media', 'styles.css'));
        const nonce = getNonce();
        return /* html */ `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src ${webview.cspSource} data:; style-src ${webview.cspSource}; script-src 'nonce-${nonce}'; font-src ${webview.cspSource};" />
    <link rel="stylesheet" href="${styleUri}" />
    <title>ONNX Inspector</title>
</head>
<body>
    <div id="app" class="app-shell">
        <div class="loading-state">
            <div class="spinner"></div>
            <div>
                <div class="loading-title">Loading model…</div>
                <div class="loading-subtitle">Parsing graph, metadata, and tensor summaries.</div>
            </div>
        </div>
    </div>
    <script nonce="${nonce}" type="module" src="${scriptUri}"></script>
</body>
</html>`;
    }
}

async function openOnnxInspector(resource) {
    let target = resource;
    if (!target) {
        const activeResource = vscode.window.activeTextEditor?.document?.uri;
        if (activeResource && modelParser.detectModelFormat(activeResource.fsPath) !== 'unknown') {
            target = activeResource;
        }
    }
    if (!target) {
        const picked = await vscode.window.showOpenDialog({
            canSelectMany: false,
            filters: { Models: ['onnx', 'pt', 'pth', 'safetensors'] },
            openLabel: 'Open in Model Inspector'
        });
        target = picked?.[0];
    }
    if (!target) {
        return;
    }
    if (modelParser.detectModelFormat(target.fsPath) === 'unknown') {
        vscode.window.showErrorMessage('ONNX Inspector can only open .onnx, .pt, .pth, or .safetensors files.');
        return;
    }
    await vscode.commands.executeCommand('vscode.openWith', target, VIEW_TYPE);
}

function getNonce() {
    const characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    let result = '';
    for (let index = 0; index < 32; index += 1) {
        result += characters.charAt(Math.floor(Math.random() * characters.length));
    }
    return result;
}

function activate(context) {
    const { provider, registration } = OnnxInspectorProvider.register(context);
    context.subscriptions.push(registration);
    context.subscriptions.push(vscode.commands.registerCommand('onnxInspector.openFile', openOnnxInspector));
    context.subscriptions.push(vscode.commands.registerCommand('onnxInspector.resetLoadConsent', async () => {
        provider.resetConsent();
        vscode.window.showInformationMessage('ONNX Inspector: load consent state cleared.');
    }));
    if (sharedOutputChannel) {
        context.subscriptions.push(sharedOutputChannel);
    }
}

function deactivate() {
    if (sharedOutputChannel) {
        sharedOutputChannel.dispose();
        sharedOutputChannel = null;
    }
}

module.exports = {
    activate,
    deactivate
};
