const vscode = require('vscode');
const path = require('node:path');
const modelParser = require('./model-parser');

const VIEW_TYPE = 'onnxInspector.viewer';

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
        const bytes = openContext.untitledDocumentData || await vscode.workspace.fs.readFile(uri);
        const parseResult = await this.parseModel(uri, bytes);
        return new OnnxInspectorDocument(uri, bytes, parseResult);
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
                    case 'copyText': {
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
                    default:
                        break;
                }
            } catch (error) {
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
        const bytes = await vscode.workspace.fs.readFile(document.uri);
        const parseResult = await this.parseModel(document.uri, bytes);
        document.update(bytes, parseResult);
    }

    async parseModel(uri, bytes) {
        try {
            const configuration = vscode.workspace.getConfiguration('onnxInspector');
            const pythonPath = configuration.get('pythonPath', '');
            return await modelParser.parseModelFile({
                filePath: uri.fsPath,
                bytes,
                extensionPath: this.context.extensionPath,
                pythonPath
            });
        } catch (error) {
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
                <div class="loading-title">Loading ONNX model…</div>
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
            filters: { Models: ['onnx', 'pt', 'pth'] },
            openLabel: 'Open in Model Inspector'
        });
        target = picked?.[0];
    }
    if (!target) {
        return;
    }
    if (modelParser.detectModelFormat(target.fsPath) === 'unknown') {
        vscode.window.showErrorMessage('ONNX Inspector can only open .onnx, .pt, or .pth files.');
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
    const { registration } = OnnxInspectorProvider.register(context);
    context.subscriptions.push(registration);
    context.subscriptions.push(vscode.commands.registerCommand('onnxInspector.openFile', openOnnxInspector));
}

function deactivate() {
    // no-op
}

module.exports = {
    activate,
    deactivate
};
