import fs from 'node:fs';
import { parseOnnxBytes } from '../media/lib/onnx-parser.mjs';
const bytes = fs.readFileSync(new URL('./sample.onnx', import.meta.url));
const parsed = parseOnnxBytes(new Uint8Array(bytes));
console.log(JSON.stringify({
  graphName: parsed.summary.graphName,
  irVersion: parsed.summary.irVersion,
  metadataKeys: parsed.metadata.map((m) => m.key),
  nodeTitles: parsed.graphView.displayNodes.map((n) => n.title),
  edgeCount: parsed.graphView.edges.length,
  inputType: parsed.inputs[0].typeSummary,
  initializerSummary: parsed.initializers[0].storage.summary
}, null, 2));
