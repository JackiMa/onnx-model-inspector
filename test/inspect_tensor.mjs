import fs from 'node:fs';
import * as protobuf from '../media/lib/protobuf.mjs';
import { onnx } from '../media/lib/onnx-proto.mjs';
const data = fs.readFileSync(new URL('./sample.onnx', import.meta.url));
const reader = protobuf.BinaryReader.open(new Uint8Array(data));
const model = onnx.ModelProto.decode(reader);
console.log(Object.keys(model.graph.initializer[0]));
console.log(model.graph.initializer[0]);
