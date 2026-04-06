import fs from 'node:fs';
import * as protobuf from '../media/lib/protobuf.mjs';
import { onnx } from '../media/lib/onnx-proto.mjs';
const data = fs.readFileSync(new URL('./sample.onnx', import.meta.url));
const reader = protobuf.BinaryReader.open(new Uint8Array(data));
const model = onnx.ModelProto.decode(reader);
console.log('tensor type', model.graph.input[0].type.tensor_type);
console.log('shape dims', model.graph.input[0].type.tensor_type.shape.dim);
console.log('tensor data type enum name?', onnx.TensorProto.DataType);
