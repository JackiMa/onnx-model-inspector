import * as protobuf from './protobuf.mjs';
import { onnx } from './onnx-proto.mjs';
import { layout as dagreLayout } from './dagre.mjs';

const TENSOR_DATA_TYPE_NAME_BY_VALUE = invertEnum(onnx.TensorProto.DataType);
const ATTRIBUTE_TYPE_NAME_BY_VALUE = invertEnum(onnx.AttributeProto.AttributeType);

const TENSOR_DTYPE_BYTES = {
    FLOAT: 4,
    UINT8: 1,
    INT8: 1,
    UINT16: 2,
    INT16: 2,
    INT32: 4,
    INT64: 8,
    BOOL: 1,
    FLOAT16: 2,
    DOUBLE: 8,
    UINT32: 4,
    UINT64: 8,
    COMPLEX64: 8,
    COMPLEX128: 16,
    BFLOAT16: 2,
    FLOAT8E4M3FN: 1,
    FLOAT8E4M3FNUZ: 1,
    FLOAT8E5M2: 1,
    FLOAT8E5M2FNUZ: 1,
    UINT4: 0.5,
    INT4: 0.5,
    FLOAT4E2M1: 0.5,
    FLOAT8E8M0: 1,
    UINT2: 0.25,
    INT2: 0.25
};

export function parseOnnxBytes(data) {
    const bytes = normalizeBytes(data);
    const reader = protobuf.BinaryReader.open(bytes);
    const model = onnx.ModelProto.decode(reader);
    return summarizeModel(model, bytes.length);
}

function summarizeModel(model, fileSizeBytes) {
    const graph = model.graph || new onnx.GraphProto();
    const inputs = (graph.input || []).map((item) => summarizeValueInfo(item, 'input'));
    const outputs = (graph.output || []).map((item) => summarizeValueInfo(item, 'output'));
    const valueInfos = (graph.value_info || []).map((item) => summarizeValueInfo(item, 'valueInfo'));
    const initializers = (graph.initializer || []).map((item) => summarizeInitializer(item));

    const initializerByName = new Map(initializers.map((item) => [item.name, item]));
    const valueInfoByName = new Map([...inputs, ...outputs, ...valueInfos].map((item) => [item.name, item]));

    const nodes = (graph.node || []).map((node, index) => summarizeNode(node, index, initializerByName, valueInfoByName));
    const graphView = buildGraphView({
        graph,
        nodes,
        inputs,
        outputs,
        initializers,
        initializerByName,
        valueInfoByName
    });

    const metadata = summarizeMetadataProps(model.metadata_props || []);
    const graphMetadata = summarizeMetadataProps(graph.metadata_props || []);
    const opsets = (model.opset_import || []).map((item) => ({
        domain: item.domain || 'ai.onnx',
        version: formatBigInt(item.version)
    }));
    const temporalMetadata = detectTemporalMetadata(metadata, graphMetadata);

    const estimatedParameterBytes = initializers.reduce((acc, item) => acc + (item.estimatedBytesKnown ? item.estimatedBytesValue : 0), 0);

    return {
        format: 'onnx',
        summary: {
            graphName: graph.name || '',
            irVersion: formatBigInt(model.ir_version),
            irVersionLabel: versionNameForIR(model.ir_version),
            producerName: model.producer_name || '',
            producerVersion: model.producer_version || '',
            domain: model.domain || '',
            modelVersion: formatBigInt(model.model_version),
            docString: model.doc_string || '',
            opsets,
            fileSizeBytes,
            fileSizeLabel: formatBytes(fileSizeBytes),
            temporalMetadata,
            stats: {
                nodeCount: nodes.length,
                inputCount: inputs.length,
                outputCount: outputs.length,
                valueInfoCount: valueInfos.length,
                initializerCount: initializers.length,
                metadataCount: metadata.length,
                graphMetadataCount: graphMetadata.length,
                externalTensorCount: initializers.filter((item) => item.externalData.length > 0).length,
                estimatedParameterBytes,
                estimatedParameterBytesLabel: estimatedParameterBytes > 0 ? formatBytes(estimatedParameterBytes) : 'Unknown'
            }
        },
        metadata,
        graphMetadata,
        inputs,
        outputs,
        valueInfos,
        initializers,
        nodes,
        graphView
    };
}

function buildGraphView({ graph, nodes, inputs, outputs, initializers, initializerByName, valueInfoByName }) {
    const producerByTensor = new Map();
    const consumerRefsByTensor = new Map();
    const terminalNodes = [];
    const displayNodes = [];
    const graphInputNames = new Set(inputs.map((item) => item.name));
    const nodeById = new Map(nodes.map((node) => [node.id, node]));

    for (const node of nodes) {
        node.inputs.forEach((inputName, inputIndex) => {
            if (!inputName) {
                return;
            }
            if (!consumerRefsByTensor.has(inputName)) {
                consumerRefsByTensor.set(inputName, []);
            }
            consumerRefsByTensor.get(inputName).push({ nodeId: node.id, inputIndex });
        });
    }

    const collapsedConstantsByTensor = buildCollapsedConstantCatalog(nodes, consumerRefsByTensor, nodeById);
    const collapsedConstantNodeIds = new Set([...collapsedConstantsByTensor.values()].map((item) => item.nodeId));

    for (const input of inputs) {
        if (initializerByName.has(input.name)) {
            continue;
        }
        const nodeId = `input:${input.name}`;
        const terminal = {
            id: nodeId,
            kind: 'input',
            name: input.name,
            title: input.name,
            subtitle: shortTypeLabel(input.typeSummary),
            details: input,
            searchTokens: [input.name, input.typeSummary].filter(Boolean)
        };
        terminalNodes.push(terminal);
        displayNodes.push(terminal);
        producerByTensor.set(input.name, nodeId);
    }

    for (const node of nodes) {
        if (collapsedConstantNodeIds.has(node.id)) {
            continue;
        }

        const parameterRows = [];
        const embeddedParameters = [];
        const embeddedConstants = [];
        const dataInputs = [];

        node.inputs.forEach((inputName, inputIndex) => {
            if (!inputName) {
                return;
            }

            const collapsedConstant = collapsedConstantsByTensor.get(inputName);
            if (collapsedConstant && collapsedConstant.consumerNodeId === node.id && collapsedConstant.inputIndex === inputIndex) {
                embeddedConstants.push(collapsedConstant);
                parameterRows.push({
                    kind: 'constant',
                    label: collapsedConstant.label,
                    shape: collapsedConstant.shapeLabel || collapsedConstant.valueSummary || collapsedConstant.valuePreview || '—',
                    dtype: collapsedConstant.dataTypeName || '',
                    name: collapsedConstant.tensorName,
                    preview: collapsedConstant.valuePreview || collapsedConstant.valueSummary || '',
                    secondaryValue: collapsedConstant.valuePreview || collapsedConstant.valueSummary || ''
                });
                return;
            }

            const initializer = initializerByName.get(inputName);
            if (initializer) {
                const parameter = buildEmbeddedParameter(node, initializer, inputIndex);
                embeddedParameters.push(parameter);
                parameterRows.push({
                    kind: 'parameter',
                    label: parameter.label,
                    shape: parameter.shapeLabel,
                    dtype: parameter.dataTypeName,
                    name: parameter.name,
                    preview: parameter.valuePreview || '',
                    secondaryValue: parameter.shapeLabel || parameter.dataTypeName || '—'
                });
                return;
            }

            const valueInfo = valueInfoByName.get(inputName) || null;
            dataInputs.push({
                index: inputIndex,
                name: inputName,
                typeSummary: valueInfo?.typeSummary || ''
            });
        });

        const displayAttributeRows = buildDisplayAttributeRows(node, embeddedConstants);
        parameterRows.push(...displayAttributeRows);

        const outputInfos = node.outputs.map((outputName, outputIndex) => ({
            index: outputIndex,
            name: outputName,
            typeSummary: valueInfoByName.get(outputName)?.typeSummary || ''
        }));
        const renderStyle = computeOperatorRenderStyle(node, parameterRows, embeddedConstants, displayAttributeRows);
        const primaryInput = dataInputs[0] || null;
        const display = {
            id: node.id,
            kind: 'operator',
            name: node.name,
            title: node.displayName,
            subtitle: node.subtitle,
            renderStyle,
            details: {
                ...node,
                embeddedParameters,
                embeddedConstants,
                dataInputs,
                outputInfos,
                primaryInput,
                renderStyle
            },
            parameterRows,
            searchTokens: [
                node.displayName,
                node.opType,
                ...embeddedParameters.flatMap((item) => [item.label, item.name, item.dataTypeName, item.shapeLabel, item.valuePreview]),
                ...embeddedConstants.flatMap((item) => [item.label, item.tensorName, item.sourceNodeTitle, item.valuePreview, item.valueSummary, item.dataTypeName, item.shapeLabel]),
                ...dataInputs.flatMap((item) => [item.name, item.typeSummary]),
                ...outputInfos.flatMap((item) => [item.name, item.typeSummary]),
                ...displayAttributeRows.flatMap((item) => [item.label, item.shape, item.dtype, item.preview])
            ].filter(Boolean)
        };
        displayNodes.push(display);
        for (const outputName of node.outputs) {
            if (outputName) {
                producerByTensor.set(outputName, node.id);
            }
        }
    }

    for (const output of outputs) {
        const nodeId = `output:${output.name}`;
        const terminal = {
            id: nodeId,
            kind: 'output',
            name: output.name,
            title: output.name,
            subtitle: shortTypeLabel(output.typeSummary),
            details: output,
            searchTokens: [output.name, output.typeSummary].filter(Boolean)
        };
        terminalNodes.push(terminal);
        displayNodes.push(terminal);
        if (!consumerRefsByTensor.has(output.name)) {
            consumerRefsByTensor.set(output.name, []);
        }
        consumerRefsByTensor.get(output.name).push({ nodeId, inputIndex: -1, terminalKind: 'output' });
    }

    const edges = [];
    const edgeKeySet = new Set();

    for (const [tensorName, consumerRefs] of consumerRefsByTensor.entries()) {
        const sourceId = producerByTensor.get(tensorName);
        if (!sourceId) {
            continue;
        }
        for (const ref of consumerRefs) {
            const targetId = ref.nodeId;
            const key = `${sourceId}|${targetId}|${tensorName}`;
            if (edgeKeySet.has(key)) {
                continue;
            }
            edgeKeySet.add(key);
            const valueInfo = valueInfoByName.get(tensorName) || null;
            const isModelInput = graphInputNames.has(tensorName) && sourceId.startsWith('input:');
            edges.push({
                id: `edge:${edges.length}`,
                sourceId,
                targetId,
                tensorName,
                typeSummary: valueInfo?.typeSummary || '',
                shapeLabel: shortTypeLabel(valueInfo?.typeSummary || ''),
                isModelInput,
                isOutputEdge: targetId.startsWith('output:')
            });
        }
    }

    const layout = layoutGraph(displayNodes, edges);

    return {
        name: graph.name || '',
        displayNodes,
        edges,
        layout,
        terminalNodes,
        stats: {
            displayNodeCount: displayNodes.length,
            edgeCount: edges.length,
            operatorNodeCount: nodes.length,
            collapsedConstantCount: collapsedConstantNodeIds.size
        }
    };
}

function summarizeNode(node, index, initializerByName = new Map(), valueInfoByName = new Map()) {
    const attributes = (node.attribute || []).map((attribute) => summarizeAttribute(attribute));
    const metadata = summarizeMetadataProps(node.metadata_props || []);
    const name = node.name || `${node.op_type || 'Node'}_${index}`;
    const displayName = node.name || node.op_type || `Node ${index + 1}`;
    const inputs = (node.input || []).slice();
    const outputs = (node.output || []).slice();
    const embeddedParameters = [];
    const dataInputs = [];

    inputs.forEach((inputName, inputIndex) => {
        if (!inputName) {
            return;
        }
        const initializer = initializerByName.get(inputName);
        if (initializer) {
            embeddedParameters.push(buildEmbeddedParameter({ opType: node.op_type || '', name, displayName }, initializer, inputIndex));
        } else {
            const valueInfo = valueInfoByName.get(inputName) || null;
            dataInputs.push({
                index: inputIndex,
                name: inputName,
                typeSummary: valueInfo?.typeSummary || ''
            });
        }
    });

    return {
        id: `node:${index}`,
        index,
        kind: 'operator',
        name,
        displayName,
        opType: node.op_type || '',
        domain: node.domain || '',
        docString: node.doc_string || '',
        inputs,
        outputs,
        subtitle: node.op_type || 'Operator',
        attributes,
        metadata,
        embeddedParameters,
        dataInputs,
        deviceConfigurationsCount: Array.isArray(node.device_configurations) ? node.device_configurations.length : 0
    };
}

function summarizeInitializer(tensor) {
    const dims = (tensor.dims || []).map((item) => formatBigInt(item));
    const dataTypeName = tensorTypeName(tensor.data_type);
    const storage = summarizeTensorStorage(tensor, dataTypeName);
    const elementCount = multiplyBigIntArray(tensor.dims || []);
    const estimatedBytes = estimateTensorBytes(dataTypeName, elementCount, storage);
    const metadata = summarizeMetadataProps(tensor.metadata_props || []);
    const decodedValue = decodeTensorValuePreview(tensor, dataTypeName, 12, elementCount);
    return {
        kind: 'initializer',
        name: tensor.name || '',
        dataType: tensor.data_type,
        dataTypeName,
        dims,
        shapeSummary: dims.length > 0 ? `[${dims.join(' × ')}]` : '',
        elementCount: elementCount !== null ? elementCount.toString() : 'Unknown',
        estimatedBytes: estimatedBytes !== null ? formatBytes(estimatedBytes) : 'Unknown',
        estimatedBytesKnown: estimatedBytes !== null,
        estimatedBytesValue: estimatedBytes || 0,
        externalData: summarizeExternalData(tensor.external_data || []),
        storage,
        metadata,
        docString: tensor.doc_string || '',
        decodedValuePreview: decodedValue?.preview || '',
        decodedValues: decodedValue?.values || [],
        decodedValueCount: decodedValue?.valueCount || 0,
        decodedValueComplete: Boolean(decodedValue && !decodedValue.truncated)
    };
}

function decodeTensorValuePreview(tensor, dataTypeName, maxElements = 12, elementCount = multiplyBigIntArray(tensor?.dims || [])) {
    const totalElements = typeof elementCount === 'bigint'
        ? Number(elementCount <= BigInt(Number.MAX_SAFE_INTEGER) ? elementCount : BigInt(Number.MAX_SAFE_INTEGER))
        : (elementCount === null || elementCount === undefined ? null : Number(elementCount));
    const readLimit = Number.isFinite(totalElements) && totalElements > 0
        ? Math.min(maxElements, Math.max(1, totalElements))
        : maxElements;
    const values = readTensorValues(tensor, dataTypeName, readLimit);
    if (!values || !values.length) {
        return null;
    }
    const formattedValues = values.map((value) => formatPreviewScalar(value));
    const truncated = Number.isFinite(totalElements) ? totalElements > values.length : false;
    const preview = formattedValues.length === 1 && !truncated
        ? `${formattedValues[0]}`
        : `[${formattedValues.join(', ')}${truncated ? ', …' : ''}]`;
    return {
        preview,
        values: formattedValues,
        valueCount: Number.isFinite(totalElements) && totalElements > 0 ? totalElements : values.length,
        truncated
    };
}

function readTensorValues(tensor, dataTypeName, limit) {
    if (!tensor || limit <= 0) {
        return [];
    }

    const directFields = [
        ['float_data', (value) => value],
        ['double_data', (value) => value],
        ['int32_data', (value) => value],
        ['int64_data', (value) => typeof value === 'bigint' ? value.toString() : formatBigInt(value)],
        ['uint64_data', (value) => typeof value === 'bigint' ? value.toString() : formatBigInt(value)],
        ['string_data', (value) => decodeUtf8(value)]
    ];

    for (const [field, convert] of directFields) {
        const values = tensor[field];
        if (Array.isArray(values) && values.length) {
            return values.slice(0, limit).map(convert);
        }
    }

    if (!(tensor.raw_data instanceof Uint8Array) || tensor.raw_data.length === 0) {
        return [];
    }

    return decodeRawTensorValues(tensor.raw_data, dataTypeName, limit);
}

function decodeRawTensorValues(rawData, dataTypeName, limit) {
    const view = new DataView(rawData.buffer, rawData.byteOffset, rawData.byteLength);
    const values = [];
    const pushValues = (byteSize, reader) => {
        const count = Math.min(limit, Math.floor(rawData.byteLength / byteSize));
        for (let index = 0; index < count; index += 1) {
            values.push(reader(index * byteSize));
        }
        return values;
    };

    switch (`${dataTypeName || ''}`.toUpperCase()) {
        case 'FLOAT':
            return pushValues(4, (offset) => view.getFloat32(offset, true));
        case 'DOUBLE':
            return pushValues(8, (offset) => view.getFloat64(offset, true));
        case 'FLOAT16':
            return pushValues(2, (offset) => halfToFloat(view.getUint16(offset, true)));
        case 'BFLOAT16':
            return pushValues(2, (offset) => bfloat16ToFloat(view.getUint16(offset, true)));
        case 'INT8':
            return pushValues(1, (offset) => view.getInt8(offset));
        case 'UINT8':
            return pushValues(1, (offset) => view.getUint8(offset));
        case 'BOOL':
            return pushValues(1, (offset) => view.getUint8(offset) !== 0);
        case 'INT16':
            return pushValues(2, (offset) => view.getInt16(offset, true));
        case 'UINT16':
            return pushValues(2, (offset) => view.getUint16(offset, true));
        case 'INT32':
            return pushValues(4, (offset) => view.getInt32(offset, true));
        case 'UINT32':
            return pushValues(4, (offset) => view.getUint32(offset, true));
        case 'INT64':
            if (typeof view.getBigInt64 === 'function') {
                return pushValues(8, (offset) => view.getBigInt64(offset, true).toString());
            }
            return [];
        case 'UINT64':
            if (typeof view.getBigUint64 === 'function') {
                return pushValues(8, (offset) => view.getBigUint64(offset, true).toString());
            }
            return [];
        default:
            return [];
    }
}

function formatPreviewScalar(value) {
    if (typeof value === 'number') {
        if (!Number.isFinite(value)) {
            return String(value);
        }
        if (Number.isInteger(value)) {
            return String(value);
        }
        const rounded = Math.abs(value) >= 1000 || Math.abs(value) < 0.001
            ? value.toExponential(3)
            : value.toFixed(4);
        return rounded.replace(/\.0+$/, '').replace(/(\.\d*?)0+$/, '$1');
    }
    if (typeof value === 'boolean') {
        return value ? 'true' : 'false';
    }
    const stringValue = `${value}`;
    return stringValue.length > 48 ? `${stringValue.slice(0, 45)}…` : stringValue;
}

function halfToFloat(bits) {
    const sign = (bits & 0x8000) ? -1 : 1;
    const exponent = (bits >> 10) & 0x1f;
    const fraction = bits & 0x03ff;
    if (exponent === 0) {
        return fraction === 0 ? sign * 0 : sign * Math.pow(2, -14) * (fraction / 1024);
    }
    if (exponent === 0x1f) {
        return fraction ? NaN : sign * Infinity;
    }
    return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
}

function bfloat16ToFloat(bits) {
    const buffer = new ArrayBuffer(4);
    const view = new DataView(buffer);
    view.setUint32(0, bits << 16, false);
    return view.getFloat32(0, false);
}

function summarizeTensorStorage(tensor, dataTypeName) {
    const fields = [];
    const pushField = (name, values) => {
        if (Array.isArray(values) && values.length > 0) {
            fields.push({
                name,
                count: values.length,
                preview: summarizeArrayPreview(values)
            });
        } else if (values instanceof Uint8Array && values.length > 0) {
            fields.push({
                name,
                count: values.length,
                preview: summarizeBytePreview(values)
            });
        }
    };

    if (tensor.raw_data instanceof Uint8Array && tensor.raw_data.length > 0) {
        fields.push({
            name: 'raw_data',
            count: tensor.raw_data.length,
            preview: summarizeBytePreview(tensor.raw_data)
        });
    }

    pushField('float_data', tensor.float_data);
    pushField('int32_data', tensor.int32_data);
    pushField('int64_data', tensor.int64_data);
    pushField('double_data', tensor.double_data);
    pushField('uint64_data', tensor.uint64_data);
    pushField('string_data', tensor.string_data);

    return {
        dataTypeName,
        fields,
        summary: fields.length > 0 ? fields.map((field) => `${field.name} (${field.count})`).join(', ') : 'No embedded data preview available'
    };
}

function summarizeExternalData(entries) {
    return (entries || []).map((entry) => ({ key: entry.key || '', value: entry.value || '' }));
}

function summarizeValueInfo(valueInfo, role) {
    const typeSummary = summarizeTypeProto(valueInfo.type || null);
    const metadata = summarizeMetadataProps(valueInfo.metadata_props || []);
    return {
        kind: role,
        name: valueInfo.name || '',
        typeSummary: typeSummary.summary,
        typeDetail: typeSummary,
        docString: valueInfo.doc_string || '',
        metadata,
        denotation: valueInfo.type?.denotation || ''
    };
}

function summarizeTypeProto(typeProto) {
    if (!typeProto) {
        return { kind: 'unknown', summary: 'Unknown' };
    }
    if (typeProto.tensor_type) {
        const tensorType = typeProto.tensor_type;
        const dtype = tensorTypeName(tensorType.elem_type);
        const shape = summarizeShape(tensorType.shape);
        return {
            kind: 'tensor',
            dataTypeName: dtype,
            shape,
            summary: `tensor<${dtype}${shape.summary ? ` ${shape.summary}` : ''}>`
        };
    }
    if (typeProto.sparse_tensor_type) {
        const tensorType = typeProto.sparse_tensor_type;
        const dtype = tensorTypeName(tensorType.elem_type);
        const shape = summarizeShape(tensorType.shape);
        return {
            kind: 'sparse_tensor',
            dataTypeName: dtype,
            shape,
            summary: `sparse_tensor<${dtype}${shape.summary ? ` ${shape.summary}` : ''}>`
        };
    }
    if (typeProto.sequence_type) {
        const elemType = summarizeTypeProto(typeProto.sequence_type.elem_type);
        return {
            kind: 'sequence',
            elementType: elemType,
            summary: `sequence<${elemType.summary}>`
        };
    }
    if (typeProto.optional_type) {
        const elemType = summarizeTypeProto(typeProto.optional_type.elem_type);
        return {
            kind: 'optional',
            elementType: elemType,
            summary: `optional<${elemType.summary}>`
        };
    }
    if (typeProto.map_type) {
        const keyType = tensorTypeName(typeProto.map_type.key_type);
        const valueType = summarizeTypeProto(typeProto.map_type.value_type);
        return {
            kind: 'map',
            keyType,
            valueType,
            summary: `map<${keyType}, ${valueType.summary}>`
        };
    }
    if (typeProto.opaque_type) {
        const domain = typeProto.opaque_type.domain || '';
        const name = typeProto.opaque_type.name || '';
        return {
            kind: 'opaque',
            domain,
            name,
            summary: `opaque<${domain ? `${domain}:` : ''}${name}>`
        };
    }
    return { kind: 'unknown', summary: 'Unknown' };
}

function summarizeShape(shapeProto) {
    const dims = (shapeProto?.dim || []).map((dim) => {
        if (dim.denotation) {
            return dim.denotation;
        }
        if (dim.dim_param) {
            return dim.dim_param;
        }
        if (dim.dim_value !== undefined && dim.dim_value !== null) {
            return formatBigInt(dim.dim_value);
        }
        return '?';
    });
    return {
        dims,
        summary: dims.length > 0 ? `[${dims.join(' × ')}]` : ''
    };
}

function summarizeAttribute(attribute) {
    const typeName = ATTRIBUTE_TYPE_NAME_BY_VALUE[attribute.type] || inferAttributeType(attribute);
    const valueSummary = summarizeAttributeValue(attribute, typeName);
    return {
        name: attribute.name || '',
        refAttrName: attribute.ref_attr_name || '',
        type: typeName,
        valueSummary: valueSummary.summary,
        valueDetail: valueSummary.detail,
        docString: attribute.doc_string || ''
    };
}

function summarizeAttributeValue(attribute, typeName) {
    switch (typeName) {
        case 'FLOAT':
            return { summary: `${attribute.f}`, detail: attribute.f };
        case 'INT':
            return { summary: formatBigInt(attribute.i), detail: formatBigInt(attribute.i) };
        case 'STRING': {
            const value = decodeUtf8(attribute.s);
            return { summary: truncateMiddle(value, 120), detail: value };
        }
        case 'TENSOR': {
            const tensor = summarizeInitializer(attribute.t || new onnx.TensorProto());
            return { summary: `${tensor.dataTypeName}${tensor.shapeSummary ? ` ${tensor.shapeSummary}` : ''}`, detail: tensor };
        }
        case 'GRAPH': {
            const graph = attribute.g || new onnx.GraphProto();
            return { summary: graph.name ? `graph:${graph.name}` : 'graph', detail: { name: graph.name || '', nodeCount: (graph.node || []).length } };
        }
        case 'FLOATS':
            return summarizeRepeatedAttribute(attribute.floats || []);
        case 'INTS':
            return summarizeRepeatedAttribute(attribute.ints || []);
        case 'STRINGS':
            return summarizeRepeatedAttribute((attribute.strings || []).map((value) => decodeUtf8(value)));
        case 'TENSORS':
            return { summary: `${(attribute.tensors || []).length} tensor(s)`, detail: (attribute.tensors || []).map((tensor) => summarizeInitializer(tensor)) };
        case 'GRAPHS':
            return { summary: `${(attribute.graphs || []).length} graph(s)`, detail: (attribute.graphs || []).map((graph) => ({ name: graph.name || '', nodeCount: (graph.node || []).length })) };
        case 'SPARSE_TENSOR':
        case 'SPARSE_TENSORS':
            return { summary: typeName.toLowerCase(), detail: typeName.toLowerCase() };
        case 'TYPE_PROTO': {
            const type = summarizeTypeProto(attribute.tp || null);
            return { summary: type.summary, detail: type };
        }
        case 'TYPE_PROTOS':
            return { summary: `${(attribute.type_protos || []).length} type proto(s)`, detail: (attribute.type_protos || []).map((item) => summarizeTypeProto(item)) };
        default:
            return { summary: '—', detail: null };
    }
}

function summarizeRepeatedAttribute(values) {
    const normalized = values.map((item) => typeof item === 'bigint' ? item.toString() : item);
    return {
        summary: `[${normalized.slice(0, 8).join(', ')}${normalized.length > 8 ? ', …' : ''}]`,
        detail: normalized
    };
}

function summarizeMetadataProps(entries) {
    return (entries || []).map((entry) => {
        const key = entry.key || '';
        const value = entry.value || '';
        const json = tryParseJson(value);
        return {
            key,
            value,
            valuePreview: truncateMiddle(value, 160),
            isJson: json.ok,
            jsonValue: json.ok ? json.value : null,
            formattedJson: json.ok ? JSON.stringify(json.value, null, 2) : ''
        };
    });
}


export function detectTemporalMetadata(modelMetadata, graphMetadata) {
    const candidates = [];
    collectTemporalCandidates(candidates, modelMetadata, 'model');
    collectTemporalCandidates(candidates, graphMetadata, 'graph');
    candidates.sort((left, right) => temporalPriority(left) - temporalPriority(right) || lexicalSort(left.path, right.path));
    return {
        primary: candidates[0] || null,
        candidates
    };
}

function collectTemporalCandidates(target, entries, scope) {
    for (const entry of entries || []) {
        scanTemporalValue(target, entry.value, `${scope}.${entry.key}`, scope, entry.key, entry.isJson ? entry.jsonValue : undefined);
        if (entry.isJson) {
            scanTemporalJson(target, entry.jsonValue, `${scope}.${entry.key}`, scope, entry.key);
        }
    }
}

function scanTemporalJson(target, value, path, scope, key) {
    if (!value || typeof value !== 'object') {
        return;
    }
    if (Array.isArray(value)) {
        value.forEach((item, index) => scanTemporalJson(target, item, `${path}[${index}]`, scope, key));
        return;
    }
    for (const [childKey, childValue] of Object.entries(value)) {
        const childPath = `${path}.${childKey}`;
        scanTemporalValue(target, childValue, childPath, scope, childKey);
        if (childValue && typeof childValue === 'object') {
            scanTemporalJson(target, childValue, childPath, scope, childKey);
        }
    }
}

function scanTemporalValue(target, value, path, scope, key) {
    if (!looksTemporalKey(key) && !looksTemporalPath(path)) {
        return;
    }
    const parsed = parseTemporalValue(value);
    if (!parsed) {
        return;
    }
    target.push({
        scope,
        key,
        path,
        value: typeof value === 'string' ? value : JSON.stringify(value),
        isoValue: parsed.isoValue,
        displayValue: parsed.isoValue,
        sourceType: parsed.sourceType
    });
}

function looksTemporalKey(key) {
    const normalized = `${key || ''}`.toLowerCase();
    return /(?:^|[_\-.])(created|generated|exported|saved|timestamp|time|date|datetime|createdat|updatedat|modifiedat)(?:$|[_\-.])/.test(normalized)
        || ['created', 'generated', 'exported', 'saved', 'timestamp', 'time', 'date', 'datetime', 'createdat', 'updatedat', 'modifiedat'].includes(normalized);
}

function looksTemporalPath(path) {
    return /(created|generated|exported|saved|timestamp|time|date)/i.test(`${path || ''}`);
}

function parseTemporalValue(value) {
    if (value === null || value === undefined) {
        return null;
    }
    if (typeof value === 'number' && Number.isFinite(value)) {
        return parseEpochNumber(value);
    }
    if (typeof value === 'bigint') {
        return parseEpochNumber(Number(value));
    }
    if (typeof value !== 'string') {
        return null;
    }
    const trimmed = value.trim();
    if (!trimmed) {
        return null;
    }
    if (/^\d{10,17}$/.test(trimmed)) {
        const epoch = Number(trimmed);
        return parseEpochNumber(epoch);
    }
    const time = Date.parse(trimmed);
    if (Number.isNaN(time)) {
        return null;
    }
    return {
        isoValue: new Date(time).toISOString(),
        sourceType: 'date-string'
    };
}

function parseEpochNumber(value) {
    if (!Number.isFinite(value) || value <= 0) {
        return null;
    }
    let millis = value;
    if (value < 1e11) {
        millis = value * 1000;
    } else if (value > 1e14) {
        millis = Math.round(value / 1000);
    }
    const date = new Date(millis);
    if (Number.isNaN(date.getTime())) {
        return null;
    }
    return {
        isoValue: date.toISOString(),
        sourceType: 'epoch'
    };
}

function temporalPriority(candidate) {
    const path = `${candidate?.path || ''}`.toLowerCase();
    const orderedPatterns = [
        /export/, /generate/, /create/, /save/, /modify/, /update/, /time/, /date/
    ];
    for (let index = 0; index < orderedPatterns.length; index += 1) {
        if (orderedPatterns[index].test(path)) {
            return index;
        }
    }
    return orderedPatterns.length;
}

function buildCollapsedConstantCatalog(nodes, consumerRefsByTensor, nodeById = new Map()) {
    const catalog = new Map();
    for (const node of nodes) {
        if (`${node?.opType || ''}`.toLowerCase() !== 'constant') {
            continue;
        }
        const tensorName = (node.outputs || [])[0] || '';
        if (!tensorName) {
            continue;
        }
        const consumers = consumerRefsByTensor.get(tensorName) || [];
        if (consumers.length !== 1) {
            continue;
        }
        const consumerRef = consumers[0];
        const consumerNode = nodeById.get(consumerRef.nodeId);
        if (!consumerNode) {
            continue;
        }
        const constantInfo = summarizeConstantNodeValue(node);
        if (!constantInfo || !constantInfo.mergeable) {
            continue;
        }
        catalog.set(tensorName, {
            ...constantInfo,
            nodeId: node.id,
            tensorName,
            sourceNodeTitle: node.displayName,
            consumerNodeId: consumerRef.nodeId,
            inputIndex: consumerRef.inputIndex,
            label: inferInlineConstantLabel(consumerNode.opType, consumerRef.inputIndex, tensorName)
        });
    }
    return catalog;
}

function summarizeConstantNodeValue(node) {
    const valueAttribute = (node?.attributes || []).find((attribute) => `${attribute.name || ''}`.toLowerCase() === 'value')
        || (node?.attributes || [])[0]
        || null;
    if (!valueAttribute) {
        return null;
    }

    if (valueAttribute.type === 'TENSOR' && valueAttribute.valueDetail) {
        const tensor = valueAttribute.valueDetail;
        const numericCount = Number.parseInt(tensor.elementCount || '', 10);
        const elementCount = Number.isFinite(numericCount) ? numericCount : null;
        return {
            attributeName: valueAttribute.name,
            dataTypeName: tensor.dataTypeName || 'TENSOR',
            shapeLabel: tensor.shapeSummary ? tensor.shapeSummary.replace(/^\[|\]$/g, '') : '',
            shapeSummary: tensor.shapeSummary || '',
            valuePreview: tensor.decodedValuePreview || '',
            valueSummary: tensor.decodedValuePreview || `${tensor.dataTypeName || 'tensor'}${tensor.shapeSummary ? ` ${tensor.shapeSummary}` : ''}`,
            rawValue: tensor.decodedValuePreview || tensor.storage?.summary || '',
            mergeable: elementCount !== null ? elementCount <= 16 : Boolean((tensor.decodedValues || []).length && (tensor.decodedValues || []).length <= 16)
        };
    }

    const primitivePreview = valueAttribute.valueSummary || 'constant';
    const detailLength = Array.isArray(valueAttribute.valueDetail) ? valueAttribute.valueDetail.length : null;
    return {
        attributeName: valueAttribute.name,
        dataTypeName: valueAttribute.type,
        shapeLabel: '',
        shapeSummary: '',
        valuePreview: primitivePreview,
        valueSummary: primitivePreview,
        rawValue: primitivePreview,
        mergeable: detailLength === null ? primitivePreview.length <= 96 : detailLength <= 16
    };
}

function inferInlineConstantLabel(opType, inputIndex, tensorName) {
    switch (`${opType || ''}`.toLowerCase()) {
        case 'unsqueeze':
        case 'squeeze':
            return 'axes';
        case 'reshape':
        case 'expand':
        case 'constantofshape':
            return 'shape';
        case 'slice':
            return ['data', 'starts', 'ends', 'axes', 'steps'][inputIndex] || `const_${inputIndex}`;
        case 'pad':
            return ['data', 'pads', 'constant_value', 'axes'][inputIndex] || `const_${inputIndex}`;
        case 'gather':
        case 'gatherelements':
        case 'gathernd':
            return 'indices';
        case 'topk':
            return 'k';
        case 'tile':
            return 'repeats';
        case 'range':
            return ['start', 'limit', 'delta'][inputIndex] || `const_${inputIndex}`;
        case 'onehot':
            return ['indices', 'depth', 'values'][inputIndex] || `const_${inputIndex}`;
        case 'clip':
            return inputIndex === 1 ? 'min' : inputIndex === 2 ? 'max' : `const_${inputIndex}`;
        default:
            return normalizedInlineConstantLabel(tensorName, inputIndex);
    }
}

function normalizedInlineConstantLabel(tensorName, inputIndex) {
    const normalized = `${tensorName || ''}`.toLowerCase();
    if (normalized.includes('axes')) return 'axes';
    if (normalized.includes('shape')) return 'shape';
    if (normalized.includes('pads')) return 'pads';
    if (normalized.includes('bias')) return 'bias';
    if (normalized.includes('scale')) return 'scale';
    return inputIndex === 1 ? 'constant' : `const_${inputIndex}`;
}

function buildDisplayAttributeRows(node, embeddedConstants = []) {
    const rows = [];
    const opType = `${node?.opType || ''}`.toLowerCase();
    const mergedLabels = new Set((embeddedConstants || []).map((item) => `${item.label || ''}`.toLowerCase()));

    if (opType === 'constant') {
        const constantInfo = summarizeConstantNodeValue(node);
        if (constantInfo) {
            rows.push({
                kind: 'attribute',
                label: 'value',
                shape: constantInfo.shapeLabel || constantInfo.valueSummary || '—',
                dtype: constantInfo.dataTypeName || 'TENSOR',
                name: node.displayName,
                preview: constantInfo.valuePreview || constantInfo.valueSummary || '',
                secondaryValue: constantInfo.valuePreview || constantInfo.valueSummary || '—'
            });
        }
        return rows;
    }

    const compactAttributeNames = new Set(['axes', 'axis', 'perm', 'pads', 'starts', 'ends', 'steps', 'keepdims']);
    for (const attribute of node?.attributes || []) {
        const attributeName = `${attribute.name || ''}`.toLowerCase();
        if (!compactAttributeNames.has(attributeName) || mergedLabels.has(attributeName)) {
            continue;
        }
        rows.push({
            kind: 'attribute',
            label: attribute.name,
            shape: attribute.valueSummary || '—',
            dtype: attribute.type || '',
            name: attribute.name,
            preview: attribute.valueSummary || '',
            secondaryValue: attribute.valueSummary || '—'
        });
    }

    return rows;
}

function computeOperatorRenderStyle(node, parameterRows = [], embeddedConstants = [], displayAttributeRows = []) {
    const family = classifyOperatorFamily(node?.opType || '');
    const attributeRows = displayAttributeRows.filter((row) => row.kind === 'attribute').length;
    const mergedConstantRows = embeddedConstants.length;
    const rowCount = Array.isArray(parameterRows) ? parameterRows.length : 0;

    let variant = 'standard';
    if (family === 'activation' && (rowCount === 0 || (rowCount <= 1 && displayAttributeRows.every((row) => ['alpha', 'beta'].includes(`${row.label || ''}`.toLowerCase()))))) {
        variant = 'micro';
    } else if ((family === 'shape' || family === 'constant') && rowCount <= 1) {
        variant = 'mini';
    } else if ((family === 'utility' || family === 'other') && rowCount <= 1) {
        variant = 'compact';
    }

    return {
        family,
        variant,
        headerHeight: variant === 'micro' ? 34 : variant === 'mini' ? 26 : variant === 'compact' ? 30 : 34,
        rowHeight: variant === 'mini' ? 17 : variant === 'compact' ? 19 : 22,
        cornerRadius: variant === 'micro' ? 14 : variant === 'mini' ? 11 : 12,
        rowCount,
        mergedConstantRows,
        attributeRows,
        bodyVisible: variant !== 'micro' && rowCount > 0
    };
}

function classifyOperatorFamily(opType) {
    const normalized = `${opType || ''}`.toLowerCase();
    if (normalized === 'constant') {
        return 'constant';
    }
    if (['relu', 'elu', 'leakyrelu', 'selu', 'sigmoid', 'tanh', 'softplus', 'softsign', 'hardsigmoid', 'clip', 'gelu'].includes(normalized)) {
        return 'activation';
    }
    if (['reshape', 'flatten', 'transpose', 'unsqueeze', 'squeeze', 'concat', 'slice', 'gather', 'gatherelements', 'gathernd', 'expand', 'tile', 'identity', 'cast', 'pad'].includes(normalized)) {
        return 'shape';
    }
    if (['batchnormalization', 'instancenormalization', 'layernormalization'].includes(normalized)) {
        return 'normalization';
    }
    if (['gru', 'lstm', 'rnn'].includes(normalized)) {
        return 'recurrent';
    }
    if (['conv', 'convtranspose', 'gemm', 'matmul', 'qlinearconv'].includes(normalized)) {
        return 'linear';
    }
    if (['add', 'sub', 'mul', 'div', 'pow', 'reducemean', 'reducesum', 'argmax', 'argmin', 'topk'].includes(normalized)) {
        return 'utility';
    }
    return 'other';
}

function buildEmbeddedParameter(node, initializer, inputIndex) {
    const label = inferParameterLabel(node?.opType || '', inputIndex, initializer.name || '');
    return {
        index: inputIndex,
        label,
        name: initializer.name,
        dataTypeName: initializer.dataTypeName,
        shapeLabel: initializer.shapeSummary ? initializer.shapeSummary.replace(/^\[|\]$/g, '') : '',
        shapeSummary: initializer.shapeSummary,
        elementCount: initializer.elementCount,
        estimatedBytes: initializer.estimatedBytes,
        metadata: initializer.metadata,
        docString: initializer.docString,
        valuePreview: initializer.decodedValuePreview || ''
    };
}

export function inferParameterLabel(opType, inputIndex, tensorName) {
    const normalizedOp = `${opType || ''}`.toLowerCase();
    const normalizedName = `${tensorName || ''}`.toLowerCase();

    switch (normalizedOp) {
        case 'gemm':
            return ['A', 'B', 'C'][inputIndex] || `Param ${inputIndex}`;
        case 'matmul':
            return inputIndex === 1 ? 'B' : `Input ${inputIndex}`;
        case 'gru':
        case 'rnn':
            return ['X', 'W', 'R', 'B', 'sequence_lens', 'initial_h'][inputIndex] || `Param ${inputIndex}`;
        case 'lstm':
            return ['X', 'W', 'R', 'B', 'sequence_lens', 'initial_h', 'initial_c', 'P'][inputIndex] || `Param ${inputIndex}`;
        default:
            break;
    }

    if (normalizedName.includes('bias')) return 'Bias';
    if (normalizedName.includes('running_mean') || normalizedName.endsWith('.mean') || normalizedName.includes('mean')) return 'Mean';
    if (normalizedName.includes('running_var') || normalizedName.includes('variance') || normalizedName.endsWith('.var')) return 'Variance';
    if (normalizedName.includes('scale') || normalizedName.includes('gamma')) return 'Scale';
    if (normalizedName.includes('beta')) return 'Bias';
    if (normalizedName.includes('filter')) return 'Filter';
    if (normalizedName.includes('weight') || normalizedName.endsWith('.w')) return 'Weight';

    switch (normalizedOp) {
        case 'conv':
        case 'convtranspose':
        case 'qlinearconv':
            return inputIndex === 1 ? 'Filter' : inputIndex === 2 ? 'Bias' : `Param ${inputIndex}`;
        case 'batchnormalization':
            return ['Input', 'Scale', 'Bias', 'Mean', 'Variance'][inputIndex] || `Param ${inputIndex}`;
        case 'layernormalization':
        case 'instancenormalization':
            return inputIndex === 1 ? 'Scale' : inputIndex === 2 ? 'Bias' : `Param ${inputIndex}`;
        case 'add':
        case 'sub':
        case 'mul':
        case 'div':
        case 'pow':
            return inputIndex === 1 ? 'B' : `Input ${inputIndex}`;
        default:
            return inputIndex === 1 ? 'Weight' : inputIndex === 2 ? 'Bias' : `Param ${inputIndex}`;
    }
}

function shortTypeLabel(typeSummary) {
    const text = `${typeSummary || ''}`;
    const tensorMatch = text.match(/^tensor<([^\s>]+)(?:\s+\[([^\]]+)\])?>$/i);
    if (tensorMatch) {
        const [, dtype, shape] = tensorMatch;
        return shape ? `${dtype} · ${shape}` : dtype;
    }
    return text.replace(/^tensor</i, '').replace(/>$/i, '');
}

function layoutGraph(nodes, edges) {
    if (!nodes.length) {
        return {
            positions: {},
            routedEdges: {},
            width: 900,
            height: 420,
            algorithm: 'empty'
        };
    }

    try {
        return dagreCompactLayout(nodes, edges);
    } catch {
        return fallbackCompactLayout(nodes, edges);
    }
}

function dagreCompactLayout(nodes, edges) {
    const measuredNodes = nodes.map((node) => {
        const size = measureDisplayNode(node);
        return {
            v: node.id,
            width: size.width,
            height: size.height
        };
    });

    const measuredEdges = edges.map((edge) => ({
        v: edge.sourceId,
        w: edge.targetId,
        minlen: 1,
        weight: edge.isModelInput ? 5 : 4,
        width: 0,
        height: 0,
        labeloffset: 10,
        labelpos: 'c'
    }));

    const layout = {
        nodesep: 28,
        ranksep: 42,
        rankdir: 'TB',
        marginx: 24,
        marginy: 24
    };

    dagreLayout(measuredNodes, measuredEdges, layout, {});

    let minX = Number.POSITIVE_INFINITY;
    let minY = Number.POSITIVE_INFINITY;
    let maxX = Number.NEGATIVE_INFINITY;
    let maxY = Number.NEGATIVE_INFINITY;

    const positions = new Map();
    for (const node of measuredNodes) {
        const x = (node.x || 0) - node.width / 2;
        const y = (node.y || 0) - node.height / 2;
        positions.set(node.v, {
            x,
            y,
            width: node.width,
            height: node.height
        });
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x + node.width);
        maxY = Math.max(maxY, y + node.height);
    }

    const routedEdges = {};
    for (let index = 0; index < measuredEdges.length; index += 1) {
        const edgeLayout = measuredEdges[index];
        const sourceEdge = edges[index];
        const points = (edgeLayout.points || []).map((point) => ({ x: point.x, y: point.y }));
        routedEdges[sourceEdge.id] = points;
        for (const point of points) {
            minX = Math.min(minX, point.x);
            minY = Math.min(minY, point.y);
            maxX = Math.max(maxX, point.x);
            maxY = Math.max(maxY, point.y);
        }
    }

    if (!Number.isFinite(minX) || !Number.isFinite(minY) || !Number.isFinite(maxX) || !Number.isFinite(maxY)) {
        return fallbackCompactLayout(nodes, edges);
    }

    const padding = 40;
    const offsetX = padding - minX;
    const offsetY = padding - minY;
    for (const [id, position] of positions.entries()) {
        position.x += offsetX;
        position.y += offsetY;
    }
    for (const [edgeId, points] of Object.entries(routedEdges)) {
        routedEdges[edgeId] = points.map((point) => ({ x: point.x + offsetX, y: point.y + offsetY }));
    }

    return {
        positions: Object.fromEntries(positions),
        routedEdges,
        width: Math.max(480, Math.ceil(maxX - minX + padding * 2)),
        height: Math.max(320, Math.ceil(maxY - minY + padding * 2)),
        algorithm: 'dagre-tb'
    };
}

function fallbackCompactLayout(nodes, edges) {
    const nodeById = new Map(nodes.map((node) => [node.id, node]));
    const incoming = new Map(nodes.map((node) => [node.id, new Set()]));
    const outgoing = new Map(nodes.map((node) => [node.id, new Set()]));

    for (const edge of edges) {
        if (!incoming.has(edge.targetId) || !outgoing.has(edge.sourceId)) {
            continue;
        }
        incoming.get(edge.targetId).add(edge.sourceId);
        outgoing.get(edge.sourceId).add(edge.targetId);
    }

    const indegree = new Map(nodes.map((node) => [node.id, incoming.get(node.id).size]));
    const queue = nodes.filter((node) => indegree.get(node.id) === 0).map((node) => node.id);
    const topo = [];
    while (queue.length > 0) {
        const id = queue.shift();
        topo.push(id);
        for (const next of outgoing.get(id)) {
            const value = indegree.get(next) - 1;
            indegree.set(next, value);
            if (value === 0) {
                queue.push(next);
            }
        }
    }

    if (topo.length !== nodes.length) {
        for (const node of nodes) {
            if (!topo.includes(node.id)) {
                topo.push(node.id);
            }
        }
    }

    const depth = new Map();
    for (const id of topo) {
        const parents = [...incoming.get(id)];
        const parentDepth = parents.length > 0 ? Math.max(...parents.map((parent) => depth.get(parent) ?? 0)) : 0;
        const currentNode = nodeById.get(id);
        const baseDepth = currentNode?.kind === 'output' ? parentDepth + 1 : parentDepth + (currentNode?.kind === 'operator' ? 1 : 0);
        depth.set(id, baseDepth);
    }

    const rows = new Map();
    for (const node of nodes) {
        const row = depth.get(node.id) ?? 0;
        if (!rows.has(row)) {
            rows.set(row, []);
        }
        rows.get(row).push(node.id);
    }

    const sortedRows = [...rows.keys()].sort((a, b) => a - b);
    const positions = new Map();
    const horizontalSpacing = 160;
    const verticalSpacing = 116;

    for (const row of sortedRows) {
        const ids = rows.get(row);
        ids.sort((left, right) => lexicalSort(nodeById.get(left)?.title, nodeById.get(right)?.title));
        const sizes = ids.map((id) => measureDisplayNode(nodeById.get(id)));
        const totalWidth = sizes.reduce((acc, size) => acc + size.width, 0) + Math.max(0, ids.length - 1) * horizontalSpacing;
        let cursorX = Math.max(40, (totalWidth > 0 ? 40 : 40));
        ids.forEach((id, index) => {
            const size = sizes[index];
            positions.set(id, {
                x: cursorX,
                y: 40 + row * verticalSpacing,
                width: size.width,
                height: size.height
            });
            cursorX += size.width + horizontalSpacing;
        });
    }

    const maxWidth = Math.max(1, ...[...positions.values()].map((item) => item.x + item.width));
    const maxHeight = Math.max(1, ...[...positions.values()].map((item) => item.y + item.height));

    return {
        positions: Object.fromEntries(positions),
        routedEdges: {},
        width: Math.max(520, maxWidth + 40),
        height: Math.max(320, maxHeight + 40),
        algorithm: 'fallback-tb'
    };
}

function measureDisplayNode(node) {
    const title = `${node?.details?.opType || node?.title || ''}`;
    const subtitle = `${node?.subtitle || ''}`;
    if (node?.kind === 'operator') {
        const style = node.renderStyle || node.details?.renderStyle || { variant: 'standard', bodyVisible: true, rowHeight: 22 };
        const parameterRows = Array.isArray(node.parameterRows) ? node.parameterRows : [];
        const titleWidth = title.length * (style.variant === 'micro' ? 7.2 : 6.8);
        const subtitleWidth = subtitle.length * 5.1;
        const parameterWidth = parameterRows.reduce((max, row) => {
            const valueText = row.shape || row.preview || row.secondaryValue || row.dtype || '';
            const estimate = (row.label?.length || 0) * 6.4 + `${valueText}`.length * 5.3 + 44;
            return Math.max(max, estimate);
        }, 0);

        if (style.variant === 'micro') {
            return {
                width: clampNumber(Math.max(60, titleWidth + 24), 60, 110),
                height: 34
            };
        }

        if (style.variant === 'mini') {
            return {
                width: clampNumber(Math.max(90, titleWidth + 28, subtitleWidth + 24, parameterWidth), 90, 176),
                height: 30 + parameterRows.length * 16 + (parameterRows.length ? 10 : 0)
            };
        }

        if (style.variant === 'compact') {
            return {
                width: clampNumber(Math.max(100, titleWidth + 32, subtitleWidth + 28, parameterWidth), 100, 208),
                height: 34 + parameterRows.length * 18 + (parameterRows.length ? 12 : 0)
            };
        }

        return {
            width: clampNumber(Math.max(132, titleWidth + 36, subtitleWidth + 32, parameterWidth), 132, 272),
            height: 40 + parameterRows.length * 20 + (parameterRows.length ? 12 : 0)
        };
    }
    const width = clampNumber(64 + Math.max(title.length * 5.4, subtitle.length * 4.1), 64, 180);
    return {
        width,
        height: 34
    };
}

function clampNumber(value, min, max) {
    return Math.min(max, Math.max(min, value));
}

function normalizeBytes(data) {
    if (data instanceof Uint8Array) {
        return data;
    }
    if (data instanceof ArrayBuffer) {
        return new Uint8Array(data);
    }
    if (Array.isArray(data)) {
        return Uint8Array.from(data);
    }
    throw new Error('Unsupported ONNX input type.');
}

function tensorTypeName(value) {
    return TENSOR_DATA_TYPE_NAME_BY_VALUE[value] || `TYPE_${value ?? 'UNKNOWN'}`;
}

function versionNameForIR(value) {
    const target = Number(value);
    for (const [name, version] of Object.entries(onnx.Version)) {
        if (version === target) {
            return name;
        }
    }
    return '';
}

function inferAttributeType(attribute) {
    if (attribute.t) return 'TENSOR';
    if (attribute.g) return 'GRAPH';
    if (attribute.tp) return 'TYPE_PROTO';
    if (attribute.f !== undefined) return 'FLOAT';
    if (attribute.i !== undefined) return 'INT';
    if (attribute.s !== undefined) return 'STRING';
    if (attribute.floats?.length) return 'FLOATS';
    if (attribute.ints?.length) return 'INTS';
    if (attribute.strings?.length) return 'STRINGS';
    return 'UNDEFINED';
}

function invertEnum(enumObject) {
    const result = {};
    for (const [name, value] of Object.entries(enumObject)) {
        result[value] = name;
    }
    return result;
}

function multiplyBigIntArray(values) {
    if (!Array.isArray(values) || values.length === 0) {
        return 0n;
    }
    let total = 1n;
    for (const value of values) {
        if (value === undefined || value === null) {
            return null;
        }
        const numeric = typeof value === 'bigint' ? value : BigInt(value);
        if (numeric < 0) {
            return null;
        }
        total *= numeric;
    }
    return total;
}

function estimateTensorBytes(dataTypeName, elementCount, storage) {
    const rawField = storage.fields.find((field) => field.name === 'raw_data');
    if (rawField) {
        return rawField.count;
    }
    if (storage.fields.length > 0 && storage.fields.every((field) => field.name === 'string_data')) {
        return null;
    }
    if (elementCount === null) {
        return null;
    }
    const bytesPerElement = TENSOR_DTYPE_BYTES[dataTypeName];
    if (bytesPerElement === undefined) {
        return null;
    }
    const numericElementCount = Number(elementCount);
    if (!Number.isFinite(numericElementCount)) {
        return null;
    }
    return Math.ceil(numericElementCount * bytesPerElement);
}

function decodeUtf8(value) {
    if (typeof value === 'string') {
        return value;
    }
    if (value instanceof Uint8Array) {
        return new TextDecoder('utf-8').decode(value);
    }
    return `${value ?? ''}`;
}

function formatBigInt(value) {
    if (value === undefined || value === null || value === '') {
        return '';
    }
    if (typeof value === 'bigint') {
        return value.toString();
    }
    return `${value}`;
}

function formatBytes(bytes) {
    if (!Number.isFinite(bytes) || bytes < 0) {
        return 'Unknown';
    }
    const units = ['B', 'KB', 'MB', 'GB', 'TB'];
    let value = bytes;
    let unitIndex = 0;
    while (value >= 1024 && unitIndex < units.length - 1) {
        value /= 1024;
        unitIndex += 1;
    }
    const formatted = value >= 10 || unitIndex === 0 ? value.toFixed(unitIndex === 0 ? 0 : 1) : value.toFixed(2);
    return `${formatted} ${units[unitIndex]}`;
}

function summarizeArrayPreview(values) {
    const normalized = values.slice(0, 8).map((value) => typeof value === 'bigint' ? value.toString() : value);
    return `[${normalized.join(', ')}${values.length > 8 ? ', …' : ''}]`;
}

function summarizeBytePreview(values) {
    const normalized = Array.from(values.slice(0, 8)).map((value) => value.toString(16).padStart(2, '0'));
    return `${normalized.join(' ')}${values.length > 8 ? ' …' : ''}`;
}

function tryParseJson(value) {
    if (typeof value !== 'string') {
        return { ok: false };
    }
    const trimmed = value.trim();
    if (!trimmed || !['{', '['].includes(trimmed[0])) {
        return { ok: false };
    }
    try {
        return { ok: true, value: JSON.parse(trimmed) };
    } catch {
        return { ok: false };
    }
}

function truncateMiddle(value, limit) {
    if (!value || value.length <= limit) {
        return value;
    }
    const head = Math.ceil((limit - 1) / 2);
    const tail = Math.floor((limit - 1) / 2);
    return `${value.slice(0, head)}…${value.slice(-tail)}`;
}

function average(values) {
    if (!values.length) {
        return 0;
    }
    return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function lexicalSort(left = '', right = '') {
    return `${left}`.localeCompare(`${right}`);
}
