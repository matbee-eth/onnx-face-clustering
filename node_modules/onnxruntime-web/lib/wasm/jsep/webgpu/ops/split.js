"use strict";
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
Object.defineProperty(exports, "__esModule", { value: true });
exports.parseSplitAttributes = exports.split = void 0;
const util_1 = require("../../util");
const attribute_with_cache_key_1 = require("../attribute-with-cache-key");
const types_1 = require("../types");
const common_1 = require("./common");
const validateInputs = (inputs) => {
    if (!inputs || inputs.length < 1) {
        throw new Error('too few inputs');
    }
};
const createSplitAttributesFromInputs = (inputs, attributes) => {
    const splitSizes = [];
    let numOutputs = attributes.numOutputs;
    if (inputs[1].dims[0] > 0) {
        inputs[1].getBigInt64Array().forEach(v => splitSizes.push(Number(v)));
        numOutputs = splitSizes.length;
    }
    return (0, attribute_with_cache_key_1.createAttributeWithCacheKey)({ numOutputs, axis: attributes.axis, splitSizes });
};
const calculateOutputIndexImpl = (numberOfTensors) => `
fn calculateOutputIndex(index: u32) -> u32 {
    for (var i: u32 = 0u; i < ${numberOfTensors}u; i += 1u ) {
    if (index < sizeInConcatAxis[i]) {
        return i;
    }
    }
    return ${numberOfTensors}u;
}`;
const writeBufferDataImpl = (outputs) => {
    const numberOfTensors = outputs.length;
    const codeLines = [];
    for (let i = 0; i < numberOfTensors; ++i) {
        const returnSnippet = outputs[i].setByIndices('indices', 'input[global_idx]');
        if (numberOfTensors === 1) {
            codeLines.push(returnSnippet);
        }
        else if (i === 0) {
            codeLines.push(`if (outputNumber == ${i}u) { ${returnSnippet} }`);
        }
        else if (i === numberOfTensors - 1) {
            codeLines.push(`else { ${returnSnippet} }`);
        }
        else {
            codeLines.push(`else if (outputNumber == ${i}) { ${returnSnippet} }`);
        }
    }
    return `
      fn writeBufferData(outputNumber: u32, indices: ${outputs[0].type.indices}, global_idx: u32) {
        ${codeLines.join('\n')}
      }`;
};
const createSplitProgramInfo = (metadata, inputs, attributes) => {
    const inputShape = inputs[0].dims;
    const inputSize = util_1.ShapeUtil.size(inputShape);
    const dataType = inputs[0].dataType;
    const rank = inputShape.length;
    const axis = attributes.axis;
    const adjustedAxis = (axis < 0) ? inputShape.length + axis : axis;
    const outputs = new Array(attributes.numOutputs);
    const input = (0, common_1.inputVariable)('input', dataType, inputShape);
    const sizeInConcatAxis = new Array(attributes.numOutputs);
    const outputsTensorInfo = [];
    const outputShapes = [];
    let previousSum = 0;
    for (let i = 0; i < attributes.numOutputs; i++) {
        previousSum += attributes.splitSizes[i];
        sizeInConcatAxis[i] = previousSum;
        const outputShape = inputShape.slice();
        outputShape[attributes.axis] = attributes.splitSizes[i];
        outputShapes.push(outputShape);
        outputs[i] = (0, common_1.outputVariable)(`output${i}`, dataType, outputShapes[i]);
        outputsTensorInfo.push({ dims: outputShapes[i], dataType: inputs[0].dataType, gpuDataType: types_1.GpuDataType.default });
    }
    const indicesAxis = rank < 2 ? 'indices' : `indices[${adjustedAxis}]`;
    const getShaderSource = (shaderHelper) => `
  ${shaderHelper.declareVariables(input, ...outputs)}
  const sizeInConcatAxis = array<u32, ${sizeInConcatAxis.length}>(${sizeInConcatAxis.map(i => `${i}u`).join(',')});
  ${calculateOutputIndexImpl(sizeInConcatAxis.length)}
  ${writeBufferDataImpl(outputs)}

  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(inputSize)}

    var indices = ${input.offsetToIndices('global_idx')};
    let outputNumber = calculateOutputIndex(${indicesAxis});
    if (outputNumber != 0) {
        ${indicesAxis} -= sizeInConcatAxis[outputNumber - 1u];
    }
    writeBufferData(outputNumber, indices, global_idx);
  }`;
    return {
        ...metadata,
        getShaderSource,
        outputs: outputsTensorInfo,
        dispatchGroup: () => ({ x: Math.ceil(inputSize / 64 /* workgroup size */) })
    };
};
const createSplitProgramInfoLoader = (inputs, attributes) => {
    const updatedAttributes = inputs.length === 1 ? attributes : createSplitAttributesFromInputs(inputs, attributes);
    const metadata = { name: 'Split', inputTypes: [types_1.GpuDataType.default], cacheHint: updatedAttributes.cacheKey };
    return { ...metadata, get: () => createSplitProgramInfo(metadata, [inputs[0]], updatedAttributes) };
};
const split = (context, attributes) => {
    validateInputs(context.inputs);
    context.compute(createSplitProgramInfoLoader(context.inputs, attributes), { inputs: [0] });
};
exports.split = split;
const parseSplitAttributes = (attributes) => {
    const axis = attributes.axis;
    const splitSizes = attributes.splitSizes;
    const numOutputs = attributes.numOutputs < 0 ? splitSizes.length : attributes.numOutputs;
    if (numOutputs !== splitSizes.length) {
        throw new Error('numOutputs and splitSizes lengh must be equal');
    }
    return (0, attribute_with_cache_key_1.createAttributeWithCacheKey)({ axis, numOutputs, splitSizes });
};
exports.parseSplitAttributes = parseSplitAttributes;
//# sourceMappingURL=split.js.map