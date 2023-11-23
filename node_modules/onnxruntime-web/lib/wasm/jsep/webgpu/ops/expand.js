"use strict";
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
Object.defineProperty(exports, "__esModule", { value: true });
exports.expand = exports.expandProgramMetadata = void 0;
const util_1 = require("../../util");
const types_1 = require("../types");
const common_1 = require("./common");
exports.expandProgramMetadata = {
    name: 'Expand',
    inputTypes: [types_1.GpuDataType.default]
};
const validateInputs = (inputs) => {
    if (!inputs || inputs.length !== 2) {
        throw new Error('Expand requires 2 input.');
    }
    const inputShape = inputs[0].dims;
    const shape = Array.from(inputs[1].getBigInt64Array(), Number);
    let shapeIndex = shape.length < inputShape.length ? 0 : shape.length - inputShape.length;
    let inputShapeIndex = inputShape.length < shape.length ? 0 : inputShape.length - shape.length;
    for (; shapeIndex < shape.length && inputShapeIndex < inputShape.length; ++shapeIndex, ++inputShapeIndex) {
        if (shape[shapeIndex] !== inputShape[inputShapeIndex] && shape[shapeIndex] !== 1 &&
            inputShape[inputShapeIndex] !== 1) {
            throw new Error('Expand requires shape to be broadcastable to input');
        }
    }
};
const getAdjustedShape = (shape1, shape2) => {
    const diff = shape1.length - shape2.length;
    const shape = [];
    for (let i = 0; i < diff; ++i) {
        shape.push(shape1[i]);
    }
    for (let i = 0; i < shape2.length; ++i) {
        shape.push(shape2[i] === 1 ? shape1[i + diff] : shape2[i]);
    }
    return shape;
};
const calculateOutputShape = (inputShape, shape) => (inputShape.length > shape.length) ? getAdjustedShape(inputShape, shape) : getAdjustedShape(shape, inputShape);
const createExpandProgramInfo = (metadata, inputs) => {
    const inputShape = inputs[0].dims;
    const shape = Array.from(inputs[1].getBigInt64Array(), Number);
    const outputShape = calculateOutputShape(inputShape, shape);
    const outputSize = util_1.ShapeUtil.size(outputShape);
    const dataType = inputs[0].dataType;
    const input = (0, common_1.inputVariable)('input', dataType, inputShape);
    const output = (0, common_1.outputVariable)('output', dataType, outputShape);
    const getShaderSource = (shaderHelper) => `
  const inputShape = ${input.indices(...inputShape)};
  ${shaderHelper.declareVariables(input, output)}
  ${shaderHelper.mainStart()}
  ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}
    let outputIndices = ${output.offsetToIndices('global_idx')};
    var inputIndices: ${input.type.indices};
    for (var i = 0; i < ${inputShape.length}; i++) {
      if (${input.indicesGet('inputShape', 'i')} == 1) {
        ${input.indicesSet('inputIndices', 'i', 0)}
      } else {
        ${input.indicesSet('inputIndices', 'i', output.indicesGet('outputIndices', `i + ${outputShape.length - inputShape.length}`))}
      }
    }
    ${output.setByOffset('global_idx', input.getByIndices('inputIndices'))}
  }`;
    return {
        ...metadata,
        getShaderSource,
        outputs: [{ dims: outputShape, dataType: inputs[0].dataType, gpuDataType: types_1.GpuDataType.default }],
        dispatchGroup: () => ({ x: Math.ceil(outputSize / 64 /* workgroup size */) })
    };
};
const expand = (context) => {
    validateInputs(context.inputs);
    const outputShape = Array.from(context.inputs[1].getBigInt64Array(), Number);
    const cacheHint = outputShape.toString();
    context.compute({ ...exports.expandProgramMetadata, cacheHint, get: () => createExpandProgramInfo(exports.expandProgramMetadata, context.inputs) }, { inputs: [0] });
};
exports.expand = expand;
//# sourceMappingURL=expand.js.map