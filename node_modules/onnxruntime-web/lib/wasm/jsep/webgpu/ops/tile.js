"use strict";
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
Object.defineProperty(exports, "__esModule", { value: true });
exports.tile = exports.createTileProgramInfo = exports.tileProgramMetadata = void 0;
const util_1 = require("../../util");
const types_1 = require("../types");
const common_1 = require("./common");
exports.tileProgramMetadata = {
    name: 'Tile',
    inputTypes: [types_1.GpuDataType.default]
};
const getRepeats = (repeatsTensorView) => Array.from(repeatsTensorView.getBigInt64Array(), Number);
const validateInputs = (inputs) => {
    if (!inputs || inputs.length !== 2) {
        throw new Error('Tile requires 2 inputs.');
    }
    if (inputs[0].dataType !== 1 /* DataType.float */ && inputs[0].dataType !== 6 /* DataType.int32 */ &&
        inputs[0].dataType !== 12 /* DataType.uint32 */) {
        throw new Error('Tile only support float, int32, and uint32 data types');
    }
    if (inputs[1].dataType !== 7 /* DataType.int64 */) {
        throw new Error('Tile `repeats` input should be of int64 data type');
    }
    if (inputs[1].dims.length !== 1) {
        throw new Error('Tile `repeats` input should be 1-D');
    }
    const repeats = getRepeats(inputs[1]);
    if (repeats.length !== inputs[0].dims.length) {
        throw new Error('Tile `repeats` input should have same number of elements as rank of input data tensor');
    }
};
const getOutputShape = (inputShape, repeats) => {
    const outputShape = [];
    for (let i = 0; i < inputShape.length; ++i) {
        outputShape.push(inputShape[i] * repeats[i]);
    }
    return outputShape;
};
const createTileProgramInfo = (tileProgramMetadata, inputs) => {
    const inputShape = inputs[0].dims;
    const repeats = getRepeats(inputs[1]);
    const outputShape = getOutputShape(inputShape, repeats);
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
        let inputDimValue = ${output.indicesGet('outputIndices', 'i')}  % ${input.indicesGet('inputShape', 'i')};

        ${input.indicesSet('inputIndices', 'i', 'inputDimValue')}
      }
      ${output.setByOffset('global_idx', input.getByIndices('inputIndices'))}
    }`;
    return {
        ...tileProgramMetadata,
        outputs: [{ dims: outputShape, dataType: inputs[0].dataType, gpuDataType: types_1.GpuDataType.default }],
        getShaderSource,
        dispatchGroup: () => ({ x: Math.ceil(outputSize / 64 /* workgroup size */) })
    };
};
exports.createTileProgramInfo = createTileProgramInfo;
const tile = (context) => {
    validateInputs(context.inputs);
    const repeats = getRepeats(context.inputs[1]);
    const cacheHint = repeats.toString();
    context.compute({ ...exports.tileProgramMetadata, cacheHint, get: () => (0, exports.createTileProgramInfo)(exports.tileProgramMetadata, context.inputs) }, { inputs: [0] });
};
exports.tile = tile;
//# sourceMappingURL=tile.js.map