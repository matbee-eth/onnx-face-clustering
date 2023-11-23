"use strict";
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
Object.defineProperty(exports, "__esModule", { value: true });
exports.parseTransposeAttributes = exports.transpose = exports.createTransposeProgramInfo = exports.transposeProgramMetadata = void 0;
const util_1 = require("../../util");
const attribute_with_cache_key_1 = require("../attribute-with-cache-key");
const types_1 = require("../types");
const common_1 = require("./common");
exports.transposeProgramMetadata = {
    name: 'Transpose',
    inputTypes: [types_1.GpuDataType.default]
};
const validateInputs = (inputs) => {
    if (!inputs || inputs.length !== 1) {
        throw new Error('Transpose requires 1 input.');
    }
    if (inputs[0].dataType !== 1 /* DataType.float */ && inputs[0].dataType !== 6 /* DataType.int32 */ &&
        inputs[0].dataType !== 12 /* DataType.uint32 */) {
        throw new Error('Transpose only support float, int32, and uint32 data types');
    }
};
const getAdjustedPerm = (inputShape, perm) => (perm && perm.length !== inputShape.length) ? [...(inputShape.keys())].reverse() : perm;
const getOutputShape = (inputShape, perm) => util_1.ShapeUtil.sortBasedOnPerm(inputShape, getAdjustedPerm(inputShape, perm));
const permFunctionBody = (perm, rank, input, output) => {
    const reverseFunc = [];
    reverseFunc.push(`fn perm(i: ${output.type.indices}) -> ${input.type.indices} {
    var a: ${input.type.indices};`);
    for (let i = 0; i < rank; ++i) {
        reverseFunc.push(input.indicesSet('a', perm[i], `i[${i}]`));
    }
    reverseFunc.push('return a;}');
    return reverseFunc.join('\n');
};
const createTransposeProgramInfo = (inputTensor, permAttr) => {
    const dataType = inputTensor.dataType;
    const inputShape = inputTensor.dims;
    const perm = getAdjustedPerm(inputShape, permAttr);
    const outputShape = getOutputShape(inputShape, perm);
    const rank = inputShape.length;
    const outputSize = util_1.ShapeUtil.size(outputShape);
    // A dims=[${inputs[0].dims.toString()}]
    // out Dims=[${unpackedOutputShape.toString()}]
    // based on perm=[${perm.toString()}]
    const output = (0, common_1.outputVariable)('output', dataType, outputShape);
    const input = (0, common_1.inputVariable)('a', dataType, inputShape);
    const getShaderSource = (shaderHelper) => `
  ${shaderHelper.declareVariables(input, output)}

  ${permFunctionBody(perm, rank, input, output)}

  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}

    let indices = ${output.offsetToIndices('global_idx')};
    let aIndices = perm(indices);

    ${output.setByOffset('global_idx', input.getByIndices('aIndices'))}
  }`;
    return {
        ...exports.transposeProgramMetadata,
        outputs: [{ dims: outputShape, dataType: inputTensor.dataType, gpuDataType: types_1.GpuDataType.default }],
        getShaderSource,
        dispatchGroup: () => ({ x: Math.ceil(outputSize / 64 /* workgroup size */) })
    };
};
exports.createTransposeProgramInfo = createTransposeProgramInfo;
const transpose = (context, attributes) => {
    validateInputs(context.inputs);
    context.compute({
        ...exports.transposeProgramMetadata,
        cacheHint: attributes.cacheKey,
        get: () => (0, exports.createTransposeProgramInfo)(context.inputs[0], attributes.perm)
    });
};
exports.transpose = transpose;
const parseTransposeAttributes = (attributes) => (0, attribute_with_cache_key_1.createAttributeWithCacheKey)({ perm: attributes.perm });
exports.parseTransposeAttributes = parseTransposeAttributes;
//# sourceMappingURL=transpose.js.map