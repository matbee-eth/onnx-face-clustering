"use strict";
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
Object.defineProperty(exports, "__esModule", { value: true });
exports.parseArgMinMaxAttributes = exports.argMax = exports.argMin = void 0;
const attribute_with_cache_key_1 = require("../attribute-with-cache-key");
const types_1 = require("../types");
const reduce_1 = require("./reduce");
const validateInputs = (inputs) => {
    if (!inputs || inputs.length === 0 || inputs.length > 2) {
        throw new Error('ArgMinMaxOp op requires 1 or 2 inputs.');
    }
    if (inputs[0].dataType !== 1 /* DataType.float */) {
        throw new Error('Invalid input type.');
    }
};
const createArgMinMaxAttributesFromInputs = (inputs, attributes) => (0, attribute_with_cache_key_1.createAttributeWithCacheKey)({ axis: attributes.axis, keepDims: attributes.keepDims, selectLastIndex: attributes.selectLastIndex });
const createArgMinMaxProgramInfoLoader = (inputs, name, attributes, reduceOp) => {
    const updatedAttributes = inputs.length === 1 ? attributes : createArgMinMaxAttributesFromInputs(inputs, attributes);
    const cacheHint = updatedAttributes.cacheKey + inputs.map(x => x.dims.toString()).join('_');
    const metadata = { name, inputTypes: [types_1.GpuDataType.default], cacheHint };
    return {
        ...metadata,
        get: () => (0, reduce_1.createReduceProgramInfo)(metadata, [inputs[0]], reduceOp, [updatedAttributes.axis], 7 /* DataType.int64 */, updatedAttributes.keepDims)
    };
};
const argMin = (context, attributes) => {
    validateInputs(context.inputs);
    const argMinMaxOp = (input, output, axes) => {
        const idxZero = [];
        for (let k = 0; k < input.shape.length; k++) {
            if (axes.indexOf(k) >= 0 || axes.length === 0) {
                idxZero.push(`inputIndices[${k}] = 0;`); // first element
            }
        }
        return [
            `${idxZero.join('\n')}`, `var value = ${input.getByOffset('inputOffset')};\nvar bestIndex : i32 = 0;`,
            `if (${input.getByOffset('inputOffset')} ${attributes.selectLastIndex > 0 ? '<=' : '<'} value) {
         value = ${input.getByOffset('inputOffset')};
         bestIndex = i32(lastIndex);
       }`,
            '', output.setByOffset('global_idx', 'bestIndex')
        ];
    };
    context.compute(createArgMinMaxProgramInfoLoader(context.inputs, 'ArgMin', attributes, argMinMaxOp), { inputs: [0] });
};
exports.argMin = argMin;
const argMax = (context, attributes) => {
    validateInputs(context.inputs);
    const argMinMaxOp = (input, output, axes) => {
        const idxZero = [];
        for (let k = 0; k < input.shape.length; k++) {
            if (axes.indexOf(k) >= 0 || axes.length === 0) {
                idxZero.push(`inputIndices[${k}] = 0;`); // first element
            }
        }
        return [
            `${idxZero.join('\n')}`, `var value = ${input.getByOffset('inputOffset')};\nvar bestIndex : i32 = 0;`,
            `if (${input.getByOffset('inputOffset')} ${attributes.selectLastIndex > 0 ? '>=' : '>'} value) {
         value = ${input.getByOffset('inputOffset')};
         bestIndex = i32(lastIndex);
       }`,
            '', output.setByOffset('global_idx', 'bestIndex')
        ];
    };
    context.compute(createArgMinMaxProgramInfoLoader(context.inputs, 'argMax', attributes, argMinMaxOp), { inputs: [0] });
};
exports.argMax = argMax;
const parseArgMinMaxAttributes = (attributes) => (0, attribute_with_cache_key_1.createAttributeWithCacheKey)(attributes);
exports.parseArgMinMaxAttributes = parseArgMinMaxAttributes;
//# sourceMappingURL=argminmax.js.map