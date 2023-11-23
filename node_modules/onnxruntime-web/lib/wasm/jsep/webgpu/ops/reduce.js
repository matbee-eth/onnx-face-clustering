"use strict";
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
Object.defineProperty(exports, "__esModule", { value: true });
exports.parseReduceAttributes = exports.reduceSumSquare = exports.reduceSum = exports.reduceProd = exports.reduceMin = exports.reduceMean = exports.reduceMax = exports.reduceLogSumExp = exports.reduceL2 = exports.reduceL1 = exports.reduceLogSum = exports.createReduceProgramInfo = void 0;
const util_1 = require("../../util");
const attribute_with_cache_key_1 = require("../attribute-with-cache-key");
const types_1 = require("../types");
const common_1 = require("./common");
const validateInputs = (inputs) => {
    if (!inputs || inputs.length === 0 || inputs.length > 2) {
        throw new Error('Reduce op requires 1 or 2 inputs.');
    }
    if (inputs.length === 2 && inputs[1].dims.length !== 1) {
        throw new Error('Invalid axes input dims.');
    }
    if (inputs[0].dataType !== 1 /* DataType.float */) {
        throw new Error('Invalid input type.');
    }
};
const noOp = (input) => ['', '', `var value = ${input.getByOffset('inputOffset')};`, ''];
const createReduceProgramInfo = (metadata, inputs, reduceOp, axesInput, outputDataType, keepDims = false, noopWithEmptyAxes = false) => {
    const outputShape = [];
    const inputShape = inputs[0].dims;
    const axes = util_1.ShapeUtil.normalizeAxes(axesInput, inputs[0].dims.length);
    const reduceOnAllAxes = !noopWithEmptyAxes && axes.length === 0;
    inputShape.forEach((d, i) => {
        if (reduceOnAllAxes || axes.indexOf(i) >= 0) {
            if (keepDims) {
                outputShape.push(1);
            } // else { // skip this axis}
        }
        else {
            outputShape.push(d);
        }
    });
    const idxCopy = []; // copy output indexes to input indexes
    const input = (0, common_1.inputVariable)('_A', inputs[0].dataType, inputShape);
    const output = (0, common_1.outputVariable)('output', outputDataType, outputShape);
    const ops = reduceOp(input, output, axes);
    const inputOffsetAssignment = `inputOffset = ${input.indicesToOffset('inputIndices')};`;
    const initinputOffsetLet = `let ${inputOffsetAssignment};`;
    const initinputOffsetVar = `var ${inputOffsetAssignment};`;
    const initinputOffset = (ops[1] === '') ? '' : initinputOffsetVar;
    let reduceOps = ((ops[1] === '') ? initinputOffsetLet : inputOffsetAssignment) + '\n' + ops[2];
    for (let k = 0, l = 0; k < inputs[0].dims.length; k++) {
        // if this axis is reduced
        if (reduceOnAllAxes || axes.indexOf(k) >= 0) {
            if (keepDims) {
                l++;
            }
            // loop over the d-th axis
            reduceOps = `for(var j${k}: u32 = 0; j${k} < ${inputs[0].dims[k]}; j${k}++) {
                ${ops[2].includes('lastIndex') ? `let lastIndex = j${k};` : ''}
                ${input.indicesSet('inputIndices', k, `j${k}`)}
                ${reduceOps}
              }`;
        }
        else {
            idxCopy.push(`${input.indicesSet('inputIndices', k, output.indicesGet('outputIndices', l))};`);
            l++;
        }
    }
    const outputSize = util_1.ShapeUtil.size(outputShape);
    const getShaderSource = (shaderHelper) => `
        ${shaderHelper.declareVariables(input, output)}

        ${shaderHelper.mainStart()}
          ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}
          var inputIndices: ${input.type.indices};
          let outputIndices = ${output.offsetToIndices('global_idx')};

          ${idxCopy.join('\n')}
          ${ops[0]}       // init ops for reduce max/min
          ${initinputOffset}
          ${ops[1]}
          ${reduceOps}
          ${ops[3]}
          ${ops.length === 4 ? output.setByOffset('global_idx', 'value') : ops.slice(4).join('\n')}
        }`;
    return {
        ...metadata,
        getShaderSource,
        outputs: [{ dims: outputShape, dataType: outputDataType, gpuDataType: types_1.GpuDataType.default }],
        dispatchGroup: () => ({ x: Math.ceil(outputSize / 64 /* workgroup size */) })
    };
};
exports.createReduceProgramInfo = createReduceProgramInfo;
const createReduceAttributesFromInputs = (inputs, attributes) => {
    const axes = [];
    if (inputs[1].dims[0] > 0) {
        inputs[1].getBigInt64Array().forEach(v => axes.push(Number(v)));
    }
    return (0, attribute_with_cache_key_1.createAttributeWithCacheKey)({ axes, keepDims: attributes.keepDims, noopWithEmptyAxes: attributes.noopWithEmptyAxes });
};
const createReduceProgramInfoLoader = (inputs, name, attributes, reduceOp) => {
    const updatedAttributes = inputs.length === 1 ? attributes : createReduceAttributesFromInputs(inputs, attributes);
    const metadata = {
        name,
        inputTypes: [types_1.GpuDataType.default],
        cacheHint: updatedAttributes.cacheKey + '_' + inputs[0].dims.map(d => d.toString()).join(',')
    };
    return {
        ...metadata,
        get: () => (0, exports.createReduceProgramInfo)(metadata, [inputs[0]], updatedAttributes.noopWithEmptyAxes && updatedAttributes.axes.length === 0 ? noOp : reduceOp, updatedAttributes.axes, inputs[0].dataType, updatedAttributes.keepDims, updatedAttributes.noopWithEmptyAxes)
    };
};
const reduceLogSum = (context, attributes) => {
    validateInputs(context.inputs);
    const reduceOp = (input, output) => [`var value = ${output.type.storage}(0);`,
        '',
        `value += ${input.getByOffset('inputOffset')};`,
        'value = log(value);',
    ];
    context.compute(createReduceProgramInfoLoader(context.inputs, 'ReduceLogSum', attributes, reduceOp), { inputs: [0] });
};
exports.reduceLogSum = reduceLogSum;
const reduceL1 = (context, attributes) => {
    validateInputs(context.inputs);
    const reduceOp = (input, output) => [`var value = ${output.type.storage}(0);`,
        '',
        `value += abs(${input.getByOffset('inputOffset')});`,
        '',
    ];
    context.compute(createReduceProgramInfoLoader(context.inputs, 'ReduceL1', attributes, reduceOp), { inputs: [0] });
};
exports.reduceL1 = reduceL1;
const reduceL2 = (context, attributes) => {
    validateInputs(context.inputs);
    const reduceOp = (input, output) => [`var t = f32(0); var value = ${output.type.storage}(0);`,
        '',
        `t = ${input.getByOffset('inputOffset')}; value += (t * t);`,
        'value = sqrt(value);',
    ];
    context.compute(createReduceProgramInfoLoader(context.inputs, 'ReduceL2', attributes, reduceOp), { inputs: [0] });
};
exports.reduceL2 = reduceL2;
const reduceLogSumExp = (context, attributes) => {
    validateInputs(context.inputs);
    const reduceOp = (input, output) => [`var value = ${output.type.storage}(0);`,
        '',
        `value += exp(${input.getByOffset('inputOffset')});`,
        'value = log(value);',
    ];
    context.compute(createReduceProgramInfoLoader(context.inputs, 'ReduceLogSumExp', attributes, reduceOp), { inputs: [0] });
};
exports.reduceLogSumExp = reduceLogSumExp;
const reduceMax = (context, attributes) => {
    validateInputs(context.inputs);
    const reduceOp = (input, _output, axes) => {
        const idxZero = [];
        for (let k = 0; k < input.shape.length; k++) {
            if (axes.indexOf(k) >= 0 || axes.length === 0) {
                idxZero.push(input.indicesSet('inputIndices', k, 0));
            }
        }
        return [
            `${idxZero.join('\n')}`,
            `var value = ${input.getByOffset('inputOffset')};`,
            `value = max(value, ${input.getByOffset('inputOffset')});`,
            '',
        ];
    };
    context.compute(createReduceProgramInfoLoader(context.inputs, 'ReduceMax', attributes, reduceOp), { inputs: [0] });
};
exports.reduceMax = reduceMax;
const reduceMean = (context, attributes) => {
    validateInputs(context.inputs);
    const reduceOp = (input, output, axes) => {
        let size = 1.0;
        for (let k = 0; k < input.shape.length; k++) {
            if (axes.indexOf(k) >= 0 || axes.length === 0) {
                size *= input.shape[k];
            }
        }
        return [
            `var value = ${output.type.storage}(0);`,
            '',
            `value += ${input.getByOffset('inputOffset')};`,
            `value = value / ${size}.;`,
        ];
    };
    context.compute(createReduceProgramInfoLoader(context.inputs, 'ReduceMean', attributes, reduceOp), { inputs: [0] });
};
exports.reduceMean = reduceMean;
const reduceMin = (context, attributes) => {
    validateInputs(context.inputs);
    const reduceOp = (input, _output, axes) => {
        const idxZero = [];
        for (let k = 0; k < input.shape.length; k++) {
            if (axes.indexOf(k) >= 0 || axes.length === 0) {
                idxZero.push(`inputIndices[${k}] = 0;`); // first element
            }
        }
        return [
            `${idxZero.join('\n')}`,
            `var value = ${input.getByOffset('inputOffset')};`,
            `value = min(value, ${input.getByOffset('inputOffset')});`,
            '',
        ];
    };
    context.compute(createReduceProgramInfoLoader(context.inputs, 'ReduceMin', attributes, reduceOp), { inputs: [0] });
};
exports.reduceMin = reduceMin;
const reduceProd = (context, attributes) => {
    validateInputs(context.inputs);
    const reduceOp = (input, output) => [`var value = ${output.type.storage}(1);`,
        '',
        `value *= ${input.getByOffset('inputOffset')};`,
        '',
    ];
    context.compute(createReduceProgramInfoLoader(context.inputs, 'ReduceProd', attributes, reduceOp), { inputs: [0] });
};
exports.reduceProd = reduceProd;
const reduceSum = (context, attributes) => {
    validateInputs(context.inputs);
    const reduceOp = (input, output) => [`var value = ${output.type.storage}(0);`,
        '',
        `value += ${input.getByOffset('inputOffset')};`,
        '',
    ];
    context.compute(createReduceProgramInfoLoader(context.inputs, 'ReduceSum', attributes, reduceOp), { inputs: [0] });
};
exports.reduceSum = reduceSum;
const reduceSumSquare = (context, attributes) => {
    validateInputs(context.inputs);
    const reduceOp = (input, output) => [`var t = f32(0); var value = ${output.type.storage}(0);`,
        '',
        `t = ${input.getByOffset('inputOffset')}; value += t * t;`,
        '',
    ];
    context.compute(createReduceProgramInfoLoader(context.inputs, 'ReduceSumSquare', attributes, reduceOp), { inputs: [0] });
};
exports.reduceSumSquare = reduceSumSquare;
const parseReduceAttributes = (attributes) => (0, attribute_with_cache_key_1.createAttributeWithCacheKey)(attributes);
exports.parseReduceAttributes = parseReduceAttributes;
//# sourceMappingURL=reduce.js.map