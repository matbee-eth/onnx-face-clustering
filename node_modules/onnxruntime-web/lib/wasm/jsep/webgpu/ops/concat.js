"use strict";
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
Object.defineProperty(exports, "__esModule", { value: true });
exports.parseConcatAttributes = exports.concat = void 0;
const util_1 = require("../../util");
const attribute_with_cache_key_1 = require("../attribute-with-cache-key");
const types_1 = require("../types");
const common_1 = require("./common");
const validateInputs = (inputs) => {
    if (!inputs || inputs.length < 1) {
        throw new Error('too few inputs');
    }
    const inputType = inputs[0].dataType;
    const inputDimensionality = inputs[0].dims.length;
    for (const input of inputs) {
        // make sure types of all inputs match
        if (input.dataType !== inputType) {
            throw new Error('input tensors should be one type');
        }
        // make sure the dimensionality of all inputs are the same
        if (input.dims.length !== inputDimensionality) {
            throw new Error('input tensors should have the same shape');
        }
    }
};
const createConcatProgramMetadata = (inputCount, cacheHint) => ({ name: 'Concat', inputTypes: Array(inputCount).fill(types_1.GpuDataType.default), cacheHint });
const calculateInputIndexImpl = (numberOfTensors) => `
  fn calculateInputIndex(index: u32) -> u32 {
    for (var i: u32 = 0u; i < ${numberOfTensors}u; i += 1u ) {
      if (index < sizeInConcatAxis[i]) {
        return i;
      }
    }
    return ${numberOfTensors}u;
  }`;
const assignOutputData = (inputs, output) => {
    const numberOfTensors = inputs.length;
    const codeLines = [];
    for (let i = 0; i < numberOfTensors; ++i) {
        const returnSnippet = output.setByOffset('global_idx', inputs[i].getByIndices('indices'));
        if (numberOfTensors === 1) {
            codeLines.push(returnSnippet);
        }
        else if (i === 0) {
            codeLines.push(`if (inputIndex == ${i}u) { ${returnSnippet} }`);
        }
        else if (i === numberOfTensors - 1) {
            codeLines.push(`else { ${returnSnippet} }`);
        }
        else {
            codeLines.push(`else if (inputIndex == ${i}) { ${returnSnippet} }`);
        }
    }
    return codeLines.join('\n');
};
const createConcatProgramInfo = (metadata, inputs, axis) => {
    const inputShape = inputs[0].dims.slice();
    if (axis >= inputShape.length || axis < (-1 * inputShape.length)) {
        throw new Error('axis specified for concat doesn\'t match input dimensionality');
    }
    const adjustedAxis = (axis < 0) ? inputShape.length + axis : axis;
    // ensure all of the non-concatenated axes match each other
    // calculate the shape of the output tensor while we do that
    const outputShape = inputShape.slice(0);
    for (let i = 1; i < inputs.length; i++) {
        const dataNShape = inputs[i].dims.slice();
        for (let axisIndex = 0; axisIndex < inputShape.length; axisIndex++) {
            // add to the placeholder for computing output shape
            if (axisIndex === adjustedAxis) {
                outputShape[adjustedAxis] += dataNShape[axisIndex];
            }
            // ensure all non-cancatenated axes match each other
            else if (inputShape[axisIndex] !== dataNShape[axisIndex]) {
                throw new Error('non concat dimensions must match');
            }
        }
    }
    const outputSize = util_1.ShapeUtil.size(outputShape);
    const sizeInConcatAxis = new Array(inputs.length);
    const inputVars = new Array(inputs.length);
    const dataType = inputs[0].dataType;
    let previousSum = 0;
    for (let i = 0; i < inputs.length; ++i) {
        previousSum += inputs[i].dims[adjustedAxis];
        sizeInConcatAxis[i] = previousSum;
        inputVars[i] = (0, common_1.inputVariable)(`input${i}`, dataType, inputs[i].dims);
    }
    const output = (0, common_1.outputVariable)('output', dataType, outputShape);
    const indicesAxis = output.indicesGet('indices', adjustedAxis);
    const getShaderSource = (shaderHelper) => `
  ${shaderHelper.declareVariables(...inputVars, output)}

  const sizeInConcatAxis = array<u32, ${sizeInConcatAxis.length}>(${sizeInConcatAxis.map(i => `${i}u`).join(',')});
  ${calculateInputIndexImpl(sizeInConcatAxis.length)}

  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize)}

    var indices = ${output.offsetToIndices('global_idx')};

    let inputIndex = calculateInputIndex(${indicesAxis});
    if (inputIndex != 0u) {
      ${indicesAxis} -= sizeInConcatAxis[inputIndex - 1u];
    }

    ${assignOutputData(inputVars, output)}
  }`;
    return {
        ...metadata,
        outputs: [{ dims: outputShape, dataType: inputs[0].dataType, gpuDataType: types_1.GpuDataType.default }],
        getShaderSource,
        dispatchGroup: () => ({ x: Math.ceil(outputSize / 64 /* workgroup size */) })
    };
};
const createConcatProgramInfoLoader = (inputs, attributes) => {
    const metadata = createConcatProgramMetadata(inputs.length, attributes.cacheKey);
    return { ...metadata, get: () => createConcatProgramInfo(metadata, inputs, attributes.axis) };
};
const concat = (context, attributes) => {
    validateInputs(context.inputs);
    context.compute(createConcatProgramInfoLoader(context.inputs, attributes));
};
exports.concat = concat;
const parseConcatAttributes = (attributes) => (0, attribute_with_cache_key_1.createAttributeWithCacheKey)({ axis: attributes.axis });
exports.parseConcatAttributes = parseConcatAttributes;
//# sourceMappingURL=concat.js.map