"use strict";
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
Object.defineProperty(exports, "__esModule", { value: true });
exports.parseSkipLayerNormAttributes = exports.skipLayerNorm = void 0;
const util_1 = require("../../util");
const attribute_with_cache_key_1 = require("../attribute-with-cache-key");
const types_1 = require("../types");
const common_1 = require("./common");
const validateInputs = (inputs) => {
    if (!inputs || inputs.length < 3) {
        throw new Error('layerNorm requires at least 3 inputs.');
    }
    if (inputs[0].dataType !== 1 /* DataType.float */ || inputs[1].dataType !== 1 /* DataType.float */) {
        throw new Error('inputs should be float type');
    }
    const input = inputs[0];
    const skip = inputs[1];
    const gamma = inputs[2];
    if (input.dataType !== skip.dataType || input.dataType !== gamma.dataType) {
        throw new Error('All inputs must have the same data type');
    }
    if (input.dims.length !== 3 && input.dims.length !== 2) {
        throw new Error('Input must be 2D or 3D');
    }
    if (skip.dims.length !== 3 && skip.dims.length !== 2) {
        throw new Error('Skip must be 2D or 3D');
    }
    const hiddenSize = input.dims[input.dims.length - 1];
    const sequenceLength = input.dims[input.dims.length - 2];
    if (skip.dims[skip.dims.length - 1] !== hiddenSize) {
        throw new Error('Skip must have the same hidden size as input');
    }
    if (skip.dims[skip.dims.length - 2] !== sequenceLength) {
        throw new Error('Skip must have the same sequence length as input');
    }
    if (gamma.dims.length !== 1) {
        throw new Error('Gamma must be 1D');
    }
    if (gamma.dims[gamma.dims.length - 1] !== hiddenSize) {
        throw new Error('Gamma must have the same hidden size as input');
    }
    if (inputs.length > 3) {
        const beta = inputs[3];
        if (beta.dims.length !== 1) {
            throw new Error('Beta must be 1D');
        }
        if (beta.dims[beta.dims.length - 1] !== hiddenSize) {
            throw new Error('Beta must have the same hidden size as input');
        }
    }
    if (inputs.length > 4) {
        const bias = inputs[4];
        if (bias.dims.length !== 1) {
            throw new Error('Bias must be 1D');
        }
        if (bias.dims[bias.dims.length - 1] !== hiddenSize) {
            throw new Error('Bias must have the same hidden size as input');
        }
    }
};
const createSkipLayerNormProgramInfo = (metadata, inputs, attributes, outputCount, isTraining) => {
    const inputShape = inputs[0].dims;
    const inputSize = util_1.ShapeUtil.size(inputShape);
    const outputShape = inputShape;
    const outputSize = inputSize;
    const hiddenSize = inputShape.slice(-1)[0];
    const meanInvStdDevDim = isTraining ? inputShape.slice(0, -1).concat(1) : [];
    const hasBetaInput = inputs.length > 3;
    const hasBiasInput = inputs.length > 4;
    const dataType = (0, common_1.tensorTypeToWsglStorageType)(inputs[0].dataType);
    const hasMeanOutput = isTraining && outputCount > 1;
    const hasInvStdDevOutput = isTraining && outputCount > 2;
    const hasInputSkipBiasSumOutput = outputCount > 3;
    let bindingNumber = 0;
    const getShaderSource = (shaderHelper) => `
      const hiddenSize: u32 = ${hiddenSize};
      const epsilon: f32 = ${attributes.epsilon};

      @group(0) @binding(${bindingNumber++}) var<storage, read> x : array<${dataType}>;
      @group(0) @binding(${bindingNumber++}) var<storage, read> skip : array<${dataType}>;
      @group(0) @binding(${bindingNumber++}) var<storage, read> gamma : array<${dataType}>;
      ${hasBetaInput ? `@group(0) @binding(${bindingNumber++}) var<storage, read> beta : array<${dataType}>;` : ''}
      ${hasBiasInput ? `@group(0) @binding(${bindingNumber++}) var<storage, read> bias : array<${dataType}>;` : ''}
      @group(0) @binding(${bindingNumber++}) var<storage, read_write> output : array<${dataType}>;
      ${hasMeanOutput ?
        `@group(0) @binding(${bindingNumber++}) var<storage, read_write> meanOutput : array<${dataType}>;` :
        ''}
      ${hasInvStdDevOutput ?
        `@group(0) @binding(${bindingNumber++}) var<storage, read_write> invStdOutput : array<${dataType}>;` :
        ''}
      ${hasInputSkipBiasSumOutput ?
        `@group(0) @binding(${bindingNumber++}) var<storage, read_write> inputSkipBiasSum : array<${dataType}>;` :
        ''}

      ${shaderHelper.mainStart()}
        ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(outputSize / hiddenSize)}
        let offset = global_idx * hiddenSize;
        var sum: f32 = 0.0;
        var squareSum: f32 = 0.0;
        for (var i: u32 = 0; i < hiddenSize; i++) {
          let skipValue = skip[offset + i];
          let biasValue = ${hasBiasInput ? 'bias[i]' : '0.0'};
          let inputValue = x[offset + i];
          let value = inputValue + skipValue + biasValue;
          ${hasInputSkipBiasSumOutput ? 'inputSkipBiasSum[offset + i] = value;' : ''}
          output[offset + i] = value;
          sum += value;
          squareSum += value * value;
        }
        let mean: f32 = sum / f32(hiddenSize);
        let variance: f32 = sqrt(squareSum / f32(hiddenSize) - mean * mean + epsilon);
        ${hasMeanOutput ? 'meanOutput[global_idx] = mean;' : ''}
        ${hasInvStdDevOutput ? 'invStdOutput[global_idx] = 1.0 / variance;' : ''}
        for (var i: u32 = 0; i < hiddenSize; i++) {
          output[offset + i] = (output[offset + i] - mean) / variance * gamma[i] + ${hasBetaInput ? 'beta[i]' : '0.0'};
        }
      }`;
    const outputs = [{ dims: outputShape, dataType: inputs[0].dataType, gpuDataType: types_1.GpuDataType.default }];
    if (outputCount > 1) {
        outputs.push({ dims: meanInvStdDevDim, dataType: inputs[0].dataType, gpuDataType: types_1.GpuDataType.default });
    }
    if (outputCount > 2) {
        outputs.push({ dims: meanInvStdDevDim, dataType: inputs[0].dataType, gpuDataType: types_1.GpuDataType.default });
    }
    if (outputCount > 3) {
        outputs.push({ dims: inputShape, dataType: inputs[0].dataType, gpuDataType: types_1.GpuDataType.default });
    }
    return {
        ...metadata,
        getShaderSource,
        outputs,
        dispatchGroup: () => ({ x: Math.ceil(outputSize / hiddenSize / 64) })
    };
};
const createSkipLayerNormProgramInfoLoader = (inputs, attributes, outputCount, isTraining) => {
    const inputTypes = new Array(inputs.length).fill(types_1.GpuDataType.default);
    const metadata = {
        name: 'SkipLayerNormalization',
        inputTypes,
        cacheHint: attributes.cacheKey,
    };
    return {
        ...metadata,
        get: () => createSkipLayerNormProgramInfo(metadata, inputs, attributes, outputCount, isTraining)
    };
};
const skipLayerNorm = (context, attributes) => {
    // TODO: initialize isTraining from ComputeContext
    const isTraining = false;
    validateInputs(context.inputs);
    // Mean and InvStdDev are only used in training mode and are not required for inference.
    // They are added here for completeness only.
    const outputs = [0];
    if (context.outputCount > 1) {
        outputs.push(isTraining ? 1 : -3);
    }
    if (context.outputCount > 2) {
        outputs.push(isTraining ? 2 : -3);
    }
    if (context.outputCount > 3) {
        outputs.push(3);
    }
    context.compute(createSkipLayerNormProgramInfoLoader(context.inputs, attributes, context.outputCount, isTraining), { outputs });
};
exports.skipLayerNorm = skipLayerNorm;
const parseSkipLayerNormAttributes = (attributes) => {
    const epsilon = attributes.epsilon;
    return (0, attribute_with_cache_key_1.createAttributeWithCacheKey)({ epsilon });
};
exports.parseSkipLayerNormAttributes = parseSkipLayerNormAttributes;
//# sourceMappingURL=skip-layer-norm.js.map