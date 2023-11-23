"use strict";
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
Object.defineProperty(exports, "__esModule", { value: true });
exports.layerNorm = exports.parseLayerNormAttributes = void 0;
const util_1 = require("../../util");
const attribute_with_cache_key_1 = require("../attribute-with-cache-key");
const types_1 = require("../types");
const common_1 = require("./common");
const validateInputs = (inputs) => {
    if (!inputs || inputs.length < 2) {
        throw new Error('layerNorm requires at least 2 inputs.');
    }
    if (inputs[0].dataType !== 1 /* DataType.float */ || inputs[1].dataType !== 1 /* DataType.float */) {
        throw new Error('inputs should be float type');
    }
};
const createLayerNormProgramInfo = (metadata, inputs, attributes, outputCount) => {
    const xShape = inputs[0].dims;
    const scale = inputs[1];
    const bias = inputs[2];
    const outputShape = xShape;
    const outputSize = util_1.ShapeUtil.size(outputShape);
    const axis = util_1.ShapeUtil.normalizeAxis(attributes.axis, xShape.length);
    const normCount = util_1.ShapeUtil.sizeToDimension(xShape, axis);
    const normSize = util_1.ShapeUtil.sizeFromDimension(xShape, axis);
    const scaleSize = util_1.ShapeUtil.size(scale.dims);
    const biasSize = bias ? util_1.ShapeUtil.size(bias.dims) : 0;
    if (scaleSize !== normSize || (bias && biasSize !== normSize)) {
        throw new Error(`Size of X.shape()[axis:] == ${normSize}.
       Size of scale and bias (if provided) must match this.
       Got scale size of ${scaleSize} and bias size of ${biasSize}`);
    }
    const meanInvStdDevDim = [];
    for (let i = 0; i < xShape.length; ++i) {
        if (i < axis) {
            meanInvStdDevDim.push(xShape[i]);
        }
        else {
            meanInvStdDevDim.push(1);
        }
    }
    const dataType = (0, common_1.tensorTypeToWsglStorageType)(inputs[0].dataType);
    const hasMeanDataOutput = outputCount > 1;
    const hasInvStdOutput = outputCount > 2;
    let bindingIndex = 0;
    const getShaderSource = (shaderHelper) => `
  const normSize: u32 = ${normSize};
  const normSizeTyped: ${dataType} = ${normSize};
  const epsilon: f32 = ${attributes.epsilon};

  @group(0) @binding(${bindingIndex++}) var<storage, read> x : array<${dataType}>;
  @group(0) @binding(${bindingIndex++}) var<storage, read> scale : array<${dataType}>;
  ${bias ? `@group(0) @binding(${bindingIndex++}) var<storage, read> bias : array<${dataType}>;` : ''}
  @group(0) @binding(${bindingIndex++}) var<storage, read_write> output : array<${dataType}>;
  ${hasMeanDataOutput ?
        `@group(0) @binding(${bindingIndex++}) var<storage, read_write> meanDataOutput : array<${dataType}>` :
        ''};
  ${hasInvStdOutput ?
        `@group(0) @binding(${bindingIndex++}) var<storage, read_write> invStdOutput : array<${dataType}>` :
        ''};

  ${shaderHelper.mainStart()}
    let offset = global_idx * normSize;
    if (offset >= ${outputSize}) { return; }
    var mean: ${dataType} = 0;
    var meanSquare: ${dataType} = 0;

    for (var h: u32 = 0u; h < normSize; h++) {
      mean = mean + x[h + offset];
      meanSquare = meanSquare + x[h + offset] * x[h + offset];
    }
    mean = mean / normSizeTyped;
    meanSquare = sqrt(meanSquare / normSizeTyped - mean * mean + epsilon);

    for (var j: u32 = 0; j < normSize; j++) {
      output[j + offset] = (x[j + offset] - mean) / meanSquare * scale[j] ${bias ? '+ bias[j]' : ''};
    }

    ${hasMeanDataOutput ? 'meanDataOutput[global_idx] = mean' : ''};
    ${hasInvStdOutput ? 'invStdOutput[global_idx] = 1 / meanSquare' : ''};
  }`;
    const outputs = [{ dims: outputShape, dataType: inputs[0].dataType, gpuDataType: types_1.GpuDataType.default }];
    if (hasMeanDataOutput) {
        outputs.push({ dims: meanInvStdDevDim, dataType: inputs[0].dataType, gpuDataType: types_1.GpuDataType.default });
    }
    if (hasInvStdOutput) {
        outputs.push({ dims: meanInvStdDevDim, dataType: inputs[0].dataType, gpuDataType: types_1.GpuDataType.default });
    }
    return {
        ...metadata,
        outputs,
        getShaderSource,
        dispatchGroup: () => ({ x: Math.ceil(normCount / 64 /* workgroup size */) })
    };
};
const parseLayerNormAttributes = (attributes) => (0, attribute_with_cache_key_1.createAttributeWithCacheKey)({ axis: attributes.axis, epsilon: attributes.epsilon });
exports.parseLayerNormAttributes = parseLayerNormAttributes;
const layerNorm = (context, attributes) => {
    validateInputs(context.inputs);
    const metadata = {
        name: 'LayerNormalization',
        inputTypes: context.inputs.length === 2 ? [types_1.GpuDataType.default, types_1.GpuDataType.default] :
            [types_1.GpuDataType.default, types_1.GpuDataType.default, types_1.GpuDataType.default],
        cacheHint: attributes.cacheKey + context.outputCount.toString(10) + context.inputs.length.toString(10),
    };
    context.compute(createLayerNormProgramInfo(metadata, context.inputs, attributes, context.outputCount));
};
exports.layerNorm = layerNorm;
//# sourceMappingURL=layer-norm.js.map