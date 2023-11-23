"use strict";
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
Object.defineProperty(exports, "__esModule", { value: true });
exports.instanceNorm = exports.parseInstanceNormAttributes = void 0;
const util_1 = require("../../util");
const attribute_with_cache_key_1 = require("../attribute-with-cache-key");
const types_1 = require("../types");
const common_1 = require("./common");
const validateInputs = (inputs) => {
    if (!inputs || inputs.length !== 3) {
        throw new Error('instanceNorm requires 3 inputs.');
    }
    if (inputs[0].dataType !== 1 /* DataType.float */ || inputs[1].dataType !== 1 /* DataType.float */) {
        throw new Error('inputs should be float type');
    }
};
const createInstanceNormProgramInfo = (metadata, inputs, attributes) => {
    const xShape = inputs[0].dims;
    const scale = inputs[1];
    const bias = inputs[2];
    const outputShape = xShape;
    const outputSize = util_1.ShapeUtil.size(outputShape);
    const axis = 2;
    const normCount = util_1.ShapeUtil.sizeToDimension(xShape, axis);
    const normSize = util_1.ShapeUtil.sizeFromDimension(xShape, axis);
    const C = xShape[1];
    const scaleSize = util_1.ShapeUtil.size(scale.dims);
    const biasSize = bias ? util_1.ShapeUtil.size(bias.dims) : 0;
    if (scaleSize !== normSize || (bias && biasSize !== normSize)) {
        throw new Error(`Size of X.shape()[axis:] == ${normSize}.
             Size of scale and bias (if provided) must match this. 
             Got scale size of ${scaleSize} and bias size of ${biasSize}`);
    }
    const dataType = (0, common_1.tensorTypeToWsglStorageType)(inputs[0].dataType);
    const getShaderSource = (shaderHelper) => `
  const C: u32 = ${C};
  const normSize: u32 = ${normSize};
  const normSizeTyped: ${dataType} = ${normSize};
  const epsilon: f32 = ${attributes.epsilon};

  @group(0) @binding(0) var<storage, read> x : array<${dataType}>;
  @group(0) @binding(1) var<storage, read> scale : array<${dataType}>;
  @group(0) @binding(2) var<storage, read> bias : array<${dataType}>;
  @group(0) @binding(3) var<storage, read_write> output : array<${dataType}>;

  ${shaderHelper.mainStart()}
    let offset = global_idx * normSize;
    if (offset + normSize >= ${outputSize}) { return; }
    var mean: ${dataType} = 0;

    for (var h: u32 = 0u; h < normSize; h++) {
        mean = mean + x[h + offset];
    }
    mean = mean / normSizeTyped;

    var squaredNorm: ${dataType} = 0;
    for (var h: u32 = 0u; h < normSize; h++) {
        let deviation: f32 = x[h + offset] - mean;
        squaredNorm = squaredNorm + deviation * deviation;
    }
    let invStdDev = 1 / sqrt(squaredNorm / normSizeTyped + epsilon);
    let channelScale = invStdDev * scale[global_idx % C];
    let channelShift = bias[global_idx % C] - mean * channelScale;
    for (var j: u32 = 0; j < normSize; j++) {
        output[j + offset] = x[j + offset] * channelScale + channelShift;
    }
  }`;
    return {
        ...metadata,
        outputs: [
            { dims: outputShape, dataType: inputs[0].dataType, gpuDataType: types_1.GpuDataType.default },
        ],
        getShaderSource,
        dispatchGroup: () => ({ x: Math.ceil(normCount / 64 /* workgroup size */) })
    };
};
const createInstanceNormNHWCProgramInfo = (metadata, inputs, attributes) => {
    const xShape = inputs[0].dims;
    const outputShape = xShape;
    const outputSize = util_1.ShapeUtil.size(outputShape);
    const N = xShape[0];
    const C = xShape[xShape.length - 1];
    const H = util_1.ShapeUtil.sizeFromDimension(xShape, 1) / C;
    const dataType = (0, common_1.tensorTypeToWsglStorageType)(inputs[0].dataType);
    const normCount = C * N;
    const getShaderSource = (shaderHelper) => `
  const N: u32 = ${N};
  const H: u32 = ${H};
  const C: u32 = ${C};
  const normSizeTyped: ${dataType} = ${H};
  const imageSize: u32 = ${H * C};
  const epsilon: f32 = ${attributes.epsilon};

  @group(0) @binding(0) var<storage, read> x : array<${dataType}>;
  @group(0) @binding(1) var<storage, read> scale : array<${dataType}>;
  @group(0) @binding(2) var<storage, read> bias : array<${dataType}>;
  @group(0) @binding(3) var<storage, read_write> output : array<${dataType}>;

  ${shaderHelper.mainStart()}
    let currentImageNumber = global_idx / C;
    let currentChannelNumber = global_idx % C;
    
    // offset is channel num * N
    let offset = currentImageNumber * imageSize;
    if (offset >= ${outputSize}) { return; }
    var mean: ${dataType} = 0;

    for (var i: u32 = 0u; i < H; i++) {
        mean = mean + x[offset + i * C + currentChannelNumber];
    }
    mean = mean / normSizeTyped;

    var squaredNorm: ${dataType} = 0;
    for (var i: u32 = 0u; i < H; i++) {
        let deviation: f32 = x[offset + i * C + currentChannelNumber] - mean;
        squaredNorm = squaredNorm + deviation * deviation;
    }
    let invStdDev = 1 / sqrt(squaredNorm / normSizeTyped + epsilon);
    let channelScale = invStdDev * scale[currentChannelNumber];
    let channelShift = bias[currentChannelNumber] - mean * channelScale;
    for (var i: u32 = 0u; i < H; i++) {
        let currentOffset = offset + i * C + currentChannelNumber;
        output[currentOffset] = x[currentOffset] * channelScale + channelShift;
    }
  }`;
    return {
        ...metadata,
        outputs: [
            { dims: outputShape, dataType: inputs[0].dataType, gpuDataType: types_1.GpuDataType.default },
        ],
        getShaderSource,
        dispatchGroup: () => ({ x: Math.ceil(normCount / 64 /* workgroup size */) })
    };
};
const parseInstanceNormAttributes = (attributes) => (0, attribute_with_cache_key_1.createAttributeWithCacheKey)({ epsilon: attributes.epsilon, format: attributes.format });
exports.parseInstanceNormAttributes = parseInstanceNormAttributes;
const instanceNorm = (context, attributes) => {
    validateInputs(context.inputs);
    const metadata = {
        name: 'InstanceNormalization',
        inputTypes: [types_1.GpuDataType.default, types_1.GpuDataType.default, types_1.GpuDataType.default],
        cacheHint: attributes.cacheKey,
    };
    if (attributes.format === 'NHWC') {
        context.compute(createInstanceNormNHWCProgramInfo(metadata, context.inputs, attributes));
    }
    else {
        context.compute(createInstanceNormProgramInfo(metadata, context.inputs, attributes));
    }
};
exports.instanceNorm = instanceNorm;
//# sourceMappingURL=instance-norm.js.map