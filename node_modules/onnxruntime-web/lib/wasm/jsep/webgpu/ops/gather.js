"use strict";
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
Object.defineProperty(exports, "__esModule", { value: true });
exports.gather = exports.parseGatherAttributes = void 0;
const util_1 = require("../../util");
const attribute_with_cache_key_1 = require("../attribute-with-cache-key");
const types_1 = require("../types");
const validateInputs = (inputs) => {
    if (!inputs || inputs.length !== 2) {
        throw new Error('Gather requires 2 inputs.');
    }
};
const createGatherProgramInfo = (metadata, inputs, attributes) => {
    const inputShape = inputs[0].dims;
    const indicesShape = inputs[1].dims;
    const inputRank = inputShape.length;
    const axis = util_1.ShapeUtil.normalizeAxis(attributes.axis, inputRank);
    const outputShape = inputShape.slice(0);
    outputShape.splice(axis, 1, ...indicesShape);
    const inputDataType = inputs[0].dataType;
    const block = util_1.ShapeUtil.sizeFromDimension(inputShape, axis + 1);
    const elementSize = [7 /* DataType.int64 */, 13 /* DataType.uint64 */, 11 /* DataType.double */].includes(inputDataType) ? 2 : 1;
    const indicesElementSize = inputs[1].dataType === 7 /* DataType.int64 */ ? 2 : 1;
    const blockSize = elementSize * block;
    const M = util_1.ShapeUtil.sizeToDimension(inputShape, axis);
    const N = util_1.ShapeUtil.size(indicesShape);
    const dataBatchElements = util_1.ShapeUtil.sizeFromDimension(inputShape, axis) * elementSize;
    const gatheredBatchElements = N * block * elementSize;
    const axisDimLimit = inputShape[axis];
    const inputSize = util_1.ShapeUtil.size(inputShape) * elementSize;
    const outputSize = util_1.ShapeUtil.size(outputShape) * elementSize;
    const totalGathers = M * N;
    // int64 indices would be treated as little endian i32 with assumption they fall in i32 limits
    // That assumption is safe as it's not possible to allocate >2gb buffer for input tensor
    // Input data will be treated as u32 or two u32 for 8-byte tensors
    const getShaderSource = (shaderHelper) => `
  const N: u32 = ${N};
  const elementSize: u32 = ${elementSize};
  const indicesElementSize: u32 = ${indicesElementSize};

  @group(0) @binding(0) var<storage, read> input : array<u32>;
  @group(0) @binding(1) var<storage, read> inputIndices : array<i32>;
  @group(0) @binding(2) var<storage, read_write> output: array<u32>;

  ${shaderHelper.mainStart()}
    let batch: u32 = global_idx / N;
    let i: u32 = global_idx % N;

    let srcOffsetBatch: u32 = batch * ${dataBatchElements};
    let dstOffsetBatch: u32 = batch * ${gatheredBatchElements};
    var idx = inputIndices[i * indicesElementSize];
    if (idx < 0) {
        idx = idx + ${axisDimLimit};
    }

    let srcOffset = srcOffsetBatch + u32(idx) * ${blockSize};
    let dstOffset = dstOffsetBatch + i * ${blockSize};
    if (srcOffset >= ${inputSize}) {
        return;
    }
    if (dstOffset >= ${outputSize}) {
        return;
    }
    for (var j: u32 = 0; j < ${blockSize}; j++) {
        output[dstOffset + j] = input[srcOffset + j];
    }
  }`;
    return {
        ...metadata,
        outputs: [
            { dims: outputShape, dataType: inputs[0].dataType, gpuDataType: types_1.GpuDataType.default },
        ],
        getShaderSource,
        dispatchGroup: () => ({ x: Math.ceil(totalGathers / 64 /* workgroup size */) })
    };
};
const parseGatherAttributes = (attributes) => (0, attribute_with_cache_key_1.createAttributeWithCacheKey)({ axis: attributes.axis });
exports.parseGatherAttributes = parseGatherAttributes;
const gather = (context, attributes) => {
    const inputs = context.inputs;
    validateInputs(inputs);
    const metadata = {
        name: 'Gather',
        inputTypes: [types_1.GpuDataType.default, types_1.GpuDataType.default],
        cacheHint: attributes.cacheKey,
    };
    context.compute(createGatherProgramInfo(metadata, context.inputs, attributes));
};
exports.gather = gather;
//# sourceMappingURL=gather.js.map