"use strict";
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
Object.defineProperty(exports, "__esModule", { value: true });
exports.sub = exports.pow = exports.mul = exports.div = exports.add = void 0;
const util_1 = require("../../util");
const types_1 = require("../types");
const common_1 = require("./common");
const createBinaryOpProgramShader = (shaderHelper, dimsA, dimsB, dimsOutput, vectorize, doBroadcast, funcCall, typeA, typeB, typeOutput, additionalImplementation) => {
    const outputSize = util_1.ShapeUtil.size(dimsOutput);
    const vecSize = Math.ceil(outputSize / 4);
    let expressionScalar;
    let expressionVector;
    if (typeof funcCall === 'string') {
        expressionScalar = expressionVector = (a, b) => `${funcCall}((${a}),(${b}))`;
    }
    else if (typeof funcCall === 'function') {
        expressionScalar = expressionVector = funcCall;
    }
    else {
        expressionScalar = funcCall.scalar;
        expressionVector = funcCall.vector;
    }
    let broadcastImpl = '';
    const output = (0, common_1.outputVariable)('outputData', typeOutput, dimsOutput, 4);
    const a = (0, common_1.inputVariable)('aData', typeA, dimsA, 4);
    const b = (0, common_1.inputVariable)('bData', typeB, dimsB, 4);
    if (doBroadcast) {
        const calcOffsetImpl = (dims) => {
            const strides = util_1.ShapeUtil.computeStrides(dims);
            const offsets = [];
            for (let i = dims.length - 1; i >= 0; i--) {
                const idx = dimsOutput.length === 0 ? '0u' :
                    (dimsOutput.length === 1) ? 'outputIndices' :
                        `outputIndices[${i + dimsOutput.length - dims.length}]`;
                offsets.push(`${strides[i]}u * (${idx} % ${dims[i]}u)`);
            }
            return offsets.length > 0 ? offsets.join('+') : '0u';
        };
        broadcastImpl = `
  fn calcOffsetA(outputIndices: ${output.type.indices}) -> u32 {
    return ${calcOffsetImpl(dimsA)};
  }

  fn calcOffsetB(outputIndices: ${output.type.indices}) -> u32 {
    return ${calcOffsetImpl(dimsB)};
  }
  `;
    }
    let assignment;
    if (vectorize) {
        if (doBroadcast) {
            assignment = `
      let outputIndices = ${output.offsetToIndices('global_idx * 4u')};
      let offsetA = calcOffsetA(outputIndices);
      let offsetB = calcOffsetB(outputIndices);
      ${output.setByOffset('global_idx', expressionVector(a.getByOffset('offsetA / 4u'), b.getByOffset('offsetB / 4u')))}`;
        }
        else {
            assignment = output.setByOffset('global_idx', expressionVector(a.getByOffset('global_idx'), b.getByOffset('global_idx')));
        }
    }
    else {
        if (!doBroadcast) {
            throw new Error('no necessary to use scalar implementation for element-wise binary op implementation.');
        }
        const singleAssignment = (x) => {
            const expressionA = `aData[indexA${x}][componentA${x}]`;
            const expressionB = `bData[indexB${x}][componentB${x}]`;
            return `
      let outputIndices${x} = ${output.offsetToIndices(`global_idx * 4u + ${x}u`)};
      let offsetA${x} = calcOffsetA(outputIndices${x});
      let offsetB${x} = calcOffsetB(outputIndices${x});
      let indexA${x} = offsetA${x} / 4u;
      let indexB${x} = offsetB${x} / 4u;
      let componentA${x} = offsetA${x} % 4u;
      let componentB${x} = offsetB${x} % 4u;
      outputData[global_idx][${x}] = ${expressionScalar(expressionA, expressionB)};`;
        };
        assignment = `
      ${singleAssignment(0)}
      ${singleAssignment(1)}
      ${singleAssignment(2)}
      ${singleAssignment(3)}`;
    }
    return `
  ${shaderHelper.declareVariables(a, b, output)}

  ${additionalImplementation ?? ''}
  ${broadcastImpl}

  ${shaderHelper.mainStart()}
    ${shaderHelper.guardAgainstOutOfBoundsWorkgroupSizes(vecSize)}
    ${assignment}
  }`;
};
const createBinaryOpProgramInfo = (metadata, a, b, funcCall, additionalImplementation, outputDataType = a.dataType) => {
    const isBroadcast = !util_1.ShapeUtil.areEqual(a.dims, b.dims);
    let outputShape = a.dims;
    let outputSize = util_1.ShapeUtil.size(a.dims);
    let vectorize = false;
    // TODO: deal with zero-sized tensors (eg. dims=[1,0])
    if (isBroadcast) {
        const calculatedShape = util_1.BroadcastUtil.calcShape(a.dims, b.dims, false);
        if (!calculatedShape) {
            throw new Error('Can\'t perform binary op on the given tensors');
        }
        outputShape = calculatedShape;
        outputSize = util_1.ShapeUtil.size(outputShape);
        // check whether vectorize can be enabled
        let sharedDimension = 1;
        for (let i = 0; i < outputShape.length; i++) {
            const dimA = a.dims[a.dims.length - i] ?? 1;
            const dimB = b.dims[b.dims.length - i] ?? 1;
            if (dimA === dimB) {
                sharedDimension *= dimA;
            }
            else {
                break;
            }
        }
        if (sharedDimension % 4 === 0) {
            vectorize = true;
        }
    }
    else {
        // element-wise
        vectorize = true;
    }
    return {
        ...metadata,
        getShaderSource: (shaderHelper) => createBinaryOpProgramShader(shaderHelper, a.dims, b.dims, outputShape, vectorize, isBroadcast, funcCall, a.dataType, b.dataType, outputDataType, additionalImplementation),
        outputs: [{ dims: outputShape, dataType: outputDataType, gpuDataType: types_1.GpuDataType.default }],
        dispatchGroup: () => ({ x: Math.ceil(outputSize / 64 /* workgroup size */ / (vectorize ? 4 : 1) /* vec size */) })
    };
};
const createBinaryOpProgramInfoLoader = (inputs, name, funcCall, additionalImplementation, cacheKey) => {
    const metadata = { name, inputTypes: [types_1.GpuDataType.default, types_1.GpuDataType.default], cacheHint: cacheKey };
    return {
        ...metadata,
        get: () => createBinaryOpProgramInfo(metadata, inputs[0], inputs[1], funcCall, additionalImplementation)
    };
};
const add = (context) => {
    context.compute(createBinaryOpProgramInfoLoader(context.inputs, 'Add', (a, b) => `${a}+${b}`));
};
exports.add = add;
const div = (context) => {
    context.compute(createBinaryOpProgramInfoLoader(context.inputs, 'Div', (a, b) => `${a}/${b}`));
};
exports.div = div;
const mul = (context) => {
    context.compute(createBinaryOpProgramInfoLoader(context.inputs, 'Mul', (a, b) => `${a}*${b}`));
};
exports.mul = mul;
const pow = (context) => {
    const type = (0, common_1.inputVariable)('input', context.inputs[0].dataType, context.inputs[0].dims).type.value;
    const roundStr = type === 'i32' ? 'round' : '';
    context.compute(createBinaryOpProgramInfoLoader(context.inputs, 'Pow', ({ scalar: (a, b) => `pow_custom(${a},${b})`, vector: (a, b) => `pow_vector_custom(${a},${b})` }), `
    fn pow_custom(a : ${type}, b : ${type}) -> ${type} {
      if (b == ${type}(0.0)) {
        return ${type}(1.0);
      } else if (a < ${type}(0.0) && f32(b) != floor(f32(b))) {
        return ${type}(pow(f32(a), f32(b))); // NaN
      }
      return select(sign(a), ${type}(1.0), round(f32(abs(b) % ${type}(2.0))) != 1.0) * ${type}(${roundStr}(pow(f32(abs(a)), f32(b))));
    }
    fn pow_vector_custom(a : vec4<${type}>, b : vec4<${type}>) -> vec4<${type}> {
      // TODO: implement vectorized pow
      return vec4<${type}>(pow_custom(a.x, b.x), pow_custom(a.y, b.y), pow_custom(a.z, b.z), pow_custom(a.w, b.w));
    }
      `));
};
exports.pow = pow;
const sub = (context) => {
    context.compute(createBinaryOpProgramInfoLoader(context.inputs, 'Sub', (a, b) => `${a}-${b}`));
};
exports.sub = sub;
//# sourceMappingURL=binary-op.js.map