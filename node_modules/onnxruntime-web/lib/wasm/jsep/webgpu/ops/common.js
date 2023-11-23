"use strict";
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
Object.defineProperty(exports, "__esModule", { value: true });
exports.createShaderHelper = exports.outputVariable = exports.inputVariable = exports.tensorTypeToWsglStorageType = exports.WORKGROUP_SIZE = void 0;
const util_1 = require("../../util");
/**
 * constant value for a workgroup size.
 *
 * We definitely can do further optimization in future, but for now we use 64.
 *
 * rule of thumb: Use [a workgroup size of] 64 unless you know what GPU you are targeting or that your workload
 *                needs something different.
 *
 * from: https://surma.dev/things/webgpu/
 **/
exports.WORKGROUP_SIZE = 64;
const getWgslMappedType = (type, components) => {
    // return type is [ storage type, runtime type ] or a single string for both
    switch (type) {
        // TODO: enable after "shader-f16" WSGL extension release
        // case DataType.float16:
        //   return components > 1 ? `vec${components}<f16>` : 'f16';
        case 1 /* DataType.float */:
            return components > 1 ? `vec${components}<f32>` : 'f32';
        case 6 /* DataType.int32 */:
            return components > 1 ? `vec${components}<i32>` : 'i32';
        case 12 /* DataType.uint32 */:
            return components > 1 ? `vec${components}<u32>` : 'u32';
        case 7 /* DataType.int64 */:
            if (components > 1) {
                throw new Error('currently not supported vecX of uint64 yet');
            }
            return ['vec2<u32>', 'i32'];
        case 13 /* DataType.uint64 */:
            if (components > 1) {
                throw new Error('currently not supported vecX of uint64 yet');
            }
            return ['vec2<u32>', 'u32'];
        case 9 /* DataType.bool */:
            if (components !== 4) {
                throw new Error('bool must be vec4');
            }
            return ['u32', 'vec4<bool>'];
        default:
            throw new Error(`Unknown data type: ${type}`);
    }
};
const tensorTypeToWsglStorageType = (type, components = 1) => {
    const mappedType = getWgslMappedType(type, components);
    return typeof mappedType === 'string' ? mappedType : mappedType[0];
};
exports.tensorTypeToWsglStorageType = tensorTypeToWsglStorageType;
/**
 * A helper function to get a IndicesHelper for a given input or output.
 *
 * @param name - the name of the input or output.
 * @param tensorType - the tensor type of the input or output.
 * @param shape - the tensor shape of the input or output.
 * @param isInput - whether the helper is for an input or an output.
 * @param components - indicates the number of components of each element. 1 for scalar, 2 for vec2, 3 for vec3, 4 for
 *    vec4.
 */
const createIndicesHelper = (name, tensorType, shape, isInput, components) => {
    const rank = shape.length;
    const indicesType = rank < 2 ? 'u32' : rank <= 4 ? `vec${rank}<u32>` : `array<u32, ${rank}>`;
    const mappedType = getWgslMappedType(tensorType, components);
    const valueType = typeof mappedType === 'string' ? mappedType : mappedType[1];
    const storageType = typeof mappedType === 'string' ? mappedType : mappedType[0];
    const type = { indices: indicesType, value: valueType, storage: storageType, tensor: tensorType };
    const normalizeDim = (dim) => typeof dim === 'string' ? dim : `${dim}u`;
    const implementationUsed = {
        offsetToIndices: false,
        indicesToOffset: false,
        set: false,
        setByIndices: false,
        get: false,
        getByIndices: false,
    };
    const strides = util_1.ShapeUtil.computeStrides(shape);
    let o2iSnippet = '';
    for (let i = 0; i < rank - 1; i++) {
        o2iSnippet += `
    let dim${i} = current / ${strides[i]}u;
    let rest${i} = current % ${strides[i]}u;
    indices[${i}] = dim${i};
    current = rest${i};
    `;
    }
    o2iSnippet += `indices[${rank - 1}] = current;`;
    const offsetToIndicesImplementation = rank < 2 ? '' : `
  fn o2i_${name}(offset: u32) -> ${type.indices} {
    var indices: ${type.indices};
    var current = offset;
    ${o2iSnippet}
    return indices;
  }`;
    const offsetToIndices = (varOffset) => {
        implementationUsed.offsetToIndices = true;
        return rank < 2 ? varOffset : `o2i_${name}(${varOffset})`;
    };
    const offsets = [];
    if (rank >= 2) {
        for (let i = rank - 1; i >= 0; i--) {
            offsets.push(`${strides[i]}u * (indices[${i}])`);
        }
    }
    const indicesToOffsetImplementation = rank < 2 ? '' : `
  fn i2o_${name}(indices: ${type.indices}) -> u32 {
    return ${offsets.join('+')};
  }`;
    const indicesToOffset = (varIndices) => {
        implementationUsed.indicesToOffset = true;
        return rank < 2 ? varIndices : `i2o_${name}(${varIndices})`;
    };
    const indices = (...init) => rank === 0 ? '0u' : `${type.indices}(${init.map(normalizeDim).join(',')})`;
    const indicesGet = (varIndices, idx) => {
        if (rank < 2) {
            return `${varIndices}`;
        }
        else {
            return `${varIndices}[${idx}]`;
        }
    };
    const indicesSet = (varIndices, idx, value) => {
        if (rank < 2) {
            return `${varIndices}=${value};`;
        }
        else {
            return `${varIndices}[${idx}]=${value};`;
        }
    };
    const setByOffset = (offset, value) => (() => {
        if (type.storage === type.value) {
            return `${name}[${offset}]=${value};`;
        }
        else if (type.storage === 'vec2<u32>' && type.value === 'i32') {
            // int64, components === 1
            return `${name}[${offset}]=vec2<u32>(u32(${value}), select(0u, 0xFFFFFFFFu, ${value} < 0));`;
        }
        else if (type.storage === 'vec2<u32>' && type.value === 'u32') {
            // uint64, components === 1
            return `${name}[${offset}]=vec2<u32>(u32(${value}), 0u);`;
        }
        else if (type.storage === 'u32' && type.value === 'vec4<bool>') {
            // bool, components === 4
            return `${name}[${offset}]=dot(vec4<u32>(0x1, 0x100, 0x10000, 0x1000000), vec4<u32>(${value}));`;
        }
        else {
            throw new Error(`not supported combination of storage type ${type.storage} and value type ${type.value} yet`);
        }
    })();
    const getByOffset = (offset) => (() => {
        if (type.storage === type.value) {
            return `${name}[${offset}]`;
        }
        else if (type.storage === 'vec2<u32>' && type.value === 'i32') {
            // int64, components === 1
            return `i32(${name}[${offset}].x)`;
        }
        else if (type.storage === 'vec2<u32>' && type.value === 'u32') {
            // uint64, components === 1
            return `u32(${name}[${offset}].x)`;
        }
        else if (type.storage === 'u32' && type.value === 'vec4<bool>') {
            // bool, components === 4
            return `vec4<bool>(bool(${name}[${offset}] & 0xFFu), bool(${name}[${offset}] & 0xFF00u), bool(${name}[${offset}] & 0xFF0000u), bool(${name}[${offset}] & 0xFF000000u))`;
        }
        else {
            throw new Error(`not supported combination of storage type ${type.storage} and value type ${type.value} yet`);
        }
    })();
    const getByIndicesImplementation = rank < 2 ? '' : `
  fn get_${name}ByIndices(indices: ${type.indices}) -> ${valueType} {
    return ${name}[i2o_${name}(indices)];
  }`;
    const getImplementation = rank < 2 ? '' : (() => {
        const params = shape.map((_, i) => `d${i}: u32`).join(', ');
        const dims = shape.map((_, i) => `d${i}`).join(', ');
        return `
  fn get_${name}(${params}) -> ${valueType} {
    return get_${name}ByIndices(${indices(dims)});
  }`;
    })();
    const get = (...indices) => {
        if (indices.length !== rank) {
            throw new Error(`indices length must be ${rank}`);
        }
        const normalizedIndices = indices.map(normalizeDim).join(',');
        if (rank === 0) {
            return getByOffset('0u');
        }
        else if (rank === 1) {
            return getByOffset(normalizedIndices[0]);
        }
        else {
            implementationUsed.get = true;
            implementationUsed.getByIndices = true;
            implementationUsed.indicesToOffset = true;
            return `get_${name}(${normalizedIndices})`;
        }
    };
    const getByIndices = (varIndices) => {
        if (rank < 2) {
            return getByOffset(varIndices);
        }
        else {
            implementationUsed.getByIndices = true;
            implementationUsed.indicesToOffset = true;
            return `get_${name}ByIndices(${varIndices})`;
        }
    };
    const setByIndicesImplementation = rank < 2 ? '' : `
  fn set_${name}ByIndices(indices: ${type.indices}, value: ${valueType}) {
    ${setByOffset(`i2o_${name}(indices)`, 'value')}
  }`;
    const setImplementation = rank < 2 ? '' : (() => {
        const params = shape.map((_, i) => `d${i}: u32`).join(', ');
        const dims = shape.map((_, i) => `d${i}`).join(', ');
        return `
  fn set_${name}(${params}, value: ${valueType}) {
    set_${name}ByIndices(${indices(dims)}, value);
  }`;
    })();
    const set = (...indicesAndValue) => {
        if (indicesAndValue.length !== rank + 1) {
            throw new Error(`indices length must be ${rank}`);
        }
        const value = indicesAndValue[rank];
        if (typeof value !== 'string') {
            throw new Error('value must be string');
        }
        const normalizedIndices = indicesAndValue.slice(0, rank).map(normalizeDim).join(',');
        if (rank === 0) {
            return setByOffset('0u', value);
        }
        else if (rank === 1) {
            return setByOffset(normalizedIndices[0], value);
        }
        else {
            implementationUsed.set = true;
            implementationUsed.setByIndices = true;
            implementationUsed.indicesToOffset = true;
            return `set_${name}(${normalizedIndices}, ${value})`;
        }
    };
    const setByIndices = (varIndices, value) => {
        if (rank < 2) {
            return setByOffset(varIndices, value);
        }
        else {
            implementationUsed.setByIndices = true;
            implementationUsed.indicesToOffset = true;
            return `set_${name}ByIndices(${varIndices}, ${value});`;
        }
    };
    const impl = () => {
        const impls = [];
        if (implementationUsed.offsetToIndices) {
            impls.push(offsetToIndicesImplementation);
        }
        if (implementationUsed.indicesToOffset) {
            impls.push(indicesToOffsetImplementation);
        }
        if (implementationUsed.set) {
            impls.push(setImplementation);
        }
        if (implementationUsed.setByIndices) {
            impls.push(setByIndicesImplementation);
        }
        if (implementationUsed.get) {
            impls.push(getImplementation);
        }
        if (implementationUsed.getByIndices) {
            impls.push(getByIndicesImplementation);
        }
        return impls.join('\n');
    };
    return {
        impl,
        type,
        offsetToIndices,
        indicesToOffset,
        indices,
        indicesGet,
        indicesSet,
        set,
        setByOffset,
        setByIndices,
        get,
        getByOffset,
        getByIndices,
        // isVec4,
        usage: isInput ? 'input' : 'output',
        name,
        shape
    };
};
/**
 * Create a IndicesHelper for an input.
 *
 * @param name - the name of the input.
 * @param type - the tensor type of the input.
 * @param shape - the tensor shape of the input.
 * @param components - the number of components of the input. available values are 1, 2, 3, 4. default is 1.
 * @returns an IndicesHelper for the input.
 */
const inputVariable = (name, type, shape, components = 1) => createIndicesHelper(name, type, shape, true, components);
exports.inputVariable = inputVariable;
/**
 * Create a IndicesHelper for an output.
 *
 * @param name - the name of the output.
 * @param type - the tensor type of the output.
 * @param shape - the tensor shape of the output.
 * @param components - the number of components of the input. available values are 1, 2, 3, 4. default is 1.
 * @returns an IndicesHelper for the output.
 */
const outputVariable = (name, type, shape, components = 1) => createIndicesHelper(name, type, shape, false, components);
exports.outputVariable = outputVariable;
class ShaderHelperImpl {
    constructor(normalizedDispatchGroup) {
        this.normalizedDispatchGroup = normalizedDispatchGroup;
        this.indicesHelpers = [];
    }
    guardAgainstOutOfBoundsWorkgroupSizes(size) {
        // Guard against out-of-bounds work group sizes
        const sizeInCode = typeof size === 'number' ? `${size}u` : size;
        return `if (global_idx >= ${sizeInCode}) { return; }`;
    }
    mainStart(workgroupSize = exports.WORKGROUP_SIZE) {
        const workgroupSizeX = typeof workgroupSize === 'number' ? workgroupSize : workgroupSize[0];
        const workgroupSizeY = typeof workgroupSize === 'number' ? 1 : workgroupSize[1];
        const workgroupSizeZ = typeof workgroupSize === 'number' ? 1 : workgroupSize[2];
        const is1DimensionDispatch = this.normalizedDispatchGroup[1] === 1 && this.normalizedDispatchGroup[2] === 1;
        const paramList = is1DimensionDispatch ? '@builtin(global_invocation_id) global_id : vec3<u32>' :
            `@builtin(local_invocation_index) local_index : u32,
    @builtin(workgroup_id) workgroup_id : vec3<u32>`;
        const globalIdxDefinition = is1DimensionDispatch ?
            'let global_idx = global_id.x;' :
            `let global_idx = (workgroup_id.z * ${this.normalizedDispatchGroup[0] * this.normalizedDispatchGroup[1]}u +
          workgroup_id.y * ${this.normalizedDispatchGroup[0]}u + workgroup_id.x) * ${workgroupSizeX * workgroupSizeY * workgroupSizeZ}u + local_index;`;
        return `@compute @workgroup_size(${workgroupSizeX}, ${workgroupSizeY}, ${workgroupSizeZ})
  fn main(${paramList}) {
    ${globalIdxDefinition}
  `;
    }
    declareVariable(variable, bindingIndex) {
        this.indicesHelpers.push(variable);
        const access = variable.usage === 'input' ? 'read' : 'read_write';
        const storageType = variable.type.storage;
        return `@group(0) @binding(${bindingIndex}) var<storage, ${access}> ${variable.name}: array<${storageType}>;`;
    }
    declareVariables(...variables) {
        let i = 0;
        return variables.filter(v => util_1.ShapeUtil.size(v.shape) > 0).map(v => this.declareVariable(v, i++)).join('\n');
    }
    get additionalImplementations() {
        return this.indicesHelpers.map(i => i.impl()).join('\n');
    }
}
const createShaderHelper = (dispatchGroup) => new ShaderHelperImpl(dispatchGroup);
exports.createShaderHelper = createShaderHelper;
//# sourceMappingURL=common.js.map