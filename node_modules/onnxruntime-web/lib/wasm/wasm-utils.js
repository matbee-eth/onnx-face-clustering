"use strict";
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
Object.defineProperty(exports, "__esModule", { value: true });
exports.checkLastError = exports.iterateExtraOptions = exports.allocWasmString = void 0;
const wasm_factory_1 = require("./wasm-factory");
const allocWasmString = (data, allocs) => {
    const wasm = (0, wasm_factory_1.getInstance)();
    const dataLength = wasm.lengthBytesUTF8(data) + 1;
    const dataOffset = wasm._malloc(dataLength);
    wasm.stringToUTF8(data, dataOffset, dataLength);
    allocs.push(dataOffset);
    return dataOffset;
};
exports.allocWasmString = allocWasmString;
const iterateExtraOptions = (options, prefix, seen, handler) => {
    if (typeof options == 'object' && options !== null) {
        if (seen.has(options)) {
            throw new Error('Circular reference in options');
        }
        else {
            seen.add(options);
        }
    }
    Object.entries(options).forEach(([key, value]) => {
        const name = (prefix) ? prefix + key : key;
        if (typeof value === 'object') {
            (0, exports.iterateExtraOptions)(value, name + '.', seen, handler);
        }
        else if (typeof value === 'string' || typeof value === 'number') {
            handler(name, value.toString());
        }
        else if (typeof value === 'boolean') {
            handler(name, (value) ? '1' : '0');
        }
        else {
            throw new Error(`Can't handle extra config type: ${typeof value}`);
        }
    });
};
exports.iterateExtraOptions = iterateExtraOptions;
/**
 * check web assembly API's last error and throw error if any error occurred.
 * @param message a message used when an error occurred.
 */
const checkLastError = (message) => {
    const wasm = (0, wasm_factory_1.getInstance)();
    const stack = wasm.stackSave();
    try {
        const paramsOffset = wasm.stackAlloc(8);
        wasm._OrtGetLastError(paramsOffset, paramsOffset + 4);
        const errorCode = wasm.HEAP32[paramsOffset / 4];
        const errorMessagePointer = wasm.HEAPU32[paramsOffset / 4 + 1];
        const errorMessage = errorMessagePointer ? wasm.UTF8ToString(errorMessagePointer) : '';
        throw new Error(`${message} ERROR_CODE: ${errorCode}, ERROR_MESSAGE: ${errorMessage}`);
    }
    finally {
        wasm.stackRestore(stack);
    }
};
exports.checkLastError = checkLastError;
//# sourceMappingURL=wasm-utils.js.map