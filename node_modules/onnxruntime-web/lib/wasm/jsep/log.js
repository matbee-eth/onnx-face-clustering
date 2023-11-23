"use strict";
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
Object.defineProperty(exports, "__esModule", { value: true });
exports.LOG_DEBUG = exports.LOG = exports.configureLogger = void 0;
const wasm_common_1 = require("../wasm-common");
const logLevelPrefix = ['V', 'I', 'W', 'E', 'F'];
const doLog = (level, message) => {
    // eslint-disable-next-line no-console
    console.log(`[${logLevelPrefix[level]},${new Date().toISOString()}]${message}`);
};
let configLogLevel;
let debug;
const configureLogger = ($configLogLevel, $debug) => {
    configLogLevel = $configLogLevel;
    debug = $debug;
};
exports.configureLogger = configureLogger;
/**
 * A simple logging utility to log messages to the console.
 */
const LOG = (logLevel, msg) => {
    const messageLevel = (0, wasm_common_1.logLevelStringToEnum)(logLevel);
    const configLevel = (0, wasm_common_1.logLevelStringToEnum)(configLogLevel);
    if (messageLevel >= configLevel) {
        doLog(messageLevel, typeof msg === 'function' ? msg() : msg);
    }
};
exports.LOG = LOG;
/**
 * A simple logging utility to log messages to the console. Only logs when debug is enabled.
 */
const LOG_DEBUG = (...args) => {
    if (debug) {
        (0, exports.LOG)(...args);
    }
};
exports.LOG_DEBUG = LOG_DEBUG;
//# sourceMappingURL=log.js.map