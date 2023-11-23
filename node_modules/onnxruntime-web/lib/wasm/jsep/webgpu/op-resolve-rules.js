"use strict";
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.WEBGPU_OP_RESOLVE_RULES = void 0;
const argminmax_1 = require("./ops/argminmax");
const binaryOps = __importStar(require("./ops/binary-op"));
const concat_1 = require("./ops/concat");
const conv_1 = require("./ops/conv");
const conv_transpose_1 = require("./ops/conv-transpose");
const expand_1 = require("./ops/expand");
const gather_1 = require("./ops/gather");
const gemm_1 = require("./ops/gemm");
const instance_norm_1 = require("./ops/instance-norm");
const layer_norm_1 = require("./ops/layer-norm");
const matmul_1 = require("./ops/matmul");
const pool = __importStar(require("./ops/pool"));
const reduce_1 = require("./ops/reduce");
const resize_1 = require("./ops/resize");
const skip_layer_norm_1 = require("./ops/skip-layer-norm");
const slice_1 = require("./ops/slice");
const softmax_1 = require("./ops/softmax");
const split_1 = require("./ops/split");
const tile_1 = require("./ops/tile");
const transpose_1 = require("./ops/transpose");
const unaryOps = __importStar(require("./ops/unary-op"));
exports.WEBGPU_OP_RESOLVE_RULES = new Map([
    ['Abs', [unaryOps.abs]],
    ['Acos', [unaryOps.acos]],
    ['Acosh', [unaryOps.acosh]],
    ['Add', [binaryOps.add]],
    ['ArgMax', [argminmax_1.argMax, argminmax_1.parseArgMinMaxAttributes]],
    ['ArgMin', [argminmax_1.argMin, argminmax_1.parseArgMinMaxAttributes]],
    ['Asin', [unaryOps.asin]],
    ['Asinh', [unaryOps.asinh]],
    ['Atan', [unaryOps.atan]],
    ['Atanh', [unaryOps.atanh]],
    // TODO: support new attributes for AveragePool-10
    ['AveragePool', [pool.averagePool, pool.parseAveragePoolAttributes]],
    ['Cast', [unaryOps.cast, unaryOps.parseCastAttributes]],
    ['Ceil', [unaryOps.ceil]],
    ['ClipV10', [unaryOps.clipV10]],
    ['Clip', [unaryOps.clip]],
    ['Concat', [concat_1.concat, concat_1.parseConcatAttributes]],
    ['Conv', [conv_1.conv, conv_1.parseConvAttributes]],
    ['ConvTranspose', [conv_transpose_1.convTranspose, conv_transpose_1.parseConvTransposeAttributes]],
    ['Cos', [unaryOps.cos]],
    ['Cosh', [unaryOps.cosh]],
    ['Div', [binaryOps.div]],
    ['Elu', [unaryOps.elu, unaryOps.parseAlphaAttributes]],
    ['Erf', [unaryOps.erf]],
    ['Exp', [unaryOps.exp]],
    ['Expand', [expand_1.expand]],
    ['Floor', [unaryOps.floor]],
    ['Gather', [gather_1.gather, gather_1.parseGatherAttributes]],
    ['Gelu', [unaryOps.gelu]],
    ['Gemm', [gemm_1.gemm, gemm_1.parseGemmAttributes]],
    ['GlobalAveragePool', [pool.globalAveragePool, pool.parseGlobalAveragePoolAttributes]],
    ['GlobalMaxPool', [pool.globalMaxPool, pool.parseGlobalMaxPoolAttributes]],
    ['InstanceNormalization', [instance_norm_1.instanceNorm, instance_norm_1.parseInstanceNormAttributes]],
    ['LayerNormalization', [layer_norm_1.layerNorm, layer_norm_1.parseLayerNormAttributes]],
    ['LeakyRelu', [unaryOps.leakyRelu, unaryOps.parseAlphaAttributes]],
    ['Log', [unaryOps.log]],
    ['MatMul', [matmul_1.matMul]],
    // TODO: support new attributes for MaxPool-8 and MaxPool-10
    ['MaxPool', [pool.maxPool, pool.parseMaxPoolAttributes]],
    ['Mul', [binaryOps.mul]],
    ['Neg', [unaryOps.neg]],
    ['Pow', [binaryOps.pow]],
    ['Reciprocal', [unaryOps.reciprocal]],
    ['ReduceMin', [reduce_1.reduceMin, reduce_1.parseReduceAttributes]],
    ['ReduceMean', [reduce_1.reduceMean, reduce_1.parseReduceAttributes]],
    ['ReduceMax', [reduce_1.reduceMax, reduce_1.parseReduceAttributes]],
    ['ReduceSum', [reduce_1.reduceSum, reduce_1.parseReduceAttributes]],
    ['ReduceProd', [reduce_1.reduceProd, reduce_1.parseReduceAttributes]],
    ['ReduceL1', [reduce_1.reduceL1, reduce_1.parseReduceAttributes]],
    ['ReduceL2', [reduce_1.reduceL2, reduce_1.parseReduceAttributes]],
    ['ReduceLogSum', [reduce_1.reduceLogSum, reduce_1.parseReduceAttributes]],
    ['ReduceLogSumExp', [reduce_1.reduceLogSumExp, reduce_1.parseReduceAttributes]],
    ['ReduceSumSquare', [reduce_1.reduceSumSquare, reduce_1.parseReduceAttributes]],
    ['Relu', [unaryOps.relu]],
    ['Resize', [resize_1.resize, resize_1.parseResizeAttributes]],
    ['Sigmoid', [unaryOps.sigmoid]],
    ['Sin', [unaryOps.sin]],
    ['Sinh', [unaryOps.sinh]],
    ['Slice', [slice_1.slice, slice_1.parseSliceAttributes]],
    ['SkipLayerNormalization', [skip_layer_norm_1.skipLayerNorm, skip_layer_norm_1.parseSkipLayerNormAttributes]],
    ['Split', [split_1.split, split_1.parseSplitAttributes]],
    ['Sqrt', [unaryOps.sqrt]],
    ['Softmax', [softmax_1.softmax, softmax_1.parseSoftmaxAttributes]],
    ['Sub', [binaryOps.sub]],
    ['Tan', [unaryOps.tan]],
    ['Tanh', [unaryOps.tanh]],
    ['ThresholdedRelu', [unaryOps.thresholdedRelu, unaryOps.parseAlphaAttributes]],
    ['Tile', [tile_1.tile]],
    ['Transpose', [transpose_1.transpose, transpose_1.parseTransposeAttributes]],
]);
//# sourceMappingURL=op-resolve-rules.js.map