const HtmlWebpackPlugin = require("html-webpack-plugin");
const CopyPlugin = require("copy-webpack-plugin");

const path = require('path');

module.exports = {
  entry: './src/index.ts',
  devtool: 'inline-source-map',
  node: false,
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        use: 'ts-loader',
        exclude: /node_modules/,
      },
    ],
  },
  resolve: {
    extensions: ['.tsx', '.ts', '.js'],
    fallback: {
      "crypto": false,
      "path": false,
      "stream": false,
      "fs": false,
    }
  },
  plugins: [
    new HtmlWebpackPlugin({
      title: 'our project',
      template: 'src/index.html'
    }),
    new CopyPlugin({
      // Use copy plugin to copy *.wasm to output folder.
      patterns: [{ from: 'node_modules/onnxruntime-web/dist/*.wasm', to: '[name][ext]' }]
    }),
    new CopyPlugin({
      // Use copy plugin to copy *.wasm to output folder.
      patterns: [{ from: 'src/models/*.onnx', to: 'models/[name][ext]' }]
    })
  ],

  devServer: {
    static: path.join(__dirname, "dist"),
    compress: true,
    port: 4000,
  },
  output: {
    filename: 'bundle.js',
    path: path.resolve(__dirname, 'dist'),
  },
};
