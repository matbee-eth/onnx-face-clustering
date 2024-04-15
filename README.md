# Project Title

## Description

This project is an implementation of the Multi-task Cascaded Convolutional Networks (MTCNN) for face detection and alignment and VGGFace for calculating embeddings.

# Based on MTCNN Python implementation
Based on https://github.com/yiyuezhuo/mtcnn-onnxruntime
- Python=3.7
- pip install onnxruntime opencv-python
- python3 app.py

## Files

- `mtcnnjs/pnet.js`: Contains the JavaScript implementation of the P-Net. (Stage 1)
- `mtcnnjs/rnet.js`: Contains the JavaScript implementation of the R-Net. (Stage 2)
- `mtcnnjs/onet.js`: Contains the JavaScript implementation of the O-Net. (Stage 3)
- `mtcnn_ort/mtcnn_ort.py`: Contains the Python implementation of MTCNN using ONNX Runtime.

## Usage

To use this project, you will need to install the necessary dependencies, then run the appropriate script for your use case.
You need to run Google Chrome/Canary with chrome://flags/#enable-webgpu-developer-features enabled.
- npm install http-server
- http-server . -p 8080
- navigate to http://localhost:8080/app.html
