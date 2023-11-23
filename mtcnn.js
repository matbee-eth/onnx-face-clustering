class MTCNN {

  stepsThreshold = [0.6, 0.7, 0.7]

  minFaceSize = 20

  _scaleFactor = 0.709

  constructor() {
    // Initialize variables for storing loaded models
    this.pNet = null;
    this.rNet = null;
    this.oNet = null;
    this.loadModels();
  }

  async loadModels() {
    this.pNet = await ort.InferenceSession.create('mtcnn_ort/pnet.onnx');
    this.rNet = await ort.InferenceSession.create('mtcnn_ort/rnet.onnx');
    this.oNet = await ort.InferenceSession.create('mtcnn_ort/onet.onnx');
  }

  preprocessImage(image) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = 160; // Assuming 160 is the required dimension
    canvas.height = 160;

    // Draw and resize the image on the canvas
    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

    // Extract image data
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;
    const float32Data = new Float32Array(3 * 160 * 160);

    for (let i = 0, j = 0; i < data.length; i += 4, j += 3) {
      // Convert from RGBA to RGB and normalize the data
      float32Data[j] = (data[i] - 127.5) / 128.0;
      float32Data[j + 1] = (data[i + 1] - 127.5) / 128.0;
      float32Data[j + 2] = (data[i + 2] - 127.5) / 128.0;
    }

    return new ort.Tensor('float32', float32Data, [1, 3, 160, 160]);
  }

  nonMaximumSuppression(boxes, threshold, method) {
    if (boxes.length === 0) {
      return [];
    }

    const areas = boxes.map(box => (box.x2 - box.x1 + 1) * (box.y2 - box.y1 + 1));
    const s = boxes.map(box => box.score);

    let sortedIndices = s.map((score, index) => index).sort((a, b) => s[b] - s[a]);
    const pick = [];

    while (sortedIndices.length > 0) {
      let last = sortedIndices.length - 1;
      let i = sortedIndices[last];
      pick.push(boxes[i]);

      let maxxx1 = [];
      let maxyy1 = [];
      let minxx2 = [];
      let minyy2 = [];
      let w = [];
      let h = [];

      for (let j = 0; j < last; j++) {
        let idx = sortedIndices[j];
        maxxx1[j] = Math.max(boxes[i].x1, boxes[idx].x1);
        maxyy1[j] = Math.max(boxes[i].y1, boxes[idx].y1);
        minxx2[j] = Math.min(boxes[i].x2, boxes[idx].x2);
        minyy2[j] = Math.min(boxes[i].y2, boxes[idx].y2);
        w[j] = Math.max(0.0, minxx2[j] - maxxx1[j] + 1);
        h[j] = Math.max(0.0, minyy2[j] - maxyy1[j] + 1);
      }

      const inter = w.map((width, index) => width * h[index]);

      let o;
      if (method === 'Min') {
        o = inter.map((_, index) => inter[index] / Math.min(areas[i], areas[sortedIndices[index]]));
      } else {
        o = inter.map((_, index) => inter[index] / (areas[i] + areas[sortedIndices[index]] - inter[index]));
      }

      sortedIndices = sortedIndices.filter((_, index) => o[index] <= threshold);
    }

    return pick;
  }


  async runPNet(image) {
    // Preprocess the image
    const inputTensor = this.preprocessImage(image);

    // Run the P-Net model
    const pNetOutputs = await this.pNet.run({ input: inputTensor });

    // Extract and process P-Net outputs to get candidate bounding boxes
    // This involves interpreting the model outputs, applying NMS, etc.
    // Pseudo-code below, replace with actual implementation
    const boxes = this.processPNetOutputs(pNetOutputs);

    return boxes;
  }

  computeScalePyramid(m, minLayer) {
    const scales = [];
    let factorCount = 0;

    while (minLayer >= 12) {
      scales.push(m * Math.pow(this._scaleFactor, factorCount));
      minLayer *= this._scaleFactor;
      factorCount++;
    }

    return scales;
  }

  async markFaces(imageData) {
    if (!this.pNet) {
      await this.loadModels();
    }
    console.info("markFaces", imageData)
    // Assuming `detectFaces` is a method that returns face detections with bounding boxes and keypoints
    const results = await this.detectFaces(imageData);
    console.info("markFaces results", results)
    // Create a canvas and get the context for drawing
    const canvas = document.createElement('canvas');
    canvas.width = imageData.width;
    canvas.height = imageData.height;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(imageData, 0, 0);

    // Draw rectangles and keypoints
    results.forEach(result => {
      const box = result.box;
      ctx.strokeStyle = 'rgb(0, 155, 255)';
      ctx.lineWidth = 2;
      ctx.strokeRect(box[0], box[1], box[2], box[3]);

      ctx.fillStyle = 'rgb(0, 155, 255)';
      Object.values(result.keypoints).forEach(point => {
        ctx.beginPath();
        ctx.arc(point[0], point[1], 2, 0, 2 * Math.PI);
        ctx.fill();
      });
    });

    return canvas; // Returns the image as a data URL
  }


  async detectFacesRaw(image) {
    if (!image) {
      throw new Error("Image not valid.");
    }
    console.info("detectFacesRaw", image.width, image.height)
    const height = image.height;
    const width = image.width;
    let stageStatus = new StageStatus(null, width, height);

    const m = 12 / this.minFaceSize; // Assuming this.minFaceSize is defined
    const minLayer = Math.min(height, width) * m;

    const scales = this.computeScalePyramid(m, minLayer);

    const stages = [this.stage1.bind(this)/*, this.stage2, this.stage3*/]; // Assuming these stages are defined
    let result = [scales, stageStatus];

    for (const stage of stages) {
      result = await stage(image, result[0], result[1]);
    }

    return result; // [total_boxes, points]
  }

  async detectFaces(image) {
    const [totalBoxes, points] = await this.detectFacesRaw(image);

    const boundingBoxes = totalBoxes.map((boundingBox, i) => {
      const keypoints = points[i];
      return {
        box: [Math.max(0, boundingBox.x1), Math.max(0, boundingBox.y1), boundingBox.x2 - boundingBox.x1, boundingBox.y2 - boundingBox.y1],
        confidence: boundingBox.score,
        keypoints: {
          left_eye: keypoints.leftEye,
          right_eye: keypoints.rightEye,
          nose: keypoints.nose,
          mouth_left: keypoints.mouthLeft,
          mouth_right: keypoints.mouthRight
        }
      };
    });

    return boundingBoxes;
  }

  imageDataToTensor(imageData) {
    const { data, width, height } = imageData;
    const tensorData = new Float32Array(width * height * 3);

    for (let i = 0, j = 0; i < data.length; i += 4, j += 3) {
        tensorData[j] = data[i];     // Red
        tensorData[j + 1] = data[i + 1]; // Green
        tensorData[j + 2] = data[i + 2]; // Blue
        // Alpha channel is ignored
    }

    // Normalize if necessary
    // Return the tensor with the shape that matches your model's input
    console.info("TENSOR???", [1, 3, height, width]);
    return new ort.Tensor('float32', tensorData, [1, 3, height, width]);
}


  async stage1(image, scales, stageStatus) {
    console.info("stage1", this, image, scales, stageStatus)
    let totalBoxes = [];
    const threshold = 0.6; // Example threshold value, adjust based on your model's requirements
    for (const scale of scales) {
      const scaledImage = this.scaleImage(image, scale);
      const imgY = this.transposeImage(scaledImage);
      const tensor = this.imageDataToTensor(imgY); // Convert to tensor

      // Run P-Net model
      const out = await this.pNet.run({ input_1: tensor});

      // Process P-Net outputs to generate bounding boxes
      const boxes = this.generateBoundingBox(out, scale, threshold);

      // Inter-scale NMS
      const pick = this.nonMaximumSuppression(boxes, 0.5, 'Union');
      if (boxes.length > 0 && pick.length > 0) {
        const pickedBoxes = pick.map(index => boxes[index]);
        totalBoxes = totalBoxes.concat(pickedBoxes);
      }
    }

    // Further processing on totalBoxes
    if (totalBoxes.length > 0) {
      const pick = this.nonMaximumSuppression(totalBoxes, 0.7, 'Union');
      totalBoxes = pick.map(index => totalBoxes[index]);

      // Refine boxes
      totalBoxes = this.refineBoxes(totalBoxes);
      console.info("totalBoxes, stageStatus", totalBoxes, stageStatus)
      // Update stage status
      stageStatus = this.updateStageStatus(totalBoxes, stageStatus);
    }

    return [totalBoxes, stageStatus];
  }

  refineBoxes(boxes) {
    return boxes.map(box => {
      const regw = box.x2 - box.x1;
      const regh = box.y2 - box.y1;

      const qq1 = box.x1 + box.reg[0] * regw;
      const qq2 = box.y1 + box.reg[1] * regh;
      const qq3 = box.x2 + box.reg[2] * regw;
      const qq4 = box.y2 + box.reg[3] * regh;

      return {
        x1: qq1,
        y1: qq2,
        x2: qq3,
        y2: qq4,
        score: box.score
      };
    });
  }

  generateBoundingBox(pNetOutput, scale, threshold) {
    const confidenceData = pNetOutput.confidences.data; // Assuming this is the confidence output
    const regressionData = pNetOutput.regressions.data; // Assuming this is the regression output
    let boxes = [];

    for (let y = 0; y < pNetOutput.confidences.shape[1]; y++) {
      for (let x = 0; x < pNetOutput.confidences.shape[2]; x++) {
        const score = confidenceData[y * pNetOutput.confidences.shape[2] + x];
        if (score > threshold) {
          const reg = regressionData.slice((y * pNetOutput.confidences.shape[2] + x) * 4, ((y * pNetOutput.confidences.shape[2] + x) + 1) * 4);
          const box = {
            x1: Math.round(x * 2 / scale),
            y1: Math.round(y * 2 / scale),
            x2: Math.round((x * 2 + 12) / scale),
            y2: Math.round((y * 2 + 12) / scale),
            score: score,
            reg: reg
          };
          boxes.push(box);
        }
      }
    }

    return boxes;
  }


  transposeImage(imageData) {
    const width = imageData.width;
    const height = imageData.height;
    const transposedData = new Uint8ClampedArray(width * height * 4);

    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const originalIndex = (y * width + x) * 4;
        const transposedIndex = (x * height + y) * 4;
        transposedData[transposedIndex] = imageData.data[originalIndex];       // R
        transposedData[transposedIndex + 1] = imageData.data[originalIndex + 1]; // G
        transposedData[transposedIndex + 2] = imageData.data[originalIndex + 2]; // B
        transposedData[transposedIndex + 3] = imageData.data[originalIndex + 3]; // A
      }
    }

    return new ImageData(transposedData, height, width);
  }


  scaleImage(image, scale) {
    const widthScaled = Math.ceil(image.width * scale);
    const heightScaled = Math.ceil(image.height * scale);

    // Create a canvas and resize the image
    const canvas = document.createElement('canvas');
    canvas.width = widthScaled;
    canvas.height = heightScaled;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(image, 0, 0, widthScaled, heightScaled);

    // Extract the image data and normalize
    const imageData = ctx.getImageData(0, 0, widthScaled, heightScaled);
    const data = imageData.data;
    for (let i = 0; i < data.length; i += 4) {
      data[i] = (data[i] - 127.5) * 0.0078125;     // Red
      data[i + 1] = (data[i + 1] - 127.5) * 0.0078125; // Green
      data[i + 2] = (data[i + 2] - 127.5) * 0.0078125; // Blue
    }

    return imageData;
  }


  // Additional class methods will be defined below
}

class StageStatus {
  constructor(padResult = null, width = 0, height = 0) {
    this.width = width;
    this.height = height;
    this.dy = this.edy = this.dx = this.edx = this.y = this.ey = this.x = this.ex = this.tmpw = this.tmph = [];

    if (padResult !== null) {
      this.update(padResult);
    }
  }

  update(padResult) {
    [this.dy, this.edy, this.dx, this.edx, this.y, this.ey, this.x, this.ex, this.tmpw, this.tmph] = padResult;
  }
}