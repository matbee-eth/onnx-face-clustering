// Assuming onnxruntime-web is already included in your environment

// 1. Loading the MTCNN ONNX Model
async function loadMtcnnModel(modelPath) {
  const session = await ort.InferenceSession.create(modelPath);
  return session;
}

// 2. Preparing the Image Data
// This function should take an HTMLImageElement and return a Float32Array in the format expected by the model
function prepareImageData(image) {
  const canvas = document.createElement('canvas');
  canvas.width = image.width;
  canvas.height = image.height;
  const ctx = canvas.getContext('2d');
  // document.body.appendChild(canvas)
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  // Convert to Float32Array and normalize pixel values
  const float32Data = Float32Array.from(imageData, (v) => v / 255.0);
  // Reshape or process the data as needed for the model
  return float32Data;
}

// 3. Running the Model
async function runMtcnn(session, inputData) {
  // Assuming the model takes a single input named 'input'
  const inputTensor = new ort.Tensor('float32', inputData, [1, 3, canvas.width, canvas.height]);
  const inputs = { input_1: inputTensor };

  // Run the model
  const output = await session.run(inputs);
  return output;
}

// Function to run a stage of MTCNN (P-Net, R-Net, or O-Net)
async function runMtcnnStage(session, inputData, width, height) {
  // Log tensor dimensions
  console.log(`Creating tensor with dimensions: [1, 3, ${height}, ${width}]`);

  const inputTensor = new ort.Tensor('float32', inputData, [1, 3, height, width]);
  const feeds = { input_1: inputTensor }; // Replace 'input_1' with the correct input name

  const output = await session.run(feeds);
  return output;
}


function preprocessImageForOnet(image) {
  const targetWidth = 48;  // O-Net specific dimensions
  const targetHeight = 48;

  const canvas = document.createElement('canvas');
  canvas.id = `preprocessImageForOnet${image.width}${image.height}`;
  // document.body.appendChild(canvas)
  canvas.width = targetWidth;
  canvas.height = targetHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(image, 0, 0, targetWidth, targetHeight);

  const imageData = ctx.getImageData(0, 0, targetWidth, targetHeight);
  const data = new Float32Array(targetWidth * targetHeight * 3); // Channels last

  for (let i = 0; i < imageData.data.length / 4; i++) {
    data[i * 3] = imageData.data[i * 4] / 255.0;     // Red
    data[i * 3 + 1] = imageData.data[i * 4 + 1] / 255.0; // Green
    data[i * 3 + 2] = imageData.data[i * 4 + 2] / 255.0; // Blue
  }

  return data;
}

function preprocessImageForPnet(canvas) {
  const ctx = canvas.getContext('2d');
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = new Float32Array(canvas.width * canvas.height * 3); // Channels last

  for (let i = 0; i < imageData.data.length / 4; i++) {
    data[i * 3] = imageData.data[i * 4] / 255.0;     // Red
    data[i * 3 + 1] = imageData.data[i * 4 + 1] / 255.0; // Green
    data[i * 3 + 2] = imageData.data[i * 4 + 2] / 255.0; // Blue
  }

  return data;
}

async function runPnet(session, inputData, width, height) {
  const inputTensor = new ort.Tensor('float32', inputData, [1, height, width, 3]);
  const feeds = { input_1: inputTensor }; // Use 'input_1' as per model's input name

  const output = await session.run(feeds);
  console.info("runPnet", output);
  return output;
}


function processPnetOutput(pnetOutput, scale, canvas, originalImage, imageData, threshold) {
  const originalWidth = originalImage.width;
  const originalHeight = originalImage.height;
  const stride = 2;
  const cellsize = 12;
  const { dims: regDims, data: regData } = pnetOutput.conv2d_4;
  const { dims: confDims, data: confData } = pnetOutput.softmax;
  const boundingBoxes = [];

  for (let y = 0; y < confDims[1]; y++) {
    for (let x = 0; x < confDims[2]; x++) {
      const scoreIndex = (y * confDims[2] + x) * 2 + 1;
      const score = confData[scoreIndex];
      if (score > threshold) {
        const regIndex = (y * regDims[2] + x) * 4;
        const reg = regData.slice(regIndex, regIndex + 4);

        // Convert bounding box to square
        let width = ((stride * x + cellsize) + reg[2] * cellsize) - (stride * x + reg[0] * cellsize);
        let height = ((stride * y + cellsize) + reg[3] * cellsize) - (stride * y + reg[1] * cellsize);
        let maxSide = Math.max(width, height);
        width = height = maxSide; // Make it a square

        const box = {
          x1: (stride * x + reg[0] * cellsize) / scale,
          y1: (stride * y + reg[1] * cellsize) / scale,
          x2: ((stride * x + width) + reg[2] * cellsize) / scale,
          y2: ((stride * y + height) + reg[3] * cellsize) / scale,
        }
        // Calculate the bounding box in the scaled image, then scale it up
        const boundingBox = {
          ...box,
          score,
          scale,
          canvas,
          imageData,
          originalImage
        };
        if (score > 0.6) {
          console.info("processPnetOutput boundingBox", boundingBox)
        }
        boundingBoxes.push(boundingBox);
      }
    }
  }

  console.info("processPnetOutput boundingBoxes", boundingBoxes);
  let finalBoxes = nonMaximumSuppression(boundingBoxes, 0.5, 'Union');
  console.info("processPnetOutput finalBoxes", finalBoxes);
  return finalBoxes;
}

/** RNET */
function preprocessImageForRnet(image) {
  const targetWidth = 24;  // R-Net specific dimensions
  const targetHeight = 24;

  const canvas = document.createElement('canvas');
  canvas.id = `preprocessImageForRnet${image.width}${image.height}`;
  // document.body.appendChild(canvas)
  canvas.width = targetWidth;
  canvas.height = targetHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(image, 0, 0, targetWidth, targetHeight);

  const imageData = ctx.getImageData(0, 0, targetWidth, targetHeight);
  const data = new Float32Array(targetWidth * targetHeight * 3); // Channels last

  for (let i = 0; i < imageData.data.length / 4; i++) {
    data[i * 3] = imageData.data[i * 4] / 255.0;     // Red
    data[i * 3 + 1] = imageData.data[i * 4 + 1] / 255.0; // Green
    data[i * 3 + 2] = imageData.data[i * 4 + 2] / 255.0; // Blue
  }
  console.info("preprocessImageForRnet", data)
  return data;
}

function prepareRnetInput(image, pnetBoxes, rnetInputSize) {
  const rnetInputs = [];

  pnetBoxes.forEach(box => {
    const srcWidth = box.x2 - box.x1;
    const srcHeight = box.y2 - box.y1;
    const srcX = box.x1;
    const srcY = box.y1;

    if (srcWidth > 0 && srcHeight > 0) {
      const canvas = document.createElement('canvas');
      canvas.id = `rnetCanvas${box.x1}${box.y1}`;
      canvas.dataset.score = box.score;
      canvas.dataset.x = box.x1;
      canvas.dataset.y = box.y1;
      canvas.dataset.width = srcWidth;
      canvas.dataset.height = srcHeight;

      var referenceElement = box.canvas; // Get the reference element

      canvas.width = rnetInputSize;
      canvas.height = rnetInputSize;
      const ctx = canvas.getContext('2d');

      // Draw the cropped and resized image region onto the canvas
      console.info("prepareRnetInput src", canvas, referenceElement, rnetInputSize, srcX, srcY, srcWidth, srcHeight)
      ctx.drawImage(image, box.x1, box.y1, srcWidth, srcHeight, 0, 0, rnetInputSize, rnetInputSize);

      referenceElement.parentNode.insertBefore(canvas, referenceElement.nextSibling); // Insert the new element after the reference element

      const imageData = ctx.getImageData(0, 0, rnetInputSize, rnetInputSize).data;
      const rgbData = new Float32Array(rnetInputSize * rnetInputSize * 3);

      // Normalize the pixel values to [-1, 1]
      for (let i = 0, j = 0; i < imageData.length; i += 4) {
        rgbData[j++] = ((imageData[i] / 255.0) - 0.5) * 2;
        rgbData[j++] = ((imageData[i + 1] / 255.0) - 0.5) * 2;
        rgbData[j++] = ((imageData[i + 2] / 255.0) - 0.5) * 2;
      }

      rnetInputs.push({ data: rgbData, originalImage: box.originalImage, box: { ...box, canvas, pnetCanvas: box.canvas } });
    }
  });

  console.info("prepareRnetInput rnetInputs", rnetInputs);
  return rnetInputs;
}

async function runRnet(session, inputData, pnetBoxes) {
  let aggregatedResults = [];
  console.info("runRnet inputData length:", inputData.length);
  for (const { data, box } of inputData) {
    if (data.length !== 24 * 24 * 3) {
      console.error(`Invalid data length: expected ${24 * 24 * 3}, got ${data.length}`);
      throw new Error(`Invalid data length: expected ${24 * 24 * 3}, got ${data.length}`);
    }
    const scale = box.scale;
    const inputTensor = new ort.Tensor('float32', data, [1, 24, 24, 3]);
    const feeds = { input_2: inputTensor };

    // console.info("Running R-Net for input data:", data);
    const output = await session.run(feeds);

    // Extract bounding box refinements and confidence scores
    const boxRefinements = output.dense_2.data;
    const confidenceScores = output.softmax_1.data;

    console.info("R-Net output - box refinements:", boxRefinements);
    console.info("R-Net output - confidence scores:", confidenceScores);

    let refinedBoxes = processRnetOutput(boxRefinements, confidenceScores, pnetBoxes, scale, box.canvas);
    console.info("pnetBoxes", pnetBoxes.length, "refined:", refinedBoxes.length)
    console.info("Processed refined boxes:", refinedBoxes);

    aggregatedResults.push(...refinedBoxes);
  }

  console.info("Aggregated results:", aggregatedResults);
  return aggregatedResults;
}

function processRnetOutput(boxRefinements, confidenceScores, pnetBoxes, scale, canvas) {
  let refinedBoxes = [];

  for (let i = 0; i < boxRefinements.length / 4; i++) {
    let score = confidenceScores[i * 2 + 1]; // Assuming the second score is the face confidence
    console.info("processRnetBoxesAndScores", i, score, pnetBoxes[i])
    if (score > 0.5) { // Threshold for confidence score
      let originalBox = pnetBoxes[i]; // Get the original box from P-Net
      let refinement = {
        x1: boxRefinements[i * 4],
        y1: boxRefinements[i * 4 + 1],
        x2: boxRefinements[i * 4 + 2],
        y2: boxRefinements[i * 4 + 3]
      };

      // Apply the refinements to the bounding box from P-Net
      let refinedBox = {
        x1: originalBox.x1 + refinement.x1 * (originalBox.x2 - originalBox.x1),
        y1: originalBox.y1 + refinement.y1 * (originalBox.y2 - originalBox.y1),
        x2: originalBox.x2 + refinement.x2 * (originalBox.x2 - originalBox.x1),
        y2: originalBox.y2 + refinement.y2 * (originalBox.y2 - originalBox.y1),
      };

      // Calculate the refined width and height
      let refinedWidth = refinedBox.x2 - refinedBox.x1;
      let refinedHeight = refinedBox.y2 - refinedBox.y1;

      let centerX = refinedBox.x1 + refinedWidth / 2;
      let centerY = refinedBox.y1 + refinedHeight / 2;

      // Adjust to square by setting new x1, y1, x2, y2 based on the maxSize
      let maxSize = Math.max(refinedWidth, refinedHeight);

      let halfMaxSize = maxSize / 2;
      refinedBox.x1 = centerX - halfMaxSize;
      refinedBox.y1 = centerY - halfMaxSize;
      refinedBox.x2 = centerX + halfMaxSize;
      refinedBox.y2 = centerY + halfMaxSize;

      // Ensure the refined box is still within the image bounds
      refinedBox.x1 = Math.max(0, refinedBox.x1);
      refinedBox.y1 = Math.max(0, refinedBox.y1);
      refinedBox.x2 = Math.min(originalBox.originalImage.width, refinedBox.x2);
      refinedBox.y2 = Math.min(originalBox.originalImage.height, refinedBox.y2);

      // // Adjust to square
      // refinedBox.x2 = refinedBox.x1 + maxSize;
      // refinedBox.y2 = refinedBox.y1 + maxSize;

      // Check if the refined box intersects with the original box
      console.info("processRnetBoxesAndScores refinedBox, originalBox", refinedBox, originalBox)
      let scaledBox = {
        x1: refinedBox.x1,
        y1: refinedBox.y1,
        x2: refinedBox.x2,
        y2: refinedBox.y2,
        originalImage: originalBox.originalImage,
        score,
        scale,
        canvas,
        pnetCanvas: originalBox.canvas,
        unscaled: originalBox,
        pnetBox: originalBox,
        refinement
      };
      if (score > 0.6) {
        console.info("processRnetBoxesAndScores Scaled refined box", scaledBox)
      }
      // console.info("processRnetBoxesAndScores Scaled refined box", scaledBox);
      refinedBoxes.push(scaledBox);
    }
  }

  // Apply Non-Maximum Suppression (NMS) here if necessary
  return nonMaximumSuppression(refinedBoxes, 0.5, 'Union');
}

/** END OF RNET */

function resizeImage(image, scale) {
  const width = Math.ceil(image.width * scale);
  const height = Math.ceil(image.height * scale);

  const canvas = document.createElement('canvas');
  canvas.id = `pnetImage${width}x${height}:${scale}`;

  // document.body.appendChild(canvas)
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');

  ctx.drawImage(image, 0, 0, width, height);
  return canvas;
}

function computeScalePyramid(minFaceSize, width, height) {
  const m = 12 / minFaceSize;
  const minLayer = Math.min(width, height) * m;

  let scales = [];
  let factor = 0.709; // Scale factor between subsequent scales; adjust as needed
  let scale = 1;

  while (minLayer * scale > 12) {
    scales.push(scale);
    scale *= factor;
  }
  console.info("computeScalePyramid", scales)
  return scales;
}

async function processImageAtScales(image, pnetSession, scales) {
  let allBoxes = [];
  scales = [0.06385132023393836]
  for (let scale of scales) {
    let resizedCanvas = resizeImage(image, scale);
    let inputData = preprocessImageForPnet(resizedCanvas);
    let pnetOutput = await runPnet(pnetSession, inputData, resizedCanvas.width, resizedCanvas.height);
    console.info("ProcessImageAtAscales:: pnetOutput", pnetOutput)
    let boxes = processPnetOutput(pnetOutput, scale, resizedCanvas, image, inputData, 0.6);
    console.info("ProcessImageAtAscales:: boxes", boxes)

    boxes.forEach(box => {
      document.body.appendChild(box.canvas)
      // drawBoundingBox(box.canvas, box, "yellow"); // Original box
      // drawBoundingBox(box.canvas, {...box, ...box.unscaled}, "green"); // Unscaled
      // drawBoundingBox(box.canvas, {...box, ...box.scaled}, "blue"); // Scaled
      // drawBoundingBox(box.canvas, {...box, ...box.square}, "red"); // Square
    });

    allBoxes.push(...boxes);
  }
  console.info("processImageAtScales allBoxes", allBoxes);
  let rnetInput = prepareRnetInput(image, allBoxes, 24); // 24 is
  console.info("processImageAtScales rnetInput", rnetInput)
  const rnetSession = await loadMtcnnModel(rnetModelPath);
  let rnetBoxes = await runRnet(rnetSession, rnetInput, allBoxes);
  console.info("rnetBoxes", rnetBoxes)
  const canvas = document.createElement('canvas');
  canvas.id = `processImageAtScales${image.width}${image.height}`;
  const ctx = canvas.getContext('2d');
  canvas.width = image.width;
  canvas.height = image.height;
  ctx.drawImage(image, 0, 0, image.width, image.height);

  // Draw R-Net bounding boxes
  // rnetBoxes.forEach(({ unscaled: box }) => {
  //   ctx.beginPath();
  //   ctx.rect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
  //   ctx.strokeStyle = 'red'; // Set bounding box color
  //   ctx.lineWidth = 2; // Set line width
  //   ctx.stroke();
  // });
  document.body.appendChild(canvas)
  console.info("processImageAtScales rnetBoxes", rnetBoxes, allBoxes.length)

  // Process image through O-Net
  let onetInput = prepareOnetInput(image, rnetBoxes, 48); // 48 is the input size for O-Net
  console.info("processImageAtScales onetInput", onetInput)
  const onetSession = await loadMtcnnModel(onetModelPath);
  let onetOutput = await runOnet(image, onetSession, onetInput, rnetBoxes, image.width, image.height);

  console.info("processImageAtScales onetOutput", onetOutput)
  onetOutput.forEach((box) => {
    ctx.beginPath();
    ctx.rect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
    ctx.strokeStyle = 'green'; // Set bounding box color
    ctx.lineWidth = 2; // Set line width
    ctx.stroke();
    console.info("processImageAtScales box", box.pnetCanvas, box.canvas, box.landmarks)
    drawLandmarks(canvas, box.landmarks, box.score, box.scale);
  });
  return onetOutput;
}

function nonMaximumSuppression(boxes, threshold, method = 'Union') {
  if (boxes.length === 0) {
    return [];
  }

  // Calculate the area of each box
  const areas = boxes.map(box => (box.x2 - box.x1 + 1) * (box.y2 - box.y1 + 1));

  // Sort the boxes by score in descending order
  let sortedIndices = boxes.map((_, index) => index)
    .sort((a, b) => boxes[b].score - boxes[a].score);

  const pick = [];

  while (sortedIndices.length > 0) {
    const last = sortedIndices.length - 1;
    const i = sortedIndices[last];
    pick.push(boxes[i]);

    let inter = [];
    for (let j = 0; j < last; j++) {
      let idx = sortedIndices[j];
      let xx1 = Math.max(boxes[i].x1, boxes[idx].x1);
      let yy1 = Math.max(boxes[i].y1, boxes[idx].y1);
      let xx2 = Math.min(boxes[i].x2, boxes[idx].x2);
      let yy2 = Math.min(boxes[i].y2, boxes[idx].y2);
      let w = Math.max(0, xx2 - xx1 + 1);
      let h = Math.max(0, yy2 - yy1 + 1);
      inter[j] = w * h;
    }

    let o = inter.map((intArea, index) => {
      let overlapArea = (method === 'Min') ? Math.min(areas[i], areas[sortedIndices[index]]) 
                                           : (areas[i] + areas[sortedIndices[index]] - intArea);
      return intArea / overlapArea;
    });

    sortedIndices = sortedIndices.filter((_, index) => o[index] <= threshold);
  }

  return pick;
}

/** ONET */

function prepareOnetInput(image, rnetBoxes, onetInputSize) {
  const onetInputs = [];

  rnetBoxes.forEach((box, index) => {
    const canvas = document.createElement('canvas');
    canvas.id = `prepareOnetCanvas${box.x1}${box.y1}`;
    canvas.dataset.score = box.score;
    canvas.dataset.x = box.x1;
    canvas.dataset.y = box.y1;
    canvas.dataset.width = box.x2 - box.x1;
    canvas.dataset.height = box.y2 - box.y1;
    console.info("prepareOnetInput src", box.pnetBox, box)

    canvas.width = onetInputSize;
    canvas.height = onetInputSize;
    const ctx = canvas.getContext('2d');

    // Calculate the coordinates and size of the source rectangle for cropping
    const sx = box.x1;
    const sy = box.y1;
    const sw = box.x2 - box.x1;
    const sh = box.y2 - box.y1;


    // Crop and resize the image region to the canvas
    ctx.drawImage(image, sx, sy, sw, sh, 0, 0, onetInputSize, onetInputSize);

    // Insert the canvas into the DOM for debugging
    var referenceElement = box.canvas;
    referenceElement.parentNode.insertBefore(canvas, referenceElement.nextSibling);

    // Extract image data
    const imageData = ctx.getImageData(0, 0, onetInputSize, onetInputSize).data;
    const rgbData = [];
    for (let i = 0; i < imageData.length; i += 4) {
      rgbData.push(imageData[i], imageData[i + 1], imageData[i + 2]); // Skip alpha channel
    }
    const normalizedData = Float32Array.from(rgbData, (v) => v / 255.0);

    // Add the data to the onetInputs array
    onetInputs.push({
      data: normalizedData,
      originalImage: box.originalImage,
      scale: box.scale, // Include other properties if needed
      score: box.score,
      box: { x1: sx, y1: sy, width: sw, height: sh },
      pnetBox: box.pnetBox,
      rnetBox: box,
      canvas: canvas,
      rnetCanvas: box.canvas,
      pnetCanvas: box.pnetCanvas
    });
  });

  console.info("prepareOnetInput onetInputs", onetInputs);
  return onetInputs;
}

async function runOnet(image, session, inputData, rnetBoxes, originalWidth, originalHeight) {
  let aggregatedResults = [];

  for (var i = 0; i < inputData.length; i++) {
    var data = inputData[i].data;
    var scale = inputData[i].scale;
    console.info("runOnet data", data.length, scale)
    if (data.length !== 48 * 48 * 3) {
      throw new Error(`Invalid data length: expected ${48 * 48 * 3}, got ${data.length}`);
    }

    const inputTensor = new ort.Tensor('float32', data, [1, 48, 48, 3]);
    const feeds = { input_3: inputTensor };
    const output = await session.run(feeds);
    console.info("runOnet output", output, scale)
    let processedOutput = processOnetOutput(image, output, rnetBoxes[i], 0.6, scale, originalWidth, originalHeight);
    aggregatedResults.push(...processedOutput);
  }

  return aggregatedResults;
}

const onetInputSize = 48; // O-Net expects 48x48 pixel input images
function processOnetOutput(image, OnetOutput, rnetBox, threshold, originalWidth, originalHeight) {
  let refinedBoxes = [];
  const boxRefinements = OnetOutput.dense_5.data;
  const confidenceScores = OnetOutput.softmax_2.data;
  const landmarks = OnetOutput.dense_6.data;

  for (let i = 0; i < boxRefinements.length / 4; i++) {
    let score = confidenceScores[i * 2 + 1]; // Assuming index 'i * 2 + 1' is the face confidence score
    console.info("processOnetOutput", i,  threshold, score, score > threshold)
    if (score > threshold) {
      let originalBox = rnetBox;
      let refinement = {
        x1: boxRefinements[i * 4],
        y1: boxRefinements[i * 4 + 1],
        x2: boxRefinements[i * 4 + 2],
        y2: boxRefinements[i * 4 + 3]
      };

      // Calculate width and height of the R-Net bounding box
      const originalWidth = originalBox.originalImage.width;
      const originalHeight = originalBox.originalImage.height;

      // Apply refinements to the R-Net bounding box
      let refinedBox = {
        x1: originalBox.x1 + refinement.x1 * (originalBox.x2 - originalBox.x1),
        y1: originalBox.y1 + refinement.y1 * (originalBox.y2 - originalBox.y1),
        x2: originalBox.x2 + refinement.x2 * (originalBox.x2 - originalBox.x1),
        y2: originalBox.y2 + refinement.y2 * (originalBox.y2 - originalBox.y1),
        score: score
      };

      console.info("processOnetOutput box", "x1", refinedBox.x1, "y1", refinedBox.y1, "x2", refinedBox.x2, "y2", refinedBox.y2)
      // Calculate the center of the refined box
      // Calculate the center of the refined box
      let centerX = (originalBox.x1 + refinement.x1 * (originalBox.x2 - originalBox.x1)) + ((originalBox.x2 + refinement.x2 * (originalBox.x2 - originalBox.x1)) - (originalBox.x1 + refinement.x1 * (originalBox.x2 - originalBox.x1))) / 2;
      let centerY = (originalBox.y1 + refinement.y1 * (originalBox.y2 - originalBox.y1)) + ((originalBox.y2 + refinement.y2 * (originalBox.y2 - originalBox.y1)) - (originalBox.y1 + refinement.y1 * (originalBox.y2 - originalBox.y1))) / 2;

      // Calculate the maximum size to make the box square
      let refinedWidth = (originalBox.x2 + refinement.x2 * (originalBox.x2 - originalBox.x1)) - (originalBox.x1 + refinement.x1 * (originalBox.x2 - originalBox.x1));
      let refinedHeight = (originalBox.y2 + refinement.y2 * (originalBox.y2 - originalBox.y1)) - (originalBox.y1 + refinement.y1 * (originalBox.y2 - originalBox.y1));
      let maxSize = Math.max(refinedWidth, refinedHeight);

      // Adjust to square while keeping the box centered
      let newHalfSize = maxSize / 2;
      refinedBox = {
        x1: centerX - newHalfSize,
        y1: centerY - newHalfSize,
        x2: centerX + newHalfSize,
        y2: centerY + newHalfSize,
        score: score
      };

      // Ensure the box is within the image bounds
      refinedBox.x1 = Math.max(0, refinedBox.x1);
      refinedBox.y1 = Math.max(0, refinedBox.y1);
      refinedBox.x2 = Math.min(originalWidth, refinedBox.x2);
      refinedBox.y2 = Math.min(originalHeight, refinedBox.y2);

      console.info("processOnetOutput box post-center", "x1", refinedBox.x1, "y1", refinedBox.y1, "x2", refinedBox.x2, "y2", refinedBox.y2)
      // Transform landmarks from local coordinates to the coordinates of the original image
      let transformedLandmarks = extractLandmarks(i, landmarks, refinedBox);
      refinedBox.landmarks = transformedLandmarks;

      // Create a canvas for each box
      const canvas = document.createElement('canvas');
      canvas.id = `onetCanvas${refinedBox.x1}${refinedBox.y1}`;
      canvas.dataset.score = score;
      canvas.width = onetInputSize;
      canvas.height = onetInputSize;
      const ctx = canvas.getContext('2d');
      canvas.dataset.x = refinedBox.x1;
      canvas.dataset.y = refinedBox.y1;
      canvas.dataset.width = refinedBox.x2 - refinedBox.x1;
      canvas.dataset.height = refinedBox.y2 - refinedBox.y1;

      // Draw the cropped and resized image region onto the canvas
      const srcWidth = refinedBox.x2 - refinedBox.x1;
      const srcHeight = refinedBox.y2 - refinedBox.y1;
      ctx.drawImage(image, refinedBox.x1, refinedBox.y1, srcWidth, srcHeight, 0, 0, onetInputSize, onetInputSize);

      // Insert the canvas into the DOM for debugging
      const referenceElement = originalBox.canvas;
      referenceElement.parentNode.insertBefore(canvas, referenceElement.nextSibling);

      refinedBox.canvas = canvas;
      refinedBox.rnetCanvas = originalBox.canvas;
      refinedBox.pnetCanvas = originalBox.pnetCanvas;
      refinedBox.rnetBox = originalBox;
      refinedBox.pnetBox = originalBox.pnetBox;
      refinedBox.refinement = refinement;
      refinedBox.originalImage = originalBox.originalImage;
      refinedBoxes.push(refinedBox);
    }
  }
  console.info("Before parsing refined boxes:", refinedBoxes);
  // Apply Non-Maximum Suppression (NMS) to filter overlapping boxes
  const newRefined = nonMaximumSuppression(refinedBoxes, 0.3, 'Union');
  console.info("After parsing refined boxes:", newRefined);
  return newRefined;
}


/** End of ONET */


function extractLandmarks(index, landmarks, boundingBox) {
  let landmarkSet = [];
  for (let j = 0; j < 5; j++) {
    // Landmarks are normalized [0, 1] relative to the bounding box
    let originalX = boundingBox.x1 + landmarks[index * 10 + j * 2] * (boundingBox.x2 - boundingBox.x1);
    let originalY = boundingBox.y1 + landmarks[index * 10 + j * 2 + 1] * (boundingBox.y2 - boundingBox.y1);
    landmarkSet.push({ x: originalX, y: originalY });
  }
  return landmarkSet;
}

function translateLandmarks(landmarks, boundingBox, originalWidth, originalHeight) {
  let translatedLandmarks = [];

  // Calculate the actual pixel coordinates of the bounding box on the original image
  const actualX1 = boundingBox.x1 * originalWidth;
  const actualY1 = boundingBox.y1 * originalHeight;
  const actualWidth = (boundingBox.x2 - boundingBox.x1) * originalWidth;
  const actualHeight = (boundingBox.y2 - boundingBox.y1) * originalHeight;

  // Translate each landmark to the original image space
  landmarks.forEach(landmark => {
    translatedLandmarks.push({
      x: actualX1 + landmark.x * actualWidth,
      y: actualY1 + landmark.y * actualHeight
    });
  });

  return translatedLandmarks;
}

const rnetInputSize = 24; // R-Net expects 24x24 pixel input images

async function processImageThroughMtcnn(imageSrc, scale) {
  const image = new Image();
  image.src = imageSrc;
  console.info("processImageThroughMtcnn", imageSrc)
  image.onload = async () => {
    console.info("processImageThroughMtcnn: pnetSession", image)
    const pnetSession = await loadMtcnnModel(pnetModelPath);

    // const onetSession = await loadMtcnnModel(onetModelPath);
    const minFaceSize = 20; // Adjust based on the smallest face size you want to detect
    const imageWidth = image.width;
    const imageHeight = image.height;

    const scales = computeScalePyramid(minFaceSize, imageWidth, imageHeight);

    const finalBoxes = (await processImageAtScales(image, pnetSession, scales));
    console.info("finalBoxes", finalBoxes)
  };
  image.onerror = (error) => {
    console.error("Error loading image:", error);
  };
}
const pnetModelPath = 'mtcnn_ort/pnet.onnx';
const rnetModelPath = 'mtcnn_ort/rnet.onnx';
const onetModelPath = 'mtcnn_ort/onet.onnx';


function drawBoundingBox(canvas, box, color) {
  const ctx = canvas.getContext("2d");
  ctx.beginPath();
  ctx.rect(box.x1, box.y1, box.x2 - box.x1, box.y2 - box.y1);
  ctx.lineWidth = 2;
  ctx.strokeStyle = color;
  ctx.stroke();
  ctx.fillStyle = color;
  ctx.fillText(`Score: ${box.score.toFixed(2)}`, box.x1 + 5, box.y1 - 5);
}

// Function to draw landmarks
function drawLandmarks(canvas, landmarks, score, scale) {
  const ctx = canvas.getContext("2d");

  const colors = ["red", "green", "blue", "yellow", "purple"];
  const map = ["leftEye", "rightEye", "nose", "leftMouth", "rightMouth"]
  for (let i = 0; i < landmarks.length; i++) {
    const landmark = landmarks[i];
    console.info('landmark', landmark);
    ctx.beginPath();
    ctx.arc(landmark.x * canvas.width, landmark.y * canvas.height, 3, 0, 2 * Math.PI);
    ctx.fillStyle = colors[i % colors.length];
    ctx.fill();
    ctx.fillText(`Score: ${score.toFixed(2)}`, 0, 0);
  }
}

function drawImage(canvas, image) {
  canvas.width = image.width;
  canvas.height = image.height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
}

// Call this function with the path to your image and MTCNN model paths
processImageThroughMtcnn('4545.jpg', 0.5);
