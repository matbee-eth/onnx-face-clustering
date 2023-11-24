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
  document.body.appendChild(canvas)
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

function preprocessImageForRnet(image) {
  const targetWidth = 24;  // R-Net specific dimensions
  const targetHeight = 24;

  const canvas = document.createElement('canvas');
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

async function runRnet(session, inputData) {
  let aggregatedResults = [];
//   console.info("runRnet inputData", inputData)
  for (const data of inputData) {
      if (data.length !== 24 * 24 * 3) {
          throw new Error(`Invalid data length: expected ${24 * 24 * 3}, got ${data.length}`);
      }

      const inputTensor = new ort.Tensor('float32', data, [1, 24, 24, 3]);
      const feeds = { input_2: inputTensor };
      const output = await session.run(feeds);

      // Extract bounding box refinements and confidence scores
      const boxRefinements = output.dense_2.data;
      const confidenceScores = output.softmax_1.data;
    //   console.info("runRnet output", boxRefinements, confidenceScores);
      let refinedBoxes = processRnetBoxesAndScores(boxRefinements, confidenceScores);
      aggregatedResults.push(...refinedBoxes);
  }

  return aggregatedResults;
}


function processRnetBoxesAndScores(boxRefinements, confidenceScores) {
  let refinedBoxes = [];

  for (let i = 0; i < boxRefinements.length / 4; i++) {
      let score = confidenceScores[i * 2 + 1]; // Confidence that the region contains a face
      console.info("refinedBoxes", score, boxRefinements)
      if (score > 0.0003) { // Example threshold
          let box = {
              x1: boxRefinements[i * 4],
              y1: boxRefinements[i * 4 + 1],
              x2: boxRefinements[i * 4 + 2],
              y2: boxRefinements[i * 4 + 3],
              score: score
          };
          refinedBoxes.push(box);
      }
  }
  // Apply Non-Maximum Suppression (NMS) here if necessary
  return nonMaximumSuppression(refinedBoxes, 0.7, 'Union');
}



function preprocessImage(image, scale) {
  const targetWidth = Math.ceil(image.width * scale);
  const targetHeight = Math.ceil(image.height * scale);

  console.log(`Rescaled image size: ${targetWidth}x${targetHeight}`);

  const canvas = document.createElement('canvas');
  canvas.width = targetWidth;
  canvas.height = targetHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(image, 0, 0, targetWidth, targetHeight);

  const imageData = ctx.getImageData(0, 0, targetWidth, targetHeight).data;
  const data = new Float32Array(3 * targetWidth * targetHeight);

  console.log(`Length of image data array: ${imageData.length}`);
  console.log(`Length of data array for tensor: ${data.length}`);

  // Reshape data into [channels, height, width] format
  for (let y = 0; y < targetHeight; y++) {
      for (let x = 0; x < targetWidth; x++) {
          let i = y * targetWidth + x;
          let j = 4 * i;
          data[i] = imageData[j] / 255.0; // Red channel
          data[targetHeight * targetWidth + i] = imageData[j + 1] / 255.0; // Green channel
          data[2 * targetHeight * targetWidth + i] = imageData[j + 2] / 255.0; // Blue channel
      }
  }

  console.log(`First few elements of the tensor data: ${data.slice(0, 10).toString()}`);


  return { data, width: targetWidth, height: targetHeight };
}





// async function chainMtcnnOutputs(image, pnetSession, rnetSession, onetSession) {
//   // Process image through P-Net
//   let pnetInput = preprocessImage(image, pNetInputSize, pNetInputSize); // Resize according to P-Net input size
//   let pnetOutput = await runMtcnnStage(pnetSession, pnetInput);
//   let pnetBoxes = processPnetOutput(pnetOutput); // Implement this to extract and process boxes from P-Net output

//   // Process image through R-Net
//   let rnetInput = prepareRnetInput(image, pnetBoxes); // Implement this to extract regions based on P-Net boxes and resize them for R-Net
//   let rnetOutput = await runMtcnnStage(rnetSession, rnetInput);
//   let rnetBoxes = processRnetOutput(rnetOutput); // Implement this to refine boxes based on R-Net output

//   // Process image through O-Net
//   let onetInput = prepareOnetInput(image, rnetBoxes); // Similar to R-Net, prepare input for O-Net based on R-Net boxes
//   let onetOutput = await runMtcnnStage(onetSession, onetInput);
//   let finalResults = processOnetOutput(onetOutput); // Extract final boxes and landmarks

//   return finalResults;
// }

function prepareOnetInput(image, rnetBoxes, onetInputSize) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  const onetInputs = [];

  rnetBoxes.forEach(box => {
      // Resize canvas for each box
      canvas.width = onetInputSize;
      canvas.height = onetInputSize;

      // Calculate the coordinates and size of the box
      const srcX = box.x1;
      const srcY = box.y1;
      const srcWidth = box.x2 - box.x1;
      const srcHeight = box.y2 - box.y1;

      // Draw the cropped region onto the canvas, resizing it to O-Net's expected input size
      ctx.drawImage(image, srcX, srcY, srcWidth, srcHeight, 0, 0, onetInputSize, onetInputSize);

      // Extract image data
      const imageData = ctx.getImageData(0, 0, onetInputSize, onetInputSize).data;
      const rgbData = [];

      for (let i = 0; i < imageData.length; i += 4) {
        // Extract RGB values (skip the Alpha channel)
        rgbData.push(imageData[i], imageData[i + 1], imageData[i + 2]);
      }
      // Normalize the pixel values and convert to Float32Array (if needed, based on your model's requirements)
      const normalizedData = Float32Array.from(rgbData, (v) => v / 255.0);

      onetInputs.push(normalizedData);
  });

  return onetInputs;
}

function processOnetOutput(onetOutput, originalBoxes) {
  // Assuming onetOutput contains final bounding boxes, scores, and facial landmarks
  // Example: { boxes: Float32Array, scores: Float32Array, landmarks: Float32Array }

  const boxes = onetOutput.boxes;
  const scores = onetOutput.scores;
  const landmarks = onetOutput.landmarks; // Assuming each landmark set contains 5 points (x1,y1,x2,y2,...,x5,y5)
  const threshold = 0.7; // Adjust based on your model and requirements

  let finalResults = [];

  for (let i = 0; i < scores.length; i++) {
      if (scores[i] > threshold) {
          const box = {
              x1: boxes[i * 4],
              y1: boxes[i * 4 + 1],
              x2: boxes[i * 4 + 2],
              y2: boxes[i * 4 + 3],
              score: scores[i]
          };

          const landmarkSet = [];
          for (let j = 0; j < 5; j++) {
              landmarkSet.push({
                  x: landmarks[i * 10 + j * 2],
                  y: landmarks[i * 10 + j * 2 + 1]
              });
          }

          finalResults.push({ box, landmarkSet });
      }
  }

  console.info("processOnetOutput boxes", boxes)
  console.info("processOnetOutput finalResults", finalResults)

  // Apply Non-Maximum Suppression (NMS)
  const nmsThreshold = 0.5; // Adjust based on your model
  let nmsResults = nonMaximumSuppression(finalResults.map(r => r.box), nmsThreshold);

  // Combine NMS results with landmarks
  let combinedResults = nmsResults.map((nmsBox, index) => {
      return { 
          box: nmsBox, 
          landmarks: finalResults[index].landmarkSet 
      };
  });

  return combinedResults;
}

function resizeImage(image, scale) {
    const width = Math.ceil(image.width * scale);
    const height = Math.ceil(image.height * scale);
    
    const canvas = document.createElement('canvas');
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

    return scales;
}


async function processImageAtScales(image, pnetSession, scales) {
    let allBoxes = [];

    for (let scale of scales) {
        let resizedCanvas = resizeImage(image, scale);
        let inputData = preprocessImageForPnet(resizedCanvas);
        let pnetOutput = await runPnet(pnetSession, inputData, resizedCanvas.width, resizedCanvas.height);
        console.info("ProcessImageAtAscales:: pnetOutput", pnetOutput)
        let boxes = processPnetOutput(pnetOutput, scale, resizedCanvas.width, resizedCanvas.height, 0.6);
        console.info("ProcessImageAtAscales:: boxes", boxes)

        allBoxes.push(...boxes);
    }
    console.info("processImageAtScales allBoxes", allBoxes);
    let rnetInputBoxes = nonMaximumSuppression(allBoxes, 0.5); // Adjust NMS threshold as needed
    console.info("processImageAtScales rnetInputBoxes", rnetInputBoxes)
    let rnetInput = prepareRnetInput(image, rnetInputBoxes, 24); // 24 is
    console.info("processImageAtScales rnetInput", rnetInput)
    const rnetSession = await loadMtcnnModel(rnetModelPath);
    let rnetBoxes = await runRnet(rnetSession, rnetInput);
    console.info("processImageAtScales rnetBoxes", rnetBoxes)
    // let rnetBoxes = processRnetOutp/ut(rnetOutput, rnetInputBoxes); // Implement this function
    
    // Apply NMS to R-Net results
    // let nmsRnetBoxes = nonMaximumSuppression(rnetBoxes, 0.5); // Adjust the NMS threshold as needed

    // Process image through O-Net
    let onetInput = prepareOnetInput(image, rnetBoxes, 48); // 48 is the input size for O-Net
    const onetSession = await loadMtcnnModel(onetModelPath);
    let onetOutput = await runOnet(onetSession, onetInput); // Implement runOnet similar to runPnet and runRnet
    console.info("processImageAtScales onetOutput", onetOutput)
    let finalBoxes = processOnetOutput(onetOutput); // Implement this function
    console.info("processImageAtScales finalBoxes", finalBoxes)
    return finalBoxes;
}

function generateBoundingBoxes(confidenceData, regData, scale, threshold) {
    const stride = 2;
    const cellsize = 12;
    const boundingBoxes = [];

    for (let y = 0; y < confidenceData.length; y++) {
        for (let x = 0; x < confidenceData[y].length; x++) {
            const score = confidenceData[y][x];
            if (score > threshold) {
                const reg = regData[y][x]; // [dx1, dy1, dx2, dy2]
                const boundingBox = {
                    x1: Math.max(0, Math.floor((stride * x + 1) / scale + reg[0] * cellsize)),
                    y1: Math.max(0, Math.floor((stride * y + 1) / scale + reg[1] * cellsize)),
                    x2: Math.min(originalWidth, Math.floor((stride * x + cellsize) / scale + reg[2] * cellsize)),
                    y2: Math.min(originalHeight, Math.floor((stride * y + cellsize) / scale + reg[3] * cellsize)),
                    score: score
                };
                boundingBoxes.push(boundingBox);
            }
        }
    }

    return boundingBoxes;
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
          w[j] = Math.max(0, minxx2[j] - maxxx1[j] + 1);
          h[j] = Math.max(0, minyy2[j] - maxyy1[j] + 1);
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
function processPnetOutput(pnetOutput, scale, originalWidth, originalHeight, threshold = 0.5) {
    const stride = 2;
    const cellsize = 12;

    // Assuming pnetOutput.conv2d_4 is the bounding box regression data
    // and pnetOutput.softmax is the confidence scores
    const { dims: regDims, data: regData } = pnetOutput.conv2d_4;
    const { dims: confDims, data: confData } = pnetOutput.softmax;
    const boundingBoxes = [];

    for (let y = 0; y < confDims[1]; y++) {
        for (let x = 0; x < confDims[2]; x++) {
            const scoreIndex = (y * confDims[2] + x) * 2 + 1; // Assuming 2nd value is the confidence score
            const score = confData[scoreIndex];
            if (score > 0.5) {
                const regIndex = (y * regDims[2] + x) * 4;
                const reg = regData.slice(regIndex, regIndex + 4); // [dx1, dy1, dx2, dy2]

                const boundingBox = {
                    x1: Math.max(0, Math.floor((stride * x + 1) / scale + reg[0] * originalWidth)),
                    y1: Math.max(0, Math.floor((stride * y + 1) / scale + reg[1] * originalHeight)),
                    x2: Math.min(originalWidth, Math.floor((stride * x + cellsize - 1) / scale + reg[2] * originalWidth)),
                    y2: Math.min(originalHeight, Math.floor((stride * y + cellsize - 1) / scale + reg[3] * originalHeight)),
                    score: score
                };

                boundingBoxes.push(boundingBox);
            }
        }
    }

    console.info("processPnetOutput boundingBoxes", boundingBoxes);
    let finalBoxes = nonMaximumSuppression(boundingBoxes, 0.5); // Adjust NMS threshold as needed
    console.info("processPnetOutput finalBoxes", finalBoxes);
    return finalBoxes;
}

    

function processRnetOutput(rnetOutput, pnetBoxes, threshold) {
    
    let refinedBoxes = [];

    for (let i = 0; i < rnetOutput.length; i++) {
        let refinement = rnetOutput[i];
        if (refinement.score > threshold) { // Threshold, can be adjusted
            let originalBox = pnetBoxes[i];
            let refinedBox = {
                x1: originalBox.x1 + refinement.x1 * (originalBox.x2 - originalBox.x1),
                y1: originalBox.y1 + refinement.y1 * (originalBox.y2 - originalBox.y1),
                x2: originalBox.x2 + refinement.x2 * (originalBox.x2 - originalBox.x1),
                y2: originalBox.y2 + refinement.y2 * (originalBox.y2 - originalBox.y1),
                score: refinement.score
            };
            refinedBoxes.push(refinedBox);
        }
    }
    console.info("processRnetOutput refinedBoxes", refinedBoxes);
    // Apply Non-Maximum Suppression (NMS) to reduce overlapping boxes
    let finalBoxes = nonMaximumSuppression(refinedBoxes, 0.5); // Adjust NMS threshold as needed
    console.info("processRnetOutput finalBoxes", finalBoxes);
    return finalBoxes;
}

function prepareRnetInput(image, pnetBoxes, rnetInputSize) {
    const rnetInputs = [];
    console.info("prepareRnetInput pnetBoxes", pnetBoxes.length, rnetInputSize)
    const canvas = document.createElement('canvas');
    canvas.width = rnetInputSize;
    canvas.height = rnetInputSize;
    const ctx = canvas.getContext('2d');
    pnetBoxes.forEach(box => {
        // Ensure coordinates and dimensions are integers
        const srcX = Math.round(box.x1);
        const srcY = Math.round(box.y1);
        const srcWidth = Math.round(box.x2 - box.x1);
        const srcHeight = Math.round(box.y2 - box.y1);

        console.info("prepareRnetInput src", rnetInputSize, srcX, srcY, srcWidth, srcHeight)

        // Draw the cropped region onto the canvas, resizing it to R-Net's expected input size
        ctx.clearRect(0, 0, rnetInputSize, rnetInputSize); // Clear previous image
        ctx.drawImage(image, srcX, srcY, srcWidth, srcHeight, 0, 0, rnetInputSize, rnetInputSize);

        // Extract image data
        const imageData = ctx.getImageData(0, 0, rnetInputSize, rnetInputSize).data;
        const rgbData = [];

        for (let i = 0; i < imageData.length; i += 4) {
          // Extract RGB values (skip the Alpha channel)
          rgbData.push(imageData[i], imageData[i + 1], imageData[i + 2]);
        }
        const normalizedData = Float32Array.from(rgbData, (v) => v / 255.0); // Normalize pixel values
        console.info("prepareRnetInput imageData", imageData.length, normalizedData.length);
        rnetInputs.push(normalizedData);
    });
    console.info("prepareRnetInput", rnetInputs, rnetInputSize, rnetInputSize)
    return rnetInputs;
}

async function runOnet(session, inputData) {
    let aggregatedResults = [];

    for (const data of inputData) {
        if (data.length !== 48 * 48 * 3) {
            throw new Error(`Invalid data length: expected ${48 * 48 * 3}, got ${data.length}`);
        }

        const inputTensor = new ort.Tensor('float32', data, [1, 48, 48, 3]);
        const feeds = { input_3: inputTensor };
        const output = await session.run(feeds);
        console.info("runOnet output", output)
        // Extract bounding box refinements, confidence scores, and facial landmarks
        const boxRefinements = output.dense_5.data;
        const confidenceScores = output.softmax_2.data;
        const landmarks = output.dense_6.data;  // Assuming this tensor contains facial landmarks

        let processedOutput = processOnetOutput(boxRefinements, confidenceScores, landmarks);
        aggregatedResults.push(...processedOutput);
    }

    return aggregatedResults;
}

function processOnetOutput(boxRefinements, confidenceScores, landmarks, threshold) {
    let refinedBoxes = [];

    for (let i = 0; i < boxRefinements.length / 4; i++) {
        let score = confidenceScores[i * 2 + 1]; // Confidence that the region contains a face
        console.info("processOnetOutput score", score)
        if (score > threshold) { // Example threshold
            let box = {
                x1: boxRefinements[i * 4],
                y1: boxRefinements[i * 4 + 1],
                x2: boxRefinements[i * 4 + 2],
                y2: boxRefinements[i * 4 + 3],
                score: score,
                landmarks: extractLandmarks(i, landmarks)  // Extract landmarks for this box
            };
            refinedBoxes.push(box);
        }
    }

    // Apply Non-Maximum Suppression (NMS) here if necessary
    return refinedBoxes;
}

function extractLandmarks(index, landmarks) {
    // Assuming each face has 5 landmarks, each with an x and y coordinate
    let landmarkSet = [];
    for (let j = 0; j < 5; j++) {
        landmarkSet.push({
            x: landmarks[index * 10 + j * 2],
            y: landmarks[index * 10 + j * 2 + 1]
        });
    }
    return landmarkSet;
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

      const finalBoxes = await processImageAtScales(image, pnetSession, scales);
        console.info("finalBoxes", finalBoxes)
      // Process through P-Net
    //   let pnetInputData = preprocessImageForPnet(image, scale);
    //   let pnetOutput = await runPnet(pnetSession, pnetInputData, Math.ceil(image.width * scale), Math.ceil(image.height * scale));
    //   let pnetRegions = processPnetOutput(pnetOutput, scale, image.width, image.height);


      // Prepare and process through R-Net
    //   let rnetInput = prepareRnetInput(image, pnetRegions, rnetInputSize); // rnetInputSize needs to be defined
    //   let rnetOutput = await runRnet(rnetSession, rnetInput);
    //   let rnetRegions = processRnetOutput(rnetOutput); // Implement this function to process R-Net output

      // // Prepare input for O-Net
      // const oNetInputSize = 48; // Adjust based on the input size required by O-Net
      // let onetInput = prepareOnetInput(image, rnetBoxes, oNetInputSize);

      // // Run O-Net
      // let onetOutput = await runMtcnnStage(onetSession, onetInput);
      // let finalResults = processOnetOutput(onetOutput, rnetBoxes);

      // console.log(finalResults);
  };
  image.onerror = (error) => {
      console.error("Error loading image:", error);
  };
}
const pnetModelPath = 'mtcnn_ort/pnet.onnx';
const rnetModelPath = 'mtcnn_ort/rnet.onnx';
const onetModelPath = 'mtcnn_ort/onet.onnx';

// Call this function with the path to your image and MTCNN model paths
processImageThroughMtcnn('t.jpeg', 0.5);
