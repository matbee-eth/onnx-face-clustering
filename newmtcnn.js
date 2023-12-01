// Assuming onnxruntime-web is already included in your environment

// 1. Loading the MTCNN ONNX Model
async function loadMtcnnModel(modelPath) {
  const session = await ort.InferenceSession.create(modelPath);
  return session;
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
  // console.log(`Creating tensor with dimensions: [1, 3, ${height}, ${width}]`);

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
    data[i * 3] = (imageData.data[i * 4 + 2] - 127.5) * 0.0078125;     // Blue
    data[i * 3 + 1] = (imageData.data[i * 4 + 1] - 127.5) * 0.0078125; // Green
    data[i * 3 + 2] = (imageData.data[i * 4] - 127.5) * 0.0078125;     // Red
  }

  return data;
}

function matToFloat32Array(mat) {
  let temp = new cv.Mat();

  // Check if the Mat has 4 channels (RGBA), and if so, convert to 3 channels (RGB)
  if (mat.channels() === 4) {
    cv.cvtColor(mat, temp, cv.COLOR_BGRA2BGR);
  } else {
    temp = mat.clone();
  }

  // Convert to 32FC3 format if not already
  if (temp.type() !== cv.CV_32FC3) {
    let converted = new cv.Mat();
    temp.convertTo(converted, cv.CV_32FC3);
    temp.delete(); // Delete the temp Mat
    temp = converted;
  }

  // Access the data and convert it to a Float32Array
  let array = new Float32Array(temp.data32F);

  return array;
}

function preprocessImageForPnet(canvas) {
  const ctx = canvas.getContext('2d');
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = new Float32Array(canvas.width * canvas.height * 3); // Channels last

  for (let i = 0; i < imageData.data.length / 4; i++) {
    // Adjust normalization to match Python's method
    data[i * 3] = (imageData.data[i * 4 + 2] - 127.5) * 0.0078125;     // Blue
    data[i * 3 + 1] = (imageData.data[i * 4 + 1] - 127.5) * 0.0078125; // Green
    data[i * 3 + 2] = (imageData.data[i * 4] - 127.5) * 0.0078125;     // Red
  }

  return data;
}

function displayImageFromFloat32Array(float32Array, width, height, canvas) {
  canvas = canvas ?? document.createElement('canvas');
  document.body.appendChild(canvas);
  // Create a cv.Mat from the Float32Array
  let mat = cv.matFromArray(height, width, cv.CV_32FC3, float32Array);

  // Convert the float image to a uchar image for displaying purposes
  let matToDisplay = new cv.Mat();
  mat.convertTo(matToDisplay, cv.CV_8UC3, 255); // Scale back up to uchar range

  // Display the image on the canvas
  cv.imshow(canvas, matToDisplay);

  // Clean up
  mat.delete();
}

function resizeImageOpenCV(image, scale) {
  const mat = cv.imread(image);
  const width = Math.ceil(mat.cols * scale);
  const height = Math.ceil(mat.rows * scale);
  const dsize = new cv.Size(width, height);
  const resized = new cv.Mat();
  cv.resize(mat, resized, dsize, 0, 0, cv.INTER_AREA);
  return resized;
}

function normalizeImage(srcMat) {
  let dst = new cv.Mat();
  cv.cvtColor(srcMat, dst, cv.COLOR_BGRA2BGR);

  let dstFloat = new cv.Mat();
  dst.convertTo(dstFloat, cv.CV_32FC3); // Convert to float without scaling

  // Normalize the image's pixels to match the Python code
  let meanScalar = new cv.Scalar(127.5, 127.5, 127.5);
  let stdScalar = new cv.Scalar(1 / 0.0078125, 1 / 0.0078125, 1 / 0.0078125);
  // Create a cv.Mat filled with the scalar value for subtraction
  let meanScalarMat = new cv.Mat(dstFloat.rows, dstFloat.cols, dstFloat.type(), meanScalar);

  // Perform the subtraction
  cv.subtract(dstFloat, meanScalarMat, dstFloat);
  meanScalarMat.delete(); // Remember to delete the temporary Mats to avoid memory leaks

  // Create a cv.Mat filled with the scalar value for division
  let stdScalarMat = new cv.Mat(dstFloat.rows, dstFloat.cols, dstFloat.type(), stdScalar);

  // Perform the division
  cv.divide(dstFloat, stdScalarMat, dstFloat);
  stdScalarMat.delete(); // Clean up

  return dstFloat;
}

function preprocessImageForPnetOpenCV(image, scale) {
  let src = cv.imread(image);
  console.info("preprocessImageForPnetOpenCV", image.width, image.height, scale, Math.ceil(src.cols * scale), Math.ceil(src.rows * scale))

  let dst = resizeImageOpenCV(image, scale);
  let dstFloat = normalizeImage(dst);

  const canvas = document.createElement('canvas');
  document.body.appendChild(canvas);
  cv.imshow(canvas, dst);
  displayImageFromFloat32Array(matToFloat32Array(dstFloat), dstFloat.cols, dstFloat.rows);

  src.delete();
  dst.delete();

  return dstFloat;
}

function cropAndResizeImageWithBoundingBox(srcMat, bbox, targetSize) {
  // Ensure the bounding box coordinates are within the dimensions of the source image
  const x1 = Math.max(0, bbox.x1);
  const y1 = Math.max(0, bbox.y1);
  const x2 = Math.min(bbox.x2, srcMat.cols);
  const y2 = Math.min(bbox.y2, srcMat.rows);

  // Calculate the width and height of the bounding box
  const width = x2 - x1;
  const height = y2 - y1;

  // Create a rectangle for the bounding box
  let rect = new cv.Rect(x1, y1, width, height);

  // Crop the image to the bounding box
  let cropped = srcMat.roi(rect);

  // Resize the cropped image to the target size for RNet or ONet
  let dsize = new cv.Size(targetSize, targetSize);
  let resizedCropped = new cv.Mat();
  cv.resize(cropped, resizedCropped, dsize, 0, 0, cv.INTER_AREA);

  let dstFloat = new cv.Mat();
  resizedCropped.convertTo(dstFloat, cv.CV_32FC3); // Convert to float without scaling
  cv.cvtColor(resizedCropped, dstFloat, cv.COLOR_BGRA2BGR);

  // // Normalize the image's pixels to match the Python code
  // let meanScalar = new cv.Scalar(127.5, 127.5, 127.5);
  // let stdScalar = new cv.Scalar(1 / 0.0078125, 1 / 0.0078125, 1 / 0.0078125);
  // // Create a cv.Mat filled with the scalar value for subtraction
  // let meanScalarMat = new cv.Mat(dstFloat.rows, dstFloat.cols, dstFloat.type(), meanScalar);

  // // Perform the subtraction
  // cv.subtract(dstFloat, meanScalarMat, dstFloat);
  // meanScalarMat.delete(); // Remember to delete the temporary Mats to avoid memory leaks

  // // Create a cv.Mat filled with the scalar value for division
  // let stdScalarMat = new cv.Mat(dstFloat.rows, dstFloat.cols, dstFloat.type(), stdScalar);

  // // Perform the division
  // cv.divide(dstFloat, stdScalarMat, dstFloat);
  // stdScalarMat.delete(); // Clean up

  cropped.delete(); // Clean up the cropped Mat as it's no longer needed
  resizedCropped.delete();
  
  return dstFloat; // Return the cropped and resized image as a Mat object
}

// function preprocessImageForRNetOrONet(image, scale, bbox, targetSize) {
//   // Read the image from canvas
//   let src = cv.imread(image);

//   // Calculate the size of the bounding box
//   const x1 = Math.round(bbox.x1 * scale);
//   const y1 = Math.round(bbox.y1 * scale);
//   const x2 = Math.round(bbox.x2 * scale);
//   const y2 = Math.round(bbox.y2 * scale);
//   const width = x2 - x1;
//   const height = y2 - y1;

//   // Crop the image to the bounding box
//   let rect = new cv.Rect(x1, y1, width, height);
//   let cropped = new cv.Mat();
//   cropped = src.roi(rect);

//   // Resize the cropped image to the target size for RNet or ONet
//   let dsize = new cv.Size(targetSize, targetSize);
//   let resizedCropped = new cv.Mat();
//   const canvas = document.createElement('canvas');
//   document.body.appendChild(canvas);
//   // cv.imshow(canvas, cropped);
//   cv.resize(cropped, resizedCropped, dsize, 0, 0, cv.INTER_AREA)
//   let recolored = new cv.Mat();
//   cv.cvtColor(resizedCropped, recolored, cv.COLOR_RGBA2BGR);
//   // Convert the resized cropped image to float32
//   const canvas2 = document.createElement('canvas');
//   document.body.appendChild(canvas2);
//   // cv.imshow(canvas2, recolored);
//   cv.resize(cropped, resizedCropped, dsize, 0, 0, cv.INTER_AREA)
//   // Convert the resized cropped image to float32
//   const canvas3 = document.createElement('canvas');
//   document.body.appendChild(canvas3);
//   // cv.imshow(canvas3, cropped);
//   let dstFloat = new cv.Mat();
//   recolored.convertTo(dstFloat, cv.CV_32FC3, 1 / 255.0);
//   const canvas4 = document.createElement('canvas');
//   document.body.appendChild(canvas4);
//   cv.imshow(canvas4, dstFloat);

//   // Normalize the pixel values (assuming mean subtraction and scaling)
//   let meanScalar = new cv.Scalar(127.5, 127.5, 127.5);
//   let stdScalar = new cv.Scalar(128.0, 128.0, 128.0);
//   let meanScalarMat = new cv.Mat(dstFloat.rows, dstFloat.cols, dstFloat.type(), meanScalar);
//   let stdScalarMat = new cv.Mat(dstFloat.rows, dstFloat.cols, dstFloat.type(), stdScalar);
//   cv.subtract(dstFloat, meanScalarMat, dstFloat);
//   cv.divide(dstFloat, stdScalarMat, dstFloat);

//   // Clean up
//   src.delete();
//   cropped.delete();
//   resizedCropped.delete();
//   meanScalarMat.delete();
//   stdScalarMat.delete();

//   // Return the preprocessed image as a Mat object
//   return dstFloat; // Ready for further processing or input into RNet or ONet
// }


async function runPnet(scaledCvMat) {
  const session = await loadMtcnnModel(pnetModelPath);
  // scaledCvMat is a cv.Mat object
  let rows = scaledCvMat.rows;  // Replace with the actual height of the source image
  let cols = scaledCvMat.cols;   // Replace with the actual width of the source image
  let channels = 3;   // For a typical color image

  const inputAsMat = matToFloat32Array(scaledCvMat);
  console.info("runPnet", inputAsMat.length, rows, cols, channels)
  if (inputAsMat.length !== rows * cols * channels) {
    throw new Error(`Data length (${inputAsMat.length}) does not match expected size (${rows * cols * channels}).`);
  }

  const inputTensor = new ort.Tensor('float32', inputAsMat, [1, rows, cols, 3]);
  const feeds = { input_1: inputTensor }; // Use 'input_1' as per model's input name

  const output = await session.run(feeds);
  // console.info("runPnet", output);
  return { output, image: inputAsMat };
}

function calculateBoundingBoxTest(data, originalWidth, originalHeight) {
  return data.map((item, index) => {
      // Simulate x and y based on the index and feature map size
      const x = item.x;
      const y = item.y;

      return calculateBoundingBox(x, y, [
          item.regX1, item.regY1, item.regX2, item.regY2
      ], item.scale, originalWidth, originalHeight, item.score);
  });
}

function calculateBoundingBox(x, y, reg, scale, originalWidth, originalHeight) {
  console.info("calculateBoundingBox", x, y, reg, scale, originalWidth, originalHeight)
  const stride = 2;
  const cellsize = 12;

  // Apply the regression values to the feature map coordinates
  const offsetX1 = reg[0] * cellsize;
  const offsetY1 = reg[1] * cellsize;
  const offsetX2 = reg[2] * cellsize;
  const offsetY2 = reg[3] * cellsize;

  // Calculate the bounding box coordinates
  let x1 = Math.round((stride * x + offsetX1) / scale);
  let y1 = Math.round((stride * y + offsetY1) / scale);
  let x2 = Math.round((stride * x + cellsize + offsetX2) / scale);
  let y2 = Math.round((stride * y + cellsize + offsetY2) / scale);

  // Clamp values within the original image dimensions
  x1 = Math.max(0, Math.min(x1, originalWidth));
  y1 = Math.max(0, Math.min(y1, originalHeight));
  x2 = Math.max(x1, Math.min(x2, originalWidth));
  y2 = Math.max(y1, Math.min(y2, originalHeight));

  return {
    x1, y1, x2, y2,
    scale
  };
}

function processPnetOutput(pnetOutput, scale, originalImage, threshold) {
  const originalWidth = originalImage.cols;
  const originalHeight = originalImage.rows;
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
              const basicBoundingBox = calculateBoundingBox(x, y, reg, scale, originalWidth, originalHeight);
              const squaredBoundingBox = rerec(basicBoundingBox, originalWidth, originalHeight);
              if (squaredBoundingBox.x2 - squaredBoundingBox.x1 == 0 || squaredBoundingBox.y2 - squaredBoundingBox.y1 == 0) {
                debugger;
              }
              squaredBoundingBox.score = score;
              boundingBoxes.push(squaredBoundingBox);
              console.info('Bounding box:', squaredBoundingBox.x2 - squaredBoundingBox.x1, squaredBoundingBox.y2 - squaredBoundingBox.y1, squaredBoundingBox.score);
              console.info(squaredBoundingBox);
          }
      }
  }

  return boundingBoxes;
}



/** RNET */
function prepareRnetInput(image, pnetBoxes, rnetInputSize) {
  console.info("prepareRnetInput", image.channels(), pnetBoxes, rnetInputSize)
  const rnetInputs = [];
  const canvas = document.createElement('canvas');
  document.body.appendChild(canvas);
  let matToDisplay = new cv.Mat();
  image.convertTo(matToDisplay, cv.CV_8UC3, 255); // Scale back up to uchar range

  pnetBoxes.forEach(box => {
    const croppedImage = cropAndResizeImageWithBoundingBox(image, box, rnetInputSize);
    console.info("prepareRnetInput", croppedImage.channels(), pnetBoxes, rnetInputSize, croppedImage.cols, croppedImage.rows)
    const data = matToFloat32Array(croppedImage);
    console.info("prepareRnetInput", croppedImage.channels(), pnetBoxes, rnetInputSize, croppedImage.cols, croppedImage.rows)
    const newCanvas = document.createElement('canvas');
    document.body.appendChild(newCanvas);
    displayImageFromFloat32Array(data, croppedImage.cols, croppedImage.rows, newCanvas);
    rnetInputs.push({ data, originalImage: box.originalImage, ...box });
    drawBoundingBoxOnImage(matToDisplay, box);

    croppedImage.delete(); // Clean up
  });

  cv.imshow(canvas, matToDisplay);
  // console.info("prepareRnetInput rnetInputs", rnetInputs);
  return rnetInputs;
}

async function runRnet(inputData) {
  const session = await loadMtcnnModel(rnetModelPath);

  const batchSize = inputData.length;
  const height = rnetInputSize;  // Height for RNet
  const width = rnetInputSize;   // Width for RNet
  const channels = 3; // Number of channels (RGB)

  // Create a single array to hold all batched data
  const batchedData = new Float32Array(batchSize * height * width * channels);

  // Batch the data (flatten and concatenate)
  inputData.forEach((imageData, index) => {
    batchedData.set(imageData.data, index * height * width * channels);
  });

  // Create the input tensor for ONNX Runtime
  const inputTensor = new ort.Tensor('float32', batchedData, [batchSize, height, width, channels]);

  // Run the model
  const feeds = { 'input_2': inputTensor }; // Use the correct input name for RNet
  const rnetOutput = await session.run(feeds);

  return rnetOutput;
}

function rerec(bbox, maxImageWidth, maxImageHeight) {
  let h = bbox.y2 - bbox.y1;
  let w = bbox.x2 - bbox.x1;
  let l = Math.max(w, h);
  bbox.y1 = bbox.y1 + h * 0.5 - l * 0.5;
  bbox.x1 = bbox.x1 + w * 0.5 - l * 0.5;
  bbox.y2 = bbox.y1 + l;
  bbox.x2 = bbox.x1 + l;
  bbox.x1 = Math.max(0, Math.min(bbox.x1, maxImageWidth));
  bbox.y1 = Math.max(0, Math.min(bbox.y1, maxImageHeight));
  bbox.x2 = Math.min(maxImageWidth, bbox.x2);
  bbox.y2 = Math.min(maxImageHeight, bbox.y2);
  return bbox;
}

function bbreg(originalBox, refinement) {
  let w = originalBox.x2 - originalBox.x1 + 1;
  let h = originalBox.y2 - originalBox.y1 + 1;

  let b1 = originalBox.x1 + refinement.x1 * w;
  let b2 = originalBox.y1 + refinement.y1 * h;
  let b3 = originalBox.x2 + refinement.x2 * w;
  let b4 = originalBox.y2 + refinement.y2 * h;
  return {
    x1: b1,
    y1: b2,
    x2: b3,
    y2: b4,
    score: originalBox.score
  };
}

function processRnetOutput(rnetOutput, rnetBoxes, image) {
  const boxRefinements = rnetOutput.dense_2.data;
  const confidenceScores = rnetOutput.softmax_1.data;

  let refinedBoxes = [];
  for (let i = 0; i < boxRefinements.length / 4; i++) {
    let score = confidenceScores[i * 2 + 1]; // Assuming the second score is the face confidence
    console.info("processRnetOutput", score, boxRefinements[i * 4], boxRefinements[i * 4 + 1], boxRefinements[i * 4 + 2], boxRefinements[i * 4 + 3])
    if (score > 0.5) { // Threshold for confidence score
      let originalBox = rnetBoxes[i]; // Get the original box from P-Net
      let refinement = {
        x1: boxRefinements[i * 4],
        y1: boxRefinements[i * 4 + 1],
        x2: boxRefinements[i * 4 + 2],
        y2: boxRefinements[i * 4 + 3]
      };
      let refinedBox = rerec(bbreg(originalBox, refinement), image.cols, image.rows);
      // console.info("processRnetOutput originalBox", "x1", originalBox.x1, "y1", originalBox.y1, "x2", originalBox.x2, "y2", originalBox.y2)
      // console.info("processRnetOutput refinedBox", "x1", refinedBox.x1, "y1", refinedBox.y1, "x2", refinedBox.x2, "y2", refinedBox.y2)

      if (refinedBox.x2 < refinedBox.x1 || refinedBox.y2 < refinedBox.y1) {
        console.warn(`Refined box is malformed: x1=${refinedBox.x1} ${refinement.x1}, y1=${refinedBox.y1} ${refinement.y1}, x2=${refinedBox.x2} ${refinement.x2}, y2=${refinedBox.y2} ${refinement.y2}`);
        continue;
      }


      // Check if the refined box intersects with the original box
      let scaledBox = {
        x1: Math.min(Math.max(0, refinedBox.x1), image.cols),
        y1: Math.min(Math.max(0, refinedBox.y1), image.rows),
        x2: Math.min(refinedBox.x2, image.cols),
        y2: Math.min(refinedBox.y2, image.rows),
        score,
        scale: originalBox.scale,
        refinement
      };
      // console.info("processRnetOutput scaledBox", "x1", scaledBox.x1, "y1", scaledBox.y1, "x2", scaledBox.x2, "y2", scaledBox.y2)
      if (score > 0.6) {
        // console.info("processRnetBoxesAndScores Scaled refined box", scaledBox)
      }
      refinedBoxes.push(scaledBox);
    }
  }
  console.info("processRnetOutput refinedBoxes", refinedBoxes)
  // Apply Non-Maximum Suppression (NMS) here if necessary
  return nonMaximumSuppression(refinedBoxes, 0.5, 'Union');
}

/** END OF RNET */

function resizeImage(image, scale) {
  const width = Math.ceil(image.width * scale);
  const height = Math.ceil(image.height * scale);

  const canvas = document.createElement('canvas');
  canvas.id = `pnetImage${width}x${height}:${scale}`;

  document.body.appendChild(canvas)
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext('2d');

  ctx.drawImage(image, 0, 0, width, height);
  console.info("resizeImage", scale, width, height)
  return canvas;
}

function computeScalePyramid(minFaceSize, width, height) {
  console.info("computeScalePyramid: ", minFaceSize, width, height)
  const m = 12 / minFaceSize;
  const minLayer = Math.min(width, height) * m;

  let scales = [];
  let factor = 0.709; // Scale factor between subsequent scales; adjust as needed
  let scale = m;

  while (minLayer * scale > 12) {
    scales.push(scale);
    scale *= factor;
  }
  console.info("computeScalePyramid scales", scales)
  return scales;
}

async function processImageAtScales(image, scales) {
  let allBoxes = [];
  scales = [0.6]
  const originalImageAsMat = preprocessImageForPnetOpenCV(image, 1)
  // console.info("ProcessImageAtAscales:: scales", scales)
  for (let scale of scales) {
    // let resizedCanvas = resizeImage(image, scale);
    // let inputData = preprocessImageForPnet(resizedCanvas);
    let inputData = resizeImageOpenCV(image, scale);
    // const inputData = preprocessImageForPnetOpenCV(image, scale)
    // previewOnCanvas(inputData, inputData.cols, inputData.rows)
    let { output: pnetOutput } = await runPnet(inputData, originalImageAsMat.cols, originalImageAsMat.rows);
    // console.info("ProcessImageAtAscales:: pnetOutput", scale, im)inputData.cols, inputData.rows
    // console.info(`Scaled image dimensions: width=${inputData.rows}, height=${inputData.cols}, original width=${image.width}, original height=${image.height}`);

    let boxes = processPnetOutput(pnetOutput, scale, originalImageAsMat, 0.6);
    // console.info("ProcessImageAtAscales:: boxes", scale, boxes)


    allBoxes.push(...boxes);
  }
  allBoxes = nonMaximumSuppression(allBoxes, 0.5, 'Union', 'PNET');
  console.info("processImageAtScales allBoxes", allBoxes);
  let rnetInput = prepareRnetInput(cv.imread(image), allBoxes, 24); // 24 is
  console.info("processImageAtScales rnetInput", rnetInput)
  let rnetOutput = await runRnet(rnetInput);
  let rnetBoxes = await processRnetOutput(rnetOutput, rnetInput, originalImageAsMat);
  console.info("processImageAtScales rnetBoxes", rnetBoxes)
  // Process image through O-Net
  let onetInput = prepareOnetInput(cv.imread(image), rnetBoxes, 48); // 48 is the input size for O-Net
  let onetOutput = await runOnet(onetInput);
  let onetBoxes = processOnetOutput(onetOutput, rnetBoxes, 0.7, originalImageAsMat);

  console.info("processImageAtScales onetBoxes", onetBoxes)
  // const pnetcanvas = document.createElement('canvas');
  // allBoxes.forEach(box => {
  //   const newMat = originalImageAsMat.clone();
  //   drawBoundingBoxOnImage(newMat, box);
  //   cv.imshow(pnetcanvas, newMat);
  // });
  // document.body.appendChild(pnetcanvas);
  // const rnetcanvas = document.createElement('canvas');
  // rnetBoxes.forEach(box => {
  //   const newMat = originalImageAsMat.clone();
  //   drawBoundingBoxOnImage(newMat, box);
  //   cv.imshow(rnetcanvas, newMat);
  // });
  // document.body.appendChild(rnetcanvas);
  // onetOutput.forEach((box) => {
  //   const newMat = originalImageAsMat.clone();
  //   drawBoundingBoxOnImage(newMat, box);
  //   const canvas = document.createElement('canvas');
  //   document.body.appendChild(canvas);
  //   cv.imshow(canvas, newMat);
  //   drawLandmarks(canvas, box.landmarks, box.score, box.scale);
  // });
  return onetOutput;
}

function nonMaximumSuppression(boxes, threshold, method = 'Union', logKey) {
  if (boxes.length === 0) {
    return [];
  }

  // Calculate the area of each box
  const areas = boxes.map(box => (box.x2 - box.x1) * (box.y2 - box.y1));

  // Sort the boxes by score in descending order
  let sortedIndices = boxes.map((_, index) => index)
    .sort((a, b) => boxes[b].score - boxes[a].score);

  const pick = [];

  while (sortedIndices.length > 0) {
    const currentIdx = sortedIndices.pop();
    const currentBox = boxes[currentIdx];
    pick.push(currentBox);

    sortedIndices = sortedIndices.filter(idx => {
      const box = boxes[idx];

      const xx1 = Math.max(currentBox.x1, box.x1);
      const yy1 = Math.max(currentBox.y1, box.y1);
      const xx2 = Math.min(currentBox.x2, box.x2);
      const yy2 = Math.min(currentBox.y2, box.y2);

      const w = Math.max(0, xx2 - xx1);
      const h = Math.max(0, yy2 - yy1);
      const inter = w * h;

      const o = (method === 'Min')
        ? inter / Math.min(areas[currentIdx], areas[idx])
        : inter / (areas[currentIdx] + areas[idx] - inter);

      // Log IoU values for debugging
      // console.log(`${logKey ?? ''} IoU between box ${currentIdx} and box ${idx} = ${o}`);

      return o <= threshold;
    });
  }

  return pick;
}

/** ONET */

function prepareOnetInput(image, rnetBoxes, onetInputSize) {
  const rnetInputs = [];
  const canvas = document.createElement('canvas');
  document.body.appendChild(canvas);
  let matToDisplay = new cv.Mat();
  image.convertTo(matToDisplay, cv.CV_8UC3, 255); // Scale back up to uchar range

  rnetBoxes.forEach(box => {
    const croppedImage = cropAndResizeImageWithBoundingBox(image, box, onetInputSize);
    console.info("prepareOnetInput", croppedImage.channels(), rnetBoxes, rnetInputSize)

    const data = matToFloat32Array(croppedImage);
    const newCanvas = document.createElement('canvas');
    document.body.appendChild(newCanvas);
    displayImageFromFloat32Array(data, croppedImage.cols, croppedImage.rows, newCanvas);
    rnetInputs.push({ data, ...box });
    drawBoundingBoxOnImage(matToDisplay, box);

    croppedImage.delete(); // Clean up
  });

  cv.imshow(canvas, matToDisplay);
  // console.info("prepareRnetInput rnetInputs", rnetInputs);
  return rnetInputs;
}

async function runOnet(inputData) {
  const session = await loadMtcnnModel(onetModelPath);

  const batchSize = inputData.length;
  const height = onetInputSize;  // Height for RNet
  const width = onetInputSize;   // Width for RNet
  const channels = 3; // Number of channels (RGB)

  // Create a single array to hold all batched data
  const batchedData = new Float32Array(batchSize * height * width * channels);

  // Batch the data (flatten and concatenate)
  inputData.forEach((imageData, index) => {
    batchedData.set(imageData.data, index * height * width * channels);
  });

  // Create the input tensor for ONNX Runtime
  const inputTensor = new ort.Tensor('float32', batchedData, [batchSize, height, width, channels]);

  // Run the model
  const feeds = { 'input_3': inputTensor }; // Use the correct input name for RNet
  const rnetOutput = await session.run(feeds);

  return rnetOutput;
}

const onetInputSize = 48; // O-Net expects 48x48 pixel input images
function processOnetOutput(OnetOutput, rnetBoxes, threshold, image) {
  // console.info("processOnetOutput", OnetOutput, rnetBox, threshold, originalWidth, originalHeight)
  let refinedBoxes = [];
  const boxRefinements = OnetOutput.dense_5.data;
  const confidenceScores = OnetOutput.softmax_2.data;
  const landmarks = OnetOutput.dense_6.data;

  for (let i = 0; i < boxRefinements.length / 4; i++) {
    let score = confidenceScores[i * 2 + 1]; // Assuming index 'i * 2 + 1' is the face confidence score
    console.info("processOnetOutput", i, score, boxRefinements[i * 4], boxRefinements[i * 4 + 1], boxRefinements[i * 4 + 2], boxRefinements[i * 4 + 3])
    if (score > 0.5) {
      let originalBox = rnetBoxes[i];
      let refinement = {
        x1: boxRefinements[i * 4],
        y1: boxRefinements[i * 4 + 1],
        x2: boxRefinements[i * 4 + 2],
        y2: boxRefinements[i * 4 + 3]
      };


      let refinedBox = bbreg(originalBox, refinement);

      // console.info("processOnetOutput originalBox", "x1", originalBox.x1, "y1", originalBox.y1, "x2", originalBox.x2, "y2", originalBox.y2)

      // console.info("processOnetOutput refinedBox", "x1", refinedBox.x1, refinement.x1, "y1", refinedBox.y1, refinement.y1, "x2", refinedBox.x2, refinement.x2, "y2", refinedBox.y2, refinement.y2)
      // console.info("processOnetOutput image.cols", image.cols, image.rows)
      // Transform landmarks from local coordinates to the coordinates of the original image
      let transformedLandmarks = extractLandmarks(i, landmarks, refinedBox);
      refinedBox.landmarks = transformedLandmarks;
      refinedBox.originalLandmarks = landmarks;
      let scaledBox = {
        x1: Math.min(Math.max(0, refinedBox.x1), image.cols),
        y1: Math.min(Math.max(0, refinedBox.y1), image.rows),
        x2: Math.min(refinedBox.x2, image.cols),
        y2: Math.min(refinedBox.y2, image.rows),
        originalImage: originalBox.originalImage,
        score,
        scale: originalBox.scale,
        refinement,
        landmarks: transformedLandmarks
      };
      refinedBoxes.push(scaledBox);
    }
  }
  // console.info("processOnetOutput Before parsing refined boxes:", refinedBoxes);
  // Apply Non-Maximum Suppression (NMS) to filter overlapping boxes
  const newRefined = nonMaximumSuppression(refinedBoxes, 0.5, 'Union', 'ONET');
  // console.info("processOnetOutput After parsing refined boxes:", newRefined);
  return newRefined;
}

function drawBoundingBoxOnImage(srcMat, bbox, color = [255, 0, 0, 255], thickness = 2) {
  // Convert color to a cv.Scalar
  let rectangleColor = new cv.Scalar(...color);

  // Create a cv.Point for the top-left and bottom-right corners of the bounding box
  let pt1 = new cv.Point(bbox.x1, bbox.y1);
  let pt2 = new cv.Point(bbox.x2, bbox.y2);

  // Draw the rectangle on the source image
  cv.rectangle(srcMat, pt1, pt2, rectangleColor, thickness);
  console.info("drawing bounding box of width", bbox.x2 - bbox.x1, "height", bbox.y2 - bbox.y1, bbox)

  // The source image now has the bounding box drawn on it
  return srcMat;
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
const minFaceSize = 20; // Adjust based on the smallest face size you want to detect

async function processImageThroughMtcnn(image, scale) {
  // const image = new Image();
  // image.src = imageSrc;
  // console.info("processImageThroughMtcnn", image)
  // image.onload = async () => {
  // console.info("processImageThroughMtcnn: pnetSession", image)

  // const onetSession = await loadMtcnnModel(onetModelPath);
  const imageWidth = image.width;
  const imageHeight = image.height;

  const scales = computeScalePyramid(minFaceSize, imageWidth, imageHeight);

  const finalBoxes = (await processImageAtScales(image, scales));
  console.info("finalBoxes", finalBoxes)
  return finalBoxes;
  // };
  // image.onerror = (error) => {
  //   console.error("Error loading image:", error);
  // };
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
    // console.info('landmark', landmark);
    ctx.beginPath();
    ctx.arc(landmark.x, landmark.y, 3, 0, 2 * Math.PI);
    ctx.fillStyle = colors[i % colors.length];
    ctx.fill();
    ctx.fillText(`Score: ${score.toFixed(2)}, Landmark: ${map[i]}`, landmark.x, landmark.y - 10);
  }
}

function drawImage(canvas, image) {
  canvas.width = image.width;
  canvas.height = image.height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
}

function previewOnCanvas(data, width, height, canvas) {
  // Create a new canvas context for the preview if one is not provided
  if (!canvas) {
    canvas = document.createElement('canvas');
    document.body.appendChild(canvas);
  }
  canvas.width = width;
  canvas.height = height;

  const ctx = canvas.getContext('2d');
  const imageData = ctx.createImageData(width, height);

  for (let i = 0; i < data.length / 3; i++) {
    // Reverse normalization and convert BGR back to RGB
    imageData.data[i * 4] = Math.round((data[i * 3] / 0.0078125) + 127.5);     // B
    imageData.data[i * 4 + 1] = Math.round((data[i * 3 + 1] / 0.0078125) + 127.5); // G
    imageData.data[i * 4 + 2] = Math.round((data[i * 3 + 2] / 0.0078125) + 127.5); // R
    imageData.data[i * 4 + 3] = 255; // Alpha channel
  }

  ctx.putImageData(imageData, 0, 0);
}

// Convert a canvas to a Mat
function canvasToMat(canvas) {
  let img = cv.imread(canvas);
  // Now you can process the image img using OpenCV.js functions
  return img;
}


// Call this function with the path to your image and MTCNN model paths