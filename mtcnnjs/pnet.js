function preparePnetInput(image, scale) {
  const resizedCanvas = scaleImageOpenCV(image, scale);
  const normalizedImage = normalizeImage(resizedCanvas, `pnetImage${resizedCanvas.cols}x${resizedCanvas.rows}:${scale}.jpg`);
  return normalizedImage;
}

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
              // const squaredBoundingBox = rerec(basicBoundingBox, originalWidth, originalHeight);
              // const paddedBoundingBox = padBoundingBoxes(squaredBoundingBox, originalWidth, originalHeight);
              
              // if (isValidBoundingBox(paddedBoundingBox, originalWidth, originalHeight)) {
              //   paddedBoundingBox.score = score;
              //   boundingBoxes.push(paddedBoundingBox);
              // } else if (isValidBoundingBox(squaredBoundingBox, originalWidth, originalHeight)) {
              //   squaredBoundingBox.score = score;
              //   boundingBoxes.push(squaredBoundingBox);
              // } else if (isValidBoundingBox(basicBoundingBox, originalWidth, originalHeight)) {
              //   basicBoundingBox.score = score;
              //   boundingBoxes.push(basicBoundingBox);
              // }
            boundingBoxes.push({...basicBoundingBox, score})
          }
      }
  }
  console.info("processPnetOutput boundingBoxes", boundingBoxes.length)
  return boundingBoxes;
}

function calculateBoundingBox(x, y, reg, scale, originalWidth, originalHeight) {
  // console.info("calculateBoundingBox", x, y, scale, originalWidth, originalHeight)
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
    via: "calculateBoundingBox",
    x1, y1, x2, y2,
    scale
  };
}