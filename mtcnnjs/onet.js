
const onetInputSize = 48; // O-Net expects 48x48 pixel input images
function prepareOnetInput(image, rnetBoxes, onetInputSize) {
  const rnetInputs = [];

  rnetBoxes.forEach(box => {
    const croppedImage = cropImageOpenCV(image, box);
    const resizedImage = resizeImageOpenCV(croppedImage, onetInputSize, onetInputSize);
    const normalizedImage = normalizeImage(resizedImage);
    const data = new Float32Array(normalizedImage.data32F);
    displayMatOnCanvas(normalizedImage);

    rnetInputs.push({ data, ...box, image: normalizedImage });
    // console.info(`Cropped image dimensions: width=${croppedImage.cols}, height=${croppedImage.rows} channel=${croppedImage.channels()}`);
    // console.info(`Resized image dimensions: width=${resizedImage.cols}, height=${resizedImage.rows} channel=${resizedImage.channels()}`);
    // console.info(`Normalized image dimensions: width=${normalizedImage.cols}, height=${normalizedImage.rows} channel=${normalizedImage.channels()}`);

    croppedImage.delete(); // Clean up
  });

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

function processOnetOutput(OnetOutput, onetInput, threshold) {
  console.info("OnetOutput", OnetOutput)
  let refinedBoxes = [];
  const { dims: regDims, data: regData } = OnetOutput.dense_5;
  const { dims: confDims, data: confidenceScores } = OnetOutput.softmax_2;
  const { dims: landmarksDims, data: landmarksData } = OnetOutput.dense_6;

  // Assuming dims[0] is the number of boxes
  for (let i = 0; i < regDims[0]; i++) {
    const score = confidenceScores[i * 2 + 1]; // confidence score index
    const originalBox = onetInput[i];
    console.info("processOnetOutput", originalBox, score, score > threshold)
    if (score > threshold) {
      const refinement = {
        x1: regData[i * 4],
        y1: regData[i * 4 + 1],
        x2: regData[i * 4 + 2],
        y2: regData[i * 4 + 3]
      };

      const refinedBox = bbreg(originalBox, refinement);
      const landmarkPoints = [];

      for (let j = 0; j < 5; j++) {
        const lx = landmarksData[i * 10 + j];
        const ly = landmarksData[i * 10 + j + 5];
        landmarkPoints.push({ x: lx, y: ly });
      }

      refinedBox.landmarks = landmarkPoints;
      refinedBox.score = score;
      refinedBox.refinement = refinement;
      refinedBox.inputBox = {...originalBox, "label": "ONetInputBox"};
      refinedBoxes.push(refinedBox);
    }
  }

  return refinedBoxes;
}