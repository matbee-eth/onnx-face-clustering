
const rnetInputSize = 24; // R-Net expects 24x24 pixel input images
function prepareRnetInput(image, pnetBoxes, rnetInputSize) {
  console.info("prepareRnetInput", image.channels(), pnetBoxes, rnetInputSize)
  const rnetInputs = [];

  pnetBoxes.forEach((box, i) => {
    const croppedImage = cropImageOpenCV(image, box);
    const resizedImage = resizeImageOpenCV(croppedImage, rnetInputSize, rnetInputSize);
    const normalizedImage = normalizeImage(resizedImage, `js_rnetImage${i}.jpg`);
    displayMatOnCanvas(normalizedImage);
    const data = new Float32Array(normalizedImage.data32F);
    rnetInputs.push({ data, ...box, image: normalizedImage });

    croppedImage.delete(); // Clean up
  });

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

function processRnetOutput(rnetOutput, rnetInput, pnetBoxes, image, threshold) {
  const boxRefinements = rnetOutput.dense_2.data;
  const confidenceScores = rnetOutput.softmax_1.data;

  let refinedBoxes = [];
  for (let i = 0; i < boxRefinements.length / 4; i++) {
    let score = confidenceScores[i * 2 + 1]; // Assuming the second score is the face confidence
    console.info("processRnetOutput", i, score, score > threshold)
    if (score > threshold) { // Threshold for confidence score
      let originalBox = rnetInput[i]; // Get the original box from P-Net
      let refinement = {
        x1: boxRefinements[i * 4],
        y1: boxRefinements[i * 4 + 1],
        x2: boxRefinements[i * 4 + 2],
        y2: boxRefinements[i * 4 + 3]
      };
      // console.info("processRnetOutput original box Width and Height", originalBox.x2 - originalBox.x1, originalBox.y2 - originalBox.y1)

      const basicBox = bbreg(originalBox, refinement);
      basicBox.refinement = refinement;
      basicBox.inputBox = {...originalBox, "label": "RNInputBox"};
      // console.info("processRnetOutput basic box Width and Height", basicBox.x2 - basicBox.x1, basicBox.y2 - basicBox.y1)

      // let refinedBox = rerec(basicBox, image.cols, image.rows);
      // const paddedBoundingBox = padBoundingBoxes(refinedBox, image.cols, image.rows);

      // if (isValidBoundingBox(paddedBoundingBox, image.cols, image.rows)) {
      //   paddedBoundingBox.score = score;
      //   paddedBoundingBox.image = rnetInput[i].image;
      //   refinedBoxes.push(paddedBoundingBox);
      // } else if (isValidBoundingBox(refinedBox, image.cols, image.rows)) {
      //   refinedBoxes.push(refinedBox);
      // } else 
      if (isValidBoundingBox(basicBox, image.cols, image.rows)) {
        refinedBoxes.push(basicBox);
      }
       else {
        console.warn(`Refined box is malformed: x1=${basicBox.x1} ${refinement.x1}, y1=${basicBox.y1} ${refinement.y1}, x2=${basicBox.x2} ${refinement.x2}, y2=${basicBox.y2} ${refinement.y2}`);
      }
      // console.info("processRnetOutput Padding box Width and Height", paddedBoundingBox.x2 - paddedBoundingBox.x1, paddedBoundingBox.y2 - paddedBoundingBox.y1, "score", score, "refinedBox", refinedBox, "paddedBoundingBox", paddedBoundingBox, "originalBox", originalBox, "refinement", refinement)
    }
  }
  console.info("processRnetOutput boundingBoxes", refinedBoxes)
  // Apply Non-Maximum Suppression (NMS) here if necessary
  return refinedBoxes
}