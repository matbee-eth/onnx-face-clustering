// Assuming onnxruntime-web is already included in your environment

async function processImageAtScales(image, scales) {
  let pnetBoxes = [];
  // scales = [0.6]
  const originalImageAsMat = cv.imread(image);
  for (let scale of scales) {
    let inputData = preparePnetInput(originalImageAsMat, scale);
    console.info("ProcessImageAtAscales:: inputData", inputData.length, inputData.cols, inputData.rows)
    let { output: pnetOutput } = await runPnet(inputData);
    console.info("pnetOutput", pnetOutput)
    let boxes = processPnetOutput(pnetOutput, scale, originalImageAsMat, 0.6);
    boxes = nonMaximumSuppression(boxes, 0.6, 'Union', 'nms PNET');
    pnetBoxes.push(...boxes);
  }
  pnetBoxes = nonMaximumSuppression(pnetBoxes, 0.7, 'Union', 'nms PNET')
    .map(box => rerec(box, originalImageAsMat.cols, originalImageAsMat.rows))
    // .map((box) => scaleBoundingBox(box, 1.30))
    // .map(box => padBoundingBoxes(box, originalImageAsMat.cols, originalImageAsMat.rows));

  console.info("processImageAtScales allBoxes", pnetBoxes);
  let rnetInput = prepareRnetInput(originalImageAsMat, pnetBoxes, 24); // 24 is
  let rnetOutput = await runRnet(rnetInput);
  let rnetBoxes = processRnetOutput(rnetOutput, rnetInput, pnetBoxes, originalImageAsMat, 0.7);
  rnetBoxes = nonMaximumSuppression(rnetBoxes, 0.7, 'Union', 'nms RNET')
    // .map(box => rerec(box, originalImageAsMat.cols, originalImageAsMat.rows));

  console.info("processImageAtScales rnetBoxes", rnetBoxes)
  // Process image through O-Net
  let onetInput = prepareOnetInput(originalImageAsMat, rnetBoxes, 48); // 48 is the input size for O-Net
  let onetOutput = await runOnet(onetInput);
  let onetBoxes = await processOnetOutput(onetOutput, onetInput, 0.7)
  onetBoxes = nonMaximumSuppression(onetBoxes, 0.2, 'Min', 'nms ONET');

  const finalBoxes = prepareApplicationOutput(onetBoxes, originalImageAsMat);

  console.info("processImageAtScales onetBoxes", onetBoxes)
  const newMat = originalImageAsMat.clone();
  pnetBoxes.forEach((box,i) => {
    drawBoundingBoxOnImage(newMat, box, undefined, undefined, i);
  });
  displayMatOnCanvas(newMat)
  const rnetMat = originalImageAsMat.clone();
  rnetBoxes.forEach((box, i) => {
    drawBoundingBoxOnImage(rnetMat, box,undefined, undefined, i);
  });
  const {canvas: rnetCanvas} = displayMatOnCanvas(rnetMat)
  const onetNewMat = originalImageAsMat.clone();
  finalBoxes.forEach((box, i) => {
    drawBoundingBoxOnImage(onetNewMat, box, [255, 255, 0, 255], undefined, i);
    drawBoundingBoxOnImage(onetNewMat, box.inputBox, undefined, undefined, i);
    drawBoundingBoxOnImage(onetNewMat, box.inputBox.inputBox, [128, 128, 0, 255], undefined, i);
    drawLandmarksOpenCV(onetNewMat, box.landmarks, box.score, box.scale)
  });
  const {canvas} = displayMatOnCanvas(onetNewMat)
  finalBoxes.forEach((box,i) => {
    const container = document.createElement("div")

    // ONet OutputBox
    const tooltipcanvas = document.createElement('canvas');
    tooltipcanvas.width = box.image.cols;
    tooltipcanvas.height = box.image.rows;
    cv.imshow(tooltipcanvas, box.image);

    // ONet InputBox
    const tooltipcanvas2 = document.createElement('canvas');
    tooltipcanvas2.width = box.inputBox.image.cols;
    tooltipcanvas2.height = box.inputBox.image.rows;
    cv.imshow(tooltipcanvas2, box.inputBox.image);

    //RNet InputBox
    const tooltipcanvas3 = document.createElement('canvas');
    tooltipcanvas3.width = box.inputBox.inputBox.image.cols;
    tooltipcanvas3.height = box.inputBox.inputBox.image.rows;
    cv.imshow(tooltipcanvas3, box.inputBox.inputBox.image);
  
    container.appendChild(tooltipcanvas)
    container.appendChild(tooltipcanvas2)
    container.appendChild(tooltipcanvas3)
    var t1 = new ToolTip(canvas, box, container, box.image.cols);
  });

  return onetBoxes;
}

// 1. Loading the MTCNN ONNX Model
async function loadMtcnnModel(modelPath) {
  const session = await ort.InferenceSession.create(modelPath);
  return session;
}

const prepareApplicationOutput = (refinedBoxes, image) => {
  // Create an Image Mat for each bounding box
  return refinedBoxes.map((box) => scaleBoundingBox(box, 1)).map((box) => {
    const croppedImage = cropImageOpenCV(image, box);
    
    return {
      image: croppedImage,
      ...box,
    };
  });
}

async function processImageThroughMtcnn(image, scale) {
  const imageWidth = image.width;
  const imageHeight = image.height;

  const scales = computeScalePyramid(minFaceSize, imageWidth, imageHeight);

  const finalBoxes = (await processImageAtScales(image, scales));
  console.info("finalBoxes", finalBoxes)
  return finalBoxes;
}
const pnetModelPath = 'mtcnn_ort/pnet.onnx';
const rnetModelPath = 'mtcnn_ort/rnet.onnx';
const onetModelPath = 'mtcnn_ort/onet.onnx';

// Debug helpers

const isValidBoundingBox = (bbox, width, height) => {
  const { x1, y1, x2, y2 } = bbox;
  return x1 >= 0 && x1 < width && y1 >= 0 && y1 < height && x2 >= 0 && x2 < width && y2 >= 0 && y2 < height && x2 > x1 && y2 > y1;
};

function scaleBoundingBox(box, scale) {
  const centerX = (box.x1 + box.x2) / 2;
  const centerY = (box.y1 + box.y2) / 2;
  const width = box.x2 - box.x1;
  const height = box.y2 - box.y1;

  const scaledWidth = width * scale;
  const scaledHeight = height * scale;

  return {
      ...box,
      x1: centerX - scaledWidth / 2,
      y1: centerY - scaledHeight / 2,
      x2: centerX + scaledWidth / 2,
      y2: centerY + scaledHeight / 2,
  };
}