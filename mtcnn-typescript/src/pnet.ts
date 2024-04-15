import { Mat, cvtColor, COLOR_BGR2RGB, COLOR_BGRA2RGB, COLOR_BGRA2BGR, COLOR_RGBA2RGB, COLOR_RGBA2BGR, CV_32FC3, ColorConversionCodes } from "@techstark/opencv-js";
import { loadMtcnnModel } from "./utils/loadMtcnnModel";
import { CELL_SIZE, PNET_MODEL_PATH } from "./shared/globals";
import { normalize } from "./image-manipulation/normalize";
import { MtcnnOptions } from "./mtcnn/MtcnnOptions";
import { pyramidDown } from "./mtcnn/pyramidDown";
import { getSizesForScale } from "./mtcnn/getSizesForScale";
import { scale as scaleImage } from "./image-manipulation/scale";
import { InferenceSession, Tensor } from "onnxruntime-web/webgpu";
import { extractBoundingBoxes } from "./utils/pnet/calculateBoundingBoxes";
import { nonMaxSuppression } from "./utils/nonMaxSuppression";
import { BoundingBox } from "./shared/BoundingBox";
import { displayMatOnCanvas } from "./utils/displayMatOnCanvas";
import { setColorMode } from "./image-manipulation/setColorMode";
import { convertTo32Bit } from "./image-manipulation/convertTo32Bit";

const stepsThreshold = [0.6, 0.7, 0.7]
const minFaceSize = 20
const scaleFactor = 0.709
const maxNumScales = 10;

export type PnetOutput = {
  conv2d_4: Tensor,
  softmax: Tensor
}

export async function run(image: Mat) {
  const scales = pyramidDown(minFaceSize, scaleFactor, [image.rows, image.cols])
    .filter(scale => {
      const sizes = getSizesForScale(scale, [image.rows, image.cols])
      return Math.min(sizes.width, sizes.height) > CELL_SIZE
    })
    .slice(0, maxNumScales)

  const session = await loadMtcnnModel(PNET_MODEL_PATH);

  const results = await Promise.all(scales.map(async scale => {

    const sizes = getSizesForScale(scale, [image.rows, image.cols])

    console.info("scale", scale, "sizes", sizes);
    let scaledImage = scaleImage(image, scale)
    let rows = scaledImage.rows;  // Replace with the actual height of the source image
    let cols = scaledImage.cols;   // Replace with the actual width of the source image
    let channels = 3;   // For a typical color image
    if (scaledImage.channels() === 4) {
      const temp = setColorMode(scaledImage, COLOR_BGRA2RGB);
      scaledImage.delete();
      scaledImage = temp;
    }
    convertTo32Bit(scaledImage);

    const normalizedImage = normalize(scaledImage)
    const inputArray = new Float32Array(normalizedImage.data32F);
    if (inputArray.length !== rows * cols * channels) {
      throw new Error(`Data length (${inputArray.length}) does not match expected size (${rows * cols * channels}).`);
    }
    const inputTensor = new Tensor('float32', inputArray, [1, rows, cols, channels]);
    const feeds = { input_1: inputTensor }; // Use 'input_1' as per model's input name

    const PnetOutput = await session.run(feeds) as unknown as PnetOutput;
    console.info("PnetOutput", PnetOutput);
    return { out: PnetOutput, scale };
  }))

  const boxesForScale = results.map(result => {
    const { scale, out } = result;
    const boundingBoxes = extractBoundingBoxes(
      out.softmax,
      out.conv2d_4,
      scale,
      0.5
    )
    console.info("boundingBoxes", boundingBoxes);

    const indices = nonMaxSuppression(
      boundingBoxes.map(bbox => bbox.cell),
      boundingBoxes.map(bbox => bbox.score),
      0.5
    )

    return indices.map(boxIdx => boundingBoxes[boxIdx])
  });

  const allBoxes = boxesForScale.reduce(
    (all, boxes) => all.concat(boxes), []
  )

  console.info("allBoxes", allBoxes);

  let finalBoxes: BoundingBox[] = []
  let finalScores: number[] = []

  if (allBoxes.length > 0) {
    const indices = nonMaxSuppression(
      allBoxes.map(bbox => bbox.cell),
      allBoxes.map(bbox => bbox.score),
      0.7
    )

    finalScores = indices.map(idx => allBoxes[idx].score)
    finalBoxes = indices
      .map(idx => allBoxes[idx])
      .map(({ cell, region }) =>
        new BoundingBox(
          cell.left + (region.left * cell.width),
          cell.top + (region.top * cell.height),
          cell.right + (region.right * cell.width),
          cell.bottom + (region.bottom * cell.height)
        ).toSquare().round()
      )
  }

  return {
    boxes: finalBoxes,
    scores: finalScores
  }
}