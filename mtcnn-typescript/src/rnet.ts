import { Mat } from '@techstark/opencv-js'
import { Box } from './shared/Box'
import { extractImagePatches } from './mtcnn/extractImagePatches'
import { RNET_INPUT_SIZE, RNET_MODEL_PATH } from './shared/globals'
import { Tensor } from 'onnxruntime-web'
import { loadMtcnnModel } from './utils/loadMtcnnModel'
import { MtcnnBox } from './mtcnn/MtcnnBox'
import { Point } from './shared/Point'
import { nonMaxSuppression } from './utils/nonMaxSuppression'

export type RnetOutput = {
  dense_2: Tensor,
  softmax_1: Tensor
}

export async function run(
  image: Mat,
  inputBoxes: Box[],
  scoreThreshold: number,
) {
  const rnetInputs = await extractImagePatches(image, inputBoxes, { width: RNET_INPUT_SIZE, height: RNET_INPUT_SIZE })
  console.log("rnetInputs", rnetInputs, inputBoxes)
  const batchSize = rnetInputs.length;
  const height = RNET_INPUT_SIZE;  // Height for RNet
  const width = RNET_INPUT_SIZE;   // Width for RNet
  const channels = 3; // Number of channels (RGB)

  // Create a single array to hold all batched data
  const batchedData = new Float32Array(batchSize * height * width * channels);
  rnetInputs.forEach((imageData, index) => {
    batchedData.set(imageData, index * height * width * channels);
  });

  const inputTensor = new Tensor('float32', batchedData, [batchSize, height, width, channels]);
  const feeds = { 'input_2': inputTensor }; // Use the correct input name for RNet
  const session = await loadMtcnnModel(RNET_MODEL_PATH);
  const rnetOutput = await session.run(feeds) as RnetOutput;

  const scoreDims = rnetOutput.softmax_1.dims
  const scoreData = rnetOutput.softmax_1.data
  const regionsData = rnetOutput.dense_2.data
  const regionsDims = rnetOutput.dense_2.dims

  const regions: { box: Box, score: number }[] = [];
  for (let i = 0; i < regionsData.length / 4; i++) {
    let score = scoreData[i * 2 + 1] as number; // Assuming the second score is the face confidence
    console.info("score", i, score)
    if (score > scoreThreshold) {
      const refinementSlice = regionsData.slice(i * 4, i * 4 + 4) as Float32Array;
      const refinement = new MtcnnBox(
        refinementSlice[0],
        refinementSlice[1],
        refinementSlice[2],
        refinementSlice[3]
      )
      const box = inputBoxes[i];
      regions.push({ box: box.calibrate(refinement), score: score });
    }
  }
  const filteredBoxes = regions.map(region => region.box)
  const filteredScores = regions.map(region => region.score)
  const nmsIndices = nonMaxSuppression(
    filteredBoxes,
    filteredScores,
    0.7,
    false
  )
  const finalBoxes = nmsIndices.map(idx => filteredBoxes[idx])
  const finalScores = nmsIndices.map(idx => filteredScores[idx])
  return {
    boxes: finalBoxes,
    scores: finalScores
  };
}