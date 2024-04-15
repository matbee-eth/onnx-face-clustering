import * as ort from 'onnxruntime-web';
import { Point } from '../../shared/Point';
import { BoundingBox } from '../../shared/BoundingBox';
import { CELL_SIZE, CELL_STRIDE } from '../../shared/globals';
import { MtcnnBox } from '../../mtcnn/MtcnnBox';
import { Tensor } from 'onnxruntime-web';

export function extractBoundingBoxes(
  scoresTensor: Tensor,
  regionsTensor: Tensor,
  scale: number,
  scoreThreshold: number
) {
  console.info("extractBoundingBoxes", scoresTensor, regionsTensor, scale, scoreThreshold)
  const scoreDims = scoresTensor.dims
  const scoreData = scoresTensor.data
  const regionsData = regionsTensor.data
  const regionsDims = regionsTensor.dims

  const indices: Point[] = []
  for (let y = 0; y < scoreDims[1]; y++) {
    for (let x = 0; x < scoreDims[2]; x++) {
      const scoreIndex = (y * scoreDims[2] + x) * 2 + 1;
      const score = scoreData[scoreIndex] as number;
      if (score >= scoreThreshold) {
        indices.push(new Point(x, y))
      }
    }
  }

  const boundingBoxes = indices.map(idx => {
    const cell = new BoundingBox(
      Math.round((idx.y * CELL_STRIDE + 1) / scale),
      Math.round((idx.x * CELL_STRIDE + 1) / scale),
      Math.round((idx.y * CELL_STRIDE + CELL_SIZE) / scale),
      Math.round((idx.x * CELL_STRIDE + CELL_SIZE) / scale)
    )

    const score = scoreData[(idx.y * scoreDims[2] + idx.x) * 2 + 1] as number
    const regIndex = (idx.y * regionsDims[2] + idx.x) * 4;
    const reg = regionsData.slice(regIndex, regIndex + 4) as Float32Array;

    const region = new MtcnnBox(
      reg[0],
      reg[1],
      reg[2],
      reg[3]
    )

    return {
      cell,
      score,
      region
    }
  })

  return boundingBoxes
}