
import { Box } from '../shared/Box'
import { IDimensions } from '../shared/Dimensions';
import { normalize } from '../image-manipulation/normalize';
import { Tensor } from 'onnxruntime-web';
import { crop } from '../image-manipulation/crop';
import { Mat, COLOR_BGRA2RGB, COLOR_RGBA2RGB } from '@techstark/opencv-js'
import { BoundingBox } from '../shared/BoundingBox';
import { setColorMode } from '../image-manipulation/setColorMode';
import { resize } from '../image-manipulation/resize';
import { displayMatOnCanvas } from '../utils/displayMatOnCanvas';
export async function extractImagePatches(
  img: Mat,
  boxes: Box[],
  { width, height }: IDimensions
): Promise<Float32Array[]> {

  const bitmaps = boxes.map(box => {
    // TODO: correct padding
    const { y, ey, x, ex } = box.padAtBorders(img.rows, img.cols)
    const boxWidth = ex - x;
    const boxHeight = ey - y;
    try {
      const croppedImage = crop(img, x, y, boxWidth, boxHeight);
      const scaledImage = resize(croppedImage, width, height)
      const coloredImage = setColorMode(scaledImage, COLOR_RGBA2RGB);
      return normalize(coloredImage);
    } catch (ex) {
      console.error(ex)
      return undefined;
    }
  }).filter(data => data !== undefined);

  return bitmaps.map(data => {
    const inputArray = new Float32Array(data.data32F)
    return inputArray;
  });
};