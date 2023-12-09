import { Mat, Size, INTER_AREA, resize as resizecv } from '@techstark/opencv-js';

export function resize(mat: Mat, width: number, height: number) {
  const dsize = new Size(width, height);
  const resized = new Mat();
  resizecv(mat, resized, dsize, 0, 0, INTER_AREA);
  return resized;
}