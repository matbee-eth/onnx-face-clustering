import * as cv from "@techstark/opencv-js";

export function scale(mat: cv.Mat, scale: number) {
  const width = Math.ceil(mat.cols * scale);
  const height = Math.ceil(mat.rows * scale);
  const dsize = new cv.Size(width, height);
  const resized = new cv.Mat();
  cv.resize(mat, resized, dsize, 0, 0, cv.INTER_AREA);
  return resized;
}