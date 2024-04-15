import { Mat, imshow, CV_32FC3, CV_8UC3 } from '@techstark/opencv-js';

export const displayMatOnCanvas = (mat: Mat, canvas?: HTMLCanvasElement) => {
  const matToDisplay = new Mat();
  canvas = canvas ?? document.createElement('canvas');
  document.body.appendChild(canvas);
  if (mat.type() !== CV_32FC3) {
    mat.convertTo(matToDisplay, CV_8UC3, 255); // Scale back up to uchar range
  }

  imshow(canvas, mat);
  return { mat: matToDisplay, canvas };
}