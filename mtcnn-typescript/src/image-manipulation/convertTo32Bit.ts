import { Mat, CV_32FC3 } from '@techstark/opencv-js'

export function convertTo32Bit(image: Mat) {
  // BGR to RGB
  let temp = image.clone();

  if (temp.type() !== CV_32FC3) {
    let converted = new Mat();
    temp.convertTo(converted, CV_32FC3);
    temp.delete(); // Delete the temp Mat
    temp = converted;
  }

  return temp;
}