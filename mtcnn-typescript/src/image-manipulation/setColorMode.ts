import { Mat, COLOR_BGRA2RGB, ColorConversionCodes, cvtColor, CV_32FC3 } from '@techstark/opencv-js'

export function setColorMode(image: Mat, color: ColorConversionCodes = COLOR_BGRA2RGB) {
  // BGR to RGB
  let temp = new Mat();

  // Check if the Mat has 4 channels (RGBA), and if so, convert to 3 channels (RGB)
  console.info("Setting color mode from BGRA to RGB")
  cvtColor(image, temp, color);

  if (temp.type() !== CV_32FC3) {
    let converted = new Mat();
    temp.convertTo(converted, CV_32FC3);
    temp.delete(); // Delete the temp Mat
    temp = converted;
  }

  return temp;
}