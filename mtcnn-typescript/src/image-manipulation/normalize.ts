import * as cv from "@techstark/opencv-js";

export function normalize(srcMat: cv.Mat) {
  // console.info("Original Mat type:", srcMat.type());
  // logPixelValues(srcMat, "Pre-Pre-normalization");

  // let colorMat = new cv.Mat();
  // cv.cvtColor(srcMat, colorMat, cv.COLOR_BGRA2BGR);

  // if (imageFilename) {
  //   saveMatAsImage(colorMat, imageFilename);
  // }

  // // Convert to CV_32FC3 if not already
  // let floatMat = new cv.Mat();
  // if (colorMat.type() !== cv.CV_32FC3) {
  //   colorMat.convertTo(floatMat, cv.CV_32FC3);
  // } else {
  //   floatMat = colorMat.clone();
  // }

  // console.info("After conversion to CV_32FC3, Mat type:", floatMat.type());

  // Log pre-normalization pixel values
  // logPixelValues(floatMat, "Pre-normalization");
  const floatMat = srcMat.clone();
  // Normalize
  let meanScalar = new cv.Scalar(127.5, 127.5, 127.5);
  let meanScalarMat = new cv.Mat(floatMat.rows, floatMat.cols, floatMat.type(), meanScalar);
  cv.subtract(floatMat, meanScalarMat, floatMat);
  meanScalarMat.delete();

  // Multiply by the normalization factor
  let normFactor = 0.0078125;
  let normFactorMat = new cv.Mat(floatMat.rows, floatMat.cols, floatMat.type(), new cv.Scalar(normFactor, normFactor, normFactor));
  cv.multiply(floatMat, normFactorMat, floatMat);

  normFactorMat.delete();

  // Log post-normalization pixel values

  return floatMat;
}
