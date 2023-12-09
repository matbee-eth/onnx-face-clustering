import { Mat, Rect } from "@techstark/opencv-js";

export function crop(srcMat: Mat, x: number, y: number, width: number, height: number) {
  // Create a rectangle for the bounding box
  let rect = new Rect(x, y, width, height);

  // Crop the image to the bounding box
  let cropped = srcMat.roi(rect);

  return cropped; // Return the cropped image as a Mat object

}