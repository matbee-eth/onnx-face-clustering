import * as ort from 'onnxruntime-web';

export function calculateBoundingBox(x: number, y: number, reg: ort.Tensor, scale: number, originalWidth: number, originalHeight: number) {
  // console.info("calculateBoundingBox", x, y, scale, originalWidth, originalHeight)
  const stride = 2;
  const cellsize = 12;

  // Apply the regression values to the feature map coordinates
  const offsetX1 = reg[0] * cellsize;
  const offsetY1 = reg[1] * cellsize;
  const offsetX2 = reg[2] * cellsize;
  const offsetY2 = reg[3] * cellsize;

  // Calculate the bounding box coordinates
  let x1 = Math.round((stride * x + offsetX1) / scale);
  let y1 = Math.round((stride * y + offsetY1) / scale);
  let x2 = Math.round((stride * x + cellsize + offsetX2) / scale);
  let y2 = Math.round((stride * y + cellsize + offsetY2) / scale);

  // Clamp values within the original image dimensions
  x1 = Math.max(0, Math.min(x1, originalWidth));
  y1 = Math.max(0, Math.min(y1, originalHeight));
  x2 = Math.max(x1, Math.min(x2, originalWidth));
  y2 = Math.max(y1, Math.min(y2, originalHeight));

  return {
    via: "calculateBoundingBox",
    x1, y1, x2, y2,
    scale
  };
}