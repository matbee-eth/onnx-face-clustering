
function matToFloat32Array(mat) {
  let temp = new cv.Mat();

  // Check if the Mat has 4 channels (RGBA), and if so, convert to 3 channels (RGB)
  if (mat.channels() === 4) {
    cv.cvtColor(mat, temp, cv.COLOR_BGRA2BGR);
  } else {
    temp = mat.clone();
  }

  // Convert to 32FC3 format if not already
  if (temp.type() !== cv.CV_32FC3) {
    console.info("matToFloat32Array converting", temp.type(), cv.CV_32FC3)
    let converted = new cv.Mat();
    temp.convertTo(converted, cv.CV_32FC3);
    temp.delete(); // Delete the temp Mat
    temp = converted;
  }

  // Access the data and convert it to a Float32Array
  let array = new Float32Array(temp.data32F);

  return array;
}

function scaleImageOpenCV(mat, scale) {
  const width = Math.ceil(mat.cols * scale);
  const height = Math.ceil(mat.rows * scale);
  const dsize = new cv.Size(width, height);
  const resized = new cv.Mat();
  cv.resize(mat, resized, dsize, 0, 0, cv.INTER_AREA);
  return resized;
}

function resizeImageOpenCV(mat, width, height) {
  const dsize = new cv.Size(width, height);
  const resized = new cv.Mat();
  cv.resize(mat, resized, dsize, 0, 0, cv.INTER_AREA);
  return resized;
}

function normalizeImage(srcMat, imageFilename) {
  // console.info("Original Mat type:", srcMat.type());
  // logPixelValues(srcMat, "Pre-Pre-normalization");
  
  let colorMat = new cv.Mat();
  cv.cvtColor(srcMat, colorMat, cv.COLOR_BGRA2BGR);

  if (imageFilename) {
    saveMatAsImage(colorMat, imageFilename);
  }

  // Convert to CV_32FC3 if not already
  let floatMat = new cv.Mat();
  if (colorMat.type() !== cv.CV_32FC3) {
    colorMat.convertTo(floatMat, cv.CV_32FC3);
  } else {
    floatMat = colorMat.clone();
  }
  
  // console.info("After conversion to CV_32FC3, Mat type:", floatMat.type());

  // Log pre-normalization pixel values
  // logPixelValues(floatMat, "Pre-normalization");

  // Normalize
  let meanScalar = new cv.Scalar(127.5, 127.5, 127.5);
  let meanScalarMat = new cv.Mat(floatMat.rows, floatMat.cols, floatMat.type(), meanScalar);
  cv.subtract(floatMat, meanScalarMat, floatMat);
  meanScalarMat.delete();

  // Multiply by the normalization factor
  let normFactor = 0.0078125;
  let normFactorMat = new cv.Mat(floatMat.rows, floatMat.cols, floatMat.type(), new cv.Scalar(normFactor, normFactor, normFactor));
  cv.multiply(floatMat, normFactorMat, floatMat);

  colorMat.delete();
  normFactorMat.delete();

  // Log post-normalization pixel values

  return floatMat;
}

function cropImageOpenCV(srcMat, bbox) {
  // Ensure the bounding box coordinates are within the dimensions of the source image
  const x1 = Math.max(0, bbox.x1);
  const y1 = Math.max(0, bbox.y1);
  const x2 = Math.min(bbox.x2, srcMat.cols);
  const y2 = Math.min(bbox.y2, srcMat.rows);

  // Calculate the width and height of the bounding box
  const width = x2 - x1;
  const height = y2 - y1;

  // Create a rectangle for the bounding box
  let rect = new cv.Rect(x1, y1, width, height);

  // Crop the image to the bounding box
  let cropped = srcMat.roi(rect);

  return cropped; // Return the cropped image as a Mat object

}

// Debug helpers
function drawLandmarksOpenCV(src, landmarks, score, scale) {
  // Convert the canvas to a cv.Mat
  const colors = [
    new cv.Scalar(255, 0, 0), // red
    new cv.Scalar(0, 255, 0), // green
    new cv.Scalar(0, 0, 255), // blue
    new cv.Scalar(255, 255, 0), // yellow
    new cv.Scalar(128, 0, 128), // purple
  ];
  const map = ["leftEye", "rightEye", "nose", "leftMouth", "rightMouth"];

  for (let i = 0; i < landmarks.length; i++) {
    const landmark = landmarks[i];
    const color = colors[i % colors.length];
    // Draw circle for the landmark
    let center = new cv.Point(landmark.x, landmark.y);
    cv.circle(src, center, 3, color, -1, cv.LINE_AA, 0);

    // Put text for the landmark
    let text = `Score: ${score.toFixed(2)}, Landmark: ${map[i]}`;
    let bottomLeftCornerOfText = new cv.Point(landmark.x, landmark.y - 10);
    cv.putText(src, text, bottomLeftCornerOfText, cv.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv.LINE_AA);
  }
}

function saveMatAsImage(mat, filename) {
  // console.info("SAVING MAT", mat, filename)
  const canvas = document.createElement('canvas');
  // Draw the Mat on the canvas
  cv.imshow(canvas, mat);

  // Convert the canvas to a data URL
  let imageURL = canvas.toDataURL('image/png');

  // Create a temporary link element
  let downloadLink = document.createElement('a');

  // Set download attribute with filename
  downloadLink.download = filename;

  // Set href to the data URL
  downloadLink.href = imageURL;

  downloadLink.textContent = 'Download image';

  // Trigger download by simulating click
  // downloadLink.click();
  // document.body.appendChild(downloadLink);
}

displayMatOnCanvas = (mat, canvas) => {
  const matToDisplay = new cv.Mat();
  canvas = canvas ?? document.createElement('canvas');
  document.body.appendChild(canvas);
  if (mat.type() !== cv.CV_32FC3) {
    mat.convertTo(matToDisplay, cv.CV_8UC3, 255); // Scale back up to uchar range
  }
  
  cv.imshow(canvas, mat);
  return {mat: matToDisplay, canvas};
}

function outputFirstRowPixels(cvMat) {
  // Ensure the Mat is of a type that contains 3 channels (e.g., CV_8UC3 for a typical color image)
  if (cvMat.type() !== cv.CV_8UC3 && cvMat.type() !== cv.CV_32FC3) {
    console.error("Unsupported Mat type for this operation. Expected a 3-channel Mat.");
    return;
  }

  // Access the first row of the image
  for (let x = 0; x < 10; x++) {
    let pixel;
    if (cvMat.type() === cv.CV_8UC3) {
      // For 8-bit unsigned integer Mat (most common for images)
      pixel = [
        cvMat.ucharPtr(0, x)[0], // Red
        cvMat.ucharPtr(0, x)[1], // Green
        cvMat.ucharPtr(0, x)[2], // Blue
      ];
    } else if (cvMat.type() === cv.CV_32FC3) {
      // For 32-bit floating point Mat
      pixel = [
        cvMat.floatPtr(0, x)[0], // Red
        cvMat.floatPtr(0, x)[1], // Green
        cvMat.floatPtr(0, x)[2], // Blue
      ];
    }

    // console.log(`Pixel ${x}: [${pixel.join(", ")}]`);
  }
}

function drawBoundingBoxOnImage(srcMat, bbox, color = [255, 0, 0, 255], thickness = 2, index) {
  // Convert color to a cv.Scalar
  let rectangleColor = new cv.Scalar(...color);

  // Create a cv.Point for the top-left and bottom-right corners of the bounding box
  let pt1 = new cv.Point(bbox.x1, bbox.y1);
  let pt2 = new cv.Point(bbox.x2, bbox.y2);

  // Draw the rectangle on the source image
  cv.rectangle(srcMat, pt1, pt2, rectangleColor, thickness);
  // console.info("drawing bounding box of width", bbox.x2 - bbox.x1, "height", bbox.y2 - bbox.y1, bbox)
  // Write Score text in box
  let text = `${index} Score: ${bbox.score.toFixed(2)}`;
  let org = new cv.Point(bbox.x1 + 5, bbox.y1 - 5);
  cv.putText(srcMat, text, org, cv.FONT_HERSHEY_SIMPLEX, 0.5, rectangleColor, 2);
  // The source image now has the bounding box drawn on it
  return srcMat;
}

function logPixelValues(mat, stage) {
  // console.info(`${stage} pixel values:`);
  for (let i = 0; i < 5; i++) {
    let pixelIndex = i * 3; // Each pixel has 3 values (B, G, R)
    let pixelValue = [
      mat.data32F[pixelIndex + 0],  // Blue
      mat.data32F[pixelIndex + 1],  // Green
      mat.data32F[pixelIndex + 2]   // Red
    ];
    // console.info(`Pixel ${i}: [${pixelValue[0].toFixed(7)}, ${pixelValue[1].toFixed(7)}, ${pixelValue[2].toFixed(7)}]`);
  }
}

function printFirstFewPixelValues(image) {
  // Accessing and printing pixel values
  console.log("First few pixel values (up to 5 pixels):");
  for (let i = 0; i < Math.min(5, image.cols); i++) {
      let pixel = image.ucharPtr(0, i); // Accessing the pixel at (0, i)
      console.log(`Pixel ${i}: [${pixel.join(", ")}]`);
  }
}