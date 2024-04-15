import { run as runPnet } from "./pnet";
import { run as runRnet } from "./rnet";
import { run as runOnet } from "./onet";

import { imread } from "@techstark/opencv-js";
import { displayMatOnCanvas } from "./utils/displayMatOnCanvas";
import { crop } from "./image-manipulation/crop";
import { drawLandmarks } from "./image-manipulation/drawLandmarks";
import { Point } from "./shared/Point";
import { Dimensions } from "./shared/Dimensions";
import { Rect } from "./shared/Rect";
let selectedFile: HTMLImageElement;
// IIFE to start the app

const processImage = async (img: HTMLImageElement) => {
  const image = imread(img)
  const imageDims = new Dimensions(image.rows, image.cols);
  const pnetoutput = await runPnet(image.clone());
  const rnetoutput = await runRnet(image.clone(), pnetoutput.boxes, 0.7);
  const onetoutput = await runOnet(image.clone(), rnetoutput.boxes, 0.7);
  const onetNewMat = image.clone();

  onetoutput.boxes.forEach((box, i) => {
    const img = crop(image, box.x, box.y, box.width, box.height);
    const unshiftedlandmarks = onetoutput.points[i].map(pt =>
      pt.sub(new Point(box.left, box.top))
        .div(new Point(box.width, box.height))
    );
    const score = onetoutput.scores[i]

    const shift = new Rect(
      box.left / imageDims.width,
      box.top / imageDims.height,
      box.width / imageDims.width,
      box.height / imageDims.height
    )

    const updatedLandmarks = unshiftedlandmarks.map(pt =>
      pt
        .sub(shift)
        .div(new Point(box.width, box.height))
        .mul(new Point(imageDims.width, imageDims.height))
        .add(shift)
    );

    drawLandmarks(img, updatedLandmarks, score);
    drawLandmarks(onetNewMat, updatedLandmarks, score);
    drawLandmarks(onetNewMat, onetoutput.points[i], score);
    displayMatOnCanvas(img)
  });
  const { mat } = displayMatOnCanvas(onetNewMat);
  console.info("onetoutput", onetoutput);
}

document.addEventListener('dragover', function (event) {
  event.preventDefault();
  event.stopPropagation();
});

document.getElementById('runModel').addEventListener('click', (e) => {
  e.preventDefault();
  console.info("runModel", selectedFile);
  if (selectedFile) {
    processImage(selectedFile);
  }
});


document.addEventListener('drop', function (event) {
  event.preventDefault();
  event.stopPropagation();

  let dt = event.dataTransfer;
  let files = dt.files;
  if (files.length > 0) {
    let file = files[0];
    let reader = new FileReader();

    reader.onload = function (event) {
      let dataUrl = event.target.result as string;
      let img = new Image();
      img.src = dataUrl;
      selectedFile = img;

      // Save the image data URL to local storage
      localStorage.setItem('savedImage', dataUrl);
    };
    // Process the files, for example, assign to a file input
    (document.getElementById('imageInput') as HTMLInputElement).files = files;

    reader.readAsDataURL(file);
  }
});

document.addEventListener('DOMContentLoaded', function () {
  // Load the image on page load
  let savedImage = localStorage.getItem('savedImage');
  if (savedImage) {
    // let img = document.getElementById('yourImage');
    // img.src = savedImage;
    let img = new Image();
    img.src = savedImage;
    selectedFile = img;
  }
});

(() => {

})();