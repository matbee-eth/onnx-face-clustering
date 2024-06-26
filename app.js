document.getElementById('runModel').addEventListener('click', (e) => {
  e.preventDefault();
  if (selectedFile) {
      processImage(selectedFile);
  }
});

let selectedFile; // HTMLImageElement

async function processImage(img) {
  // console.info("img", img);
  // const reader = new FileReader();
  // reader.onload = function(e) {
  //     const img = new Image();
  //     img.src = e.target.result;
  //     img.onload = async () => {
          // Image is ready to be displayed
          // document.getElementById('imageContainer').appendChild(img);
          const faces = await processImageThroughMtcnn(img, 0.5);

          // // Draw the bounding boxes
          // faces.forEach(async face => {
          //     const { x1, y1, x2, y2 } = face;
          //     const canvas = document.createElement('canvas');
          //     document.body.appendChild(canvas);
          //     const ctx = canvas.getContext('2d');
          //     const width = x2 - x1;
          //     const height = y2 - y1;
          //     canvas.width = 160;
          //     canvas.height = 160;
          //     ctx.drawImage(face.originalImage, x1, y1, width, height, 0, 0, 160, 160);
          //     const dataURL = canvas.toDataURL();
          //     document.getElementById('croppedImage').src = dataURL;

          //     const inputTensor = preprocessImage(canvas);
          //     await runModel(inputTensor);
          // });

  //     };
  // };
  // reader.readAsDataURL(imageFile);

}

function loadImage(file) {
  return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (event) => {
          const img = new Image();
          img.onload = () => resolve(img);
          img.onerror = reject;
          img.src = event.target.result;
      };
      reader.readAsDataURL(file);
  });
}

function preprocessImage(canvas) {
  const ctx = canvas.getContext('2d');
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = imageData.data;
  const float32Data = new Float32Array(3 * 160 * 160);

  for (let i = 0, j = 0; i < data.length; i += 4, j += 3) {
      // Map RGBA to RGB
      float32Data[j] = (data[i] - 127.5) / 128.0;     // Red
      float32Data[j + 1] = (data[i + 1] - 127.5) / 128.0; // Green
      float32Data[j + 2] = (data[i + 2] - 127.5) / 128.0; // Blue
  }

  return new ort.Tensor('float32', float32Data, [1, 3, 160, 160]);
}

async function runModel(inputTensor) {
  let session;
  try {
    const session = await ort.InferenceSession.create('InceptionResnetV1_vggface2.onnx');
    
    const {output} = await session.run({ input: inputTensor });
    console.info("output", Array.from(output.data));
  } catch (error) {
      console.error(error);
  }
}

document.addEventListener('drop', function(event) {
  event.preventDefault();
  event.stopPropagation();

  let dt = event.dataTransfer;
  let files = dt.files;
  if (files.length > 0) {
    let file = files[0];
    let reader = new FileReader();
    
    reader.onload = function(event) {
        let dataUrl = event.target.result;
        let img = new Image();
        img.src = dataUrl;
        selectedFile = img;
    
        // Save the image data URL to local storage
        localStorage.setItem('savedImage', dataUrl);
    };
    // Process the files, for example, assign to a file input
    document.getElementById('imageInput').files = files;

    reader.readAsDataURL(file);
}

  
});

document.addEventListener('dragover', function(event) {
  event.preventDefault();
  event.stopPropagation();
});

document.addEventListener('keydown', function(event) {
  if (event.key === 'p' || event.key === 'P') {
      let coords = prompt("Enter X and Y coordinates separated by a comma:");
      let [x, y] = coords.split(',');

      // Check if x and y are valid numbers
      if (!isNaN(x) && !isNaN(y)) {
          // Draw a large purple circle on the canvas at (x, y)
          let canvas = document.getElementById('processImageAtScales');
          if (canvas?.getContext) {
              let ctx = canvas.getContext('2d');
              ctx.beginPath();
              ctx.arc(x, y, 50, 0, 2 * Math.PI); // Draw a large circle
              ctx.fillStyle = 'purple'; // Set the fill color to purple
              ctx.fill();
          }
      }
  }
});

document.addEventListener('DOMContentLoaded', function() {
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

// The Tool-Tip instance:
function ToolTip(canvas, region, text, width, timeout) {

  var me = this,                                // self-reference for event handlers
      div = document.createElement("div"),      // the tool-tip div
      parent = canvas.parentNode,               // parent node for canvas
      visible = false;                          // current status
  
  // set some initial styles, can be replaced by class-name etc.
  div.style.cssText = "position:fixed;padding:7px;background:gold;pointer-events:none;width:" + width + "px";
  div.appendChild(text);
  
  // show the tool-tip
  this.show = function(pos) {
    console.info("show", div);
    if (!visible) {                             // ignore if already shown (or reset time)
      visible = true;                           // lock so it's only shown once
      setDivPos(pos);                           // set position
      parent.appendChild(div);                  // add to parent of canvas
    }
  }
  
  // hide the tool-tip
  function hide() {
    visible = false;                            // hide it after timeout
    parent.removeChild(div);                    // remove from DOM
  }

  // check mouse position, add limits as wanted... just for example:
  function check(e) {
    var pos = getPos(e),
        posAbs = {x: e.clientX, y: e.clientY};  // div is fixed, so use clientX/Y
    if (pos.x >= region.x1 && pos.x < region.x2 &&
        pos.y >= region.y1 && pos.y < region.y2) {
          setDivPos(posAbs);
          if (!visible) {
            me.show(posAbs);
          }
    }
    else {
      setDivPos(posAbs);                     // otherwise, update position
      if (visible) hide()
    }
  }
  
  // get mouse position relative to canvas
  function getPos(e) {
    var r = canvas.getBoundingClientRect();
    return {x: e.clientX - r.left, y: e.clientY - r.top}
  }
  
  // update and adjust div position if needed (anchor to a different corner etc.)
  function setDivPos(pos) {
    if (visible){
      if (pos.x < 0) pos.x = 0;
      if (pos.y < 0) pos.y = 0;
      // other bound checks here
      div.style.left = pos.x + "px";
      div.style.top = pos.y + "px";
    }
  }
  
  // we need to use shared event handlers:
  canvas.addEventListener("mousemove", check);
  canvas.addEventListener("click", check);
  
}