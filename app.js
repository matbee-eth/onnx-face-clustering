document.getElementById('runModel').addEventListener('click', (e) => {
  e.preventDefault();
  const inputElement = document.getElementById('imageInput');
  if (inputElement.files.length > 0) {
      const file = inputElement.files[0];
      processImage(file);
  }
});

async function processImage(imageFile) {

  const t = new MTCNN();
  // const tensor = t.preprocessImage(imageFile);
  const image = await loadImage(imageFile);
  const canvas = await t.markFaces(image);
  document.body.appendChild(canvas)
  const inputTensor = preprocessImage(image);
  await runModel(inputTensor);
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

function preprocessImage(image) {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  canvas.width = 160;
  canvas.height = 160;

  ctx.drawImage(image, 0, 0, canvas.width, canvas.height);

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
    console.info("output", output);
  } catch (error) {
      console.error(error);
  }
}