import * as ort from 'onnxruntime-web';

export async function loadMtcnnModel(modelPath) {
  const session = await ort.InferenceSession.create(modelPath);
  return session;
}