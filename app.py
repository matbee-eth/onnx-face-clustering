import cv2
import onnxruntime
import numpy as np
from mtcnn_ort import MTCNN
from matplotlib import pyplot
from PIL import Image
from numpy import asarray

detector = MTCNN()
test_pic = "t.jpeg"
checkpoint_path = "InceptionResnetV1_vggface2.onnx"
pixels = pyplot.imread(test_pic)
image = cv2.cvtColor(cv2.imread(test_pic), cv2.COLOR_BGR2RGB)
result = detector.detect_faces(image)

# Result is an array with all the bounding boxes detected. Show the first.
detector.detect_faces_raw(image)

if len(result) > 0:
    bounding_box = result[0]["box"]
    keypoints = result[0]['keypoints']
    
    cv2.rectangle(image,
                  (bounding_box[0], bounding_box[1]),
                  (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
                  (0,155,255),
                  2)
    
    cv2.circle(image,(keypoints['left_eye']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['right_eye']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['nose']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['mouth_left']), 2, (0,155,255), 2)
    cv2.circle(image,(keypoints['mouth_right']), 2, (0,155,255), 2)
    
    cv2.imwrite("result.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    x1, y1, width, height = result[0]['box']
    x2, y2 = x1 + width, y1 + height

    # extract the face
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize((160, 160))
    face_array = asarray(image)
    face_array = np.transpose(face_array, (2, 0, 1))  # Transpose the array to match channel dimension
    # cv2.imwrite("result2.jpg", cv2.cvtColor(face_array, cv2.COLOR_RGB2BGR))

    resnet = onnxruntime.InferenceSession(
        checkpoint_path, providers=["CPUExecutionProvider"]
    )
    input_arr = np.expand_dims((face_array - 127.5) / 128.0, axis=0)  # Adjust input array
    embeddings = resnet.run(["output"], {"input": input_arr.astype(np.float32)})[0]

    print(embeddings.shape)

# Generate labeled images
with open(test_pic, "rb") as fp:
    marked_data = detector.mark_faces(fp.read())
with open("marked.jpg", "wb") as fp:
    fp.write(marked_data)

