import cv2
import onnxruntime
import numpy as np
from mtcnn_ort import MTCNN
from matplotlib import pyplot
from PIL import Image
from numpy import asarray

def __bbreg(boundingbox, reg):
    # calibrate bounding boxes
    if reg.shape[1] == 1:
        reg = np.reshape(reg, (reg.shape[2], reg.shape[3]))

    w = boundingbox[:, 2] - boundingbox[:, 0] + 1
    h = boundingbox[:, 3] - boundingbox[:, 1] + 1
    b1 = boundingbox[:, 0] + reg[:, 0] * w
    b2 = boundingbox[:, 1] + reg[:, 1] * h
    b3 = boundingbox[:, 2] + reg[:, 2] * w
    b4 = boundingbox[:, 3] + reg[:, 3] * h
    print(f"__bbreg: pre boundingbox.shape: {boundingbox[:, 0]}  {reg[:, 0]}")
    print(f"__bbreg: pre boundingbox.shape: {boundingbox[:, 1]}  {reg[:, 1]}")
    print(f"__bbreg: pre boundingbox.shape: {boundingbox[:, 2]}  {reg[:, 2]}")
    print(f"__bbreg: pre boundingbox.shape: {boundingbox[:, 3]}  {reg[:, 3]}")
    boundingbox[:, 0:4] = np.transpose(np.vstack([b1, b2, b3, b4]))

    if b1 > 512 or b2 > 512 or b3 > 512 or b4 > 512:
        print(f"__bbreg: post boundingbox.shape: {boundingbox[:, 0]} {reg[:, 0]}")
        print(f"__bbreg: post boundingbox.shape: {boundingbox[:, 1]} {reg[:, 1]}")
        print(f"__bbreg: post boundingbox.shape: {boundingbox[:, 2]} {reg[:, 2]}")
        print(f"__bbreg: post boundingbox.shape: {boundingbox[:, 3]} {reg[:, 3]}")

    print(f"__bbreg: post boundingbox.shape: {boundingbox[:, 0]} {reg[:, 0]}")
    print(f"__bbreg: post boundingbox.shape: {boundingbox[:, 1]} {reg[:, 1]}")
    print(f"__bbreg: post boundingbox.shape: {boundingbox[:, 2]} {reg[:, 2]}")
    print(f"__bbreg: post boundingbox.shape: {boundingbox[:, 3]} {reg[:, 3]}")
    return boundingbox
detector = MTCNN()
test_pic = "t.jpg"
checkpoint_path = "InceptionResnetV1_vggface2.onnx"
pixels = pyplot.imread(test_pic)
image = cv2.cvtColor(cv2.imread(test_pic), cv2.COLOR_BGR2RGB)
result = detector.detect_faces(image)
print(result)
# # Result is an array with all the bounding boxes detected. Show the first.
# detector.detect_faces_raw(image)

# if len(result) > 0:
#     bounding_box = result[0]["box"]
#     keypoints = result[0]['keypoints']
    
#     cv2.rectangle(image,
#                   (bounding_box[0], bounding_box[1]),
#                   (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
#                   (0,155,255),
#                   2)
    
#     cv2.circle(image,(keypoints['left_eye']), 2, (0,155,255), 2)
#     cv2.circle(image,(keypoints['right_eye']), 2, (0,155,255), 2)
#     cv2.circle(image,(keypoints['nose']), 2, (0,155,255), 2)
#     cv2.circle(image,(keypoints['mouth_left']), 2, (0,155,255), 2)
#     cv2.circle(image,(keypoints['mouth_right']), 2, (0,155,255), 2)
    
#     cv2.imwrite("result.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

#     x1, y1, width, height = result[0]['box']
#     x2, y2 = x1 + width, y1 + height

#     # extract the face
#     face = pixels[y1:y2, x1:x2]
#     image = Image.fromarray(face)
#     image = image.resize((160, 160))
#     face_array = asarray(image)
#     face_array = np.transpose(face_array, (2, 0, 1))  # Transpose the array to match channel dimension
#     # cv2.imwrite("result2.jpg", cv2.cvtColor(face_array, cv2.COLOR_RGB2BGR))

#     resnet = onnxruntime.InferenceSession(
#         checkpoint_path, providers=["CPUExecutionProvider"]
#     )
#     input_arr = np.expand_dims((face_array - 127.5) / 128.0, axis=0)  # Adjust input array
#     embeddings = resnet.run(["output"], {"input": input_arr.astype(np.float32)})[0]

#     print(embeddings.shape)

# Generate labeled images
with open(test_pic, "rb") as fp:
    marked_data = detector.mark_faces(fp.read())
with open("marked.jpg", "wb") as fp:
    fp.write(marked_data)

