from keras_vggface.vggface import VGGFace

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
COLOR_CHANNELS = 3

output_path = 'vggface_model'
model = VGGFace(model='vgg16',
                include_top=False,
                input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, COLOR_CHANNELS),
                pooling='avg')
model.save(output_path)
