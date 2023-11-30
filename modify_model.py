from models import model as M
from models import config
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from PIL import Image
import numpy as np
import cv2

C = config.Config()

# LOADS the pretrained model
model = M.EndToEndModel(gamma=C.gamma, theta=C.theta, stage='test').BuildModel()
model.load_weights(C.model)
# model.summary()

img_input = model.layers[0].input
fpp_input = Input((4,))

# BUILD encoder model
encoder_model = Model(img_input, model.get_layer('block5_conv2').output)
# encoder_model.summary()

# BUILD the rest of the model
feature_map = encoder_model.get_layer('block5_conv2').output
# saliency_box layer is no longer included
y = model.get_layer('roi_pooling')([feature_map, fpp_input])
for layer in model.layers[-4:]:
    y = layer(y)

# New model
new_model = Model([img_input, fpp_input], y)
# new_model.summary()

# Test the model

# get the image and normalize it
# img_object = Image.open("resized_testing\\resized_888.jpg")  # use resized 244 images
# img_object = img_object.convert('RGB')
# image = np.asarray(img_object)
#
# h1, w1 = image.shape[0], image.shape[1]
#
# h2, w2 = (image.shape[0] // 16 + 1) * 16, (image.shape[1] // 16 + 1) * 16
# image = cv2.copyMakeBorder(image, top=0, bottom=h2 - h1, left=0, right=w2 - w1,
#                            borderType=cv2.BORDER_CONSTANT,
#                            value=0)
#
# image = image.astype('float32') / 255.0
# image = np.expand_dims(image, axis=0)
#
# sr = tf.convert_to_tensor([156, 55, 132, 96])
# sr = tf.reshape(sr, (1, 4))
# sr = tf.cast(sr, dtype='float32')
# sr = sr / 16.0
# fpp = sr
#
# print("modify_model's [image, fpp]:", [image, fpp])
# boxes = new_model.predict([image, fpp], batch_size=1, verbose=0)
# print("boxes:", boxes)
