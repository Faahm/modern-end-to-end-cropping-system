import warnings

with warnings.catch_warnings():
    import os, sys
    import cv2
    import tensorflow as tf
    import keras
    from utils import *
    import numpy as np
    from models import config
    from models import model as M
    from PIL import Image, ImageDraw
    from keras.models import Model
    from keras.preprocessing.image import array_to_img
    import keras.backend as K
    import matplotlib.pyplot as plt

C = config.Config()


def get_shape(img, w, h, scale, ratio):
    size = (w, h)
    if ratio:
        if scale is not None:
            size = (scale, scale)
    else:
        if scale is not None:
            if w <= h:
                size = (scale, scale * h // w)
            else:
                size = (scale * w // h, scale)
    return img.resize(size, Image.ANTIALIAS)


def print_image_summary(conv2d_image, cols=8):
    channels = conv2d_image.shape[-1]
    images = conv2d_image[0]
    rows = channels // cols
    plt.figure(figsize=(cols * 2, rows * 2))
    for i in range(channels):
        plt.subplot(rows, cols, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(images[:, :, i], cmap='gray')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


img_object = Image.open('images/561_h.jpg')
w3, h3 = img_object.size
img_object = img_object.convert('RGB')
# imgs = img.copy()
img_reshape = get_shape(img_object, w3, h3, C.scale, C.ratio)
image = np.asarray(img_reshape)

h1, w1 = image.shape[0], image.shape[1]

h2, w2 = (image.shape[0] // 16 + 1) * 16, (image.shape[1] // 16 + 1) * 16
image = cv2.copyMakeBorder(image, top=0, bottom=h2 - h1, left=0, right=w2 - w1,
                           borderType=cv2.BORDER_CONSTANT,
                           value=0)

image = image.astype('float32') / 255.0

image = np.expand_dims(image, axis=0)

model = M.EndToEndModel(gamma=C.gamma, theta=C.theta, stage='test').BuildModel()
model.load_weights(C.model)

# With Keras function
# get_all_layer_outputs = K.function([model.layers[0].input],
#                                    [model.get_layer('segmentation').output, model.get_layer('block5_conv2').output,
#                                     model.get_layer('roi_pooling').output])
get_all_layer_outputs = K.function([model.layers[0].input],
                                   [model.get_layer('segmentation').output,
                                    model.get_layer('block5_conv2').output,
                                    model.get_layer('roi_pooling').output,
                                    model.get_layer('offset').output])

# outputs and an array of the layer outputs
layer_output = get_all_layer_outputs([image])
first_layer = layer_output[0]
print("first_layer: ", first_layer)
second_layer = layer_output[1]
print("second_layer: ", second_layer)
third_layer = layer_output[2]
print("third_layer: ", third_layer)
fourth_layer = layer_output[3]
print("fourth_layer: ", fourth_layer)

# 2D array, saliency
plt.matshow(first_layer[0, :, :, :], cmap='viridis')
plt.savefig('first_layer.png')


# 3D array, feature
second_layer = layer_output[1][0]
combined_second_layer = np.sum(second_layer, axis=-1)
plt.matshow(combined_second_layer, cmap='viridis')
plt.savefig('second_layer.png')

# 4D array, roi pooling
third_layer = layer_output[2][0]
combined_third_layer = np.mean(third_layer, axis=-1)
combined_third_layer = combined_third_layer.squeeze()
plt.matshow(combined_third_layer, cmap='viridis')
plt.savefig('third_layer.png')

fourth_layer = layer_output[2][0]
combined_fourth_layer = np.mean(fourth_layer, axis=-1)
combined_fourth_layer = combined_fourth_layer.squeeze()
plt.matshow(combined_fourth_layer, cmap='viridis')
plt.savefig('fourth_layer.png')
