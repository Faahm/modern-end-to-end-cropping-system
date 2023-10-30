import os
import sys
import time
import urllib.request
import json

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Input,
    concatenate,
    UpSampling2D,
    Lambda,
    Flatten,
    Dense,
    Conv2D,
    MaxPooling2D
)

from models.RoiPoolingConv import RoiPoolingConv


def face_plus_plus(filepath):
    # API configuration
    key = "ILUS28ltigP0UirSewQPZmKLCqFQEtg1"
    secret = "vMLb1ufvKfjIgIcLShervHy723o7cdLI"

    face_http_url = 'https://api-us.faceplusplus.com/facepp/v3/detect'
    humanbody_http_url = 'https://api-us.faceplusplus.com/humanbodypp/v1/detect'

    boundary = '----------%s' % hex(int(time.time() * 1000))
    data = [b'--%s' % boundary.encode('utf-8'), b'Content-Disposition: form-data; name="%s"\r\n' % b'api_key',
            key.encode('utf-8'), b'--%s' % boundary.encode('utf-8'),
            b'Content-Disposition: form-data; name="%s"\r\n' % b'api_secret', secret.encode('utf-8'),
            b'--%s' % boundary.encode('utf-8')]
    fr = open(filepath, 'rb')
    data.append(b'Content-Disposition: form-data; name="%s"; filename="12263.jpg"' % b'image_file')
    data.append(b'Content-Type: %s\r\n' % b'application/octet-stream')
    data.append(fr.read())
    fr.close()
    data.append(b'--%s--\r\n' % boundary.encode('utf-8'))

    # Join data as bytes
    http_body = b'\r\n'.join(data)

    # Build HTTP request
    req = urllib.request.Request(humanbody_http_url)

    # Header
    req.add_header('Content-Type', 'multipart/form-data; boundary=%s' % boundary)
    req.data = http_body

    try:
        # Post data to server
        resp = urllib.request.urlopen(req, timeout=5)
        # Get response
        qrcont = resp.read()
        # Parse the JSON response
        parsed_response = json.loads(qrcont)

        # Check if human body detection was successful
        if len(parsed_response['humanbodies']) != 0:
            human_body_rectangle = parsed_response['humanbodies'][0]['humanbody_rectangle']

            # Extract and print the face rectangle details
            top = human_body_rectangle['top']
            left = human_body_rectangle['left']
            width = human_body_rectangle['width']
            height = human_body_rectangle['height']

            return [top, left, height, width]

        else:
            req = urllib.request.Request(face_http_url)

            # Header
            req.add_header('Content-Type', 'multipart/form-data; boundary=%s' % boundary)
            req.data = http_body
            # Post data to server
            resp = urllib.request.urlopen(req, timeout=5)
            # Get response
            qrcont = resp.read()
            # Parse the JSON response
            parsed_response = json.loads(qrcont)
            if len(parsed_response['faces']) != 0:
                face_rectangle = parsed_response['faces'][0]['face_rectangle']

                # Extract and print the face rectangle details
                top = face_rectangle['top']
                left = face_rectangle['left']
                width = face_rectangle['width']
                height = face_rectangle['height']

                return [top, left, height, width]
            else:
                return None

    except urllib.error.HTTPError as e:
        print(e.read())
        return None


class EndToEndModel(object):

    def __init__(self, weights=None, gamma=3.0, pooling_regions=7, num_rois=1, theta=0.01, stage='train', image_file=None):
        self.weights = weights
        self.stage = stage
        self.gamma = gamma
        self.pooling_regions = pooling_regions
        self.num_rois = num_rois
        self.theta = theta
        self.image_file = image_file

    def cal_salient_region(self):
        image_filepath = f"resized_testing\\resized_{self.image_file}"
        face_plus_plus(image_filepath)

        # sr = tf.convert_to_tensor([top, left, height, width])
        sr = tf.convert_to_tensor(face_plus_plus(image_filepath))

        # Normalize the salient region coordinates by dividing by 16.0 (a scaling factor)
        sr = tf.cast(sr, dtype='float32')
        sr = sr / 16.0
        # Tensor("saliency_box/map/while/truediv_6:0", shape=(4,), dtype=float32)
        # This tensor likely represents the bounding box coordinates of the salient region. The four elements
        # typically correspond to:
        # The Y-coordinate of the top-left corner of the bounding box.
        # The X-coordinate of the top-left corner of the bounding box.
        # The height (or vertical size) of the bounding box.
        # The width (or horizontal size) of the bounding box.
        return sr

    def cal_salient_regions(self, samples):
        salient_regions = tf.map_fn(lambda sample: self.cal_salient_region(), samples)
        return salient_regions

    def cal_salient_regions_output_shape(self, input_shape):
        return input_shape[0], 1, 4

    def EncodeLayer(self, input_tensor=None):
        input_shape = (None, None, 3)
        if input_tensor is None:
            img_input = Input(shape=input_shape)
        else:
            if not K.is_keras_tensor(input_tensor):
                img_input = Input(tensor=input_tensor, shape=input_shape)
            else:
                img_input = input_tensor

        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block1_conv1')(
            img_input)
        conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block1_conv2')(
            conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2), name='block1_pool')(conv1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block2_conv1')(
            pool1)
        conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block2_conv2')(
            conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2), name='block2_pool')(conv2)

        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block3_conv1')(
            pool2)
        conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block3_conv2')(
            conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2), name='block3_pool')(conv3)

        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       name='block4_conv1')(
            pool3)
        conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       name='block4_conv2')(
            conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2), name='feature_map4')(conv4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       name='block5_conv1')(pool4)
        conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       name='block5_conv2')(conv5)

        return [conv1, conv2, conv3, conv4, conv5]

    def DecodeLayer(self, X):  # (conv1,conv2,conv3,drop4,drop5)

        up6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                     name='block6_conv1') \
            (UpSampling2D(size=(2, 2), name='upsampling_1')(X[4]))
        merge6 = concatenate([X[3], up6], axis=-1, name='concat_1')
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       name='block6_conv2')(
            merge6)
        conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       name='block6_conv3')(
            conv6)

        up7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                     name='block7_conv1') \
            (UpSampling2D(size=(2, 2), name='upsampling_2')(conv6))
        merge7 = concatenate([X[2], up7], axis=-1, name='concat_2')
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       name='block7_conv2')(
            merge7)
        conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       name='block7_conv3')(
            conv7)

        up8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                     name='block8_conv1') \
            (UpSampling2D(size=(2, 2), name='upsampling_3')(conv7))
        merge8 = concatenate([X[1], up8], axis=-1, name='concat_3')
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       name='block8_conv2')(
            merge8)
        conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       name='block8_conv3')(
            conv8)

        up9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='block9_conv1') \
            (UpSampling2D(size=(2, 2), name='upsampling_4')(conv8))
        merge9 = concatenate([X[0], up9], axis=-1, name='concat_4')
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       name='block9_conv2')(
            merge9)
        conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       name='block9_conv3')(
            conv9)
        conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal',
                       name='block9_conv4')(
            conv9)
        conv10 = Conv2D(1, 1, activation='sigmoid', name='segmentation')(conv9)

        return conv10

    def AELayer(self, X, Y, stage='train'):
        sr = Lambda(self.cal_salient_regions, output_shape=self.cal_salient_regions_output_shape, name='saliency_box')(
            Y)

        out_roi_pool = RoiPoolingConv(self.pooling_regions, self.num_rois, name='roi_pooling')([X, sr])
        out = Flatten(name='flatten')(out_roi_pool)
        out = Dense(2048, activation='relu', name='fc1')(out)
        out = Dense(1024, activation='relu', name='fc2')(out)
        out = Dense(4, activation='linear', name='offset')(out)
        if stage == 'train':
            return out
        elif stage == 'test':
            return [out, sr]

    def BuildSaliencyModel(self):
        inputs = Input((None, None, 3))
        encoded_layer = self.EncodeLayer(inputs)
        decoded_layer = self.DecodeLayer(encoded_layer)
        model_saliency = tf.keras.Model(inputs, decoded_layer)
        if self.weights is not None:
            model_saliency.load_weights(self.weights)
        return model_saliency

    def BuildModel(self):
        model_saliency = self.BuildSaliencyModel()
        if self.weights is not None:
            model_saliency.load_weights(self.weights)
        saliency_input = model_saliency.get_layer('segmentation').output
        feature_input = model_saliency.get_layer('block5_conv2').output
        ae_layers = self.AELayer(feature_input, saliency_input, self.stage)
        model_total = tf.keras.Model(model_saliency.inputs, ae_layers)
        return model_total
